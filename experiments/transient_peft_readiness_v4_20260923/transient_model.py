"""TransientPEFTModel — v3 의 P1~P5 를 고친 명시적 wrapper. 배선 검사 전용, 성능 실험 아님.

P1 고침: 모듈을 클래스(TimeSelfAttention)로 고르고, 축을 이름으로 안다. shape 매칭 금지.
P2 고침: c 를 task 별로 만들어 각 row 의 group_id 로 lookup 한다.
P4 고침: Modulator 는 bias 없음 + token_valid_mask 를 곱한다.
P5 고침: 모든 arm 이 raw command 를 받는다 (fixture 쪽 책임, 여기선 계약만 강제).
"""
import torch, torch.nn as nn
from chronos.chronos2.layers import TimeSelfAttention, GroupSelfAttention

class ShapeMismatch(RuntimeError):
    """silent skip 금지 (v4 §2). 축이 안 맞으면 즉시 중단한다."""

class ResponseState(nn.Module):
    """m[j,t] = exp(-dt/tau_j)*m[j,t-1] + phi_j(u_t - u_{t-1}).  FIXED: buffer / LEARNED: Parameter"""
    def __init__(self, tau_init, learned: bool, dt: float = 1.0):
        super().__init__()
        t = torch.tensor(tau_init, dtype=torch.float32)
        if learned: self.log_tau = nn.Parameter(torch.log(t))
        else:       self.register_buffer("log_tau", torch.log(t))
        self.learned = learned; self.dt = dt
    def forward(self, u):                      # (T,) -> (J, T)
        tau = torch.exp(self.log_tau); decay = torch.exp(-self.dt / tau)
        du = torch.zeros_like(u); du[1:] = u[1:] - u[:-1]
        sig = torch.relu(du) - torch.relu(-du)
        prev = torch.zeros(len(tau), dtype=u.dtype); cols = []
        for t_ in range(len(u)):
            prev = decay * prev + sig[t_]; cols.append(prev)
        return torch.stack(cols, 1)

class Modulator(nn.Module):
    """m -> c.  bias 를 두지 않는다 (P4): m=0 인 토큰에서 c=0 이 학습 후에도 유지돼야 한다."""
    def __init__(self, J, r):
        super().__init__()
        self.lin = nn.Linear(J, r, bias=False)
        nn.init.zeros_(self.lin.weight)
    def forward(self, m):                      # (..., J) -> (..., r)
        return self.lin(m)

def patch_aggregate(m_raw, obs_mask, p, n_ctx_patch, n_out_patch, ctx_len, H):
    """마지막 '유효' 시점 상태로 패치 집계. obs_mask 를 실제 인자로 받는다 (P4).
    반환 (state (J, n_tok), token_valid (n_tok,)).  토큰 = [ctx patches][REG][out patches]"""
    J = m_raw.shape[0]; n_tok = n_ctx_patch + 1 + n_out_patch
    pad = n_ctx_patch * p - ctx_len
    st = torch.zeros(J, n_tok, dtype=m_raw.dtype)
    valid = torch.zeros(n_tok, dtype=m_raw.dtype)
    for k in range(n_ctx_patch):
        s, e = max(k*p - pad, 0), min((k+1)*p - pad, ctx_len)
        if e <= s: continue                                   # 전부 padding -> 중립
        idx = [i for i in range(s, e) if obs_mask[i] > 0]
        if not idx: continue                                  # 전부 masked -> 중립
        st[:, k] = m_raw[:, idx[-1]]; valid[k] = 1.0
    # REG (index n_ctx_patch) 는 항상 중립
    for k in range(n_out_patch):
        s = ctx_len + k*p; e = min(ctx_len + (k+1)*p, ctx_len + H)
        if e <= s: continue                                   # H 밖 -> 중립
        idx = [i for i in range(s, e) if obs_mask[i] > 0]
        if not idx: continue
        st[:, n_ctx_patch+1+k] = m_raw[:, idx[-1]]; valid[n_ctx_patch+1+k] = 1.0
    return st, valid

def time_attention_lora_modules(peft_model):
    """클래스로 TimeSelfAttention 하위 LoRA Linear 만 고른다 (P1). 문자열 매칭 금지."""
    import peft.tuners.lora as L
    base = peft_model.base_model.model
    time_names, group_names = set(), set()
    for name, mod in base.named_modules():
        if isinstance(mod, TimeSelfAttention):
            for pn, pm in mod.named_modules():
                if isinstance(pm, L.Linear): time_names.add(f"{name}.{pn}")
        elif isinstance(mod, GroupSelfAttention):
            for pn, pm in mod.named_modules():
                if isinstance(pm, L.Linear): group_names.add(f"{name}.{pn}")
    return time_names, group_names

class TransientPEFTModel(nn.Module):
    """peft_model + state generator + modulator 를 소유하고 lifecycle 을 명시한다 (v4 §9)."""
    def __init__(self, peft_model, state_gen, modulator=None, rank=8):
        super().__init__()
        self.peft_model = peft_model
        self.state_gen = state_gen
        self.modulator = modulator
        self.rank = rank
        self._holder = {}
        self._audit = []
        self.time_names, self.group_names = time_attention_lora_modules(peft_model)
        self.hook_calls = 0
        self._handles = self._install()

    def _install(self):
        """expected time module 전부에 hook 을 건다. 커버리지는 호출자가 assert 한다."""
        import peft.tuners.lora as L
        base = self.peft_model.base_model.model
        named = dict(base.named_modules())
        handles = []
        for nm in sorted(self.time_names):
            mod = named[nm]
            A = mod.lora_A["default"]
            handles.append(A.register_forward_hook(self._mk(nm)))
        return handles

    def _mk(self, module_name):
        def f(mod, inp, out):
            c = self._holder.get("c_rows")          # (rows, n_tok, r)
            if c is None: return out
            # TimeSelfAttention 입력은 (rows, tokens, d) 이므로 A out 은 (rows, tokens, r)
            if out.dim() != 3 or out.shape[0] != c.shape[0] or out.shape[1] != c.shape[1] or out.shape[2] != c.shape[2]:
                raise ShapeMismatch(
                    f"{module_name}: A out {tuple(out.shape)} vs c {tuple(c.shape)}. "
                    "축이 맞지 않으면 조용히 건너뛰지 않는다 (v4 §2).")
            self.hook_calls += 1
            g = self._holder.get("group_ids")
            self._audit.append({
                "module": module_name,
                "a_out": list(out.shape), "c": list(c.shape),
                "n_rows": int(c.shape[0]), "n_tokens": int(c.shape[1]),
                "group_ids": [int(x) for x in g] if g is not None else None,
                # token index -> 그 토큰의 row별 |c| 최대 (v4 §4 debug audit)
                "c_absmax_by_token": [float(c[:, t].abs().max()) for t in range(c.shape[1])],
                "c_absmax_by_row": [float(c[r].abs().max()) for r in range(c.shape[0])],
                "c_absmax": float(c.abs().max())})
            return out * (1.0 + c)
        return f

    def build_c_rows(self, m_task, token_valid, group_ids):
        """task state -> row state (P2). 각 row 는 자기 group_id 의 state 를 받는다.
        m_task (n_tasks, J, n_tok) / token_valid (n_tasks, n_tok) / group_ids (rows,)"""
        uniq = sorted(set(int(g) for g in group_ids))
        assert len(uniq) == m_task.shape[0], f"group {uniq} vs m_task {m_task.shape[0]}"
        lut = {g: i for i, g in enumerate(uniq)}
        rows = []
        for g in group_ids:
            i = lut[int(g)]
            mt = m_task[i].transpose(0, 1)                    # (n_tok, J)
            c = self.modulator(mt) if self.modulator is not None else torch.zeros(mt.shape[0], self.rank)
            rows.append(c * token_valid[i][:, None])          # P4: 중립 토큰 강제 0
        return torch.stack(rows, 0)                            # (rows, n_tok, r)

    def set_state(self, c_rows, group_ids=None):
        self._holder["c_rows"] = c_rows
        if group_ids is not None: self._holder["group_ids"] = group_ids
    def clear_state(self): self._holder.clear()                # stale holder 방지 (v4 §9)
    def audit(self): return list(self._audit)
    def reset_audit(self): self._audit = []; self.hook_calls = 0
    def remove_hooks(self):
        for h in self._handles: h.remove()
        self._handles = []
