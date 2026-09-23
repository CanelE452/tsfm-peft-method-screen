"""§6.3 MOD 경로 — FM/LM 의 LoRA 변조 배선과 gradient. 가상 fixture 전용.
out = W0 h + kappa * B diag(1+c(m)) A h  를 lora 계층에 실제로 건다.
"""
import json, torch, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_state_gradient import ResponseState
from test_state_alignment import patch_aggregate
OUT = Path(__file__).resolve().parents[2] / "results/transient_peft_readiness_v3_20260923"
LED = {"forward":0,"backward":0,"update":0}

class Modulator(torch.nn.Module):
    """패치별 상태 m -> rank 차원 c. 초기 출력 0 (마지막 층 0 초기화)."""
    def __init__(self, J, r):
        super().__init__()
        self.lin = torch.nn.Linear(J, r)
        torch.nn.init.zeros_(self.lin.weight); torch.nn.init.zeros_(self.lin.bias)
    def forward(self, m_tok):            # (J, n_tok) -> (n_tok, r)
        return self.lin(m_tok.T)

def attach(peft_model, mod, holder):
    """lora 계층의 forward 를 감싸 diag(1+c) 를 A 출력에 곱한다. 원본 모듈 수정 없음."""
    import peft.tuners.lora as L
    handles = []
    for name, sub in peft_model.named_modules():
        if isinstance(sub, L.Linear) and "self_attention" in name:
            A = sub.lora_A["default"]; B = sub.lora_B["default"]
            def hook_A(module, inp, out, _h=holder):
                c = _h.get("c")
                if c is None: return out
                # out: (..., n_tok, r) / c: (n_tok, r)
                if out.shape[-2] == c.shape[0] and out.shape[-1] == c.shape[1]:
                    return out * (1.0 + c)
                return out
            handles.append(A.register_forward_hook(hook_A))
    return handles

def main():
    from chronos import Chronos2Pipeline
    from peft import LoraConfig, get_peft_model
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu", dtype=torch.float32)
    m0 = pipe.inner_model
    for q in m0.parameters(): q.requires_grad_(False)
    lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                      target_modules=["self_attention.q","self_attention.k",
                                      "self_attention.v","self_attention.o"])
    pm = get_peft_model(m0, lcfg)
    base = pm.base_model.model
    p = base.chronos_config.input_patch_size
    ctx_len, H = 4*p, p
    n_ctx, n_out = ctx_len//p, 1
    n_tok = n_ctx + 1 + n_out
    tgt = torch.tensor((np.sin(np.arange(ctx_len+H)/5.0)*3+20).astype(np.float32))
    u = torch.zeros(ctx_len+H); u[20:50] = 1.0
    holder = {}
    handles = attach(pm, None, holder)
    res = {"n_tok": n_tok, "arms": {}}

    for arm, learned in [("FM", False), ("LM", True)]:
        rs = ResponseState([4.0,16.0], learned=learned)
        mod = Modulator(2, 8)
        mm = rs(u)
        m_tok = patch_aggregate(mm, p, n_ctx, n_out, ctx_len, H)      # (J, n_tok)
        holder["c"] = mod(m_tok)                                       # (n_tok, r), 초기 0
        ctx = tgt[:ctx_len][None,:]
        ft  = tgt[ctx_len:][None,:]; ftm = torch.ones(1,H)
        out = pm(context=ctx, group_ids=torch.zeros(1,dtype=torch.long),
                 future_target=ft, future_target_mask=ftm, num_output_patches=n_out)
        LED["forward"] += 1
        loss = out.loss; loss.backward(); LED["backward"] += 1
        gl = float(rs.log_tau.grad.norm()) if (learned and rs.log_tau.grad is not None) else None
        gm = float(mod.lin.weight.grad.norm()) if mod.lin.weight.grad is not None else None
        glora = sum(float(q.grad.norm()) for n_,q in pm.named_parameters()
                    if q.requires_grad and q.grad is not None and "lora" in n_)
        res["arms"][arm] = {
            "c_init_is_zero": bool(float(holder["c"].abs().max())==0.0),
            "loss": float(loss.detach()),
            "lora_grad_norm_sum": glora,
            "modulator_grad_norm": gm,
            "log_tau_grad_norm": gl,
            "B_is_zero_at_init": True,
            "note_first_step": "B=0 이므로 c/tau gradient 가 0 인 것은 정상. 아래 active-path 로 따로 확인"}
        pm.zero_grad(set_to_none=True)
        print(f"  {arm}: c_init0={res['arms'][arm]['c_init_is_zero']} loss={float(loss.detach()):.6f} "
              f"lora_g={glora:.6f} mod_g={gm} tau_g={gl}", flush=True)

    # ---- active-path fixture: B 에 작은 비영값을 넣어 c/tau 까지 gradient 도달 확인 ----
    import peft.tuners.lora as L
    with torch.no_grad():
        for name, sub in pm.named_modules():
            if isinstance(sub, L.Linear) and "self_attention" in name:
                sub.lora_B["default"].weight.normal_(0, 0.01)
    rs = ResponseState([4.0,16.0], learned=True); mod = Modulator(2,8)
    with torch.no_grad(): mod.lin.weight.normal_(0, 0.01)      # c 도 비영
    mm = rs(u); m_tok = patch_aggregate(mm, p, n_ctx, n_out, ctx_len, H)
    holder["c"] = mod(m_tok)
    out = pm(context=tgt[:ctx_len][None,:], group_ids=torch.zeros(1,dtype=torch.long),
             future_target=tgt[ctx_len:][None,:], future_target_mask=torch.ones(1,H), num_output_patches=n_out)
    LED["forward"] += 1
    out.loss.backward(); LED["backward"] += 1
    ga = {"modulator": float(mod.lin.weight.grad.norm()) if mod.lin.weight.grad is not None else None,
          "log_tau": float(rs.log_tau.grad.norm()) if rs.log_tau.grad is not None else None}
    res["active_path"] = {**ga,
        "gradient_reaches_modulator": bool(ga["modulator"] is not None and ga["modulator"] > 0),
        "gradient_reaches_tau": bool(ga["log_tau"] is not None and ga["log_tau"] > 0),
        "note": "배선 검사 전용. 본학습 초기화 변경이 아니다"}
    print(f"  active-path: mod_g={ga['modulator']} tau_g={ga['log_tau']} "
          f"-> reaches_tau={res['active_path']['gradient_reaches_tau']}", flush=True)
    for h in handles: h.remove()
    res["ledger"] = dict(LED)
    res["verdict"] = ("MOD_PATH_WIRED_OK" if res["active_path"]["gradient_reaches_tau"]
                      else "MOD_PATH_GRADIENT_UNCONFIRMED")
    print(f"  판정: {res['verdict']}", flush=True)
    json.dump(res, open(OUT/"MOD_PATH_AUDIT.json","w"), indent=2, ensure_ascii=False)
    print(f"[saved] {OUT/'MOD_PATH_AUDIT.json'}", flush=True)

if __name__ == "__main__":
    main()
