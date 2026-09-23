"""§6.2 INPUT 경로 gradient 검사 — LI 의 tau 까지 gradient 가 도달하는가.
가상 fixture 전용. 연구 fit 이 아니라 배선 검사다.

사용: python test_state_gradient.py
"""
import json, torch, numpy as np
from pathlib import Path
OUT = Path(__file__).resolve().parents[2] / "results/transient_peft_readiness_v3_20260923"
LED = {"model_load":0,"forward":0,"backward":0,"update":0}

class ResponseState(torch.nn.Module):
    """m[j,t] = exp(-dt/tau_j) * m[j,t-1] + phi_j(u_t - u_{t-1})
    FIXED: tau 는 buffer.  LEARNED: log_tau 는 Parameter (tau>0 보장)."""
    def __init__(self, tau_init, learned: bool, dt: float = 1.0):
        super().__init__()
        t = torch.tensor(tau_init, dtype=torch.float32)
        if learned: self.log_tau = torch.nn.Parameter(torch.log(t))
        else:       self.register_buffer("log_tau", torch.log(t))
        self.learned = learned; self.dt = dt
    def forward(self, u):                      # u: (T,) 명령 이력
        tau = torch.exp(self.log_tau)          # (J,)
        decay = torch.exp(-self.dt / tau)      # (J,)
        du = torch.zeros_like(u); du[1:] = u[1:] - u[:-1]
        up, dn = torch.relu(du), torch.relu(-du)     # phi: 상승·하강 분리(고정 변환)
        m = torch.zeros(len(tau), len(u), dtype=u.dtype)
        prev = torch.zeros(len(tau), dtype=u.dtype)
        cols = []
        for t_ in range(len(u)):
            prev = decay * prev + (up[t_] - dn[t_])
            cols.append(prev)
        return torch.stack(cols, dim=1)        # (J, T)

def main():
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu", dtype=torch.float32)
    LED["model_load"] += 1
    m = pipe.inner_model
    for p_ in m.parameters(): p_.requires_grad_(False)     # backbone 원 가중치 동결
    # §6: 네 arm 모두 attention LoRA A/B 는 학습 대상이다. 출력층은 대상에서 제외한다.
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                      target_modules=["self_attention.q","self_attention.k",
                                      "self_attention.v","self_attention.o"])
    m = get_peft_model(m, lcfg)
    m.print_trainable_parameters()
    base = m.base_model.model          # 실제 Chronos2Model
    p = base.chronos_config.input_patch_size
    ctx_len, H = 4*p, p
    g = np.random.default_rng(20260923)
    tgt = torch.tensor((np.sin(np.arange(ctx_len+H)/5.0)*3+20).astype(np.float32))
    u_np = np.zeros(ctx_len+H, np.float32); u_np[20:] = 1.0; u_np[50:] = 0.0   # 명령 계단
    u = torch.tensor(u_np)
    res = {}

    for name, learned in [("FI", False), ("LI", True)]:
        rs = ResponseState([4.0, 16.0], learned=learned)
        mm = rs(u)                                   # (2, T)  graph 위에서 생성
        ctx = torch.cat([tgt[:ctx_len][None,:], mm[:, :ctx_len]], 0)
        # target 행에는 미래 공변량이 없다. NaN 을 넣으면 backward 로 NaN 이 전파되므로
        # 값은 0 으로 두고 마스크로 무효화한다 (시도 2 의 NaN gradient 수정).
        fc   = torch.cat([torch.zeros(1, H), mm[:, ctx_len:]], 0)
        fcm  = torch.cat([torch.zeros(1, H), torch.ones(mm.shape[0], H)], 0)
        # target 이 없는 행은 NaN 대신 0 + mask 0 으로 넘긴다.
        # NaN 을 넘기면 _compute_loss 의 instance_norm -> arcsinh backward 가 NaN 을 만든다
        # (시도 3 에서 anomaly detection 으로 확인: AsinhBackward0, chronos_bolt.py:120).
        ft   = torch.zeros(3, H); ft[0] = tgt[ctx_len:]
        ftm  = torch.zeros(3, H); ftm[0] = 1.0
        out = m(context=ctx, group_ids=torch.zeros(3,dtype=torch.long),
                future_covariates=fc, future_covariates_mask=fcm,
                future_target=ft, future_target_mask=ftm, num_output_patches=1)
        LED["forward"] += 1
        loss = out.loss
        loss.backward(); LED["backward"] += 1
        gnorm = (float(rs.log_tau.grad.norm()) if (learned and rs.log_tau.grad is not None) else None)
        lora_g = [float(q.grad.norm()) for n_,q in m.named_parameters()
                  if q.requires_grad and q.grad is not None and "lora" in n_]
        n_lora = sum(q.numel() for n_,q in m.named_parameters() if q.requires_grad and "lora" in n_)
        m.zero_grad(set_to_none=True)
        res[name] = {
            "learned": learned,
            "state_requires_grad": bool(mm.requires_grad),
            "loss": float(loss.detach()),
            "log_tau_is_parameter": bool(isinstance(getattr(rs,'log_tau',None), torch.nn.Parameter)),
            "log_tau_grad_norm": gnorm,
            "gradient_reaches_tau": bool(gnorm is not None and gnorm > 0),
            "trainable_params_in_state": sum(q.numel() for q in rs.parameters() if q.requires_grad),
            "lora_trainable_params": n_lora,
            "lora_grad_norms_nonzero": int(sum(1 for x in lora_g if x > 0)),
            "lora_grad_count": len(lora_g)}
        print(f"  {name}: state.requires_grad={mm.requires_grad}  loss={float(loss):.6f}  "
              f"log_tau.grad_norm={gnorm}  -> gradient_reaches_tau={res[name]['gradient_reaches_tau']}", flush=True)

    # 판정
    ok_li = res["LI"]["gradient_reaches_tau"]
    ok_fi = (res["FI"]["log_tau_grad_norm"] is None) and (res["FI"]["trainable_params_in_state"] == 0)
    res["verdict"] = {
        "LI_gradient_path": "OK" if ok_li else "LI_GRADIENT_PATH_BLOCKED",
        "FI_is_truly_fixed": ok_fi,
        "note": ("LI 의 tau 까지 gradient 가 도달하고 FI 는 학습 파라미터가 0 이어야 2x2 의 F/L 축이 성립한다. "
                 "이 검사는 배선 확인이며 성능·수렴을 뜻하지 않는다.")}
    res["ledger"] = dict(LED)
    print(f"\n  판정: LI={res['verdict']['LI_gradient_path']}  FI_truly_fixed={ok_fi}", flush=True)
    json.dump(res, open(OUT/"GRADIENT_AUDIT.json","w"), indent=2, ensure_ascii=False)
    print(f"[saved] {OUT/'GRADIENT_AUDIT.json'}", flush=True)

if __name__ == "__main__":
    main()
