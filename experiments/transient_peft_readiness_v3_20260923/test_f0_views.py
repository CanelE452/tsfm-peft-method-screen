"""§5 모델 단위검사 — F0_RAW 대 F0_FEATURE, arm 별 초기 동일성. 가상 fixture 전용."""
import json, torch, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_state_gradient import ResponseState
OUT = Path(__file__).resolve().parents[2] / "results/transient_peft_readiness_v3_20260923"
LED = {"forward":0}

def main():
    from chronos import Chronos2Pipeline
    from peft import LoraConfig, get_peft_model
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu", dtype=torch.float32)
    m = pipe.inner_model; m.eval()
    for q in m.parameters(): q.requires_grad_(False)
    p = m.chronos_config.input_patch_size
    ctx_len, H = 4*p, p
    tgt = torch.tensor((np.sin(np.arange(ctx_len+H)/5.0)*3+20).astype(np.float32))
    u = torch.zeros(ctx_len+H); u[20:50] = 1.0
    mm = ResponseState([4.0,16.0], learned=False)(u)

    def pred(ctx, fc=None, fcm=None):
        n = ctx.shape[0]
        with torch.no_grad():
            o = m(context=ctx, group_ids=torch.zeros(n,dtype=torch.long),
                  future_covariates=fc, future_covariates_mask=fcm, num_output_patches=1)
        LED["forward"] += 1
        return o.quantile_preds[0].clone()          # target 행

    f0_raw = pred(tgt[:ctx_len][None,:])
    ctx_f = torch.cat([tgt[:ctx_len][None,:], mm[:,:ctx_len]],0)
    fc_f  = torch.cat([torch.zeros(1,H), mm[:,ctx_len:]],0)
    fcm_f = torch.cat([torch.zeros(1,H), torch.ones(2,H)],0)
    f0_feat = pred(ctx_f, fc_f, fcm_f)

    scale = float(tgt[:ctx_len].std())
    d = float((f0_raw - f0_feat).abs().max()) / max(scale, 1e-8)
    res = {"F0_RAW_vs_F0_FEATURE": {
             "normalized_max_abs_diff": d, "context_scale": scale,
             "differ": bool(d > 1e-5),
             "note": ("§2-E 확인: 입력 view 가 다르면 F0 예측도 다르다. "
                      "따라서 FI/LI 의 초기 동일성은 F0_FEATURE 기준으로만 요구한다.")}}
    print(f"  F0_RAW vs F0_FEATURE  normalized_max_diff={d:.6e}  differ={res['F0_RAW_vs_F0_FEATURE']['differ']}", flush=True)

    # LoRA B=0 일 때 각 view 의 F0 보존
    lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                      target_modules=["self_attention.q","self_attention.k",
                                      "self_attention.v","self_attention.o"])
    pm = get_peft_model(m, lcfg)     # peft 는 lora_B 를 0 으로 초기화한다
    def pred_pm(ctx, fc=None, fcm=None):
        n = ctx.shape[0]
        with torch.no_grad():
            o = pm(context=ctx, group_ids=torch.zeros(n,dtype=torch.long),
                   future_covariates=fc, future_covariates_mask=fcm, num_output_patches=1)
        LED["forward"] += 1
        return o.quantile_preds[0].clone()
    l0_raw = pred_pm(tgt[:ctx_len][None,:])
    fi_feat = pred_pm(ctx_f, fc_f, fcm_f)
    B_max = max(float(q.abs().max()) for n_,q in pm.named_parameters() if "lora_B" in n_)
    res["lora_B_init_zero"] = {"max_abs": B_max, "pass": bool(B_max == 0.0)}
    res["L0_vs_F0_RAW"]   = {"normalized_max_diff": float((l0_raw-f0_raw).abs().max())/max(scale,1e-8),
                             "atol":1e-5, "pass": bool(float((l0_raw-f0_raw).abs().max())/max(scale,1e-8) < 1e-5),
                             "note":"raw view 를 쓰는 L0/FM/LM 은 B=0 에서 F0_RAW 와 같아야 한다"}
    res["FI_vs_F0_FEATURE"]= {"normalized_max_diff": float((fi_feat-f0_feat).abs().max())/max(scale,1e-8),
                             "atol":1e-5, "pass": bool(float((fi_feat-f0_feat).abs().max())/max(scale,1e-8) < 1e-5),
                             "note":"feature view 를 쓰는 FI/LI 는 B=0 에서 F0_FEATURE 와 같아야 한다"}
    for k in ["lora_B_init_zero","L0_vs_F0_RAW","FI_vs_F0_FEATURE"]:
        print(f"  [{'PASS' if res[k]['pass'] else 'FAIL'}] {k}", flush=True)
    res["ledger"] = dict(LED)
    json.dump(res, open(OUT/"F0_VIEWS.json","w"), indent=2, ensure_ascii=False)
    print(f"[saved] {OUT/'F0_VIEWS.json'}", flush=True)

if __name__ == "__main__":
    main()
