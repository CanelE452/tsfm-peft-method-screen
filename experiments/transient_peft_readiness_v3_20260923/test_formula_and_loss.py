"""§5 수식 단위검사 + §6.4 native loss 집계 검산. 가상 fixture 전용, 성능 실험 아님.

사용: python test_formula_and_loss.py
"""
import json, sys
from pathlib import Path
import numpy as np, torch

OUT = Path(__file__).resolve().parents[2] / "results/transient_peft_readiness_v3_20260923"
OUT.mkdir(parents=True, exist_ok=True)
LED = {"model_load": 0, "forward": 0, "backward": 0, "update": 0}
res = {"formula": {}, "loss_reduction": {}}

# ================= §5 수식 단위검사 (CPU float64, 모델 불필요) =================
def lora_out(W0, A, B, h, kappa, c):
    """out = W0 h + kappa * B diag(1+c) A h"""
    return W0 @ h + kappa * (B @ (np.diag(1.0 + c) @ (A @ h)))

d, r = 3, 2
g = np.random.default_rng(20260923)
W0 = g.normal(size=(d, d)); A = g.normal(size=(r, d)); B = g.normal(size=(d, r))
h = g.normal(size=d); kappa = 2.0 / r

# (1) B=0 -> 원래 선형층과 동일
o_B0 = lora_out(W0, A, np.zeros_like(B), h, kappa, np.zeros(r))
res["formula"]["B0_equals_W0h"] = {
    "max_abs_diff": float(np.abs(o_B0 - W0 @ h).max()), "atol": 1e-10,
    "pass": bool(np.allclose(o_B0, W0 @ h, atol=1e-10, rtol=1e-8))}

# (2) c=0, B!=0 -> 일반 LoRA 와 동일. 원래 선형층과 같지 않다 (v2 오류 D 정정)
o_c0 = lora_out(W0, A, B, h, kappa, np.zeros(r))
plain = W0 @ h + kappa * (B @ (A @ h))
res["formula"]["c0_equals_plain_lora"] = {
    "max_abs_diff": float(np.abs(o_c0 - plain).max()), "atol": 1e-10,
    "pass": bool(np.allclose(o_c0, plain, atol=1e-10, rtol=1e-8))}
res["formula"]["c0_NOT_equal_W0h"] = {
    "max_abs_diff": float(np.abs(o_c0 - W0 @ h).max()),
    "pass": bool(not np.allclose(o_c0, W0 @ h, atol=1e-10)),
    "note": "c=0 은 F0 를 보장하지 않는다. B=0 이 필요하다 (v2 문서의 오류)"}

# (3) c!=0, B!=0 -> 상태가 추가 경로에 반영
o_c1 = lora_out(W0, A, B, h, kappa, np.array([0.5, -0.3]))
res["formula"]["c_nonzero_changes_output"] = {
    "max_abs_diff_vs_c0": float(np.abs(o_c1 - o_c0).max()),
    "pass": bool(not np.allclose(o_c1, o_c0, atol=1e-10))}

# (4) 스칼라 fixture: W0=2, A=B=h=1, kappa=1, c=0 -> 3 (원래는 2)
s = lora_out(np.array([[2.0]]), np.array([[1.0]]), np.array([[1.0]]), np.array([1.0]), 1.0, np.array([0.0]))
res["formula"]["scalar_fixture"] = {
    "got": float(s[0]), "expected": 3.0, "W0h_alone": 2.0,
    "pass": bool(abs(float(s[0]) - 3.0) < 1e-12)}

print("=== §5 수식 단위검사 ===", flush=True)
for k, v in res["formula"].items():
    print(f"  {k:28s} pass={v['pass']}  {v.get('max_abs_diff', v.get('got',''))}", flush=True)

# ================= §6.4 native loss 집계 검산 =================
from chronos import Chronos2Pipeline
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu", dtype=torch.float32)
LED["model_load"] += 1
m = pipe.inner_model; m.eval()
p = m.chronos_config.input_patch_size
ctx_len, H = 4*p, p     # 배수로 두어 padding 교란 제거
rng = np.random.default_rng(20260923)
tgt = (np.sin(np.arange(ctx_len + H)/5.0)*3 + 20).astype(np.float32)
cov = rng.normal(0, 1, (2, ctx_len + H)).astype(np.float32)

def run(n_cov):
    """n_cov 개의 공변량 행을 붙였을 때 native loss"""
    ctx = torch.tensor(tgt[:ctx_len])[None, :]                 # (1, ctx)
    fut = torch.tensor(tgt[ctx_len:])[None, :]                 # (1, H)
    if n_cov:
        ctx = torch.cat([ctx, torch.tensor(cov[:n_cov, :ctx_len])], 0)
        fut_cov = torch.tensor(cov[:n_cov, ctx_len:])
    else:
        fut_cov = None
    gids = torch.zeros(ctx.shape[0], dtype=torch.long)
    ft = torch.full((ctx.shape[0], H), float("nan"))
    ft[0] = fut[0]                                             # target 행에만 정답
    with torch.no_grad():
        out = m(context=ctx, group_ids=gids,
                future_covariates=(torch.cat([torch.full((1,H), float('nan')), fut_cov],0)
                                   if fut_cov is not None else None),
                future_target=ft, num_output_patches=1)
    LED["forward"] += 1
    return float(out.loss), ctx.shape[0]

l0, n0 = run(0)
l2, n2 = run(2)
res["loss_reduction"] = {
    "rows_target_only": n0, "loss_target_only": l0,
    "rows_with_2cov": n2, "loss_with_2cov": l2,
    "ratio_l2_over_l0": l2 / l0 if l0 else None,
    "expected_if_diluted_by_rows": n0 / n2,
    "reduction_in_source": "loss.mean(dim=-1).sum(dim=-1).mean()  # horizon 평균, quantile 합, batch(행) 평균",
    "diluted": bool(abs(l2/l0 - n0/n2) < 0.05) if l0 else None}
print("\n=== §6.4 native loss 집계 검산 ===", flush=True)
print(f"  target 행만        행수 {n0}  loss {l0:.6f}", flush=True)
print(f"  target + 공변량 2행 행수 {n2}  loss {l2:.6f}", flush=True)
print(f"  비율 {l2/l0:.4f}   행수로 희석되면 기대 {n0/n2:.4f}", flush=True)
print(f"  -> {'행 수로 희석됨 (LEARNING_SCALE_CONFOUND)' if res['loss_reduction']['diluted'] else '단순 행 희석은 아님 — 추가 분석 필요'}", flush=True)
res["ledger"] = dict(LED)
json.dump(res, open(OUT/"LOSS_REDUCTION_AUDIT.json","w"), indent=2, ensure_ascii=False)
print(f"\n[saved] {OUT/'LOSS_REDUCTION_AUDIT.json'}", flush=True)
