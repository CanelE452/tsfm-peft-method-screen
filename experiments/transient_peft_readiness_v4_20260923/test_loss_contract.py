"""§6 TARGET_ROW_MEAN_CORRECTION — 고정 prediction/target/mask 로 모델과 분리해 검산.
FP64, atol 1e-10. 결과 후 tolerance 완화 금지."""
import json, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT
import loss_contract as LC

DT = torch.float64
ATOL = 1e-10

def native_rowwise(pred, tgt, tmask, quantiles):
    """native _compute_loss 의 수식을 그대로 재현 (행별 반환).
    loss = 2*|(y - q)*(1[y<=q] - tau)| * mask, horizon 평균, quantile 합"""
    y = tgt.unsqueeze(1)                       # (rows, 1, H)
    mk = tmask.unsqueeze(1)
    y = torch.where(mk > 0, y, torch.zeros_like(y))
    qs = quantiles.view(1, -1, 1)
    ql = 2 * torch.abs((y - pred) * ((y <= pred).to(DT) - qs))
    L = ql * mk
    return L.mean(dim=-1).sum(dim=-1)          # (rows,)

def main():
    torch.manual_seed(20260923)
    nq, H = 5, 4
    quantiles = torch.tensor([0.1,0.3,0.5,0.7,0.9], dtype=DT)
    tgt_row = torch.tensor([[1.0, 2.0, 3.0, 2.5]], dtype=DT)
    pred_row = torch.randn(1, nq, H, dtype=DT).sort(dim=1).values + 2.0
    T = {}; rows_report = []

    base_scalar = None; base_grad = None
    for n_cov in [0, 1, 2, 4]:
        pred = torch.cat([pred_row] + [torch.randn(1, nq, H, dtype=DT) for _ in range(n_cov)], 0)
        pred = pred.detach().requires_grad_(True)
        tgt  = torch.cat([tgt_row] + [torch.zeros(1, H, dtype=DT) for _ in range(n_cov)], 0)
        tmask= torch.cat([torch.ones(1, H, dtype=DT)] + [torch.zeros(1, H, dtype=DT) for _ in range(n_cov)], 0)

        per_row = native_rowwise(pred, tgt, tmask, quantiles)
        native = per_row.mean()                                   # native: 전체 행 평균
        corrected, info = LC.corrected_loss(native, tmask)
        ref = LC.reference_rowmean(per_row, tmask)                # 방법 B

        pred.grad = None; corrected.backward(retain_graph=True)
        g_target = pred.grad[0].clone()
        g_cov = pred.grad[1:].abs().max().item() if n_cov else 0.0
        pred.grad = None; native.backward()
        g_native_target = pred.grad[0].clone()

        if n_cov == 0:
            base_scalar = float(corrected); base_grad = g_target.clone()
        rows_report.append({
            "n_cov_rows": n_cov, "total_rows": info["total_rows"],
            "active_target_rows": info["active_target_rows"],
            "correction_factor": info["correction_factor"],
            "native_scalar": float(native), "corrected_scalar": float(corrected),
            "method_B_scalar": float(ref),
            "A_equals_B": bool(abs(float(corrected)-float(ref)) < ATOL),
            "corrected_equals_base": bool(abs(float(corrected)-base_scalar) < ATOL),
            "corrected_grad_equals_base": bool(torch.allclose(g_target, base_grad, atol=ATOL, rtol=0)),
            "native_grad_ratio_vs_base": (float(g_native_target.abs().max()/base_grad.abs().max())
                                          if base_grad.abs().max() > 0 else None),
            "zero_loss_row_grad_max": g_cov,
            "inactive_rows_are_exactly_zero": bool(float(per_row[1:].abs().max()) == 0.0) if n_cov else True})
        print(f"  cov {n_cov}: native {float(native):.8f}  corrected {float(corrected):.8f}  "
              f"factor {info['correction_factor']:.1f}  A==B {rows_report[-1]['A_equals_B']}", flush=True)

    T["corrected_scalar_invariant"] = {"pass": all(r["corrected_equals_base"] for r in rows_report)}
    T["corrected_gradient_invariant"] = {"pass": all(r["corrected_grad_equals_base"] for r in rows_report)}
    T["native_dilutes_by_row_count"] = {
        "ratios": [r["native_grad_ratio_vs_base"] for r in rows_report],
        "expected": [1/ (1+r["n_cov_rows"]) for r in rows_report],
        "pass": all(abs(r["native_grad_ratio_vs_base"] - 1/(1+r["n_cov_rows"])) < 1e-9 for r in rows_report)}
    T["zero_loss_rows_have_zero_grad"] = {"max": max(r["zero_loss_row_grad_max"] for r in rows_report),
        "pass": all(r["zero_loss_row_grad_max"] == 0.0 for r in rows_report)}
    T["method_A_equals_method_B"] = {"pass": all(r["A_equals_B"] for r in rows_report)}
    T["inactive_rows_exactly_zero_loss"] = {"pass": all(r["inactive_rows_are_exactly_zero"] for r in rows_report),
        "note": "방법 A(배수) 가 유효하려면 inactive row loss 가 정확히 0 이어야 한다"}
    n = sum(1 for v in T.values() if v["pass"])
    res = {"contract": LC.CONTRACT_NAME, "version": LC.CONTRACT_VERSION,
           "dtype": "float64", "atol": ATOL, "cases": rows_report,
           "tests": T, "summary": {"pass": n, "total": len(T)}, "all_pass": n == len(T),
           "disclaimer": ("underlying quantile loss 는 native 와 동일하고 row reduction 만 고쳤다. "
                          "'완전히 native scalar loss 와 동일' 이라고 부르지 않는다.")}
    print()
    for k,v in T.items(): print(f"  [{'PASS' if v['pass'] else 'FAIL'}] {k}", flush=True)
    json.dump(res, open(OUT/"LOSS_CONTRACT.json","w"), indent=2, ensure_ascii=False)
    return res["all_pass"]

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
