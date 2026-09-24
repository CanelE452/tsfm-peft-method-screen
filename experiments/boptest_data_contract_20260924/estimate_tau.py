"""전환 직후 응답에서 시상수를 잰다. NEXT_STAGE_CONTRACT §2 의 tau0 칸을 채우는 근거.

두 방식을 함께 낸다.
  tau63  전체 변화량의 63.2% 에 처음 도달한 시각 (모델 가정 없음)
  tau_fit 1차 지수 y = y_inf + (y0-y_inf)exp(-t/tau) 최소제곱 (가정 있음, 적합도 함께 보고)
둘이 크게 다르면 1차 근사가 안 맞는다는 뜻이므로 그대로 기록한다.
"""
import csv, json
import numpy as np
from pathlib import Path

O = Path("/home/minjae/Documents/github/tsfm-peft-method-screen/results/boptest_data_contract_20260924")
SIG = ["reaTSup_y", "reaTRet_y", "reaTZon_y", "reaPHeaPum_y", "reaQHeaPumCon_y"]

def fit_exp(t, y):
    """tau 격자 탐색 + y0,y_inf 선형 최소제곱. scipy 의존 없음."""
    best = None
    for tau in np.geomspace(10, 40000, 400):
        b = np.exp(-t/tau)
        A = np.vstack([np.ones_like(b), b]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ coef
        ss = float(r @ r)
        if best is None or ss < best[0]: best = (ss, tau, coef)
    ss, tau, coef = best
    sst = float(((y - y.mean())**2).sum())
    return tau, (1 - ss/sst if sst > 0 else float("nan")), float(coef[0]), float(coef[0]+coef[1])

def main():
    rows = list(csv.DictReader(open(O/"transient_response.csv")))
    t = np.array([float(r["time"]) for r in rows]); t -= t[0]
    u = np.array([float(r["oveHeaPumY_u"]) for r in rows])
    d = np.diff(u)
    trans = [{"i": int(i+1), "t_s": float(t[i+1]), "t_h": round(float(t[i+1])/3600, 2),
              "dir": "ON" if d[i] > 0 else "OFF", "from": float(u[i]), "to": float(u[i+1])}
             for i in np.where(np.abs(d) > 0.5)[0]]
    out = {"n_transitions": len(trans), "transitions": trans,
           "sample_interval_s": float(t[1]-t[0]), "signals": {}}

    for s in SIG:
        y = np.array([float(r[s]) for r in rows])
        per = []
        for k, tr in enumerate(trans):
            i0 = tr["i"]
            i1 = trans[k+1]["i"] if k+1 < len(trans) else len(t)
            seg_t = t[i0:i1] - t[i0]; seg_y = y[i0:i1]
            if len(seg_t) < 10: continue
            y0, yinf = float(seg_y[0]), float(seg_y[-1])
            span = yinf - y0
            rec = {"transition_h": tr["t_h"], "dir": tr["dir"], "n_points": int(len(seg_t)),
                   "y_start": y0, "y_end": yinf, "span": span}
            if abs(span) > 1e-6:
                target = y0 + 0.632*span
                cross = np.where((seg_y - target)*np.sign(span) >= 0)[0]
                rec["tau63_s"] = float(seg_t[cross[0]]) if len(cross) else None
                tau, r2, a, b = fit_exp(seg_t, seg_y)
                rec.update({"tau_fit_s": float(tau), "fit_r2": round(float(r2), 4),
                            "fit_y_inf": a, "fit_y0": b})
            else:
                rec["note"] = "변화 없음"
            per.append(rec)
        taus = [p["tau63_s"] for p in per if p.get("tau63_s")]
        fits = [p["tau_fit_s"] for p in per if p.get("tau_fit_s")]
        out["signals"][s] = {"per_transition": per,
                             "tau63_median_s": float(np.median(taus)) if taus else None,
                             "tau63_min_s": float(np.min(taus)) if taus else None,
                             "tau63_max_s": float(np.max(taus)) if taus else None,
                             "tau_fit_median_s": float(np.median(fits)) if fits else None}
        if taus:
            print(f"  {s:18} tau63 중앙값 {np.median(taus):7.0f}s ({np.median(taus)/60:5.1f}분)  "
                  f"범위 {np.min(taus):.0f}~{np.max(taus):.0f}s  fit중앙값 {np.median(fits):.0f}s", flush=True)
    json.dump(out, open(O/"TAU_ESTIMATES.json", "w"), indent=2, ensure_ascii=False)
    print("  ==> TAU_ESTIMATES.json 저장", flush=True)
    return True

if __name__ == "__main__":
    import sys; sys.exit(0 if main() else 1)
