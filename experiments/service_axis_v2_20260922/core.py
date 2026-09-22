"""v2 공통 모듈: 시뮬레이터(§7.4), ERP(§7.2), 지표(§7.5), 판정 통계(§7.6)."""
import numpy as np

# ---------- §7.4 시뮬레이터 ----------
def simulate(S, L, Y):
    """S:(N,) 발주점, L:리드타임, Y:(N,H) 실제수요.
    규약: t일 마감 발주분은 t+L+1 일의 입고 단계에서 들어온다 -> pipe 길이 L+1.
    반환 (served:(N,), demand:(N,), avg_closing:(N,))"""
    S = np.asarray(S, float); Y = np.asarray(Y, float)
    N, H = Y.shape
    oh = S.copy(); pipe = np.zeros((L + 1, N)); served = np.zeros(N); closing = np.zeros(N)
    for t in range(H):
        oh = oh + pipe[0]
        pipe = np.roll(pipe, -1, 0); pipe[-1] = 0.0
        d = Y[:, t]
        s = np.minimum(oh, d); served += s; oh = np.maximum(0.0, oh - d)
        ip = oh + pipe[:-1].sum(0)
        pipe[-1] = np.maximum(0.0, S - ip)
        closing += oh
    return served, Y.sum(1), closing / H

# ---------- §7.2 ERP ----------
def erp_residuals(mu_cal, Y_cal, P):
    """mu_cal: list of (N,) 창별 점예측 평균, Y_cal: list of (N,H).
    창마다 연속 P일 합 블록 - mu*P. 반환 (N, n_blocks_total)."""
    out = []
    for mu, Y in zip(mu_cal, Y_cal):
        H = Y.shape[1]; nb = H - P + 1
        cs = np.cumsum(np.concatenate([np.zeros((Y.shape[0], 1)), Y], 1), 1)
        blocks = cs[:, P:] - cs[:, :-P]              # (N, H-P+1)
        out.append(blocks - mu[:, None] * P)
    return np.concatenate(out, 1)

def erp_S(mu, resid, P, tau):
    """S(tau) = max(0, mu*P + Q_tau(resid))"""
    return np.maximum(0.0, mu * P + np.quantile(resid, tau, axis=1))

# ---------- §7.5 지표 ----------
def unit_metrics(served, demand, closing):
    """단위 가중 쌍: U = 총충족/총수요, I = 계열 평균보유의 합"""
    tot_d = demand.sum()
    return (served.sum() / tot_d if tot_d > 0 else 1.0), float(closing.sum())

def interp_at(U, I, target=0.90):
    """tau 곡선(U 오름차순 가정)에서 U=target 의 I 를 선형보간. 반환 (I, flag)"""
    U = np.asarray(U, float); I = np.asarray(I, float)
    hit = np.where(U >= target)[0]
    if len(hit) == 0: return np.nan, "not_reached"
    k = hit[0]
    if k == 0: return float(I[0]), "reached_at_min"
    f0, f1 = U[k - 1], U[k]
    w = (target - f0) / (f1 - f0) if f1 > f0 else 0.0
    return float(I[k - 1] + w * (I[k] - I[k - 1])), "interp"

def nonmonotone(U):
    return int((np.diff(np.asarray(U, float)) < 0).sum())

# ---------- §7.6 판정 ----------
T_CRIT_DF7 = 2.364624
def verdict(deltas, T_stock):
    """deltas: 창별 Δ_w. 반환 dict"""
    d = np.asarray(deltas, float)
    if np.isnan(d).any():
        return {"pass": False, "reason": "INCONCLUSIVE_UNREACHED", "mean": None,
                "t_low": None, "n_pos": None, "n": len(d)}
    m = d.mean(); sd = d.std(ddof=1); n = len(d)
    lo = m - T_CRIT_DF7 * sd / np.sqrt(n)
    npos = int((d > 0).sum())
    return {"pass": bool(m >= T_stock and lo > 0 and npos >= 7),
            "mean": float(m), "sd": float(sd), "t_low": float(lo),
            "n_pos": npos, "n": n, "T_stock": float(T_stock)}

# ---------- 점예측 ----------
def qmean21(Q):
    """§6.2 QMEAN21: 학습 분위수 21개를 0에서 자른 뒤 사다리꼴 적분, 양끝 상수."""
    lv = np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,
                   0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99])
    Qc = np.clip(Q, 0, None)
    lve = np.concatenate([[0.0], lv, [1.0]])
    ve = np.concatenate([Qc[..., :1], Qc, Qc[..., -1:]], -1)
    return np.clip(np.trapezoid(ve, x=lve, axis=-1), 0, None)
