"""Core library for tail-complexity validation TCV-1 (contract 00_MASTER_CLI_tail_complexity_v1_20260924.txt).

Difficulty estimators (contract section 5), synthetic generator (4.1), out-of-grid extensions (6.2),
scores (6.3) and statistics (6.4). No TEST value is used by anything in this module except `pinball`,
which is called by the scoring step after SEAL.
"""
from __future__ import annotations
import hashlib, json, math, time
from pathlib import Path
import numpy as np
from scipy.stats import genpareto, norm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results'/'tail_complexity_v1_20260924'
CACHE = ROOT/'.cache'/'tail_complexity_v1_20260924'
SCRATCH = Path(r'C:/Users/User/AppData/Local/Temp/claude/E--CODING-proj-hierarchical-tsfm-peft/68839535-3971-4221-82f1-10ee16fe17b4/scratchpad/tcv1')
for p in (OUT, CACHE, SCRATCH): p.mkdir(parents=True, exist_ok=True)

TAUS = [0.9, 0.95, 0.99, 0.995, 0.999]
GRID_TAUS = [0.9, 0.95, 0.99]           # inside the Chronos-2 training grid
OUT_TAUS = [0.995, 0.999]               # outside the grid: extensions apply here only
SEED_BASE = 20260924
PHIS = [0.5, 0.9]
EPSS = ['gauss', 't6', 't3.5']
GARCHS = ['off', 'on']
CONDITIONS = [f'phi{p}_{e}_g{g}' for p in PHIS for e in EPSS for g in GARCHS]
N_SERIES = 40
LEN_TOTAL, LEN_TRAIN, LEN_FIT, LEN_BURN = 4000, 3000, 2500, 1000
EVAL_STRIDE = 5
CONTEXT = 512
U_Q = 0.95                              # threshold quantile for xi/theta (contract 5.2/5.3)
MIN_EXCEEDANCES = 50
MIN_EXC_EXTENSION = 20                  # EVT-S fit on CAL (contract 6.2 is silent; CAL 500 x 5% = 25)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float)+'\n', encoding='utf-8')


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()


# ----------------------------------------------------------------- synthetic data (contract 4.1)
def standardized_innovations(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == 'gauss': return rng.standard_normal(n)
    nu = 6.0 if kind == 't6' else 3.5
    return rng.standard_t(nu, n)/math.sqrt(nu/(nu-2))


def simulate(phi: float, eps_kind: str, garch: str, n: int, seed: int, keep_state=False):
    """y_t = sin(2 pi t / 24) + x_t, x_t = phi x_{t-1} + sigma_t eps_t, GARCH(1,1) with unit unconditional variance."""
    rng = np.random.default_rng(seed)
    a, b = (0.10, 0.85) if garch == 'on' else (0.0, 0.0)
    omega = 1.0 - a - b
    total = n + LEN_BURN
    eps = standardized_innovations(eps_kind, total, rng)
    sig2 = np.empty(total); shock = np.empty(total); x = np.empty(total)
    sig2[0] = 1.0; shock[0] = math.sqrt(sig2[0])*eps[0]; x[0] = shock[0]
    for t in range(1, total):
        sig2[t] = omega + a*shock[t-1]**2 + b*sig2[t-1] if (a or b) else 1.0
        shock[t] = math.sqrt(sig2[t])*eps[t]
        x[t] = phi*x[t-1] + shock[t]
    idx = np.arange(total) - LEN_BURN          # seasonal phase is defined on the retained index (t = 0 after burn-in)
    y = np.sin(2*np.pi*idx/24) + x
    out = dict(y=y[LEN_BURN:].astype(np.float64), x=x[LEN_BURN:].astype(np.float64))
    if keep_state:
        out.update(sigma2=sig2[LEN_BURN:].astype(np.float64), eps=eps[LEN_BURN:].astype(np.float64),
                   sigma2_next=np.r_[sig2[LEN_BURN+1:], np.nan].astype(np.float64), a=a, b=b, omega=omega)
    return out


def innovation_quantile(kind: str, tau: float) -> float:
    from scipy.stats import t as student
    if kind == 'gauss': return float(norm.ppf(tau))
    nu = 6.0 if kind == 't6' else 3.5
    return float(student.ppf(tau, nu)/math.sqrt(nu/(nu-2)))


def oracle_quantiles(sim: dict, phi: float, eps_kind: str, origins: np.ndarray, taus=TAUS) -> np.ndarray:
    """q_tau(t+1) = sin(2 pi (t+1)/24) + phi x_t + sigma_{t+1} F_eps^{-1}(tau)  (contract 4.1)."""
    seasonal = np.sin(2*np.pi*(origins+1)/24)
    mean = seasonal + phi*sim['x'][origins]
    sigma_next = np.sqrt(sim['sigma2_next'][origins])
    z = np.array([innovation_quantile(eps_kind, t) for t in taus])
    return mean[:, None] + sigma_next[:, None]*z[None, :]


# ----------------------------------------------------------------- difficulty (contract section 5)
def dominant_period(series: np.ndarray) -> int:
    x = series - series.mean()
    n = len(x); freq = np.fft.rfftfreq(n); power = np.abs(np.fft.rfft(x))**2
    power[0] = 0.0
    k = int(np.argmax(power))
    return int(round(1/freq[k])) if freq[k] > 0 else 1


def ar_residuals(series: np.ndarray, p: int) -> np.ndarray:
    """Least-squares AR(p) with intercept, residuals for t >= p."""
    n = len(series)
    X = np.column_stack([np.ones(n-p)] + [series[p-i-1:n-i-1] for i in range(p)])
    beta, *_ = np.linalg.lstsq(X, series[p:], rcond=None)
    return series[p:] - X @ beta


def mad(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x))))


def tail_xi(r: np.ndarray, u_q: float = U_Q):
    u = float(np.quantile(r, u_q))
    exc = r[r > u] - u
    if len(exc) < MIN_EXCEEDANCES: return float('nan'), u, len(exc)
    c, loc, scale = genpareto.fit(exc, floc=0)
    return float(c), u, int(len(exc))


def theta_intervals(r: np.ndarray, u: float):
    """Ferro & Segers (2003) intervals estimator, contract 5.3."""
    idx = np.flatnonzero(r > u); N = len(idx)
    if N < 3: return float('nan')
    T = np.diff(idx)
    if T.max() <= 2: th = 2*T.sum()**2/((N-1)*np.sum(T**2))
    else: th = 2*np.sum(T-1.)**2/((N-1)*np.sum((T-1.)*(T-2.)))
    return float(min(1.0, th))


def spectral_entropy(series: np.ndarray) -> float:
    """Standard normalized spectral entropy (contract section 2 fallback; Q5 could not be verified)."""
    x = np.asarray(series, float); x = x - x.mean()
    power = np.abs(np.fft.rfft(x*np.hanning(len(x))))**2
    power = power[1:]                      # drop DC
    p = power/power.sum()
    p = p[p > 0]
    return float(-np.sum(p*np.log(p))/np.log(len(power)))


def excess_kurtosis(r: np.ndarray) -> float:
    z = (r - r.mean())/r.std(ddof=1)
    return float(np.mean(z**4) - 3.0)


def difficulty_from_series(train: np.ndarray, kind: str):
    """Contract 5.1-5.5. kind='synth' uses AR(P+1) residuals, kind='fin' uses median/MAD standardisation."""
    if kind == 'synth':
        p = dominant_period(train) + 1
        res = ar_residuals(train, p)
        r = res/mad(res)
    else:
        p = 0
        r = (train - np.median(train))/mad(train)
    xi, u, n_exc = tail_xi(r)
    th = theta_intervals(r, u)
    return dict(ar_order=p, xi=xi, theta=th, threshold=u, n_exceedances=n_exc,
                se=spectral_entropy(train), se_resid=spectral_entropy(r), kurtosis=excess_kurtosis(r))


# ----------------------------------------------------------------- extensions (contract 6.2)
def lin_extension(q: dict[float, np.ndarray], tau: float) -> np.ndarray:
    z99, z95, zt = norm.ppf(0.99), norm.ppf(0.95), norm.ppf(tau)
    return q[0.99] + (zt - z99)*(q[0.99] - q[0.95])/(z99 - z95)


def fit_evt_static(q_cal: dict[float, np.ndarray], y_cal: np.ndarray):
    """CAL-only static EVT on z = (y - q.5)/(q.9 - q.1); returns GPD parameters and the threshold."""
    scale = q_cal[0.9] - q_cal[0.1]
    ok = np.isfinite(scale) & (scale > 0)
    z = (y_cal[ok] - q_cal[0.5][ok])/scale[ok]
    if len(z) < 100: return None
    u = float(np.quantile(z, 0.95)); exc = z[z > u] - u
    # The contract fixes CAL at ~500 origins with a 0.95 threshold, so ~25 exceedances is the design's
    # own maximum; MIN_EXCEEDANCES (50, contract 5.2 for the difficulty estimator) would make EVT-S
    # unselectable by construction and contradict 6.2. The extension uses its own floor.
    if len(exc) < MIN_EXC_EXTENSION: return None
    c, loc, sc = genpareto.fit(exc, floc=0)
    return dict(u=u, c=float(c), scale=float(sc), rate=float(len(exc)/len(z)), n=int(len(z)))


def evt_extension(q: dict[float, np.ndarray], par: dict, tau: float) -> np.ndarray:
    """q_tau = q.5 + (q.9 - q.1) * z_tau with z_tau from the static GPD tail (contract 6.2)."""
    p_exceed = (1 - tau)/par['rate']
    z_tau = par['u'] + genpareto.ppf(1 - p_exceed, par['c'], loc=0, scale=par['scale'])
    return q[0.5] + (q[0.9] - q[0.1])*z_tau


def drop_horizon(arr: np.ndarray) -> np.ndarray:
    """Saved forecasts keep the prediction_length axis (..., 1, 21). Remove it explicitly: indexing the
    quantile grid on the wrong axis would silently succeed for level 0.1 (column 0)."""
    a = np.asarray(arr)
    if a.shape[-2] != 1: raise ValueError(f'expected a single-step horizon axis, got {a.shape}')
    return a[..., 0, :]


# ----------------------------------------------------------------- scores (contract 6.3)
def pinball(q: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    return ((y < q).astype(float) - tau)*(q - y)


def gap(qs_arm: float, qs_ref: float, noise_floor: float = 1e-12):
    if not np.isfinite(qs_ref) or qs_ref <= noise_floor: return float('nan')
    return float(qs_arm/qs_ref - 1.0)


# ----------------------------------------------------------------- statistics (contract 6.4)
def ols(y: np.ndarray, X: np.ndarray):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid); ss_tot = float(((y - y.mean())**2).sum())
    return beta, 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')


def bootstrap_regression(y, X0, X1, strata=None, draws=2000, seed=20260924):
    """Returns coefficient CIs for M1 and the CI of Delta R^2 = R2(M1) - R2(M0)."""
    rng = np.random.default_rng(seed); n = len(y)
    b1, r1 = ols(y, X1); _, r0 = ols(y, X0)
    coefs = np.empty((draws, X1.shape[1]+1)); dr2 = np.empty(draws)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)] if strata is not None else [np.arange(n)]
    for d in range(draws):
        idx = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in groups])
        try:
            bb, rr1 = ols(y[idx], X1[idx]); _, rr0 = ols(y[idx], X0[idx])
        except np.linalg.LinAlgError:
            bb, rr1, rr0 = np.full(X1.shape[1]+1, np.nan), np.nan, np.nan
        coefs[d] = bb; dr2[d] = rr1 - rr0
    ci = lambda a: [float(np.nanquantile(a, .025)), float(np.nanquantile(a, .975))]
    return dict(coef=[float(v) for v in b1], coef_ci=[ci(coefs[:, j]) for j in range(coefs.shape[1])],
                r2_m1=float(r1), r2_m0=float(r0), delta_r2=float(r1-r0), delta_r2_ci=ci(dr2), n=int(n), draws=int(draws))


def bootstrap_mean(values, strata=None, draws=2000, seed=20260924, statistic=np.nanmean):
    rng = np.random.default_rng(seed); values = np.asarray(values, float)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)] if strata is not None else [np.arange(len(values))]
    stats = np.empty(draws)
    for d in range(draws):
        idx = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in groups])
        stats[d] = statistic(values[idx])
    return dict(value=float(statistic(values)), ci=[float(np.nanquantile(stats, .025)), float(np.nanquantile(stats, .975))])


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3: return float('nan')
    ra = np.argsort(np.argsort(a[ok])).astype(float); rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float(ra @ rb/math.sqrt((ra @ ra)*(rb @ rb)))
