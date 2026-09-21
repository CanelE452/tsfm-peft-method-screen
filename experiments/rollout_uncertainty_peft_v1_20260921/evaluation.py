"""Fixed-contract evaluation; callers enforce CAL-only fitting and TEST sealing.

Inputs use [origin, series, lead, quantile] with nine quantiles and 256 leads.
PCE and RMSE aggregate within each series before equal-series averaging.
"""

import math

import numpy as np
import pandas as pd


QUANTILES = np.arange(1, 10, dtype=np.float64) / 10
ALPHAS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
BETAS = (-0.5, -0.25, 0.0, 0.25, 0.5)
WINDOWS = {
    "full_256": (0, 256),
    "tail_129_256": (128, 256),
    **{f"block_{i + 1}": (64 * i, 64 * (i + 1)) for i in range(4)},
    **{f"prefix_{end}": (0, end) for end in (64, 128, 192, 256)},
}


def _validate(y, q, sigma):
    y, q, sigma = np.asarray(y), np.asarray(q), np.asarray(sigma)
    if y.ndim != 3 or y.shape[-1] != 256 or q.shape != (*y.shape, 9):
        raise ValueError("Expected y[O,S,256] and q[O,S,256,9].")
    if not y.shape[0] or not y.shape[1] or sigma.shape != (y.shape[1],):
        raise ValueError("Nonempty origins/series and sigma[S] required.")
    if not all(np.isfinite(x).all() for x in (y, q, sigma)) or (sigma <= 0).any():
        raise ValueError("Inputs must be finite and TRAIN sigma must be positive.")
    return y, q, sigma


def scaled_pinball(y, q, sigma):
    """Mean twice-pinball, equally weighted over O/S/H/nine quantiles."""
    error = np.asarray(y, dtype=np.float64)[..., None] - np.asarray(q, dtype=np.float64)
    loss = 2 * np.maximum(QUANTILES * error, (QUANTILES - 1) * error)
    return float(np.mean(loss / np.asarray(sigma)[None, :, None, None]))


def fit_affine(y, q, sigma):
    """Fit the 35-point grid independently per block on CALIBRATION only.

    The caller supplies ordered final outputs. No feedback is changed. Exact
    score ties use squared Euclidean distance from identity, alpha, |beta|,
    and then signed beta solely to resolve the remaining otherwise equal tie.
    """
    y, q, sigma = _validate(y, q, sigma)
    if (np.diff(q, axis=-1) < 0).any():
        raise ValueError("Affine fitting requires ordered final quantiles.")
    blocks = []
    for block in range(4):
        sl = slice(block * 64, (block + 1) * 64)
        base = q[:, :, sl].astype(np.float64)
        median = base[..., 4:5]
        candidates = []
        for alpha in ALPHAS:
            for beta in BETAS:
                corrected = base if alpha == 1 and beta == 0 else (
                    median + beta * sigma[None, :, None, None] + alpha * (base - median)
                )
                candidates.append({"alpha": alpha, "beta": beta,
                                   "scaled_pinball": scaled_pinball(y[:, :, sl], corrected, sigma)})
        winner = min(candidates, key=lambda c: (
            c["scaled_pinball"], (c["alpha"] - 1) ** 2 + c["beta"] ** 2,
            c["alpha"], abs(c["beta"]), c["beta"],
        ))
        blocks.append({"block": block + 1, "lead_start": block * 64 + 1,
                       "lead_end": (block + 1) * 64, **winner, "grid_scores": candidates})
    return {"role_required": "CALIBRATION", "criterion": "mean_train_scaled_twice_pinball",
            "quantile_input": "ordered_final_output", "feedback_changed": False,
            "tie_break": "squared_identity_distance,alpha,abs(beta),beta",
            "alpha": [b["alpha"] for b in blocks], "beta": [b["beta"] for b in blocks],
            "blocks": blocks}


def apply_affine(q, sigma, params):
    q, sigma = np.asarray(q), np.asarray(sigma)
    if q.ndim != 4 or q.shape[-2:] != (256, 9) or sigma.shape != (q.shape[1],):
        raise ValueError("Expected q[O,S,256,9], sigma[S].")
    if len(params["alpha"]) != 4 or len(params["beta"]) != 4:
        raise ValueError("Four alpha/beta parameters are required.")
    # Float64 retains fit precision. The identity branch avoids cancellation.
    out = q.astype(np.float64, copy=True)
    for block, (alpha, beta) in enumerate(zip(params["alpha"], params["beta"])):
        if alpha <= 0 or not np.isfinite([alpha, beta]).all():
            raise ValueError("Positive finite alpha and finite beta required.")
        if alpha == 1 and beta == 0:
            continue
        sl = slice(block * 64, (block + 1) * 64)
        base = q[:, :, sl].astype(np.float64)
        median = base[..., 4:5]
        out[:, :, sl] = median + beta * sigma[None, :, None, None] + alpha * (base - median)
    return out


def _metrics(y, q, sigma, axes, raw_q):
    """Reduce only origin/lead axes, retaining the series dimension."""
    y, q = y.astype(np.float64), q.astype(np.float64)
    error = y[..., None] - q
    scale = sigma[None, :, None]
    pinball = (2 * np.maximum(QUANTILES * error, (QUANTILES - 1) * error)).mean(-1) / scale
    median_error = y - q[..., 4]
    coverage = ((y >= q[..., 0]) & (y <= q[..., 8])).mean(axis=axes)
    quantile_coverage = (y[..., None] <= q).mean(axis=axes)
    return {
        "scaled_pinball": pinball.mean(axis=axes),
        "pce": np.abs(quantile_coverage - QUANTILES).mean(-1),
        "coverage80": coverage,
        "coverage_deficit80": 0.8 - coverage,
        "scaled_width80": ((q[..., 8] - q[..., 0]) / scale).mean(axis=axes),
        "scaled_mae": (np.abs(median_error) / scale).mean(axis=axes),
        "raw_mae": np.abs(median_error).mean(axis=axes),
        "scaled_rmse": np.sqrt(((median_error / scale) ** 2).mean(axis=axes)),
        "raw_quantile_crossing_rate": (np.diff(raw_q, axis=-1) < 0).mean(-1).mean(axis=axes),
        "raw_quantile_crossing_any_rate": (np.diff(raw_q, axis=-1) < 0).any(-1).mean(axis=axes),
    }


def score_tables(y, q, sigma, source, method, seed, variant, origins, ids, *, raw_q=None):
    """Return series_block, origin, lead and summary DataFrames.

    variant is a label (raw/ordered/affine); this function never sorts or fits.
    Pass pre-sort raw_q for crossing statistics on ordered/affine outputs.
    Origin rows average per-series horizon metrics; lead rows retain each
    series. Summary recomputes PCE/RMSE across all origins within each series.
    Duplicate full_256/prefix_256 and block_1/prefix_64 rows are intentional.
    """
    y, q, sigma = _validate(y, q, sigma)
    origins, ids = list(origins), list(ids)
    if len(origins) != y.shape[0] or len(ids) != y.shape[1]:
        raise ValueError("Origin/series labels must match array axes.")
    crossing_reference = "pre_sort_raw_q" if raw_q is not None else "input_q"
    raw_q = q if raw_q is None else np.asarray(raw_q)
    if raw_q.shape != q.shape or not np.isfinite(raw_q).all():
        raise ValueError("raw_q must be finite and match q.")
    common = {"source": source, "method": method, "seed": seed, "variant": variant,
              "crossing_reference": crossing_reference}
    series_rows, origin_rows, summary_rows = [], [], []
    for window, (start, end) in WINDOWS.items():
        yw, qw, rw = y[:, :, start:end], q[:, :, start:end], raw_q[:, :, start:end]
        metadata = {**common, "window": window, "lead_start": start + 1, "lead_end": end}
        by_series = _metrics(yw, qw, sigma, (0, 2), rw)
        for i, series_id in enumerate(ids):
            series_rows.append({**metadata, "series_id": series_id, "n_origins": len(origins),
                                **{key: float(value[i]) for key, value in by_series.items()}})
        summary_rows.append({**metadata, "n_origins": len(origins), "n_series": len(ids),
                             **{key: float(value.mean()) for key, value in by_series.items()}})
        by_origin = {key: value.mean(axis=1) for key, value in _metrics(yw, qw, sigma, (2,), rw).items()}
        for i, origin in enumerate(origins):
            origin_rows.append({**metadata, "origin": origin, "n_series": len(ids),
                                **{key: float(value[i]) for key, value in by_origin.items()}})
    by_lead = _metrics(y, q, sigma, (0,), raw_q)
    lead_rows = [{**common, "series_id": series_id, "lead": lead + 1,
                  "block": lead // 64 + 1, "n_origins": len(origins),
                  **{key: float(value[i, lead]) for key, value in by_lead.items()}}
                 for i, series_id in enumerate(ids) for lead in range(256)]
    return {"series_block": pd.DataFrame(series_rows), "origin": pd.DataFrame(origin_rows),
            "lead": pd.DataFrame(lead_rows), "summary": pd.DataFrame(summary_rows)}


def bootstrap_effect(a, b, seed=1701, reps=2000):
    """Paired noncircular moving blocks of seven ordered origins.

    a/b are equal-series/equal-horizon origin scores. Positive a-b favors b.
    To report the fixed two-repeat mean, first average corresponding origin
    scores across the two repeat seeds, then call this function. No seed or
    series resampling occurs. Origins with gaps must be reviewed by caller.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 7 or reps < 1:
        raise ValueError("Paired origin vectors with at least 7 origins and positive reps required.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Scores must be finite.")
    n = len(a)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - 7 + 1, size=(reps, math.ceil(n / 7)))
    indices = (starts[..., None] + np.arange(7)).reshape(reps, -1)[:, :n]
    difference = a - b
    draws = difference[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975], method="linear")
    baseline = float(a.mean())
    effect = float(difference.mean())
    return {"a_mean": baseline, "b_mean": float(b.mean()), "effect_a_minus_b": effect,
            "relative_gain_percent": 100 * effect / baseline if baseline != 0 else None,
            "ci95_low": float(low), "ci95_high": float(high), "bootstrap_seed": int(seed),
            "bootstrap_reps": int(reps), "block_origins": 7, "n_origins": n,
            "method": "paired_non_circular_moving_block_percentile",
            "positive_favors": "b", "quantile_method": "linear",
            "scope": "origin uncertainty conditional on fixed seeds and selected series"}
