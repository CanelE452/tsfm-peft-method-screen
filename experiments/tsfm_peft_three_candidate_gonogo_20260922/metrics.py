"""CPU-only native64 metrics and fixed-seed paired origin-block uncertainty.

Scaling statistics and role/selection permissions belong to the caller.
No model, torch, data loader, training or experiment runner is imported.
"""

import math

import numpy as np


ARRAY_KEYS = ("pinball", "nmae", "nmse", "coverage80", "width80", "raw_crossing", "raw_mae")
QUANTILES = np.arange(1, 10, dtype=np.float64) / 10.0


def metric_arrays(q, y, sigma):
    """Return metric arrays [origin, series] after common quantile sorting.

    raw_crossing is the proportion of the 64*8 adjacent quantile pairs that
    decrease, measured before sorting. All other metrics use sorted outputs.
    nmse retains squared normalized median errors for later nRMSE aggregation.
    sigma must be the source/client's frozen TRAIN population standard deviation.
    """
    q, y, sigma = (np.asarray(value, dtype=np.float64) for value in (q, y, sigma))
    if y.ndim != 3 or y.shape[-1] != 64 or q.shape != (*y.shape, 9):
        raise ValueError("Expected q[O,S,64,9] and y[O,S,64].")
    if not y.shape[0] or not y.shape[1] or sigma.shape != (y.shape[1],):
        raise ValueError("Nonempty origins/series and sigma[S] required.")
    if not all(np.isfinite(value).all() for value in (q, y, sigma)) or (sigma <= 0).any():
        raise ValueError("Inputs must be finite and sigma must be positive.")
    crossing = (np.diff(q, axis=-1) < 0).mean(axis=(2, 3))
    ordered = np.sort(q, axis=-1)
    error = y[..., None] - ordered
    scale = sigma[None, :, None]
    twice_pinball = 2.0 * np.maximum(QUANTILES * error, (QUANTILES - 1.0) * error)
    median_error = y - ordered[..., 4]
    return {
        "pinball": (twice_pinball / scale[..., None]).mean(axis=(2, 3)),
        "nmae": (np.abs(median_error) / scale).mean(axis=2),
        "nmse": ((median_error / scale) ** 2).mean(axis=2),
        "coverage80": ((y >= ordered[..., 0]) & (y <= ordered[..., 8])).mean(axis=2),
        "width80": ((ordered[..., 8] - ordered[..., 0]) / scale).mean(axis=2),
        "raw_crossing": crossing,
        "raw_mae": np.abs(median_error).mean(axis=2),
    }


def summarize(arrays):
    """Return JSON-compatible macro and per-series metrics.

    by_series maps each metric name to a list in the original series order.
    nRMSE = mean_series sqrt(mean_origin(nmse)); it is not sqrt(macro nmse)
    and not mean_origin sqrt(nmse). Call separately per seed, then average
    seed scores; this function does not pool or ensemble seed predictions.
    """
    if set(arrays) != set(ARRAY_KEYS):
        raise ValueError(f"Expected exactly these array keys: {ARRAY_KEYS}")
    converted = {key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()}
    shape = converted["pinball"].shape
    if len(shape) != 2 or not all(shape):
        raise ValueError("Metric arrays must have nonempty [O,S] axes.")
    if any(value.shape != shape or not np.isfinite(value).all() for value in converted.values()):
        raise ValueError("Metric arrays must have identical shapes and finite values.")
    if any((value < 0).any() for value in converted.values()):
        raise ValueError("Metrics cannot be negative.")
    for key in ("coverage80", "raw_crossing"):
        if (converted[key] > 1).any():
            raise ValueError(f"{key} must be a proportion in [0,1].")
    per_series = {key: value.mean(axis=0) for key, value in converted.items()}
    per_series["nrmse"] = np.sqrt(per_series["nmse"])
    return {"macro": {key: float(value.mean()) for key, value in per_series.items()},
            "by_series": {key: value.tolist() for key, value in per_series.items()},
            "n_origins": shape[0], "n_series": shape[1]}


def bootstrap_effect(baseline_pinball, candidate_pinball, seed=92200, reps=2000, block=56):
    """Paired noncircular moving-origin-block bootstrap of fixed-seed scores.

    Both inputs must be [seeds,O,S] in exactly matching seed/origin/series order.
    A fixed baseline can be broadcast by the caller to each candidate seed.
    To obtain one seed's interval, call with slices [i:i+1].

    The same sampled origin indices retain every series and every seed; seed
    scores are averaged before resampling. Horizon is already aggregated in
    each origin/series pinball. No optimizer-seed or series resampling occurs.
    Default 56 consecutive 6-hour origins spans 14 days. Caller verifies the
    actual origin spacing. Blocks may overlap, cannot wrap, and are concatenated
    then truncated to O observations, preserving the moving-block convention.

    Positive difference/gain favors the candidate. ci_difference is in score
    units; ci_gain_pct is in percentage points. Zero denominators produce None
    for the affected percentage summaries; no bootstrap draws are discarded.
    """
    baseline = np.asarray(baseline_pinball, dtype=np.float64)
    candidate = np.asarray(candidate_pinball, dtype=np.float64)
    if baseline.ndim != 3 or candidate.shape != baseline.shape or not all(baseline.shape):
        raise ValueError("Expected matching nonempty [seeds,O,S] pinball arrays.")
    if not all(np.isfinite(value).all() and (value >= 0).all() for value in (baseline, candidate)):
        raise ValueError("Pinball arrays must be finite and nonnegative.")
    if not isinstance(reps, (int, np.integer)) or reps < 1:
        raise ValueError("reps must be a positive integer.")
    if not isinstance(block, (int, np.integer)) or block < 1 or baseline.shape[1] < block:
        raise ValueError("block must be a positive integer no larger than O; it is never silently shortened.")
    n_seeds, n_origins, n_series = baseline.shape
    baseline_by_origin = baseline.mean(axis=(0, 2))
    candidate_by_origin = candidate.mean(axis=(0, 2))
    difference_by_origin = baseline_by_origin - candidate_by_origin
    baseline_mean = float(baseline_by_origin.mean())
    candidate_mean = float(candidate_by_origin.mean())
    difference = float(difference_by_origin.mean())
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_origins - block + 1, size=(reps, math.ceil(n_origins / block)))
    indices = (starts[..., None] + np.arange(block)).reshape(reps, -1)[:, :n_origins]
    bootstrap_baseline = baseline_by_origin[indices].mean(axis=1)
    bootstrap_difference = difference_by_origin[indices].mean(axis=1)
    ci_difference = np.quantile(bootstrap_difference, [0.025, 0.975], method="linear").tolist()
    gain_ci = None
    if (bootstrap_baseline > 0).all():
        gain_draws = 100.0 * bootstrap_difference / bootstrap_baseline
        gain_ci = np.quantile(gain_draws, [0.025, 0.975], method="linear").tolist()
    seed_effects = []
    for index in range(n_seeds):
        a, b = float(baseline[index].mean()), float(candidate[index].mean())
        seed_effects.append({"seed_index": index, "baseline_mean": a, "candidate_mean": b,
                             "difference": a - b, "gain_pct": 100.0 * (a - b) / a if a > 0 else None})
    return {"baseline_mean": baseline_mean, "candidate_mean": candidate_mean,
            "difference": difference, "gain_pct": 100.0 * difference / baseline_mean if baseline_mean > 0 else None,
            "ci_difference": ci_difference, "ci_gain_pct": gain_ci,
            "seed_effects": seed_effects, "n_seeds": n_seeds, "n_origins": n_origins,
            "n_series": n_series, "bootstrap_seed": int(seed), "reps": int(reps), "block": int(block),
            "zero_denominator_draws": int((bootstrap_baseline == 0).sum()),
            "positive_favors": "candidate", "method": "paired_non_circular_moving_block_percentile",
            "quantile_method": "linear", "aggregation": "equal_series_and_fixed_seed_score_mean",
            "scope": "origin uncertainty conditional on fixed seeds and series; not optimizer-population uncertainty"}
