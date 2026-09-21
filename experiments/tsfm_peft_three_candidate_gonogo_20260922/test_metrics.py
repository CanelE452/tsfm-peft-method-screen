import json

import numpy as np
import pytest

from metrics import ARRAY_KEYS, bootstrap_effect, metric_arrays, summarize


def example():
    rng = np.random.default_rng(225)
    return (rng.normal(size=(3, 2, 64, 9)), rng.normal(size=(3, 2, 64)), np.array([0.5, 3.0]))


def test_independent_scalar_all_metrics_and_ordering():
    q, y, sigma = example()
    before = q.copy()
    actual = metric_arrays(q, y, sigma)
    assert set(actual) == set(ARRAY_KEYS)
    for origin in range(3):
        for series in range(2):
            total = {key: 0.0 for key in ARRAY_KEYS}
            for horizon in range(64):
                original = list(q[origin, series, horizon])
                ordered = sorted(original)
                truth = y[origin, series, horizon]
                for i in range(9):
                    tau = (i + 1) / 10.0
                    e = truth - ordered[i]
                    total["pinball"] += 2 * (tau * e if e >= 0 else (tau - 1) * e) / sigma[series] / (64 * 9)
                median_error = truth - ordered[4]
                total["nmae"] += abs(median_error) / sigma[series] / 64
                total["nmse"] += (median_error / sigma[series]) ** 2 / 64
                total["coverage80"] += float(ordered[0] <= truth <= ordered[8]) / 64
                total["width80"] += (ordered[8] - ordered[0]) / sigma[series] / 64
                total["raw_crossing"] += sum(original[j + 1] < original[j] for j in range(8)) / (64 * 8)
                total["raw_mae"] += abs(median_error) / 64
            for key, value in total.items():
                assert actual[key].shape == (3, 2)
                assert actual[key][origin, series] == pytest.approx(value, abs=1e-12)
    np.testing.assert_array_equal(q, before)


def test_reversal_preserves_scores_but_changes_raw_crossing():
    q, y, sigma = example()
    ordered = np.sort(q, axis=-1)
    ascending = metric_arrays(ordered, y, sigma)
    descending = metric_arrays(ordered[..., ::-1], y, sigma)
    for key in ARRAY_KEYS:
        if key != "raw_crossing":
            np.testing.assert_array_equal(ascending[key], descending[key])
    np.testing.assert_array_equal(ascending["raw_crossing"], np.zeros((3, 2)))
    np.testing.assert_array_equal(descending["raw_crossing"], np.ones((3, 2)))


def test_per_series_scale_invariance_and_raw_units():
    q, y, sigma = example()
    factor = np.array([5.0, 0.25])
    a = metric_arrays(q, y, sigma)
    b = metric_arrays(q * factor[None, :, None, None], y * factor[None, :, None], sigma * factor)
    for key in ARRAY_KEYS:
        np.testing.assert_allclose(b[key], a[key] * factor[None, :] if key == "raw_mae" else a[key], atol=1e-12)


def test_nrmse_roots_after_origin_average_before_series_average():
    arrays = {key: np.zeros((2, 2)) for key in ARRAY_KEYS}
    arrays["nmse"] = np.array([[0.0, 9.0], [8.0, 9.0]])
    result = summarize(arrays)
    assert result["by_series"]["nrmse"] == [2.0, 3.0]
    assert result["macro"]["nrmse"] == 2.5
    assert result["macro"]["nrmse"] != pytest.approx(np.sqrt(arrays["nmse"].mean()))
    assert result["macro"]["nrmse"] != pytest.approx(np.sqrt(arrays["nmse"]).mean())
    json.dumps(result, allow_nan=False)


def test_summarize_scalar_equal_series_means():
    q, y, sigma = example()
    arrays = metric_arrays(q, y, sigma)
    result = summarize(arrays)
    for key in ARRAY_KEYS:
        assert result["macro"][key] == pytest.approx(sum(sum(row) for row in arrays[key]) / 6)
        np.testing.assert_allclose(result["by_series"][key], arrays[key].mean(0))
    assert result["n_origins"] == 3 and result["n_series"] == 2


def test_paired_bootstrap_matches_manual_origin_blocks_and_percentage_ci():
    rng = np.random.default_rng(11)
    baseline = rng.uniform(1, 4, size=(2, 113, 3))
    candidate = baseline * rng.uniform(0.8, 1.1, size=(2, 113, 3))
    result = bootstrap_effect(baseline, candidate)
    generator = np.random.default_rng(92200)
    origin_a = baseline.mean((0, 2))
    origin_b = candidate.mean((0, 2))
    differences, percentages = [], []
    for _ in range(2000):
        starts = generator.integers(0, 113 - 56 + 1, size=3)
        indices = [index for start in starts for index in range(start, start + 56)][:113]
        a = sum(origin_a[index] for index in indices) / 113
        b = sum(origin_b[index] for index in indices) / 113
        differences.append(a - b)
        percentages.append(100 * (a - b) / a)
    np.testing.assert_allclose(result["ci_difference"], np.quantile(differences, [0.025, 0.975]), atol=1e-12)
    np.testing.assert_allclose(result["ci_gain_pct"], np.quantile(percentages, [0.025, 0.975]), atol=1e-12)
    assert result["difference"] == pytest.approx(baseline.mean() - candidate.mean())
    assert result["gain_pct"] == pytest.approx(100 * (baseline.mean() - candidate.mean()) / baseline.mean())
    assert result["block"] == 56 and result["reps"] == 2000
    json.dumps(result, allow_nan=False)


def test_seed_score_mean_does_not_resample_optimizer_seeds():
    baseline = np.stack([np.full((112, 2), 2.0), np.full((112, 2), 4.0)])
    candidate = np.stack([np.full((112, 2), 1.0), np.full((112, 2), 5.0)])
    result = bootstrap_effect(baseline, candidate)
    assert result["difference"] == 0 and result["ci_difference"] == [0.0, 0.0]
    assert [row["difference"] for row in result["seed_effects"]] == [1.0, -1.0]
    assert [row["gain_pct"] for row in result["seed_effects"]] == [50.0, -25.0]
    assert result["gain_pct"] == 0  # not mean([50,-25])
    one = bootstrap_effect(baseline[:1], candidate[:1])
    assert one["n_seeds"] == 1 and one["ci_difference"] == [1.0, 1.0]


def test_zero_baseline_does_not_invent_relative_gain_or_discard_draws():
    baseline = np.zeros((1, 56, 2))
    result = bootstrap_effect(baseline, np.ones_like(baseline))
    assert result["difference"] == -1
    assert result["gain_pct"] is None and result["ci_gain_pct"] is None
    assert result["zero_denominator_draws"] == 2000
    json.dumps(result, allow_nan=False)


def test_zero_error_and_inclusive_interval_boundaries():
    y = np.ones((2, 3, 64))
    q = np.ones((*y.shape, 9))
    arrays = metric_arrays(q, y, np.ones(3))
    for key, value in arrays.items():
        np.testing.assert_array_equal(value, np.ones((2, 3)) if key == "coverage80" else np.zeros((2, 3)))


@pytest.mark.parametrize("fault", ["shape", "sigma", "nan", "empty"])
def test_metric_validation(fault):
    q, y, sigma = example()
    if fault == "shape":
        q = q[..., :8]
    elif fault == "sigma":
        sigma[0] = 0
    elif fault == "nan":
        y[0, 0, 0] = np.nan
    else:
        q, y = q[:0], y[:0]
    with pytest.raises(ValueError):
        metric_arrays(q, y, sigma)


def test_bootstrap_rejects_unpaired_shape_and_silent_block_shortening():
    a = np.ones((2, 55, 3))
    with pytest.raises(ValueError):
        bootstrap_effect(a, a)
    with pytest.raises(ValueError):
        bootstrap_effect(a, a[:1], block=10)
    with pytest.raises(ValueError):
        bootstrap_effect(a, -a, block=10)
    with pytest.raises(ValueError):
        bootstrap_effect(a, a, reps=0, block=10)


def test_summary_rejects_invalid_inputs():
    q, y, sigma = example()
    arrays = metric_arrays(q, y, sigma)
    arrays["coverage80"][0, 0] = 2.0
    with pytest.raises(ValueError):
        summarize(arrays)
    arrays["coverage80"][0, 0] = 1.0
    arrays["nmse"][0, 0] = -1.0
    with pytest.raises(ValueError):
        summarize(arrays)
