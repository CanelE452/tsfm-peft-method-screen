import json

import numpy as np
import pytest

from evaluation import apply_affine, bootstrap_effect, fit_affine, score_tables


def fixture_arrays():
    rng = np.random.default_rng(37)
    y = rng.normal(size=(8, 2, 256)) * np.array([1.0, 3.0])[None, :, None]
    q = np.sort(rng.normal(size=(*y.shape, 9)), axis=-1)
    q *= np.array([1.0, 3.0])[None, :, None, None]
    return y, q, np.array([1.0, 3.0])


def score(y, q, sigma, **kwargs):
    return score_tables(y, q, sigma, "synthetic", "R", 92121, "ordered",
                        np.arange(len(y)) * 24, ["s0", "s1"], **kwargs)


def test_scalar_pinball_pce_coverage_and_point_metrics():
    y, q, sigma = fixture_arrays()
    frames = score(y, q, sigma)
    row = frames["summary"].query("window == 'full_256'").iloc[0]
    scores, pces, coverages, widths, maes, raw_maes, rmses = [], [], [], [], [], [], []
    for s in range(2):
        losses, cover, width, absolute_errors, square_errors = [], [], [], [], []
        counts = [0] * 9
        for o in range(8):
            for h in range(256):
                truth = float(y[o, s, h])
                median = float(q[o, s, h, 4])
                for j in range(9):
                    tau = (j + 1) / 10
                    error = truth - float(q[o, s, h, j])
                    losses.append(2 * (tau * error if error >= 0 else (tau - 1) * error) / sigma[s])
                    counts[j] += truth <= q[o, s, h, j]
                cover.append(q[o, s, h, 0] <= truth <= q[o, s, h, 8])
                width.append((q[o, s, h, 8] - q[o, s, h, 0]) / sigma[s])
                absolute_errors.append(abs(truth - median))
                square_errors.append((truth - median) ** 2)
        scores.append(sum(losses) / len(losses))
        pces.append(sum(abs(counts[j] / (8 * 256) - (j + 1) / 10) for j in range(9)) / 9)
        coverages.append(sum(cover) / len(cover))
        widths.append(sum(width) / len(width))
        raw_maes.append(sum(absolute_errors) / len(absolute_errors))
        maes.append(raw_maes[-1] / sigma[s])
        rmses.append((sum(square_errors) / len(square_errors)) ** 0.5 / sigma[s])
    expected = {"scaled_pinball": np.mean(scores), "pce": np.mean(pces),
                "coverage80": np.mean(coverages), "coverage_deficit80": 0.8 - np.mean(coverages),
                "scaled_width80": np.mean(widths), "scaled_mae": np.mean(maes),
                "raw_mae": np.mean(raw_maes), "scaled_rmse": np.mean(rmses)}
    for metric, value in expected.items():
        assert row[metric] == pytest.approx(value, abs=1e-12)
    assert frames["series_block"].shape[0] == 2 * 10
    assert frames["origin"].shape[0] == 8 * 10
    assert frames["lead"].shape[0] == 2 * 256


def test_pce_must_be_computed_within_series_not_pooled():
    y = np.zeros((8, 2, 256))
    q = np.ones((*y.shape, 9))
    q[:, 1] = -1
    frames = score(y, q, np.ones(2))
    assert frames["summary"].iloc[0]["pce"] == pytest.approx(0.5)
    pooled_pce = np.mean(np.abs(np.full(9, 0.5) - np.arange(1, 10) / 10))
    assert frames["summary"].iloc[0]["pce"] != pytest.approx(pooled_pce)


def test_affine_exact_identity_and_flat_grid_identity_tie():
    y, q, sigma = fixture_arrays()
    identity = {"alpha": [1.0] * 4, "beta": [0.0] * 4}
    np.testing.assert_array_equal(apply_affine(q.astype(np.float32), sigma, identity), q.astype(np.float32))
    perfect = np.zeros_like(q)
    params = fit_affine(np.zeros_like(y), perfect, sigma)
    assert params["alpha"] == [1.0] * 4
    assert params["beta"] == [0.0] * 4
    assert all(len(block["grid_scores"]) == 35 for block in params["blocks"])
    json.dumps(params, allow_nan=False)


def test_affine_bias_by_block_and_calibration_improvement():
    y, q, sigma = fixture_arrays()
    q.fill(0)
    for block, beta in enumerate((-0.5, -0.25, 0.25, 0.5)):
        y[:, :, block * 64:(block + 1) * 64] = beta * sigma[None, :, None]
    params = fit_affine(y, q, sigma)
    assert params["alpha"] == [1.0] * 4
    assert params["beta"] == [-0.5, -0.25, 0.25, 0.5]
    corrected = apply_affine(q, sigma, params)
    np.testing.assert_array_equal(corrected[..., 4], y)
    assert (np.diff(corrected, axis=-1) >= 0).all()


def test_crossing_original_preserved_for_ordered_variant_and_window_consistency():
    y, q, sigma = fixture_arrays()
    raw = q[..., ::-1]
    frames = score(y, q, sigma, raw_q=raw)
    assert (frames["summary"]["raw_quantile_crossing_rate"] == 1).all()
    assert (frames["summary"]["crossing_reference"] == "pre_sort_raw_q").all()
    summary = frames["summary"].set_index("window")
    origin = frames["origin"].query("window == 'full_256'")
    assert summary.loc["full_256", "scaled_pinball"] == pytest.approx(origin.scaled_pinball.mean())
    assert summary.loc["full_256", "scaled_pinball"] == pytest.approx(summary.loc["prefix_256", "scaled_pinball"])
    block_mean = summary.loc[["block_1", "block_2", "block_3", "block_4"], "scaled_pinball"].mean()
    assert summary.loc["full_256", "scaled_pinball"] == pytest.approx(block_mean)
    lead_mean = frames["lead"].scaled_pinball.mean()
    assert summary.loc["full_256", "scaled_pinball"] == pytest.approx(lead_mean)


def test_bootstrap_uses_paired_seven_origin_blocks_and_sign():
    a = np.arange(21, dtype=float) + 1
    b = a - np.sin(a)
    result = bootstrap_effect(a, b, seed=79)
    rng = np.random.default_rng(79)
    draws = []
    for _ in range(2000):
        starts = rng.integers(0, 15, size=3)
        sample = [index for start in starts for index in range(start, start + 7)]
        draws.append(sum(a[index] - b[index] for index in sample) / 21)
    assert result["effect_a_minus_b"] == pytest.approx(np.mean(a - b))
    np.testing.assert_allclose([result["ci95_low"], result["ci95_high"]], np.quantile(draws, [0.025, 0.975]))
    constant = bootstrap_effect(a, a - 0.5)
    assert constant["effect_a_minus_b"] == constant["ci95_low"] == constant["ci95_high"] == 0.5
    assert bootstrap_effect(a, a, seed=79)["effect_a_minus_b"] == 0
    json.dumps(result, allow_nan=False)


def test_two_seed_mean_bootstraps_origins_without_resampling_seeds():
    a = np.stack([np.arange(21), np.arange(21) + 4.0])
    b = a - np.stack([np.ones(21), np.full(21, 3.0)])
    result = bootstrap_effect(a.mean(axis=0), b.mean(axis=0))
    assert result["effect_a_minus_b"] == result["ci95_low"] == result["ci95_high"] == 2


def test_invalid_metric_inputs_fail_before_silent_axis_mixing():
    y, q, sigma = fixture_arrays()
    with pytest.raises(ValueError):
        score(y, q.transpose(1, 0, 2, 3), sigma)
    with pytest.raises(ValueError):
        score(y, q, np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        fit_affine(y, q[..., ::-1], sigma)
    with pytest.raises(ValueError):
        bootstrap_effect(np.ones(6), np.ones(6))
