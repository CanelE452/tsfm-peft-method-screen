import hashlib
import json

import numpy as np
import pytest

import scoring
from metrics import metric_arrays


def effect(gain=1.0, differences=(0.1, 0.2), seed_gains=(1.0, 2.0)):
    return {"gain_pct": gain, "seed_effects": [
        {"difference": d, "gain_pct": g} for d, g in zip(differences, seed_gains)]}


def test_numeric_thresholds_and_no_ci_gate():
    assert scoring.numeric_decision("T", effect(), 1.0)["mechanical_category"] == "GO_SCREEN"
    assert scoring.numeric_decision("T", effect(gain=0.999), 0)["mechanical_category"] == "HOLD_NO_AUTO_TRAIN"
    assert scoring.numeric_decision("T", effect(), 1.001)["mechanical_category"] == "HOLD_NO_AUTO_TRAIN"
    assert scoring.numeric_decision("Q", effect(gain=2), 0, resource_ok=False)["mechanical_category"] == "HOLD_NO_AUTO_TRAIN"
    assert scoring.numeric_decision("Q", effect(gain=2), 0, allocation_mismatch=True)["mechanical_category"] == "HOLD_NO_AUTO_TRAIN"


def test_seed_counterexample_and_optimization_override():
    assert scoring.numeric_decision("T", effect(gain=2, differences=(0.1, -0.01)), 0)["mechanical_category"] == "HOLD_NO_AUTO_TRAIN"
    assert scoring.numeric_decision("T", effect(gain=2), 0, optimization_flag=True)["mechanical_category"] == "HOLD_NO_AUTO_TRAIN"
    negative = effect(gain=0, differences=(0.0, 0.0), seed_gains=(0, 0))
    assert scoring.numeric_decision("T", negative, 0)["mechanical_category"] == "NO_GO_CURRENT"
    signal = effect(gain=-0.1, differences=(0.1, -0.2), seed_gains=(1, -1.2))
    assert scoring.numeric_decision("T", signal, 0)["mechanical_category"] == "HOLD_NO_AUTO_TRAIN"


def test_client_threshold_is_strictly_greater_than_five():
    assert scoring.numeric_decision("F", effect(), 0, f_client_harm_pct=5)["mechanical_category"] == "GO_SCREEN"
    assert scoring.numeric_decision("F", effect(), 0, f_client_harm_pct=5.01)["mechanical_category"] == "TRADEOFF_HOLD"
    assert scoring.numeric_decision("T", effect(gain=None), 0)["manual_interpretation_required"]


def test_full_independent_and_sampled_scalar_replay_detects_corruption():
    rng = np.random.default_rng(92220)
    q = rng.normal(size=(5, 3, 64, 9))
    y = rng.normal(size=(5, 3, 64))
    sigma = np.array([0.2, 2.0, 20.0])
    arrays = metric_arrays(q, y, sigma)
    audit = scoring.independent_replay(q, y, sigma, arrays)
    assert audit["sample_origin_indices"] == [0, 2, 4]
    assert audit["full_pinball_max_abs_error"] < 1e-12
    arrays["pinball"][1, 0] += 0.01  # unsampled origin is still checked by full replay
    with pytest.raises(AssertionError):
        scoring.independent_replay(q, y, sigma, arrays)


def test_comparison_broadcasts_fixed_baseline_and_averages_scores():
    metrics = {"N0_0": {"pinball": np.full((56, 2), 2.0)},
               "T_DELTA_92201": {"pinball": np.full((56, 2), 1.8)},
               "T_DELTA_92202": {"pinball": np.full((56, 2), 2.1)}}
    rows = scoring.comparison_records(metrics, "T", "fixture", ["N0"] * 2, ["T_DELTA"] * 2, True)
    assert [row["seed"] for row in rows] == [92201, 92202, "score_mean"]
    assert rows[-1]["baseline_mean"] == pytest.approx(2.0)
    assert rows[-1]["candidate_mean"] == pytest.approx(1.95)
    assert rows[-1]["gain_pct"] == pytest.approx(2.5)
    assert rows[-1]["n_seeds"] == 2


def test_f_client_reports_keep_V_fixed_tail_distinct_from_TEST_worst():
    summaries, fits, metrics = {}, {}, {}
    selections = {"F": {str(seed): {"baseline": "F_LOCAL"} for seed in scoring.SEEDS}}
    for arm in scoring.F:
        for seed in scoring.SEEDS:
            key = scoring.run_key(arm, seed)
            values = [1.0, 2.0, 3.0, 4.0] if arm != "F_PERIODIC" else [1.0, 2.0, 5.0, 4.0]
            summaries[key] = {"by_series": {"pinball": values}}
            metrics[key] = {"pinball": np.tile(values, (56, 1))}
            fits[key] = {"selected_step": 8, "validation": [{"step": 8, "client_scalars": [9, 3, 2, 1]}]}
    rows, tails = scoring.f_client_evidence(metrics, summaries, selections, fits, ["a", "b", "c", "d"])
    assert len(rows) == 60 and len(tails) == 10
    periodic = [row for row in tails if row["arm"] == "F_PERIODIC"]
    assert all(row["own_V_worst_client"] == row["selected_baseline_V_worst_client"] == 0 for row in periodic)
    assert all(row["TEST_worst_client_descriptive_only"] == 2 for row in periodic)
    harm = [row for row in rows if row["arm"] == "F_PERIODIC" and row["client"] == 2]
    assert all(row["gain_pct_vs_F_LOCAL"] == pytest.approx(-200 / 3) for row in harm)


def test_validation_monotonic_definition_has_no_hidden_convergence_threshold():
    fit = {"arm": "Q_STD", "validation": [{"step": step, "score": {"macro": {"pinball": value}}}
                                            for step, value in [(256, 1), (0, 3), (128, 2)]]}
    assert scoring.strictly_decreasing(fit)
    fit["validation"][0]["score"]["macro"]["pinball"] = 2
    assert not scoring.strictly_decreasing(fit)


def build_guard_fixture(tmp_path, monkeypatch):
    root = tmp_path
    exp = root / "experiments" / "synthetic"
    results = root / "results" / "synthetic"
    cache = root / ".cache" / "synthetic"
    for name, value in (("ROOT", root), ("EXP", exp), ("RESULTS", results), ("CACHE", cache)):
        monkeypatch.setattr(scoring, name, value)
    for name in ("metrics.py", "scoring.py", "PROTOCOL.md", "contract/MASTER_PLAN.txt"):
        path = exp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture only", encoding="utf-8")
    packet = cache / "packet.npz"
    packet.parent.mkdir(parents=True, exist_ok=True)
    packet.write_bytes(b"synthetic data receipt; never loaded")
    scoring.write(results / "DATA_AND_SPLIT_AUDIT.json", {"packets": {"packet": str(packet.relative_to(root))},
                                                         "hashes": {"packet_sha256": scoring.sha(packet)}})
    scoring.write(results / "Q_ALLOCATION.json", {"resource_mismatch": False, "unused_fraction": 0})
    for category in ("Q", "T", "F"):
        scoring.write(results / f"{category}_SOURCE_SEAL.json", {"source": {}})
    selections = {category: {} for category in ("Q", "T", "F")}
    selections["checkpoints"] = {}
    ledger = {"main": 8192, "smoke": 24, "Q": 3072, "T": 2560, "F": 2560, "runs": {}}
    predictions = {}
    for key in scoring.expected_runs():
        arm, seed = key.rsplit("_", 1)
        checkpoint_hash = None
        if arm not in scoring.FIXED:
            steps = [0, 8, 16] if arm.startswith("F_") else [0, 128, 256]
            prefix = "round" if arm.startswith("F_") else "step"
            checkpoint = cache / "fits" / key / f"{prefix}{steps[1]}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(key.encode())
            checkpoint_hash = scoring.sha(checkpoint)
            history = [{"step": step, "pinball": value} if arm.startswith("F_")
                       else {"step": step, "score": {"macro": {"pinball": value}}}
                       for step, value in zip(steps, [3.0, 1.0, 2.0])]
            scoring.write(results / "fits" / key / "FIT.json", {
                "arm": arm, "seed": int(seed), "run": key, "updates": 256,
                "frozen_unchanged": True, "validation": history, "selected_step": steps[1],
                "selected_checkpoint_sha256": checkpoint_hash})
            selections["checkpoints"][key] = checkpoint_hash
            ledger["runs"][key] = 256
        scoring.write(results / "resources" / f"{key}.json", {
            "arm": arm, "seed": int(seed), "storage_ratio": 0.4,
            "checkpoint_sha256": checkpoint_hash,
            "measurements": [{"batch": batch, "finite": True} for batch in (1, 8)]})
        prediction = cache / "test_predictions" / f"{key}.npy"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_bytes(key.encode())  # guard hashes metadata only, no numerical targets
        predictions[key] = {"path": str(prediction.relative_to(root)), "bytes": prediction.stat().st_size,
                            "sha256": scoring.sha(prediction), "shape": [866, 4 if arm.startswith("F_") else 16, 64, 9]}
    for category, arms in scoring.BASELINES.items():
        for seed in scoring.SEEDS:
            selections[category][str(seed)] = {"baseline": arms[0], "validation_scores": {arm: 1.0 for arm in arms},
                                              "selection_uses_test": False}
            scoring.write(results / "teacher" / f"N0_{seed}.json", {"bridge_validation": {"macro": {"pinball": 1.0}}})
    scoring.write(results / "OPTIMIZER_LEDGER.json", ledger)
    scoring.write(results / "SELECTIONS.json", selections)
    scoring.write(results / "TEST_PREDICTIONS_SEAL.json", {
        "selection_sha256": scoring.sha(results / "SELECTIONS.json"), "all_saved_before_scoring": True,
        "predictions": predictions})
    return results, cache


def test_complete_guard_schema_with_34_predictions_32_workflows(tmp_path, monkeypatch):
    results, cache = build_guard_fixture(tmp_path, monkeypatch)
    selections, seal, fits, resources, audit, hashes = scoring.guard()
    assert len(fits) == 32 and len(resources) == len(seal["predictions"]) == 34
    assert selections["T"]["92201"]["baseline"] == "N0"
    assert not (results / "TEST_SCORES.json").exists()
    assert not (cache / "metrics").exists()


@pytest.mark.parametrize("fault", ["unfinished", "selection_hash", "prediction_hash", "validation_winner"])
def test_guard_rejects_incomplete_or_changed_artifacts(tmp_path, monkeypatch, fault):
    results, cache = build_guard_fixture(tmp_path, monkeypatch)
    if fault == "unfinished":
        ledger = scoring.read(results / "OPTIMIZER_LEDGER.json")
        ledger["main"] = 8191
        scoring.write(results / "OPTIMIZER_LEDGER.json", ledger)
    elif fault == "selection_hash":
        (results / "SELECTIONS.json").write_text("{}")
    elif fault == "prediction_hash":
        (cache / "test_predictions/A0_0.npy").write_bytes(b"modified")
    else:
        selection = scoring.read(results / "SELECTIONS.json")
        selection["Q"]["92201"]["baseline"] = "Q_QERA"
        scoring.write(results / "SELECTIONS.json", selection)
        seal = scoring.read(results / "TEST_PREDICTIONS_SEAL.json")
        seal["selection_sha256"] = scoring.sha(results / "SELECTIONS.json")
        scoring.write(results / "TEST_PREDICTIONS_SEAL.json", seal)
    with pytest.raises(AssertionError):
        scoring.guard()
    assert not (results / "TEST_SCORES.json").exists()
