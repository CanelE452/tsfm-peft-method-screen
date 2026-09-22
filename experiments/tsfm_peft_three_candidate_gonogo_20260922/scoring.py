"""CPU-only scoring of a completed, selection-sealed Q/F/T screen.

Importing this module performs no file writes or experiment execution. main()
is the only entry that accesses real TEST targets and writes score artifacts.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import bootstrap_effect, metric_arrays, summarize

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
RESULTS = ROOT / "results" / EXP.name
CACHE = ROOT / ".cache" / EXP.name
SEEDS = (92201, 92202)
Q = ("Q_FP", "Q_STD", "Q_LOFTQ", "Q_QERA", "Q_IO16", "Q_FORECAST")
T = ("T_A1", "T_RECENT", "T_KD", "T_BLEND", "T_DELTA")
F = ("F_LOCAL", "F_SHARED", "F_AFFINE", "F_HEAD", "F_PERIODIC")
FIXED = ("A0", "N0")
CANDIDATES = {"Q": Q[-1], "T": T[-1], "F": F[-1]}
BASELINES = {"Q": Q[1:-1], "T": ("N0", "T_RECENT", "T_KD", "T_BLEND"), "F": F[:-1]}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_key(arm, seed):
    return f"{arm}_{0 if arm in FIXED else seed}"


def expected_runs():
    return [run_key(arm, seed) for arm in (*Q, *T, *F, *FIXED)
            for seed in ((0,) if arm in FIXED else SEEDS)]


def validation_values(fit):
    rows = sorted(fit["validation"], key=lambda row: row["step"])
    return [(int(row["step"]), float(row["pinball"] if fit["arm"].startswith("F_")
                                    else row["score"]["macro"]["pinball"])) for row in rows]


def strictly_decreasing(fit):
    values = [value for _, value in validation_values(fit)]
    return len(values) == 3 and values[0] > values[1] > values[2]


def guard():
    """Validate completion and every prediction hash before loading TEST targets."""
    ledger_path = RESULTS / "OPTIMIZER_LEDGER.json"
    ledger = read(ledger_path)
    assert ledger["main"] == 8192 and 0 <= ledger["smoke"] <= 24
    assert {k: ledger[k] for k in ("Q", "T", "F")} == {"Q": 3072, "T": 2560, "F": 2560}
    selections = read(RESULTS / "SELECTIONS.json")
    prediction_seal = read(RESULTS / "TEST_PREDICTIONS_SEAL.json")
    assert prediction_seal["selection_sha256"] == sha(RESULTS / "SELECTIONS.json")
    assert prediction_seal["all_saved_before_scoring"] is True
    assert set(prediction_seal["predictions"]) == set(expected_runs())
    fits, resources = {}, {}
    inputs = {str(ledger_path.relative_to(ROOT)): sha(ledger_path)}
    for name in ("SELECTIONS.json", "TEST_PREDICTIONS_SEAL.json", "DATA_AND_SPLIT_AUDIT.json", "Q_ALLOCATION.json"):
        inputs[str((RESULTS / name).relative_to(ROOT))] = sha(RESULTS / name)
    for path in (EXP / "metrics.py", EXP / "scoring.py", EXP / "PROTOCOL.md", EXP / "contract/MASTER_PLAN.txt"):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    for category in ("Q", "T", "F"):
        path = RESULTS / f"{category}_SOURCE_SEAL.json"
        for relative, expected_hash in read(path)["source"].items():
            assert sha(ROOT / relative) == expected_hash, relative
        inputs[str(path.relative_to(ROOT))] = sha(path)
    for key in expected_runs():
        arm, seed = key.rsplit("_", 1)
        resource_path = RESULTS / "resources" / f"{key}.json"
        resource = read(resource_path)
        assert resource["arm"] == arm and resource["seed"] == int(seed)
        assert {row["batch"] for row in resource["measurements"]} == {1, 8}
        assert all(row["finite"] for row in resource["measurements"])
        resources[key] = resource
        inputs[str(resource_path.relative_to(ROOT))] = sha(resource_path)
        if arm not in FIXED:
            path = RESULTS / "fits" / key / "FIT.json"
            fit = read(path)
            assert fit["run"] == key and fit["updates"] == ledger["runs"][key] == 256
            assert fit["frozen_unchanged"] is True
            vals = validation_values(fit)
            expected_steps = [0, 8, 16] if arm.startswith("F_") else [0, 128, 256]
            assert [step for step, _ in vals] == expected_steps
            assert fit["selected_step"] == min(vals, key=lambda pair: (pair[1], pair[0]))[0]
            prefix = "round" if arm.startswith("F_") else "step"
            checkpoint = CACHE / "fits" / key / f"{prefix}{fit['selected_step']}.pt"
            digest = sha(checkpoint)
            assert digest == fit["selected_checkpoint_sha256"] == selections["checkpoints"][key]
            assert resource["checkpoint_sha256"] == digest
            fits[key] = fit
            inputs[str(path.relative_to(ROOT))] = sha(path)
        record = prediction_seal["predictions"][key]
        path = CACHE / "test_predictions" / f"{key}.npy"
        assert (ROOT / record["path"]).resolve() == path.resolve()
        assert sha(path) == record["sha256"]
        assert path.stat().st_size == record["bytes"]
        assert record["shape"][1:] == [4 if arm.startswith("F_") else 16, 64, 9]
        inputs[str(path.relative_to(ROOT))] = record["sha256"]
    assert len(fits) == 32 and set(selections["checkpoints"]) == set(fits)
    for category, arms in BASELINES.items():
        assert set(selections[category]) == {str(seed) for seed in SEEDS}
        for seed in SEEDS:
            selected = selections[category][str(seed)]
            assert selected["selection_uses_test"] is False
            allowed = {}
            for arm in arms:
                if category == "Q" and resources[run_key(arm, seed)]["storage_ratio"] > 0.6:
                    continue
                if arm == "N0":
                    teacher_path = RESULTS / "teacher" / f"N0_{seed}.json"
                    value = read(teacher_path)["bridge_validation"]["macro"]["pinball"]
                    inputs[str(teacher_path.relative_to(ROOT))] = sha(teacher_path)
                else:
                    fit = fits[run_key(arm, seed)]
                    value = dict(validation_values(fit))[fit["selected_step"]]
                allowed[arm] = value
            assert selected["validation_scores"] == allowed
            assert selected["baseline"] == min(allowed, key=allowed.get)
    audit = read(RESULTS / "DATA_AND_SPLIT_AUDIT.json")
    packet = ROOT / audit["packets"]["packet"]
    assert packet.exists() and sha(packet) == audit["hashes"]["packet_sha256"]
    inputs[str(packet.relative_to(ROOT))] = audit["hashes"]["packet_sha256"]
    return selections, prediction_seal, fits, resources, audit, inputs


def independent_replay(q, y, sigma, arrays):
    """Full pinball replay plus scalar checks on first/middle/last origins."""
    ordered = np.sort(np.asarray(q, dtype=np.float64), axis=-1)
    truth = np.asarray(y, dtype=np.float64)
    independent = np.zeros(truth.shape[:2], dtype=np.float64)
    for j in range(9):
        tau = (j + 1) / 10.0
        residual = truth - ordered[..., j]
        independent += (2 * np.where(residual >= 0, tau * residual, (tau - 1) * residual)
                        / sigma[None, :, None]).mean(axis=2) / 9
    np.testing.assert_allclose(independent, arrays["pinball"], rtol=1e-12, atol=1e-10)
    maximum = 0.0
    sample_origins = sorted({0, len(y) // 2, len(y) - 1})
    for o in sample_origins:
        for s in range(y.shape[1]):
            losses, absolute, squared, covers, widths, crosses = [], [], [], [], [], []
            for h in range(64):
                raw = [float(value) for value in q[o, s, h]]
                forecast = sorted(raw)
                target = float(y[o, s, h])
                for j, estimate in enumerate(forecast):
                    residual = target - estimate
                    tau = (j + 1) / 10
                    losses.append(2 * (tau * residual if residual >= 0 else (tau - 1) * residual) / float(sigma[s]))
                absolute.append(abs(target - forecast[4]))
                squared.append(((target - forecast[4]) / float(sigma[s])) ** 2)
                covers.append(float(forecast[0] <= target <= forecast[8]))
                widths.append((forecast[8] - forecast[0]) / float(sigma[s]))
                crosses.append(sum(raw[j + 1] < raw[j] for j in range(8)) / 8)
            expected = {"pinball": math.fsum(losses) / (64 * 9),
                        "raw_mae": math.fsum(absolute) / 64,
                        "nmae": math.fsum(absolute) / 64 / float(sigma[s]),
                        "nmse": math.fsum(squared) / 64, "coverage80": math.fsum(covers) / 64,
                        "width80": math.fsum(widths) / 64, "raw_crossing": math.fsum(crosses) / 64}
            for metric, value in expected.items():
                np.testing.assert_allclose(value, arrays[metric][o, s], rtol=1e-12, atol=1e-10)
                maximum = max(maximum, abs(value - float(arrays[metric][o, s])))
    return {"full_mean_independent_pinball": float(independent.mean()),
            "full_pinball_max_abs_error": float(np.max(np.abs(independent - arrays["pinball"]))),
            "scalar_max_abs_error": maximum, "sample_origin_indices": sample_origins,
            "sample_series_count": y.shape[1], "scalar_metrics": list(arrays)}


def numeric_decision(category, effect, nmae_harm_pct, resource_ok=True, allocation_mismatch=False,
                     optimization_flag=False, f_client_harm_pct=None):
    """Mechanical contract evidence only; sufficient-baseline interpretation is manual."""
    gains = [item["gain_pct"] for item in effect["seed_effects"]]
    gain = effect["gain_pct"]
    both_positive = all(item["difference"] > 0 for item in effect["seed_effects"])
    reasons = []
    if not resource_ok or allocation_mismatch:
        decision = "HOLD_NO_AUTO_TRAIN"
        reasons.append("resource_cap_or_allocation_mismatch")
    elif category == "F" and f_client_harm_pct is not None and f_client_harm_pct > 5:
        decision = "TRADEOFF_HOLD"
        reasons.append("at_least_one_client_exceeds_5pct_pinball_harm_vs_LOCAL")
    elif optimization_flag:
        decision = "HOLD_NO_AUTO_TRAIN"
        reasons.append("candidate_and_selected_trainable_baseline_strictly_improve_through_last_V_checkpoint_in_both_seeds")
    elif gain is None or nmae_harm_pct is None or any(value is None for value in gains):
        decision = "HOLD_NO_AUTO_TRAIN"
        reasons.append("undefined_relative_gain_or_harm")
    elif gain >= 1.0 and both_positive and nmae_harm_pct <= 1.0:
        decision = "GO_SCREEN"
        reasons.append("predeclared_numeric_screen_conditions_met")
    elif gain <= 0 and max(gains) < 1.0:
        decision = "NO_GO_CURRENT"
        reasons.append("nonpositive_mean_without_any_seed_at_least_1pct_positive_signal")
    else:
        decision = "HOLD_NO_AUTO_TRAIN"
        reasons.append("subthreshold_or_mixed_seed_gain_or_nMAE_tradeoff")
    return {"mechanical_category": decision, "reasons": reasons, "mean_gain_pct": gain,
            "seed_gain_pct": gains, "both_seed_differences_positive": both_positive,
            "nmae_harm_pct": nmae_harm_pct, "resource_ok": resource_ok,
            "allocation_mismatch": allocation_mismatch, "optimization_flag": optimization_flag,
            "max_F_client_harm_pct_vs_LOCAL": f_client_harm_pct,
            "manual_interpretation_required": True,
            "GO_STANDARD_ONLY": "Not automatically assigned; report whether existing methods suffice using displayed comparisons, without a new threshold.",
            "automatic_followup_training": False}


def comparison_records(metrics, category, label, baselines, candidate, primary=False):
    """baselines/candidate have one arm name per seed; fixed arms broadcast at run_key."""
    a = np.stack([metrics[run_key(arm, seed)]["pinball"] for arm, seed in zip(baselines, SEEDS)])
    b = np.stack([metrics[run_key(arm, seed)]["pinball"] for arm, seed in zip(candidate, SEEDS)])
    output = []
    for index, seed in enumerate((*SEEDS, "score_mean")):
        effect = bootstrap_effect(a if seed == "score_mean" else a[index:index + 1],
                                  b if seed == "score_mean" else b[index:index + 1])
        output.append({"category": category, "comparison": label, "seed": seed,
                       "baseline": baselines if seed == "score_mean" else baselines[index],
                       "candidate": candidate if seed == "score_mean" else candidate[index],
                       "primary_V_selected": primary, "test_reselection": False, **effect})
    return output


def f_client_evidence(metrics, summaries, selections, fits, ids):
    rows, tails = [], []
    for seed in SEEDS:
        chosen = selections["F"][str(seed)]["baseline"]
        baseline_fit = fits[run_key(chosen, seed)]
        baseline_val = next(row for row in baseline_fit["validation"] if row["step"] == baseline_fit["selected_step"])
        fixed_client = int(np.argmax(baseline_val["client_scalars"]))
        for arm in F:
            fit = fits[run_key(arm, seed)]
            own_val = next(row for row in fit["validation"] if row["step"] == fit["selected_step"])
            own_worst = int(np.argmax(own_val["client_scalars"]))
            values = summaries[run_key(arm, seed)]["by_series"]["pinball"]
            test_worst = int(np.argmax(values))
            for client in range(4):
                score = values[client]
                row = {"seed": seed, "arm": arm, "client": client, "series_id": ids[client], "pinball": score,
                       "is_own_V_fixed_worst": client == own_worst,
                       "is_selected_baseline_V_fixed_worst": client == fixed_client,
                       "is_TEST_worst_descriptive_only": client == test_worst}
                for baseline in ("F_LOCAL", "F_SHARED"):
                    base = summaries[run_key(baseline, seed)]["by_series"]["pinball"][client]
                    row[f"gain_pct_vs_{baseline}"] = 100 * (base - score) / base if base > 0 else None
                    effect = bootstrap_effect(metrics[run_key(baseline, seed)]["pinball"][None, :, client:client + 1],
                                              metrics[run_key(arm, seed)]["pinball"][None, :, client:client + 1])
                    row[f"difference_ci_low_vs_{baseline}"] = effect["ci_difference"][0]
                    row[f"difference_ci_high_vs_{baseline}"] = effect["ci_difference"][1]
                rows.append(row)
            tails.append({"seed": seed, "arm": arm, "own_V_worst_client": own_worst,
                          "own_V_worst_TEST_pinball": values[own_worst],
                          "selected_baseline_V_worst_client": fixed_client,
                          "selected_baseline_V_worst_TEST_pinball": values[fixed_client],
                          "TEST_worst_client_descriptive_only": test_worst, "TEST_worst_pinball": values[test_worst]})
    for arm in F:
        for client in range(4):
            value = float(np.mean([summaries[run_key(arm, seed)]["by_series"]["pinball"][client] for seed in SEEDS]))
            row = {"seed": "score_mean", "arm": arm, "client": client, "series_id": ids[client], "pinball": value}
            for baseline in ("F_LOCAL", "F_SHARED"):
                base = float(np.mean([summaries[run_key(baseline, seed)]["by_series"]["pinball"][client] for seed in SEEDS]))
                row[f"gain_pct_vs_{baseline}"] = 100 * (base - value) / base if base > 0 else None
                effect = bootstrap_effect(np.stack([metrics[run_key(baseline, seed)]["pinball"][:, client:client + 1] for seed in SEEDS]),
                                          np.stack([metrics[run_key(arm, seed)]["pinball"][:, client:client + 1] for seed in SEEDS]))
                row[f"difference_ci_low_vs_{baseline}"] = effect["ci_difference"][0]
                row[f"difference_ci_high_vs_{baseline}"] = effect["ci_difference"][1]
            rows.append(row)
    return rows, tails


def main():
    selections, seal, fits, resources, data_audit, input_hashes = guard()
    # Imported only after all selections, run completion and prediction hashes pass.
    import data
    d = data.load()
    origins = d["origins"]["Q_TEST"]
    np.testing.assert_array_equal(origins, d["origins"]["T_TEST"])
    assert len(origins) == 866 and (np.diff(origins) == 6).all()
    assert hashlib.sha256(origins.tobytes()).hexdigest() == seal["origins_sha256"]
    split = data_audit["split"]
    assert (origins >= split["q_test_start"]).all() and (origins + 64 <= split["q_test_end"]).all()
    y = np.stack([d["values"][o:o + 64].T for o in origins])
    summaries, metrics, replays, cache_records = {}, {}, {}, {}
    run_rows, series_rows = [], []
    folder = CACHE / "metrics"
    folder.mkdir(parents=True, exist_ok=True)
    for key in expected_runs():
        arm, seed = key.rsplit("_", 1)
        series_count = 4 if arm.startswith("F_") else 16
        q = np.load(ROOT / seal["predictions"][key]["path"], mmap_mode="r")
        assert list(q.shape) == [len(origins), series_count, 64, 9]
        arrays = metric_arrays(q, y[:, :series_count], d["sigma"][:series_count])
        replays[key] = independent_replay(q, y[:, :series_count], d["sigma"][:series_count], arrays)
        summary = summarize(arrays)
        np.testing.assert_allclose(summary["macro"]["pinball"], replays[key]["full_mean_independent_pinball"], atol=1e-10, rtol=1e-12)
        summaries[key], metrics[key] = summary, arrays
        path = folder / f"{key}.npz"
        np.savez_compressed(path, origins=origins, series_ids=np.asarray(d["ids"][:series_count]), **arrays)
        cache_records[key] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size,
                              "shape": [len(origins), series_count]}
        run_rows.append({"run": key, "arm": arm, "seed": int(seed), **summary["macro"]})
        for index in range(series_count):
            series_rows.append({"run": key, "arm": arm, "seed": int(seed), "series_index": index,
                                "series_id": d["ids"][index], **{name: values[index] for name, values in summary["by_series"].items()}})
        del q
    effects = []
    primary = {}
    for category, candidate in CANDIDATES.items():
        selected = [selections[category][str(seed)]["baseline"] for seed in SEEDS]
        records = comparison_records(metrics, category, "candidate_vs_V_selected", selected, [candidate] * 2, True)
        effects.extend(records)
        primary[category] = records[-1]
        references = {"Q": Q[:-1], "T": ("A0", "T_A1", "N0", "T_RECENT", "T_KD", "T_BLEND"), "F": F[:-1]}[category]
        for baseline in references:
            effects.extend(comparison_records(metrics, category, "diagnostic_vs_" + baseline, [baseline] * 2, [candidate] * 2))
    effects.extend(comparison_records(metrics, "T", "old_teacher_A1_vs_A0", ["A0"] * 2, ["T_A1"] * 2))
    effects.extend(comparison_records(metrics, "T", "new_base_N0_vs_old_teacher_A1", ["T_A1"] * 2, ["N0"] * 2))
    client_rows, fixed_tails = f_client_evidence(metrics, summaries, selections, fits, d["ids"])
    allocation = read(RESULTS / "Q_ALLOCATION.json")
    decisions = {}
    for category, candidate in CANDIDATES.items():
        selected = [selections[category][str(seed)]["baseline"] for seed in SEEDS]
        bn = float(np.mean([summaries[run_key(arm, seed)]["macro"]["nmae"] for arm, seed in zip(selected, SEEDS)]))
        cn = float(np.mean([summaries[run_key(candidate, seed)]["macro"]["nmae"] for seed in SEEDS]))
        monotonic_pairs = [arm not in FIXED and strictly_decreasing(fits[run_key(arm, seed)])
                           and strictly_decreasing(fits[run_key(candidate, seed)]) for arm, seed in zip(selected, SEEDS)]
        client_harm = None
        if category == "F":
            harms = [-row["gain_pct_vs_F_LOCAL"] for row in client_rows
                     if row["arm"] == candidate and row["gain_pct_vs_F_LOCAL"] is not None]
            client_harm = max(harms)
        evidence = numeric_decision(category, primary[category], 100 * (cn - bn) / bn if bn > 0 else None,
                                    resource_ok=all(resources[run_key(candidate, seed)]["storage_ratio"] <= 0.6 for seed in SEEDS) if category == "Q" else True,
                                    allocation_mismatch=bool(allocation["resource_mismatch"]) if category == "Q" else False,
                                    optimization_flag=all(monotonic_pairs), f_client_harm_pct=client_harm)
        evidence.update(selected_baselines=selected, candidate=candidate, monotonic_pairs_by_seed=monotonic_pairs,
                        nmae_baseline_score_mean=bn, nmae_candidate_score_mean=cn)
        if category == "Q":
            evidence["unused_parameter_fraction"] = allocation["unused_fraction"]
            evidence["baseline_vs_Q_FP_gap_pct"] = [100 * (summaries[run_key(arm, seed)]["macro"]["pinball"]
                    - summaries[run_key("Q_FP", seed)]["macro"]["pinball"]) / summaries[run_key("Q_FP", seed)]["macro"]["pinball"]
                    for arm, seed in zip(selected, SEEDS)]
            evidence["candidate_storage_ratio_by_seed"] = [resources[run_key(candidate, seed)]["storage_ratio"] for seed in SEEDS]
        decisions[category] = evidence
    # Scalar macro-score subtraction independently verifies every bootstrap point effect.
    for row in effects:
        row_seeds = SEEDS if row["seed"] == "score_mean" else (row["seed"],)
        baselines = row["baseline"] if isinstance(row["baseline"], list) else [row["baseline"]]
        candidates = row["candidate"] if isinstance(row["candidate"], list) else [row["candidate"]]
        a = float(np.mean([summaries[run_key(arm, seed)]["macro"]["pinball"] for arm, seed in zip(baselines, row_seeds)]))
        b = float(np.mean([summaries[run_key(arm, seed)]["macro"]["pinball"] for arm, seed in zip(candidates, row_seeds)]))
        np.testing.assert_allclose(row["difference"], a - b, atol=1e-10, rtol=1e-12)
        if a > 0:
            np.testing.assert_allclose(row["gain_pct"], 100 * (a - b) / a, atol=1e-10, rtol=1e-12)
    for relative, expected_hash in input_hashes.items():
        assert sha(ROOT / relative) == expected_hash, f"Input changed during CPU scoring: {relative}"
    pd.DataFrame(run_rows).to_csv(RESULTS / "scores.csv", index=False)
    pd.DataFrame(series_rows).to_csv(RESULTS / "series_scores.csv", index=False)
    pd.DataFrame(client_rows).to_csv(RESULTS / "F_client_effects.csv", index=False)
    pd.DataFrame(fixed_tails).to_csv(RESULTS / "F_validation_fixed_tail.csv", index=False)
    flat_effects = []
    for row in effects:
        flat_effects.append({**{k: v for k, v in row.items() if k not in ("seed_effects", "ci_difference", "ci_gain_pct")},
                             "baseline": json.dumps(row["baseline"]), "candidate": json.dumps(row["candidate"]),
                             "difference_ci_low": row["ci_difference"][0], "difference_ci_high": row["ci_difference"][1],
                             "gain_pct_ci_low": row["ci_gain_pct"][0] if row["ci_gain_pct"] else None,
                             "gain_pct_ci_high": row["ci_gain_pct"][1] if row["ci_gain_pct"] else None})
    pd.DataFrame(flat_effects).to_csv(RESULTS / "seed_effects.csv", index=False)
    write(RESULTS / "EFFECTS.json", {"comparisons": effects, "decision_evidence": decisions,
                                   "diagnostic_baselines_are_not_TEST_reselection": True})
    write(RESULTS / "TEST_SCORES.json", {"status": "PASS", "summaries": summaries,
          "replays": replays, "metric_cache_manifest": cache_records, "input_hashes_unchanged": input_hashes,
          "selection_sha256": seal["selection_sha256"], "origins_sha256": seal["origins_sha256"],
          "gains_independently_recomputed_from_macro_scores": True,
          "optimizer_ledger_unchanged": True, "scientific_classification": "REVIEW_DECISION_EVIDENCE",
          "seed_aggregation": "fixed seed score mean; not prediction ensemble",
          "F_tradeoff_check": "any individual-seed or seed-mean client pinball harm >5% vs matched independent LOCAL",
          "F_validation_fixed_tail": "each arm own-V worst and selected-baseline V worst; TEST worst is descriptive only",
          "resource_scope_caveat": "Current F resource inference measures client0 only; storage is complete four-client workflow state. See FIT client runtime/peak separately."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-sealed-test", action="store_true", required=True)
    parser.parse_args()
    main()
