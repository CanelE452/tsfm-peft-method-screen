"""Independent, CPU-only CAL grid audit, executed only after training finishes.

This helper imports no experiment modules, torch, model, optimizer or launcher.
It reads CALIBRATION predictions and constructs targets only in the CAL interval.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name
OUTPUT = HERE / "CALIBRATION_INDEPENDENT_AUDIT.json"
ALPHAS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
BETAS = (-0.5, -0.25, 0.0, 0.25, 0.5)
SOURCES = ("Electricity", "ETTh1")
FIXED = ("F0_MEDIAN", "F0_NATIVE", "CHRONOS2_DIRECT")
LEARNED = ("R_ROLLOUT_LORA", "S_STATE_ADAPTER", "U_UNCERTAINTY_ADAPTER", "R_NATIVE", "R_MC16")
SEEDS = (92121, 92122)
TOLERANCE = 1e-10


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expected_keys():
    return {(source, f"{method}_seed{seed}") for source in SOURCES
            for method in (*FIXED, *LEARNED)
            for seed in ((0,) if method in FIXED else SEEDS)}


def tie_key(candidate):
    alpha, beta = candidate["alpha"], candidate["beta"]
    return (candidate["score"], (alpha - 1) ** 2 + beta ** 2, alpha, abs(beta), beta)


def independent_grid_score(target, ordered, train_sigma, alpha, beta):
    """Quantile-by-quantile np.where loss; no production scoring function."""
    median = ordered[..., 4]
    scale = train_sigma[None, :, None]
    means = []
    for index in range(9):
        tau = (index + 1) / 10.0
        if alpha == 1.0 and beta == 0.0:
            estimate = ordered[..., index]
        else:
            estimate = median + beta * scale + alpha * (ordered[..., index] - median)
        error = target - estimate
        loss = np.where(error >= 0.0, tau * error, (tau - 1.0) * error)
        means.append(float(np.mean((2.0 * loss) / scale, dtype=np.float64)))
    return sum(means) / 9.0


def require_finished_training():
    ledger_path = HERE / "UPDATE_LEDGER.jsonl"
    ledger_hash = digest(ledger_path)
    entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    spent = {kind: {(row["fit_id"], int(row["step"])) for row in entries
                    if row["kind"] == kind and row["status"].startswith("committed")}
             for kind in ("main", "smoke")}
    assert len(spent["main"]) == 12288, "Run this helper only after all main updates finish."
    assert len(spent["smoke"]) == 12
    main_ids = {fit_id for fit_id, _ in spent["main"]}
    assert len(main_ids) == 24
    assert all({step for fid, step in spent["main"] if fid == fit_id} == set(range(1, 513))
               for fit_id in main_ids)
    assert not list((CACHE / "fits").glob("*/pending.json")), "Pending optimizer transaction exists."
    with (HERE / "FIT_LEDGER.csv").open(newline="", encoding="utf-8") as handle:
        fits = list(csv.DictReader(handle))
    assert len(fits) == 24 and {fit["fit_id"] for fit in fits} == main_ids
    assert all(int(fit["main_updates"]) == 512 for fit in fits)
    return ledger_hash


def perform_audit(report):
    report["optimizer_ledger_sha256_before"] = require_finished_training()
    selection_seal_path = HERE / "ALL_SELECTIONS_SEALED.json"
    calibration_path = HERE / "CALIBRATION_PARAMETERS.json"
    report["selection_seal_sha256_before"] = digest(selection_seal_path)
    seal = read_json(selection_seal_path)
    report["calibration_parameters_sha256_before"] = digest(calibration_path)
    assert seal["calibration_sha256"] == report["calibration_parameters_sha256_before"]
    assert seal["model_selection_sha256"] == digest(HERE / "MODEL_SELECTION.json")
    assert seal["lr_selection_sha256"] == digest(HERE / "LR_SELECTION.json")
    selected = read_json(HERE / "MODEL_SELECTION.json")
    assert set(selected) == {f"{source}/{arm}/{seed}" for source in SOURCES
                             for arm in LEARNED[:3] for seed in SEEDS}
    params = read_json(calibration_path)
    assert set(params) == {f"{source}/{key}" for source, key in expected_keys()}

    # Verify every CAL receipt before reading numerical predictions or targets.
    receipts = {}
    for source, key in sorted(expected_keys()):
        path = CACHE / "predictions" / source / "CALIBRATION" / f"{key}.npy"
        receipt = read_json(path.with_suffix(".json"))
        assert receipt["source"] == source and receipt["model"] == key
        assert receipt["role"] == "CALIBRATION"
        assert (ROOT / receipt["path"]).resolve() == path.resolve()
        assert digest(path) == receipt["sha256"]
        receipts[(source, key)] = receipt
    report["calibration_prediction_receipts"] = [
        {"source": source, "model": key, "path": receipt["path"],
         "sha256_before": receipt["sha256"], "shape": receipt["shape"]}
        for (source, key), receipt in receipts.items()]

    audit_path = HERE / "DATA_AND_SPLIT_AUDIT.json"
    report["data_audit_sha256"] = digest(audit_path)
    data_audit = read_json(audit_path)["sources"]
    report["data_cache_receipts"] = []
    report["blocks"] = []
    maximum_error = 0.0
    for source in SOURCES:
        data_path = CACHE / "data" / f"{source}.npz"
        data_hash_before = digest(data_path)
        with np.load(data_path, allow_pickle=True) as data:
            values = data["values"].astype(np.float32)
            sigma = data["sigma"].astype(np.float64)
            origins = data["origins_CALIBRATION"].astype(np.int64)
            ids = [str(item) for item in data["ids"].tolist()]
        assert ids == data_audit[source]["ids"]
        np.testing.assert_array_equal(sigma, [data_audit[source]["train_sigma_population"][sid] for sid in ids])
        assert np.isfinite(sigma).all() and (sigma > 0).all()
        lower, upper = int(0.60 * len(values)), int(0.70 * len(values))
        assert data_audit[source]["split_edges"]["CALIBRATION"] == {"start": lower, "end": upper}
        assert len(origins) == data_audit[source]["origin_audit"]["CALIBRATION"]["n_origins"]
        assert (origins % 24 == 0).all() and (origins >= lower).all()
        assert (origins + 256 <= upper).all() and (origins >= 512).all()
        # No TRAIN/VALIDATION/TEST target panel is constructed or scored.
        targets = np.stack([values[origin:origin + 256].T for origin in origins]).astype(np.float64)
        assert np.isfinite(targets).all()
        del values
        for _, key in sorted(pair for pair in expected_keys() if pair[0] == source):
            receipt = receipts[(source, key)]
            predictions = np.load(ROOT / receipt["path"], mmap_mode="r")
            assert list(predictions.shape) == receipt["shape"] == [len(origins), len(ids), 256, 9]
            assert np.isfinite(predictions).all()
            model_parameters = params[f"{source}/{key}"]
            assert model_parameters["role_required"] == "CALIBRATION"
            assert model_parameters["quantile_input"] == "ordered_final_output"
            assert model_parameters["feedback_changed"] is False
            assert len(model_parameters["blocks"]) == 4
            for block in range(4):
                begin, end = block * 64, (block + 1) * 64
                ordered = np.sort(predictions[:, :, begin:end], axis=-1).astype(np.float64)
                target = targets[:, :, begin:end]
                recorded = model_parameters["blocks"][block]
                assert recorded["block"] == block + 1
                assert recorded["lead_start"] == begin + 1 and recorded["lead_end"] == end
                stored = {(float(row["alpha"]), float(row["beta"])): float(row["scaled_pinball"])
                          for row in recorded["grid_scores"]}
                assert len(recorded["grid_scores"]) == len(stored) == 35
                assert set(stored) == {(a, b) for a in ALPHAS for b in BETAS}
                candidates = []
                for alpha in ALPHAS:
                    for beta in BETAS:
                        score = independent_grid_score(target, ordered, sigma, alpha, beta)
                        error = abs(score - stored[(alpha, beta)])
                        maximum_error = max(maximum_error, error)
                        assert error <= TOLERANCE, (source, key, block, alpha, beta, score, stored[(alpha, beta)])
                        candidates.append({"alpha": alpha, "beta": beta, "score": score,
                                           "stored_score": stored[(alpha, beta)], "absolute_error": error})
                winner = min(candidates, key=tie_key)
                stored_winner = min(({"alpha": a, "beta": b, "score": score} for (a, b), score in stored.items()), key=tie_key)
                selected_pair = (float(recorded["alpha"]), float(recorded["beta"]))
                assert selected_pair == (winner["alpha"], winner["beta"]), (
                    "Independent winner differs; report rather than choose new parameters", source, key, block)
                assert selected_pair == (stored_winner["alpha"], stored_winner["beta"])
                assert selected_pair == (model_parameters["alpha"][block], model_parameters["beta"][block])
                assert abs(float(recorded["scaled_pinball"]) - winner["score"]) <= TOLERANCE
                second = sorted(candidates, key=tie_key)[1]
                report["blocks"].append({"source": source, "model": key, "block": block + 1,
                                         "winner": winner, "runner_up_score_gap": second["score"] - winner["score"],
                                         "winner_and_exact_tie_break_agree": True, "grid": candidates})
                del ordered
            del predictions
        assert digest(data_path) == data_hash_before
        report["data_cache_receipts"].append({"source": source, "path": str(data_path.relative_to(ROOT)),
                                              "sha256_before_and_after": data_hash_before,
                                              "target_role": "CALIBRATION", "n_origins": len(origins),
                                              "n_series": len(ids), "first_origin": int(origins[0]),
                                              "last_target_exclusive": int(origins[-1] + 256)})
        del targets

    for record in report["calibration_prediction_receipts"]:
        record["sha256_after"] = digest(ROOT / record["path"])
        assert record["sha256_after"] == record["sha256_before"]
    report["optimizer_ledger_sha256_after"] = digest(HERE / "UPDATE_LEDGER.jsonl")
    report["selection_seal_sha256_after"] = digest(selection_seal_path)
    report["calibration_parameters_sha256_after"] = digest(calibration_path)
    assert report["optimizer_ledger_sha256_before"] == report["optimizer_ledger_sha256_after"]
    assert report["selection_seal_sha256_before"] == report["selection_seal_sha256_after"]
    assert report["calibration_parameters_sha256_before"] == report["calibration_parameters_sha256_after"]
    assert len(report["blocks"]) == 104
    report.update(models_audited=26, blocks_audited=104, grid_points_audited=3640,
                  maximum_absolute_grid_score_error=maximum_error,
                  all_independent_winners_and_tie_breaks_agree=True,
                  optimizer_ledger_unchanged=True, parameters_changed=False,
                  calibration_parameters_and_prediction_hashes_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-calibration-audit", action="store_true", required=True)
    parser.parse_args()
    started = time.perf_counter()
    report = {"status": "RUNNING", "audit_script_path": str(Path(__file__).resolve().relative_to(ROOT)),
              "audit_script_sha256_before": digest(__file__),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "score_absolute_tolerance": TOLERANCE,
              "loss_implementation": "nine independent np.where residual-sign losses, twice-pinball / TRAIN sigma",
              "production_scoring_functions_called": False, "target_role": "CALIBRATION",
              "test_targets_or_test_predictions_read": False,
              "tie_break": "exact score, squared distance from (1,0), alpha, abs(beta), signed beta"}
    try:
        perform_audit(report)
        report["audit_script_sha256_after"] = digest(__file__)
        assert report["audit_script_sha256_after"] == report["audit_script_sha256_before"]
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if (HERE / "UPDATE_LEDGER.jsonl").exists():
            report["optimizer_ledger_sha256_after"] = digest(HERE / "UPDATE_LEDGER.jsonl")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        with OUTPUT.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    print(json.dumps({"status": report["status"], "models": report["models_audited"],
                      "grid_points": report["grid_points_audited"],
                      "max_error": report["maximum_absolute_grid_score_error"],
                      "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
