#!/usr/bin/env python3
"""Read-only NumPy/scalar replay of two preserved completed evaluation cohorts.

This does not import experiment/model code, load checkpoints, infer, train, alter
historical files, or reproduce every result in the repository. JSON is printed to
stdout; --output creates a new report and refuses to overwrite an existing file.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

OLD_ROOT = "/home/minjae/Documents/github/tsfm-peft-method-screen"
FAMILIES = {
    "channel_phase_balance_20260916": dict(count=8, channels=32, tolerance=1e-12,
        contract="seal.json", seeds=[41000, 41001], arms=["BALANCED_LH"],
        roles=["selected", "matched_old_epoch"]),
    "channel_basis_pilot_resume_20260915": dict(count=28, channels=64, tolerance=1e-10,
        contract="contract.json", seeds=[40000, 40001],
        arms=["SPECIFIC", "SHARED", "SHARED_WIDE", "LORA_HEAD", "GROUP4", "BASIS4"],
        roles=["selected"]),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(root, recorded):
    """Resolve relative or legacy absolute paths inside an arbitrary restoration."""
    if recorded == OLD_ROOT or recorded.startswith(OLD_ROOT + "/"):
        recorded = recorded[len(OLD_ROOT):].lstrip("/")
    path = Path(recorded)
    if path.is_absolute():
        raise ValueError("Unmapped absolute project input: " + recorded)
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Project input escapes the restored root: " + recorded)
    return resolved


def require_close(actual, expected, tolerance, context):
    if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
        raise ValueError("Nonfinite metric: " + context)
    error = abs(float(actual) - float(expected))
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"Metric mismatch {context}: {actual!r} vs {expected!r}")
    return error


def independent_metrics(prediction, target, std):
    """Scalar math.fsum accumulation, with equal weight for each channel."""
    p = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if p.ndim != 3 or p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError("Expected finite predictions and matching [origin,channel,horizon] arrays")
    std = np.asarray(std, dtype=np.float64)
    if std.shape != (p.shape[1],) or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Invalid TRAIN channel standard deviations")
    channel_mse, channel_mae, counts = [], [], []
    for channel in range(p.shape[1]):
        valid = np.isfinite(y[:, channel, :])
        pp, yy = p[:, channel, :][valid], y[:, channel, :][valid]
        count = len(yy)
        if count == 0:
            raise ValueError("A recorded evaluation channel has no finite targets")
        channel_mse.append(math.fsum((float(a) - float(b)) ** 2 for a, b in zip(pp, yy)) / count)
        channel_mae.append(math.fsum(abs(float(a) - float(b)) for a, b in zip(pp, yy)) / count)
        counts.append(count)
    mse = np.asarray(channel_mse)
    mae = np.asarray(channel_mae)
    result = dict(mse=math.fsum(channel_mse) / len(channel_mse),
                  mae=math.fsum(channel_mae) / len(channel_mae),
                  channel_mse=channel_mse, channel_mae=channel_mae,
                  channel_standardized_mse=channel_mse,
                  channel_raw_mse=(mse * std ** 2).tolist(),
                  channel_raw_mae=(mae * std).tolist(),
                  valid_counts=counts, eligible_channels=len(counts))
    # Separate NumPy reduction checks the scalar implementation without calling
    # any of the project's original metric functions.
    valid = np.isfinite(y)
    errors = np.where(valid, p - y, 0.0)
    count = valid.sum(axis=(0, 2))
    vector = dict(mse=float(((errors ** 2).sum(axis=(0, 2)) / count).mean()),
                  mae=float((np.abs(errors).sum(axis=(0, 2)) / count).mean()))
    return result, vector, int(sum(counts))


def check_receipt_cells(rows, specification):
    expected = set(itertools.product(["electricity", "traffic"], specification["seeds"],
                                     specification["arms"], specification["roles"]))
    if specification["count"] == 28:
        expected |= set(itertools.product(["electricity", "traffic"], specification["seeds"],
                                         ["BASIS4"], ["posthoc_coefficient_swap"]))
    actual = [(r["dataset"], r["seed"], r["arm"], r["role"]) for r in rows]
    if len(rows) != specification["count"] or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Evaluation receipt cell coverage differs from the preserved cohort")


def validate(root):
    reports = []
    all_targets = all_metric_cells = scalar_receipt_cells = 0
    for family, specification in FAMILIES.items():
        folder = project_path(root, "results/" + family)
        receipt = folder / "evaluation.json"
        rows = json.loads(receipt.read_text())
        check_receipt_cells(rows, specification)
        contract_path = folder / specification["contract"]
        contract = json.loads(contract_path.read_text())
        if json.loads((folder / "status.json").read_text())["status"] != "COMPLETE":
            raise ValueError("The selected cohort no longer has its recorded COMPLETE status")
        checks, max_scalar_error, max_metric_error = [], 0.0, 0.0
        metric_cells = target_cells = family_scalar_cells = 0
        for row in rows:
            path = project_path(root, row["prediction_path"])
            if sha256(path) != row["prediction_sha256"]:
                raise ValueError("Prediction SHA-256 mismatch: " + str(path))
            with np.load(path, allow_pickle=False) as arrays:
                p, y, std, origins = (arrays[k] for k in ["prediction", "target", "std", "origins"])
                if p.shape[1:] != (specification["channels"], 96):
                    raise ValueError("Prediction channel/horizon dimensions changed")
                data = contract["data"][row["dataset"]]
                if not np.array_equal(std, np.asarray(data["std"])):
                    raise ValueError("TRAIN scale differs from the original contract")
                if not np.array_equal(origins, np.asarray(data["origins"]["evaluation"])):
                    raise ValueError("Evaluation origins differ from the original contract")
                computed, vector, points = independent_metrics(p, y, std)
                shape = list(p.shape)
            tolerance = specification["tolerance"]
            for key in ["mse", "mae"]:
                max_scalar_error = max(max_scalar_error, require_close(computed[key], vector[key], tolerance, family + ":scalar-vs-numpy:" + key))
                if "scalar_" + key in row:
                    max_scalar_error = max(max_scalar_error, require_close(computed[key], row["scalar_" + key], tolerance, family + ":scalar-receipt:" + key))
                    family_scalar_cells += 1
            for key, expected in row["metrics"].items():
                if key not in computed:
                    raise ValueError("Unsupported recorded metric: " + key)
                observed = np.asarray(computed[key]).reshape(-1)
                expected_array = np.asarray(expected).reshape(-1)
                if observed.shape != expected_array.shape:
                    raise ValueError("Recorded metric dimensions changed: " + key)
                for index, (a, b) in enumerate(zip(observed, expected_array)):
                    error = require_close(a, b, tolerance, f"{family}:{key}[{index}]")
                    max_metric_error = max(max_metric_error, error)
                    metric_cells += 1
            target_cells += points
            checks.append(dict(dataset=row["dataset"], seed=row["seed"], arm=row["arm"],
                               role=row["role"], path=str(path.relative_to(root)), shape=shape,
                               finite_target_values=points, mse=computed["mse"], mae=computed["mae"],
                               sha256_verified=True, origins_and_train_scale_verified=True))
        all_targets += target_cells
        all_metric_cells += metric_cells
        scalar_receipt_cells += family_scalar_cells
        reports.append(dict(family=family, receipts=len(rows), metric_scalar_cells=metric_cells,
                            additional_recorded_scalar_cells=family_scalar_cells,
                            finite_target_values=target_cells, tolerance_abs_and_rel=specification["tolerance"],
                            original_receipt_sha256=sha256(receipt), contract_sha256=sha256(contract_path),
                            max_scalar_vs_numpy_or_receipt_abs_error=max_scalar_error,
                            max_recorded_metric_abs_error=max_metric_error,
                            roles=dict(Counter(row["role"] for row in rows)), checks=checks))
    return dict(schema_version=1, verified=True, restored_root=str(root),
                scope="36 existing evaluation receipts in two completed development cohorts; includes four labeled posthoc coefficient swaps",
                prediction_receipts=sum(r["receipts"] for r in reports),
                metric_scalar_cells=all_metric_cells, additional_recorded_scalar_cells=scalar_receipt_cells,
                finite_target_values=all_targets, families=reports,
                runtime=dict(python=platform.python_version(), numpy=np.__version__),
                side_effects=dict(historical_files_written=0, checkpoint_loads=0, model_imports=0,
                                  model_forwards=0, optimizer_updates=0),
                limits=["Does not verify every experiment, checkpoint or training trajectory.",
                        "Recomputes saved-prediction metrics; does not reproduce predictions from model weights.",
                        "Does not change historical FAIL/STOP decisions or establish new scientific evidence."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Create a new JSON report; existing files are never overwritten")
    args = parser.parse_args()
    report = validate(args.root.expanduser().resolve())
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(content)
        print(json.dumps({k: v for k, v in report.items() if k != "families"}, indent=2))
    else:
        sys.stdout.write(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
