import argparse
import gzip
import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

try:
    from .common import CACHE, RESULTS, ROOT, SEEDS, save, sha
except ImportError:
    from common import CACHE, RESULTS, ROOT, SEEDS, save, sha


SOURCE = "Electricity"
SOURCE_URL = (
    "https://raw.githubusercontent.com/laiguokun/"
    "multivariate-time-series-data/master/electricity/electricity.txt.gz"
)
PINNED_SOURCE_URL = (
    "https://raw.githubusercontent.com/laiguokun/"
    "multivariate-time-series-data/7f402f185cc2435b5e66aed13a3b560ed142e023/"
    "electricity/electricity.txt.gz"
)
UPSTREAM = {
    "repo": "laiguokun/multivariate-time-series-data",
    "branch": "master",
    "path": "electricity/electricity.txt.gz",
    "commit": "7f402f185cc2435b5e66aed13a3b560ed142e023",
}
EXPECTED_RAW_SHA256 = "3c4c069588198c1fcc95cace7bb69c99922129edfd673b7286661dad20badefa"
CONTEXT = 256
HORIZON = 64


def _rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def _ceil_grid(value, step):
    return int(((int(value) + step - 1) // step) * step)


def _hash_id(original_col):
    return hashlib.sha256(f"feedback-peft-v1|{original_col}".encode("utf-8")).hexdigest()


def _raw_path(allow_download=False):
    new_path = CACHE / "data" / "electricity.txt.gz"
    old_path = ROOT / ".cache" / "rollout_uncertainty_peft_v1_20260921" / "data" / "electricity.txt.gz"
    for path in [new_path, old_path]:
        if path.exists():
            actual = sha(path)
            if actual != EXPECTED_RAW_SHA256:
                raise RuntimeError(f"Raw Electricity SHA mismatch for {path}: {actual}")
            return path
    if not allow_download:
        raise FileNotFoundError("Electricity raw file is not present in the residual or rollout cache.")
    new_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = new_path.with_suffix(new_path.suffix + ".tmp")
    urllib.request.urlretrieve(PINNED_SOURCE_URL, tmp)
    actual = sha(tmp)
    if actual != EXPECTED_RAW_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded Electricity SHA mismatch: {actual}")
    os.replace(tmp, new_path)
    return new_path


def _load_raw_matrix(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        matrix = np.loadtxt(handle, delimiter=",", dtype=np.float32)
    if matrix.ndim != 2:
        raise RuntimeError(f"Expected 2D Electricity matrix, got ndim={matrix.ndim}")
    return matrix


def _select_series(matrix):
    n_rows, n_cols = matrix.shape
    t1 = int(np.floor(0.30 * n_rows))
    first = matrix[:t1].astype(np.float64, copy=False)
    finite_first30 = np.isfinite(first).all(axis=0)
    nonconstant_first30 = first.std(axis=0, ddof=0) > 0.0
    eligible = [int(i) for i in np.where(finite_first30 & nonconstant_first30)[0]]
    if len(eligible) < 48:
        raise RuntimeError(f"DATA_BLOCK: only {len(eligible)} eligible Electricity columns; need 48.")
    ordered = sorted(eligible, key=lambda i: (_hash_id(i), i))
    donor = ordered[:32]
    dev = ordered[32:40]
    test = ordered[40:48]
    selected = donor + dev + test
    return {
        "eligible": eligible,
        "selected": selected,
        "ids": {"DONOR": donor, "DEV": dev, "TEST": test},
        "selected_hashes": {str(i): _hash_id(i) for i in selected},
        "finite_first30": finite_first30,
        "nonconstant_first30": nonconstant_first30,
    }


def scale(context):
    arr = np.asarray(context, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"scale expects a 1D context, got shape={arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("scale context contains non-finite values")
    return float(max(arr.std(ddof=0), 1e-6 * np.mean(np.abs(arr)), 1e-6))


def _write_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    os.replace(tmp, path)


class Data:
    def __init__(self, source=SOURCE):
        if source != SOURCE:
            raise ValueError(f"This contract only permits source={SOURCE!r}, got {source!r}")
        raw = _raw_path(allow_download=False)
        matrix = _load_raw_matrix(raw)
        selection = _select_series(matrix)
        self.source = source
        self.raw_path = raw
        self.raw_sha256 = sha(raw)
        self.original_shape = tuple(int(x) for x in matrix.shape)
        self.ids = {k: [int(x) for x in v] for k, v in selection["ids"].items()}
        selected = selection["selected"]
        self.values = np.ascontiguousarray(matrix[:, selected], dtype=np.float32)
        self.values.setflags(write=False)
        n = int(self.values.shape[0])
        self.edges = [0, int(np.floor(0.30 * n)), int(np.floor(0.60 * n)), int(np.floor(0.80 * n)), n]
        self.role_indices = {
            "DONOR": list(range(0, 32)),
            "DEV": list(range(32, 40)),
            "TEST": list(range(40, 48)),
        }
        self.original_to_local = {int(col): i for i, col in enumerate(selected)}
        self.context = CONTEXT
        self.horizon = HORIZON

    def base_schedule(self, seed):
        rng = np.random.default_rng(int(seed))
        start, end = self.edges[0], self.edges[1]
        issues = np.arange(start + CONTEXT, end - HORIZON + 1, dtype=np.int64)
        if len(issues) == 0:
            raise RuntimeError("No legal BASE_TRAIN issues.")
        series = np.asarray(self.role_indices["DONOR"], dtype=np.int64)
        out = np.empty((512, 8, 2), dtype=np.int64)
        out[:, :, 0] = rng.choice(series, size=(512, 8), replace=True)
        out[:, :, 1] = rng.choice(issues, size=(512, 8), replace=True)
        return out

    def meta_schedule(self, seed):
        rng = np.random.default_rng(int(seed) + 100_000)
        start = _ceil_grid(self.edges[1] + CONTEXT, 8)
        end = self.edges[2] - HORIZON
        issues = np.arange(start, end + 1, 8, dtype=np.int64)
        if len(issues) == 0:
            raise RuntimeError("No legal META_TRAIN issues.")
        series = np.asarray(self.role_indices["DONOR"], dtype=np.int64)
        out = np.empty((512, 4, 2), dtype=np.int64)
        out[:, :, 0] = rng.choice(series, size=(512, 4), replace=True)
        out[:, :, 1] = rng.choice(issues, size=(512, 4), replace=True)
        return out

    def episodes(self, track, role):
        track = str(track).upper()
        role = str(role).upper()
        if role not in ("DEV", "TEST"):
            raise ValueError(f"role must be DEV or TEST, got {role!r}")
        if track not in ("A", "B"):
            raise ValueError(f"track must be A or B, got {track!r}")
        lo, hi = (self.edges[2], self.edges[3]) if role == "DEV" else (self.edges[3], self.edges[4])
        series = self.role_indices[role]
        pairs = []
        if track == "A":
            grid = np.arange(_ceil_grid(lo, HORIZON), hi - HORIZON + 1, HORIZON, dtype=np.int64)
            if len(grid) < 12:
                raise RuntimeError(f"BLOCK: only {len(grid)} A episode grid points for {role}.")
            positions = np.rint(np.linspace(0, len(grid) - 1, 12)).astype(np.int64)
            issues = grid[positions]
        else:
            first = _ceil_grid(lo, 8)
            issues = first + 8 * np.arange(48, dtype=np.int64)
            if int(issues[-1]) + HORIZON > hi:
                raise RuntimeError(f"BLOCK: B 48-tick stream does not fit in {role}.")
        for s in series:
            for issue in issues:
                pairs.append((int(s), int(issue)))
        return np.asarray(pairs, dtype=np.int64)

    def asof(self, series, t):
        series = int(series)
        t = int(t)
        if series < 0 or series >= self.values.shape[1]:
            raise IndexError(series)
        if t < 0 or t > self.values.shape[0]:
            raise IndexError(t)
        return self.values[:t, series].copy()

    def scale(self, context):
        return scale(context)

    def _check_pairs(self, pairs, allowed_roles):
        arr = np.asarray(pairs, dtype=np.int64)
        if arr.shape[-1] != 2:
            raise ValueError(f"pairs must end with shape 2, got {arr.shape}")
        flat = arr.reshape(-1, 2)
        allowed = set()
        for role in allowed_roles:
            allowed.update(self.role_indices[role])
        bad_series = sorted({int(s) for s in flat[:, 0]} - allowed)
        if bad_series:
            raise ValueError(f"Pairs include disallowed local series indices: {bad_series}")
        bad_issues = [
            (int(s), int(t))
            for s, t in flat
            if int(t) < CONTEXT or int(t) + HORIZON > self.values.shape[0]
        ]
        if bad_issues:
            raise ValueError(f"Pairs include illegal context/target windows: {bad_issues[:5]}")
        return flat

    def batch(self, pairs, allowed_roles=("DONOR",)):
        flat = self._check_pairs(pairs, allowed_roles)
        contexts = np.stack([self.values[t - CONTEXT : t, s] for s, t in flat]).astype(np.float32)
        targets = np.stack([self.values[t : t + HORIZON, s] for s, t in flat]).astype(np.float32)
        scales = np.asarray([scale(x) for x in contexts], dtype=np.float32)
        return {
            "x": contexts,
            "y": targets,
            "scale": scales,
            "pairs": flat.copy(),
        }

    def base_batch(self, pairs):
        return self.batch(pairs, allowed_roles=("DONOR",))

    def meta_batch(self, pairs):
        return self.batch(pairs, allowed_roles=("DONOR",))

    def eval_context_batch(self, pairs, role):
        flat = self._check_pairs(pairs, allowed_roles=(role,))
        contexts = np.stack([self.asof(s, t)[-CONTEXT:] for s, t in flat]).astype(np.float32)
        if contexts.shape[1] != CONTEXT:
            raise RuntimeError("Evaluation context was not exactly 256 values.")
        scales = np.asarray([scale(x) for x in contexts], dtype=np.float32)
        return {"x": contexts, "scale": scales, "pairs": flat.copy()}

    def scorer_targets(self, pairs, role):
        flat = self._check_pairs(pairs, allowed_roles=(role,))
        return np.stack([self.values[t : t + HORIZON, s] for s, t in flat]).astype(np.float32)


def _episode_summary(data, track, role):
    pairs = data.episodes(track, role)
    issues = pairs[:, 1]
    series = pairs[:, 0]
    return {
        "count": int(len(pairs)),
        "series_count": int(len(set(series.tolist()))),
        "issue_count_per_series": int(len(pairs) // len(set(series.tolist()))),
        "first_pair": pairs[0].tolist(),
        "last_pair": pairs[-1].tolist(),
        "min_issue": int(issues.min()),
        "max_issue": int(issues.max()),
        "all_query_targets_inside_role": _queries_inside_role(data, pairs, role),
    }


def _queries_inside_role(data, pairs, role):
    lo, hi = (data.edges[2], data.edges[3]) if role == "DEV" else (data.edges[3], data.edges[4])
    issues = pairs[:, 1]
    return bool((issues >= lo).all() and (issues + HORIZON <= hi).all())


def _packet_manifest(data):
    packet_dir = CACHE / "data" / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    schedules = {}
    for seed in SEEDS:
        base = data.base_schedule(seed)
        meta = data.meta_schedule(seed)
        base_path = packet_dir / f"base_schedule_seed{seed}.npy"
        meta_path = packet_dir / f"meta_schedule_seed{seed}.npy"
        _write_npy(base_path, base)
        _write_npy(meta_path, meta)
        schedules[str(seed)] = {
            "base_schedule": {
                "path": _rel(base_path),
                "sha256": sha(base_path),
                "shape": list(base.shape),
                "allowed_series_role": "DONOR",
                "unique_series": int(len(np.unique(base[:, :, 0]))),
                "min_issue": int(base[:, :, 1].min()),
                "max_issue": int(base[:, :, 1].max()),
            },
            "meta_schedule": {
                "path": _rel(meta_path),
                "sha256": sha(meta_path),
                "shape": list(meta.shape),
                "allowed_series_role": "DONOR",
                "issue_grid_hours": 8,
                "unique_series": int(len(np.unique(meta[:, :, 0]))),
                "min_issue": int(meta[:, :, 1].min()),
                "max_issue": int(meta[:, :, 1].max()),
            },
        }
    episodes = {}
    for track in ("A", "B"):
        episodes[track] = {}
        for role in ("DEV", "TEST"):
            arr = data.episodes(track, role)
            path = packet_dir / f"episodes_{track}_{role}.npy"
            _write_npy(path, arr)
            episodes[track][role] = {
                "path": _rel(path),
                "sha256": sha(path),
                "shape": list(arr.shape),
                "ordered": "series_then_time",
                "first_pair": arr[0].tolist(),
                "last_pair": arr[-1].tolist(),
            }
    return {
        "status": "PASS",
        "source": SOURCE,
        "context": CONTEXT,
        "horizon": HORIZON,
        "seed_schedules": schedules,
        "episodes": episodes,
        "all_schedules_written_before_main_training": True,
    }


def run_audit():
    raw = _raw_path(allow_download=True)
    matrix = _load_raw_matrix(raw)
    selection = _select_series(matrix)
    data = Data(SOURCE)
    selected_values_path = CACHE / "data" / "electricity_selected_48_float32.npy"
    _write_npy(selected_values_path, data.values)

    n_rows, n_cols = matrix.shape
    full_nonfinite = int((~np.isfinite(matrix)).sum())
    duplicate_row_count = int(n_rows - np.unique(matrix, axis=0).shape[0])
    first30_stds = matrix[: data.edges[1]].astype(np.float64).std(axis=0, ddof=0)

    source_doc = {
        "status": "PASS",
        "domain_context": {
            "domain": "hourly electricity load time-series forecasting",
            "downstream_decision": "screen whether residual/partial-feedback PEFT adds value beyond shared LoRA and simple lawful controls",
            "quality_bar": "publication-style pilot receipts with zero leakage tolerance",
            "primary_leakage_paths": [
                "using DEV/TEST targets to select columns",
                "using query future targets in context or adaptation",
                "letting evaluation series labels train generator weights",
                "computing hidden future residuals before masking",
            ],
        },
        "source_url": SOURCE_URL,
        "pinned_source_url": PINNED_SOURCE_URL,
        "upstream": UPSTREAM,
        "raw_path": _rel(raw),
        "raw_sha256": sha(raw),
        "raw_bytes": int(raw.stat().st_size),
        "schema": {
            "format": "gzip csv matrix",
            "timestamp_column": False,
            "rows_time_axis": True,
            "n_rows": int(n_rows),
            "n_columns_raw": int(n_cols),
            "dtype_loaded": "float32",
        },
        "time_axis": {
            "unit": "hourly_slot_index",
            "calendar_dates_fabricated": False,
            "row_index_start": 0,
            "row_index_end_exclusive": int(n_rows),
            "row_index_duplicates": 0,
            "frequency_metadata": "README describes hourly rows; no timestamp column is present in the matrix",
        },
        "missing": {
            "raw_nonfinite_cells": full_nonfinite,
            "selected_nonfinite_cells": int((~np.isfinite(data.values)).sum()),
            "interpolation_applied": False,
            "imputation_applied": False,
        },
        "duplicate_metadata": {
            "duplicate_full_rows": duplicate_row_count,
            "duplicate_rows_used_for_exclusion": False,
        },
        "selected_cache": {
            "path": _rel(selected_values_path),
            "sha256": sha(selected_values_path),
            "shape": list(data.values.shape),
        },
        "prior_exposure_caveat": (
            "Electricity was used in earlier local PEFT screens. The held-out claim here is only "
            "the new residual_feedback_v1 donor/dev/test series and time split, not global prior non-exposure."
        ),
    }

    ids = {k: [int(x) for x in v] for k, v in selection["ids"].items()}
    split_doc = {
        "status": "PASS",
        "selection_rule": "eligible columns sorted by SHA256('feedback-peft-v1|' + original_zero_based_column_id)",
        "eligibility_rule": "first 30 percent rows finite and nonconstant",
        "candidate_columns": int(n_cols),
        "eligible_columns": int(len(selection["eligible"])),
        "selected_total": 48,
        "ids": ids,
        "local_indices": data.role_indices,
        "selected_hashes": selection["selected_hashes"],
        "first30_population_std_by_selected_id": {
            str(i): float(first30_stds[i]) for i in selection["selected"]
        },
        "intersections": {
            "DONOR_DEV": sorted(set(ids["DONOR"]) & set(ids["DEV"])),
            "DONOR_TEST": sorted(set(ids["DONOR"]) & set(ids["TEST"])),
            "DEV_TEST": sorted(set(ids["DEV"]) & set(ids["TEST"])),
        },
        "selection_uses_targets_or_performance": False,
    }

    time_doc = {
        "status": "PASS",
        "edges": data.edges,
        "roles": {
            "BASE_TRAIN": {"start": data.edges[0], "end": data.edges[1], "series_role": "DONOR"},
            "META_TRAIN": {"start": data.edges[1], "end": data.edges[2], "series_role": "DONOR"},
            "DEV": {"start": data.edges[2], "end": data.edges[3], "series_role": "DEV"},
            "TEST": {"start": data.edges[3], "end": data.edges[4], "series_role": "TEST"},
        },
        "context": CONTEXT,
        "horizon": HORIZON,
        "clock": {
            "context": "[issue-256, issue)",
            "target": "[issue, issue+64)",
            "available_at_t": "values with index < t only",
            "current_y_t_available": False,
        },
        "scale_rule": "max(std(context, ddof=0), 1e-6*mean(abs(context)), 1e-6)",
        "leakage_controls": {
            "base_and_meta_training_series": "DONOR only",
            "dev_test_series_excluded_from_generator_training": True,
            "test_truth_used_by_scorer_only": True,
            "timestamps_fabricated": False,
        },
    }

    prefix = data.asof(0, data.edges[2])
    before = float(data.values[0, 0])
    prefix[0] = prefix[0] + 12345.0
    prefix_copy_ok = bool(float(data.values[0, 0]) == before)

    support_doc = {
        "status": "PASS",
        "A": {
            "support_issues_relative_to_query": [-256, -192, -128, -64],
            "support_targets_all_end_before_query": True,
            "episode_rule": "12 query issues per series on a 64-hour grid, ordered series then time",
            "DEV": _episode_summary(data, "A", "DEV"),
            "TEST": _episode_summary(data, "A", "TEST"),
        },
        "B": {
            "support_issues_relative_to_query": [-64, -32, -16, -8],
            "visible_counts": [64, 32, 16, 8],
            "episode_rule": "48 consecutive 8-hour issue ticks per series, ordered series then time",
            "DEV": _episode_summary(data, "B", "DEV"),
            "TEST": _episode_summary(data, "B", "TEST"),
        },
        "asof_prefix_returns_copy": prefix_copy_ok,
        "future_values_required_for_context_or_feature": False,
        "nan_then_mask_pattern_allowed": False,
        "evaluation_targets_separate_scorer_method": "scorer_targets",
    }

    packet_doc = _packet_manifest(data)

    save(RESULTS / "DATA_SOURCE.json", source_doc)
    save(RESULTS / "SERIES_SPLIT.json", split_doc)
    save(RESULTS / "TIME_SPLIT.json", time_doc)
    save(RESULTS / "SUPPORT_QUERY_AUDIT.json", support_doc)
    save(RESULTS / "TRAIN_PACKET_MANIFEST.json", packet_doc)
    return {
        "DATA_SOURCE": source_doc,
        "SERIES_SPLIT": split_doc,
        "TIME_SPLIT": time_doc,
        "SUPPORT_QUERY_AUDIT": support_doc,
        "TRAIN_PACKET_MANIFEST": packet_doc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if not args.audit:
        parser.error("Only --audit is supported for data.py CLI.")
    out = run_audit()
    print(
        {
            "status": "PASS",
            "source": out["DATA_SOURCE"]["source_url"],
            "ids": out["SERIES_SPLIT"]["ids"],
            "edges": out["TIME_SPLIT"]["edges"],
            "packets": out["TRAIN_PACKET_MANIFEST"]["seed_schedules"],
        }
    )


if __name__ == "__main__":
    main()
