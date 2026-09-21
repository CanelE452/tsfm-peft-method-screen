from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from common import CACHE, RESULTS, ROOT, save_json, sha


SOURCES = ("Electricity", "ETTh1")
CONTEXT = 512
HORIZON = 256
ROLES = ("TRAIN", "CALIBRATION", "VALIDATION", "TEST")
ECL_URL = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz"
ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
ETT_COLUMNS = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def _data_dir() -> Path:
    path = CACHE / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with urllib.request.urlopen(url, timeout=180) as response:
        path.write_bytes(response.read())


def _github_commit(repo: str, branch: str, file_path: str) -> dict:
    api = f"https://api.github.com/repos/{repo}/commits/{branch}?path={file_path}"
    with urllib.request.urlopen(api, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {"repo": repo, "branch": branch, "path": file_path, "commit": payload["sha"]}


def _role_edges(T: int) -> dict:
    edges = [0, int(0.60 * T), int(0.70 * T), int(0.80 * T), T]
    return {role: {"start": int(lo), "end": int(hi)} for role, lo, hi in zip(ROLES, edges[:-1], edges[1:])}


def role_origins(T: int) -> dict[str, np.ndarray]:
    edges = _role_edges(T)
    out = {}
    for role in ROLES:
        lo, hi = edges[role]["start"], edges[role]["end"]
        first = max(lo, CONTEXT)
        first = ((first + 23) // 24) * 24
        out[role] = np.arange(first, hi - HORIZON + 1, 24, dtype=np.int64)
    return out


def _filter_finite_origins(values: np.ndarray, origins: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    finite = np.isfinite(values)
    for role, candidates in origins.items():
        keep = []
        for origin in candidates:
            window = finite[origin - CONTEXT: origin + HORIZON]
            if window.shape == (CONTEXT + HORIZON, values.shape[1]) and bool(window.all()):
                keep.append(int(origin))
        out[role] = np.asarray(keep, dtype=np.int64)
    return out


def _origin_audit(values: np.ndarray, origins: dict[str, np.ndarray]) -> dict:
    roles = {}
    for role, role_origins_ in origins.items():
        if len(role_origins_):
            target_starts = role_origins_
            target_ends = role_origins_ + HORIZON
            overlaps = np.maximum(0, target_ends[:-1] - target_starts[1:]) if len(role_origins_) > 1 else np.asarray([], dtype=np.int64)
            roles[role] = {
                "n_origins": int(len(role_origins_)),
                "first_origin": int(role_origins_[0]),
                "last_origin": int(role_origins_[-1]),
                "origin_stride_hours_unique": sorted(np.unique(np.diff(role_origins_)).astype(int).tolist()) if len(role_origins_) > 1 else [],
                "all_origin_mod_24_zero": bool(np.all(role_origins_ % 24 == 0)),
                "target_overlap_hours_min": int(overlaps.min()) if len(overlaps) else 0,
                "target_overlap_hours_max": int(overlaps.max()) if len(overlaps) else 0,
                "target_window_first": [int(target_starts[0]), int(target_ends[0])],
                "target_window_last": [int(target_starts[-1]), int(target_ends[-1])],
            }
        else:
            roles[role] = {"n_origins": 0}
    roles["all_windows_finite"] = {
        role: bool(all(np.isfinite(values[o - CONTEXT: o + HORIZON]).all() for o in role_origins_))
        for role, role_origins_ in origins.items()
    }
    return roles


def _select_electricity_columns(matrix: np.ndarray, train_end: int) -> tuple[list[int], dict]:
    train = matrix[:train_end]
    finite = np.isfinite(train).all(axis=0)
    sigma = np.nanstd(train, axis=0, ddof=0)
    eligible = [int(i) for i in range(matrix.shape[1]) if finite[i] and sigma[i] > 0]
    ordered = sorted(eligible, key=lambda i: hashlib.sha256(f"rollout-v1|ECL|{i}".encode("utf-8")).hexdigest())
    selected = ordered[:32]
    return selected, {
        "candidate_columns": int(matrix.shape[1]),
        "eligible_columns_train_finite_positive_sigma": int(len(eligible)),
        "selection_rule": "first 32 by sha256('rollout-v1|ECL|'+zero_based_column_id), no zero-rate/performance filter",
        "selected_zero_based_columns": selected,
        "selected_hashes": {str(i): hashlib.sha256(f"rollout-v1|ECL|{i}".encode("utf-8")).hexdigest() for i in selected},
    }


def _read_electricity() -> tuple[np.ndarray, list[str], dict]:
    raw = _data_dir() / "electricity.txt.gz"
    _download(ECL_URL, raw)
    with gzip.open(raw, "rt", encoding="utf-8") as handle:
        matrix = pd.read_csv(handle, header=None).to_numpy(dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 32:
        raise RuntimeError("BLOCKED_DATA: unexpected Electricity matrix schema")
    train_end = int(0.60 * matrix.shape[0])
    columns, selection = _select_electricity_columns(matrix, train_end)
    values = matrix[:, columns].astype(np.float32, copy=True)
    ids = [str(i) for i in columns]
    audit = {
        "source_url": ECL_URL,
        "upstream": _github_commit("laiguokun/multivariate-time-series-data", "master", "electricity/electricity.txt.gz"),
        "raw_path": str(raw.relative_to(ROOT)),
        "raw_sha256": sha(raw),
        "raw_bytes": raw.stat().st_size,
        "schema": {"format": "gzip csv matrix", "rows_time_axis": True, "timestamp_column": False, "n_rows": int(matrix.shape[0]), "n_columns_raw": int(matrix.shape[1])},
        "time_axis": {"unit": "hourly_slot_index", "timezone_invented": False, "calendar_dates_verified": False},
        "selection": selection,
    }
    return values, ids, audit


def _read_ett() -> tuple[np.ndarray, list[str], dict]:
    raw = _data_dir() / "ETTh1.csv"
    _download(ETT_URL, raw)
    frame = pd.read_csv(raw)
    if list(frame.columns) != ETT_COLUMNS:
        raise RuntimeError(f"BLOCKED_DATA: unexpected ETTh1 header {list(frame.columns)}")
    times = pd.to_datetime(frame["date"], format="%Y-%m-%d %H:%M:%S")
    diffs = times.diff().dropna()
    expected = pd.Timedelta(hours=1)
    if not bool((diffs == expected).all()) or int(times.duplicated().sum()) != 0:
        raise RuntimeError("BLOCKED_DATA: ETTh1 timestamp axis is not unique hourly")
    numeric = frame[ETT_COLUMNS[1:]].to_numpy(dtype=np.float64)
    values = numeric.astype(np.float32, copy=True)
    audit = {
        "source_url": ETT_URL,
        "upstream": _github_commit("zhouhaoyi/ETDataset", "main", "ETT-small/ETTh1.csv"),
        "raw_path": str(raw.relative_to(ROOT)),
        "raw_sha256": sha(raw),
        "raw_bytes": raw.stat().st_size,
        "schema": {"format": "csv", "header": list(frame.columns), "n_rows": int(len(frame)), "target_columns": ETT_COLUMNS[1:]},
        "time_axis": {
            "timestamp_column": "date",
            "timezone_invented": False,
            "first_timestamp": str(times.iloc[0]),
            "last_timestamp": str(times.iloc[-1]),
            "all_hourly_diffs": bool((diffs == expected).all()),
            "duplicate_timestamps": int(times.duplicated().sum()),
        },
        "selection": {"selected_columns": ETT_COLUMNS[1:], "selection_rule": "all seven official target columns, independent univariate requests"},
    }
    return values, ETT_COLUMNS[1:], audit


def _build_source(source: str) -> dict:
    if source == "Electricity":
        values, ids, audit = _read_electricity()
    elif source == "ETTh1":
        values, ids, audit = _read_ett()
    else:
        raise ValueError(source)
    missing_total = int((~np.isfinite(values)).sum())
    edges = _role_edges(len(values))
    train_values = values[edges["TRAIN"]["start"]: edges["TRAIN"]["end"]].astype(np.float64)
    sigma = train_values.std(axis=0, ddof=0)
    if not np.isfinite(sigma).all() or bool((sigma <= 0).any()):
        raise RuntimeError(f"BLOCKED_DATA: nonpositive TRAIN sigma in {source}")
    origins = _filter_finite_origins(values, role_origins(len(values)))
    counts = {role: int(len(origins[role])) for role in ROLES}
    if any(counts[role] <= 0 for role in ROLES):
        raise RuntimeError(f"BLOCKED_DATA: empty legal role origins in {source}: {counts}")
    audit.update({
        "source": source,
        "selected_shape": {"T": int(values.shape[0]), "N": int(values.shape[1])},
        "ids": ids,
        "split_edges": edges,
        "missing": {"selected_nonfinite_cells": missing_total, "interpolation_applied": False},
        "train_sigma_population": {ids[i]: float(sigma[i]) for i in range(len(ids))},
        "origin_audit": _origin_audit(values, origins),
        "quality_decision": "PASS",
        "leakage_controls": {
            "selection_uses_test_target": False,
            "sigma_fit_role": "TRAIN only",
            "origins_require_future_finite_for_quality_only": True,
            "future_targets_used_in_context": False,
            "timestamps_fabricated_for_electricity": False,
        },
        "prediction0_state_audit": {
            "initial_context_length": CONTEXT,
            "horizon": HORIZON,
            "observed_m_all_zero_before_first_prediction": True,
            "generated_width_all_zero_before_first_prediction": True,
        },
    })
    np.savez_compressed(_data_dir() / f"{source}.npz", values=values, sigma=sigma.astype(np.float64), ids=np.asarray(ids, dtype=object), **{f"origins_{role}": origins[role] for role in ROLES})
    return {"values": values, "sigma": sigma.astype(np.float64), "ids": ids, "origins": origins, "audit": audit}


def prepare() -> dict:
    out = {
        "domain_context": {
            "domain": "hourly time-series probabilistic forecasting",
            "downstream_decision": "screen whether uncertainty-width metadata adds value beyond rollout LoRA/state metadata/baselines",
            "quality_bar": "publication-style pilot receipts; no interpolation; zero leakage tolerance",
            "primary_leakage_paths": ["future targets in rollout context", "TEST-based model or calibration selection", "column selection by performance", "non-TRAIN scaling"],
        },
        "sources": {},
    }
    for source in SOURCES:
        out["sources"][source] = _build_source(source)["audit"]
    for source in SOURCES:
        npz_path = _data_dir() / f"{source}.npz"
        out["sources"][source]["prepared_npz"] = {
            "path": str(npz_path.relative_to(ROOT)),
            "sha256": sha(npz_path),
            "bytes": npz_path.stat().st_size,
        }
        out["sources"][source]["train_schedules"] = {}
        for seed in (92120, 92121, 92122):
            packet = schedule(source, seed)
            schedule_path = _data_dir() / f"schedule_{source}_seed{seed}.npz"
            out["sources"][source]["train_schedules"][str(seed)] = {
                "path": str(schedule_path.relative_to(ROOT)),
                "sha256": sha(schedule_path),
                "bytes": schedule_path.stat().st_size,
                "shape": list(packet.shape),
                "unique_series_indices": int(len(np.unique(packet[..., 0]))),
                "unique_train_origins": int(len(np.unique(packet[..., 1]))),
            }
    save_json(RESULTS / "DATA_AND_SPLIT_AUDIT.json", out)
    return out


def load(source: str) -> dict:
    path = _data_dir() / f"{source}.npz"
    if not path.exists():
        prepare()
    with np.load(path, allow_pickle=True) as z:
        ids = [str(x) for x in z["ids"].tolist()]
        origins = {role: z[f"origins_{role}"].astype(np.int64) for role in ROLES}
        return {
            "values": z["values"].astype(np.float32),
            "sigma": z["sigma"].astype(np.float64),
            "ids": ids,
            "origins": origins,
        }


def schedule(source: str, seed: int) -> np.ndarray:
    path = _data_dir() / f"schedule_{source}_seed{seed}.npz"
    if path.exists():
        with np.load(path) as z:
            return z["tuples"].astype(np.int64)
    data = load(source)
    train_origins = data["origins"]["TRAIN"]
    if len(train_origins) == 0:
        raise RuntimeError(f"BLOCKED_DATA: no TRAIN origins for {source}")
    salt = int(hashlib.sha256(f"rollout-v1|schedule|{source}|{seed}".encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(salt)
    tuples = np.empty((512, 8, 2), dtype=np.int64)
    tuples[..., 0] = rng.integers(0, len(data["ids"]), size=(512, 8), endpoint=False)
    tuples[..., 1] = train_origins[rng.integers(0, len(train_origins), size=(512, 8), endpoint=False)]
    np.savez_compressed(path, tuples=tuples, source=source, seed=np.int64(seed))
    return tuples


if __name__ == "__main__":
    prepare()
