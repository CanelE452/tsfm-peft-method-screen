from __future__ import annotations

import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NAME = "branch_mixture_lora_v1_20260922"
ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / NAME
RESULTS = ROOT / "results" / NAME
CACHE = ROOT / ".cache" / NAME

SOURCE_REPO = "zhouhaoyi/ETDataset"
SOURCE_BRANCH = "main"
SOURCE_PATH = "ETT-small/ETTh2.csv"
SOURCE_URL = f"https://github.com/{SOURCE_REPO}/blob/{SOURCE_BRANCH}/{SOURCE_PATH}"
ETT_COLUMNS = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

CONTEXT = 512
HORIZON = 128
STRIDE = 24
TRAIN_UPDATES = 512
BATCH_SIZE = 4
SEEDS = (92231, 92232)
ROLES = ("TRAIN", "CALIBRATION", "VALIDATION", "TEST")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _data_dir() -> Path:
    path = CACHE / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _github_commit() -> dict[str, str]:
    api = f"https://api.github.com/repos/{SOURCE_REPO}/commits/{SOURCE_BRANCH}?path={SOURCE_PATH}"
    request = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {
        "repo": SOURCE_REPO,
        "branch": SOURCE_BRANCH,
        "path": SOURCE_PATH,
        "commit": payload["sha"],
        "api_url": api,
    }


def _raw_url(commit: str) -> str:
    return f"https://raw.githubusercontent.com/{SOURCE_REPO}/{commit}/{SOURCE_PATH}"


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    request = urllib.request.Request(url, headers={"User-Agent": "codex-data-audit"})
    with urllib.request.urlopen(request, timeout=180) as response:
        path.write_bytes(response.read())


def _role_edges(T: int) -> dict[str, dict[str, int]]:
    edges = [0, int(0.60 * T), int(0.70 * T), int(0.80 * T), T]
    return {
        role: {"start": int(lo), "end": int(hi)}
        for role, lo, hi in zip(ROLES, edges[:-1], edges[1:])
    }


def role_origins(T: int) -> dict[str, np.ndarray]:
    edges = _role_edges(T)
    origins = {}
    for role in ROLES:
        lo = edges[role]["start"]
        hi = edges[role]["end"]
        first = max(lo, CONTEXT)
        first = ((first + STRIDE - 1) // STRIDE) * STRIDE
        origins[role] = np.arange(first, hi - HORIZON + 1, STRIDE, dtype=np.int64)
    return origins


def _filter_finite_origins(values: np.ndarray, origins: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    finite = np.isfinite(values)
    filtered = {}
    for role, candidates in origins.items():
        keep = []
        for origin in candidates:
            window = finite[origin - CONTEXT : origin + HORIZON]
            if window.shape == (CONTEXT + HORIZON, values.shape[1]) and bool(window.all()):
                keep.append(int(origin))
        filtered[role] = np.asarray(keep, dtype=np.int64)
    return filtered


def _origin_audit(values: np.ndarray, origins: dict[str, np.ndarray]) -> dict[str, Any]:
    edges = _role_edges(len(values))
    audit: dict[str, Any] = {}
    for role, role_origins_ in origins.items():
        lo = edges[role]["start"]
        hi = edges[role]["end"]
        if len(role_origins_) == 0:
            audit[role] = {"n_origins": 0, "target_windows_inside_role": False}
            continue
        target_starts = role_origins_
        target_ends = role_origins_ + HORIZON
        context_starts = role_origins_ - CONTEXT
        overlaps = (
            np.maximum(0, target_ends[:-1] - target_starts[1:])
            if len(role_origins_) > 1
            else np.asarray([], dtype=np.int64)
        )
        audit[role] = {
            "n_origins": int(len(role_origins_)),
            "first_origin": int(role_origins_[0]),
            "last_origin": int(role_origins_[-1]),
            "origin_stride_hours_unique": (
                sorted(np.unique(np.diff(role_origins_)).astype(int).tolist())
                if len(role_origins_) > 1
                else []
            ),
            "all_origin_mod_24_zero": bool(np.all(role_origins_ % STRIDE == 0)),
            "context_window_first": [int(context_starts[0]), int(role_origins_[0])],
            "context_window_last": [int(context_starts[-1]), int(role_origins_[-1])],
            "target_window_first": [int(target_starts[0]), int(target_ends[0])],
            "target_window_last": [int(target_starts[-1]), int(target_ends[-1])],
            "target_windows_inside_role": bool(np.all(target_starts >= lo) and np.all(target_ends <= hi)),
            "context_ends_before_target_starts": bool(np.all(role_origins_ <= target_starts)),
            "target_overlap_hours_min": int(overlaps.min()) if len(overlaps) else 0,
            "target_overlap_hours_max": int(overlaps.max()) if len(overlaps) else 0,
        }
    audit["all_windows_finite"] = {
        role: bool(all(np.isfinite(values[o - CONTEXT : o + HORIZON]).all() for o in role_origins_))
        for role, role_origins_ in origins.items()
    }
    return audit


def _repository_exposure() -> dict[str, Any]:
    command = [
        "rg",
        "-n",
        "--glob",
        "!.git/**",
        "--glob",
        "!.cache/**",
        "--glob",
        f"!experiments/{NAME}/**",
        "--glob",
        f"!results/{NAME}/**",
        "--glob",
        "!results/**/*.json",
        "--glob",
        "!results/**/*.csv",
        "--glob",
        "!results/**/*.jsonl",
        "--glob",
        "!results/**/*.xml",
        "ETTh2",
        "docs",
        "experiments",
        "results",
        "research",
        "papers",
        "src",
        "tests",
        "configs",
        "scripts",
        "README.md",
    ]
    try:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        return {
            "tool": "rg",
            "returncode": int(proc.returncode),
            "match_count": len(lines),
            "matches": lines[:50],
            "conclusion": (
                "ETTh2 is mentioned in this repository; treat this as a same-family public ETT source,"
                " not a fully unseen independent source."
                if lines
                else "No repository ETTh2 mention found by scoped rg."
            ),
        }
    except Exception as exc:  # pragma: no cover - rg may be unavailable on a new machine.
        return {"tool": "rg", "error": repr(exc), "match_count": None, "matches": []}


def _read_ett() -> tuple[np.ndarray, list[str], dict[str, Any]]:
    upstream = _github_commit()
    raw = _data_dir() / f"ETTh2_{upstream['commit']}.csv"
    pinned_url = _raw_url(upstream["commit"])
    _download(pinned_url, raw)
    frame = pd.read_csv(raw)
    if list(frame.columns) != ETT_COLUMNS:
        raise RuntimeError(f"BLOCKED_DATA: unexpected ETTh2 header {list(frame.columns)}")
    times = pd.to_datetime(frame["date"], format="%Y-%m-%d %H:%M:%S")
    diffs = times.diff().dropna()
    hourly = pd.Timedelta(hours=1)
    if int(times.duplicated().sum()) != 0 or not bool((diffs == hourly).all()):
        raise RuntimeError("BLOCKED_DATA: ETTh2 timestamp axis is not unique hourly")
    numeric = frame[ETT_COLUMNS[1:]].to_numpy(dtype=np.float64)
    if numeric.ndim != 2 or numeric.shape[1] != 7:
        raise RuntimeError("BLOCKED_DATA: ETTh2 numeric matrix is not T x 7")
    values = numeric.astype(np.float32, copy=True)
    audit = {
        "source_url": SOURCE_URL,
        "pinned_raw_url": pinned_url,
        "upstream": upstream,
        "raw_path": str(raw.relative_to(ROOT)),
        "raw_sha256": sha(raw),
        "raw_bytes": raw.stat().st_size,
        "schema": {
            "format": "csv",
            "header": list(frame.columns),
            "n_rows": int(len(frame)),
            "target_columns": ETT_COLUMNS[1:],
            "values_shape": [int(values.shape[0]), int(values.shape[1])],
        },
        "time_axis": {
            "timestamp_column": "date",
            "timezone_invented": False,
            "first_timestamp": str(times.iloc[0]),
            "last_timestamp": str(times.iloc[-1]),
            "all_hourly_diffs": bool((diffs == hourly).all()),
            "duplicate_timestamps": int(times.duplicated().sum()),
            "expected_rows_from_timestamp_span": int((times.iloc[-1] - times.iloc[0]) / hourly) + 1,
        },
        "selection": {
            "selected_columns": ETT_COLUMNS[1:],
            "selection_rule": "all seven official ETTh2 target columns; no performance or missingness based column selection",
        },
    }
    return values, ETT_COLUMNS[1:], audit


def _build_dataset() -> dict[str, Any]:
    values, ids, audit = _read_ett()
    missing_total = int((~np.isfinite(values)).sum())
    edges = _role_edges(len(values))
    train = values[edges["TRAIN"]["start"] : edges["TRAIN"]["end"]].astype(np.float64)
    sigma = train.std(axis=0, ddof=0)
    if missing_total != 0:
        raise RuntimeError(f"BLOCKED_DATA: ETTh2 selected cells contain {missing_total} nonfinite values")
    if not np.isfinite(sigma).all() or bool((sigma <= 0).any()):
        raise RuntimeError("BLOCKED_DATA: nonpositive TRAIN sigma in ETTh2")
    origins = _filter_finite_origins(values, role_origins(len(values)))
    counts = {role: int(len(origins[role])) for role in ROLES}
    if any(counts[role] <= 0 for role in ROLES):
        raise RuntimeError(f"BLOCKED_DATA: empty legal role origins: {counts}")
    npz_path = _data_dir() / "ETTh2.npz"
    np.savez_compressed(
        npz_path,
        values=values,
        sigma=sigma.astype(np.float64),
        ids=np.asarray(ids, dtype=object),
        **{f"origins_{role}": origins[role] for role in ROLES},
    )
    audit.update(
        {
            "dataset": "ETTh2",
            "selected_shape": {"T": int(values.shape[0]), "N": int(values.shape[1])},
            "ids": ids,
            "split_policy": "time-ordered fractions TRAIN 60%, CALIBRATION 10%, VALIDATION 10%, TEST 20%",
            "split_edges": edges,
            "missing": {"selected_nonfinite_cells": missing_total, "interpolation_applied": False},
            "train_sigma_population": {ids[i]: float(sigma[i]) for i in range(len(ids))},
            "origin_audit": _origin_audit(values, origins),
            "leakage_controls": {
                "sigma_fit_role": "TRAIN only",
                "origin_generation_uses_test_targets_for_selection": False,
                "origins_require_future_finite_for_quality_only": True,
                "future_targets_used_in_context": False,
                "target_wholly_inside_role": True,
                "schedule_uses_training_origins_only": True,
            },
            "prepared_npz": {
                "path": str(npz_path.relative_to(ROOT)),
                "sha256": sha(npz_path),
                "bytes": npz_path.stat().st_size,
            },
            "quality_decision": "PASS",
        }
    )
    return {"values": values, "sigma": sigma.astype(np.float64), "ids": ids, "origins": origins, "audit": audit}


def prepare() -> dict[str, Any]:
    built = _build_dataset()
    audit = {
        "domain_context": {
            "domain": "hourly multivariate time-series forecasting used as independent univariate targets",
            "downstream_decision": "screen whether native branching mixture-loss LoRA adds value over per-branch component loss under identical Chronos rollout data",
            "quality_bar": "publication-style pilot receipts; zero leakage tolerance; no interpolation",
            "primary_leakage_paths": [
                "future target values entering rollout context",
                "TEST-based model, LR, checkpoint, or loss selection",
                "non-TRAIN scaling",
                "choosing columns, origins, or seeds after seeing outcomes",
            ],
        },
        "budget": {
            "train_updates": TRAIN_UPDATES,
            "batch_size": BATCH_SIZE,
            "seeds": list(SEEDS),
            "schedule_shape": [TRAIN_UPDATES, BATCH_SIZE, 2],
            "context": CONTEXT,
            "horizon": HORIZON,
            "stride": STRIDE,
        },
        "repository_exposure": _repository_exposure(),
        "source": built["audit"],
    }
    audit["source"]["train_schedules"] = {}
    for seed in SEEDS:
        packet = schedule(seed)
        schedule_path = _data_dir() / f"schedule_ETTh2_seed{seed}.npz"
        audit["source"]["train_schedules"][str(seed)] = {
            "path": str(schedule_path.relative_to(ROOT)),
            "sha256": sha(schedule_path),
            "bytes": schedule_path.stat().st_size,
            "shape": list(packet.shape),
            "unique_series_indices": int(len(np.unique(packet[..., 0]))),
            "unique_train_origins": int(len(np.unique(packet[..., 1]))),
            "selection_rule": "fixed before outcome by sha256-seeded numpy RNG; identical schedule is reused across arms",
        }
    save_json(RESULTS / "DATA_AUDIT.json", audit)
    return audit


def load() -> dict[str, Any]:
    path = _data_dir() / "ETTh2.npz"
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


def schedule(seed: int) -> np.ndarray:
    path = _data_dir() / f"schedule_ETTh2_seed{seed}.npz"
    if path.exists():
        with np.load(path) as z:
            return z["tuples"].astype(np.int64)
    data = load()
    train_origins = data["origins"]["TRAIN"]
    if len(train_origins) == 0:
        raise RuntimeError("BLOCKED_DATA: no TRAIN origins for ETTh2")
    salt = int(hashlib.sha256(f"{NAME}|schedule|ETTh2|{seed}".encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(salt)
    tuples = np.empty((TRAIN_UPDATES, BATCH_SIZE, 2), dtype=np.int64)
    tuples[..., 0] = rng.integers(0, len(data["ids"]), size=(TRAIN_UPDATES, BATCH_SIZE), endpoint=False)
    tuples[..., 1] = train_origins[
        rng.integers(0, len(train_origins), size=(TRAIN_UPDATES, BATCH_SIZE), endpoint=False)
    ]
    np.savez_compressed(path, tuples=tuples, dataset="ETTh2", seed=np.int64(seed))
    return tuples


def batch(data: dict[str, Any], pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(pairs, dtype=np.int64)
    flat = arr.reshape(-1, 2)
    values = np.asarray(data["values"], dtype=np.float32)
    sigma_all = np.asarray(data["sigma"], dtype=np.float64)
    x = np.empty((len(flat), CONTEXT), dtype=np.float32)
    y = np.empty((len(flat), HORIZON), dtype=np.float32)
    sigma = np.empty((len(flat),), dtype=np.float64)
    for i, (series_idx, origin) in enumerate(flat):
        if series_idx < 0 or series_idx >= values.shape[1]:
            raise IndexError(f"series index out of range: {series_idx}")
        if origin - CONTEXT < 0 or origin + HORIZON > values.shape[0]:
            raise IndexError(f"origin does not support context+horizon: {origin}")
        x[i] = values[origin - CONTEXT : origin, series_idx]
        y[i] = values[origin : origin + HORIZON, series_idx]
        sigma[i] = sigma_all[series_idx]
    return x.reshape(arr.shape[:-1] + (CONTEXT,)), y.reshape(arr.shape[:-1] + (HORIZON,)), sigma.reshape(arr.shape[:-1])


if __name__ == "__main__":
    prepare()
