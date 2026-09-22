from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NAME = "continuation_lora_v1_20260922"
ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / NAME
RESULTS = ROOT / "results" / NAME
CACHE = ROOT / ".cache" / NAME

SOURCE_REPO = "laiguokun/multivariate-time-series-data"
SOURCE_BRANCH = "master"
SOURCE_PATH = "solar-energy/solar_AL.txt.gz"
SOURCE_URL = f"https://github.com/{SOURCE_REPO}/blob/{SOURCE_BRANCH}/{SOURCE_PATH}"
README_URL = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_BRANCH}/README.md"

RAW_ROWS = 52560
RAW_COLUMNS = 137
AGGREGATION_ROWS = 6
HOURLY_ROWS = 8760
SELECTED_COLUMNS = 8
CONTEXT = 512
HORIZON = 128
STRIDE = 24
TRAIN_UPDATES = 512
BATCH_SIZE = 4
SEEDS = (92241, 92242)
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


def _github_json(url: str) -> dict[str, Any] | list[Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "codex-data-audit"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _github_commit() -> dict[str, str]:
    api = f"https://api.github.com/repos/{SOURCE_REPO}/commits/{SOURCE_BRANCH}?path={SOURCE_PATH}"
    payload = _github_json(api)
    assert isinstance(payload, dict)
    return {
        "repo": SOURCE_REPO,
        "branch": SOURCE_BRANCH,
        "path": SOURCE_PATH,
        "commit": payload["sha"],
        "commit_date": payload["commit"]["committer"]["date"],
        "html_url": payload["html_url"],
        "api_url": api,
    }


def _contents_metadata() -> dict[str, Any]:
    api = f"https://api.github.com/repos/{SOURCE_REPO}/contents/{SOURCE_PATH}?ref={SOURCE_BRANCH}"
    payload = _github_json(api)
    assert isinstance(payload, dict)
    return {
        "api_url": api,
        "name": payload["name"],
        "path": payload["path"],
        "type": payload["type"],
        "size": int(payload["size"]),
        "git_blob_sha": payload["sha"],
        "download_url": payload["download_url"],
    }


def _readme_evidence() -> dict[str, Any]:
    request = urllib.request.Request(README_URL, headers={"User-Agent": "codex-data-audit"})
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8")
    lines = text.splitlines()
    keep = []
    for i, line in enumerate(lines, start=1):
        if "T lines" in line or "Solar Energy" in line or "solar power production" in line:
            keep.append({"line": i, "text": line.strip()})
    return {
        "url": README_URL,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "evidence_lines": keep,
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


def aggregate_hourly(raw: np.ndarray) -> np.ndarray:
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] % AGGREGATION_ROWS != 0:
        raise RuntimeError(f"BLOCKED_DATA: cannot aggregate shape {matrix.shape}")
    return matrix.reshape(matrix.shape[0] // AGGREGATION_ROWS, AGGREGATION_ROWS, matrix.shape[1]).mean(axis=1)


def _select_columns(hourly: np.ndarray, train_end: int) -> tuple[list[int], dict[str, Any]]:
    train = hourly[:train_end]
    finite = np.isfinite(train).all(axis=0)
    sigma = np.nanstd(train, axis=0, ddof=0)
    eligible = [int(i) for i in range(hourly.shape[1]) if finite[i] and sigma[i] > 0]
    ordered = sorted(eligible, key=lambda i: hashlib.sha256(f"continuation-v1|solar|{i}".encode("utf-8")).hexdigest())
    selected = ordered[:SELECTED_COLUMNS]
    return selected, {
        "candidate_columns": int(hourly.shape[1]),
        "eligible_columns_train_finite_positive_sigma": int(len(eligible)),
        "selected_zero_based_columns": selected,
        "selection_rule": "first 8 by sha256('continuation-v1|solar|'+zero_based_column_id) among TRAIN-only finite positive-variance columns; no zero-rate or performance filter",
        "selected_hashes": {
            str(i): hashlib.sha256(f"continuation-v1|solar|{i}".encode("utf-8")).hexdigest()
            for i in selected
        },
    }


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
        "!results/**/*.npz",
        "--glob",
        "!results/**/*.pt",
        "solar_AL|solar-energy|solarenergy",
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
        "AGENTS.md",
    ]
    try:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        return {
            "tool": "rg",
            "pattern": "solar_AL|solar-energy|solarenergy",
            "returncode": int(proc.returncode),
            "match_count": len(lines),
            "matches": lines[:50],
            "conclusion": (
                "Scoped rg found prior Solar source mentions in repository docs/code/results."
                if lines
                else "Scoped rg found no prior solar_AL/solar-energy/solarenergy mentions; this does not rule out upstream pretraining exposure."
            ),
        }
    except Exception as exc:  # pragma: no cover
        return {"tool": "rg", "error": repr(exc), "match_count": None, "matches": []}


def _read_solar() -> tuple[np.ndarray, list[str], dict[str, Any]]:
    upstream = _github_commit()
    contents = _contents_metadata()
    readme = _readme_evidence()
    raw_path = _data_dir() / f"solar_AL_{upstream['commit']}.txt.gz"
    pinned_url = _raw_url(upstream["commit"])
    _download(pinned_url, raw_path)
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        raw = pd.read_csv(handle, header=None).to_numpy(dtype=np.float64)
    if raw.shape != (RAW_ROWS, RAW_COLUMNS):
        raise RuntimeError(f"BLOCKED_DATA: expected raw shape {(RAW_ROWS, RAW_COLUMNS)}, got {raw.shape}")
    hourly = aggregate_hourly(raw)
    if hourly.shape != (HOURLY_ROWS, RAW_COLUMNS):
        raise RuntimeError(f"BLOCKED_DATA: expected hourly shape {(HOURLY_ROWS, RAW_COLUMNS)}, got {hourly.shape}")
    edges = _role_edges(len(hourly))
    selected, selection = _select_columns(hourly, edges["TRAIN"]["end"])
    values = hourly[:, selected].astype(np.float32, copy=True)
    ids = [str(i) for i in selected]
    audit = {
        "source_url": SOURCE_URL,
        "pinned_raw_url": pinned_url,
        "upstream": upstream,
        "contents_metadata": contents,
        "readme_evidence": readme,
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": sha(raw_path),
        "raw_bytes": raw_path.stat().st_size,
        "schema": {
            "format": "gzip csv matrix",
            "raw_shape": [int(raw.shape[0]), int(raw.shape[1])],
            "hourly_shape": [int(hourly.shape[0]), int(hourly.shape[1])],
            "timestamp_column": False,
            "selected_columns": ids,
        },
        "aggregation": {
            "input_resolution": "10-minute rows by official README",
            "output_resolution": "hourly row index",
            "method": "chronological disjoint six-row arithmetic means",
            "rows_per_hour": AGGREGATION_ROWS,
            "raw_rows": RAW_ROWS,
            "hourly_rows": HOURLY_ROWS,
            "invented_timestamps": False,
            "first_hour_raw_row_span": [0, 6],
            "last_hour_raw_row_span": [RAW_ROWS - AGGREGATION_ROWS, RAW_ROWS],
        },
        "time_axis": {
            "unit": "hourly_slot_index_after_disjoint_mean",
            "timezone_invented": False,
            "calendar_dates_invented": False,
            "readme_year": "2006",
        },
        "raw_missing": {"nonfinite_cells": int((~np.isfinite(raw)).sum()), "interpolation_applied": False},
        "hourly_missing": {"nonfinite_cells": int((~np.isfinite(hourly)).sum()), "interpolation_applied": False},
        "selection": selection,
    }
    return values, ids, audit


def _build_dataset() -> dict[str, Any]:
    values, ids, audit = _read_solar()
    missing_total = int((~np.isfinite(values)).sum())
    edges = _role_edges(len(values))
    train = values[edges["TRAIN"]["start"] : edges["TRAIN"]["end"]].astype(np.float64)
    sigma = train.std(axis=0, ddof=0)
    if missing_total != 0:
        raise RuntimeError(f"BLOCKED_DATA: selected hourly values contain {missing_total} nonfinite cells")
    if not np.isfinite(sigma).all() or bool((sigma <= 0).any()):
        raise RuntimeError("BLOCKED_DATA: nonpositive TRAIN sigma in selected Solar series")
    origins = _filter_finite_origins(values, role_origins(len(values)))
    counts = {role: int(len(origins[role])) for role in ROLES}
    if any(counts[role] <= 0 for role in ROLES):
        raise RuntimeError(f"BLOCKED_DATA: empty legal role origins: {counts}")
    npz_path = _data_dir() / "SolarEnergy_hourly8.npz"
    np.savez_compressed(
        npz_path,
        values=values,
        sigma=sigma.astype(np.float64),
        ids=np.asarray(ids, dtype=object),
        **{f"origins_{role}": origins[role] for role in ROLES},
    )
    audit.update(
        {
            "dataset": "SolarEnergy-hourly8",
            "selected_shape": {"T": int(values.shape[0]), "N": int(values.shape[1])},
            "ids": ids,
            "split_policy": "time-ordered hourly rows: TRAIN 60%, CALIBRATION 10%, VALIDATION 10%, TEST 20%",
            "split_edges": edges,
            "selected_missing": {"selected_nonfinite_cells": missing_total, "interpolation_applied": False},
            "train_sigma_population": {ids[i]: float(sigma[i]) for i in range(len(ids))},
            "origin_audit": _origin_audit(values, origins),
            "leakage_controls": {
                "sigma_fit_role": "TRAIN only",
                "column_selection_uses_test_target_or_performance": False,
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
            "domain": "10-minute solar PV multivariate time series aggregated to hourly forecasting targets",
            "downstream_decision": "fixed-source continuation LoRA pilot data contract; do not use scores to choose data",
            "quality_bar": "publication-style pilot receipts; zero leakage tolerance; no interpolation",
            "primary_leakage_paths": [
                "future target values entering context",
                "TEST-based model, LR, checkpoint, or data selection",
                "non-TRAIN scaling",
                "column choice based on zero-rate or forecasting performance",
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
        schedule_path = _data_dir() / f"schedule_SolarEnergy_seed{seed}.npz"
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
    path = _data_dir() / "SolarEnergy_hourly8.npz"
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
    path = _data_dir() / f"schedule_SolarEnergy_seed{seed}.npz"
    if path.exists():
        with np.load(path) as z:
            return z["tuples"].astype(np.int64)
    data = load()
    train_origins = data["origins"]["TRAIN"]
    if len(train_origins) == 0:
        raise RuntimeError("BLOCKED_DATA: no TRAIN origins for SolarEnergy")
    salt = int(hashlib.sha256(f"{NAME}|schedule|SolarEnergy|{seed}".encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(salt)
    tuples = np.empty((TRAIN_UPDATES, BATCH_SIZE, 2), dtype=np.int64)
    tuples[..., 0] = rng.integers(0, len(data["ids"]), size=(TRAIN_UPDATES, BATCH_SIZE), endpoint=False)
    tuples[..., 1] = train_origins[
        rng.integers(0, len(train_origins), size=(TRAIN_UPDATES, BATCH_SIZE), endpoint=False)
    ]
    np.savez_compressed(path, tuples=tuples, dataset="SolarEnergy-hourly8", seed=np.int64(seed))
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
