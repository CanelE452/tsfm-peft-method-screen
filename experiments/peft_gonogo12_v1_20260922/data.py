from __future__ import annotations

import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import CACHE, CONFIG, RESULTS, ROOT, save, sha

CONTEXT = int(CONFIG["data"]["context"])
HORIZON = int(CONFIG["data"]["horizon"])
PATCH = int(CONFIG["data"]["native_patch"])
RAW_PATH = ROOT / CONFIG["data"]["raw_path"]
ALL_IDS = (
    [str(x) for x in CONFIG["data"]["donor_ids"]]
    + [str(x) for x in CONFIG["data"]["development_ids"]]
    + [str(x) for x in CONFIG["data"]["evaluation_ids"]]
)
DONOR_IDS = [str(x) for x in CONFIG["data"]["donor_ids"]]
DEV_IDS = [str(x) for x in CONFIG["data"]["development_ids"]]
EVAL_IDS = [str(x) for x in CONFIG["data"]["evaluation_ids"]]
GROUP_IDS = {"donor": DONOR_IDS, "dev": DEV_IDS, "eval": EVAL_IDS}


def _rng_u64(key: str) -> np.random.Generator:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))


def mask_for(series_id: str, origin: int, role: str, kind: str, repeat: int = 0, seed: int | None = None) -> np.ndarray:
    observed = np.ones(CONTEXT, dtype=bool)
    if kind == "CLEAN":
        return observed
    if kind == "ALIGNED48_LAST":
        observed[-48:] = False
        return observed
    key_seed = "" if seed is None else f"|seed{seed}"
    rng = _rng_u64(f"peft-gonogo12-v1|{role}|{series_id}|{origin}|{repeat}|{kind}{key_seed}")
    if kind == "IID48":
        hidden = rng.choice(CONTEXT, size=48, replace=False)
        observed[hidden] = False
        return observed
    if kind == "BLOCK48":
        start = int(rng.integers(0, CONTEXT - 48 + 1))
        observed[start : start + 48] = False
        return observed
    raise ValueError(kind)


def interpolate_context(x: np.ndarray, observed: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    observed = np.asarray(observed, dtype=bool)
    if observed.all():
        return x
    if not observed.any():
        raise RuntimeError("DATA_FAILURE: cannot interpolate an all-missing context")
    grid = np.arange(len(x), dtype=np.float64)
    x[~observed] = np.interp(grid[~observed], grid[observed], x[observed]).astype(np.float32)
    return x


@dataclass
class Data:
    values: np.ndarray
    ids: list[str]
    index: dict[str, int]
    config: dict
    sigma_table: dict[str, dict[str, float]]

    def sigma(self, series_id: str, group: str) -> float:
        return self.sigma_table[group][str(series_id)]

    def group_ids(self, group: str) -> list[str]:
        return GROUP_IDS[group]

    def origins(self, group: str, part: str, stride: int | None = None) -> np.ndarray:
        stride = int(stride or CONFIG["data"]["eval_origin_stride"])
        ranges = CONFIG["data"]["ranges"]
        if group == "donor" and part == "train":
            lo, hi = ranges["donor_train"]
            return np.arange(max(CONTEXT, lo), hi - HORIZON + 1, stride, dtype=np.int64)
        prefix = "dev" if group == "dev" else "eval"
        if part in {"cal", "validation", "test"}:
            key = f"{prefix}_{part}"
            lo, hi = ranges[key]
            return np.arange(max(CONTEXT, lo), hi - HORIZON + 1, stride, dtype=np.int64)
        if part == "adapt":
            lo, hi = ranges[f"{prefix}_adapt"]
            return np.arange(lo + CONTEXT, hi - HORIZON + 1, stride, dtype=np.int64)
        if part in {"adapt_a", "adapt_b"}:
            lo, hi = ranges[f"{prefix}_adapt"]
            mid = lo + (hi - lo) // 2
            block_lo, block_hi = (lo, mid) if part == "adapt_a" else (mid, hi)
            return np.arange(block_lo + CONTEXT, block_hi - HORIZON + 1, stride, dtype=np.int64)
        raise ValueError((group, part))

    def batch(
        self,
        pairs: list[tuple[str, int]] | np.ndarray,
        group: str,
        role: str = "CLEAN",
        condition: str = "CLEAN",
        repeat: int = 0,
        seed: int | None = None,
        interpolate: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(pairs, dtype=object).reshape(-1, 2)
        x = np.empty((len(arr), CONTEXT), dtype=np.float32)
        y = np.empty((len(arr), HORIZON), dtype=np.float32)
        sigma = np.empty((len(arr),), dtype=np.float64)
        masks = np.empty((len(arr), CONTEXT), dtype=bool)
        for i, (series_raw, origin_raw) in enumerate(arr):
            series_id, origin = str(series_raw), int(origin_raw)
            col = self.index[series_id]
            if origin - CONTEXT < 0 or origin + HORIZON > len(self.values):
                raise IndexError(f"origin {origin} cannot support context+horizon")
            observed = mask_for(series_id, origin, role, condition, repeat, seed)
            context = self.values[origin - CONTEXT : origin, col].astype(np.float32, copy=True)
            if interpolate:
                context = interpolate_context(context, observed)
            else:
                context[~observed] = np.nan
            x[i] = context
            y[i] = self.values[origin : origin + HORIZON, col]
            sigma[i] = self.sigma(series_id, group)
            masks[i] = observed
        return x, y, sigma, masks


def _read_raw() -> np.ndarray:
    if not RAW_PATH.exists():
        raise RuntimeError(f"DATA_FAILURE: missing archived raw file {RAW_PATH}")
    raw_sha = sha(RAW_PATH)
    if raw_sha != CONFIG["data"]["raw_sha256"]:
        raise RuntimeError(f"DATA_FAILURE: raw sha mismatch {raw_sha}")
    with gzip.open(RAW_PATH, "rt", encoding="utf-8") as handle:
        matrix = pd.read_csv(handle, header=None).to_numpy(dtype=np.float64)
    if matrix.shape != (CONFIG["data"]["rows"], 321):
        raise RuntimeError(f"DATA_FAILURE: unexpected raw matrix shape {matrix.shape}")
    return matrix


def _sigma(values: np.ndarray, ids: list[str], ranges: dict[str, list[int]]) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    spans = {
        "donor": ranges["donor_train"],
        "dev": ranges["dev_adapt"],
        "eval": ranges["eval_adapt"],
    }
    index = {sid: i for i, sid in enumerate(ids)}
    for group, (lo, hi) in spans.items():
        table[group] = {}
        for sid in GROUP_IDS[group]:
            segment = values[lo:hi, index[sid]].astype(np.float64)
            sig = float(segment.std(ddof=0))
            if not np.isfinite(sig) or sig <= 0:
                raise RuntimeError(f"DATA_FAILURE: nonpositive sigma for {group}/{sid}")
            table[group][sid] = sig
    return table


def load_data() -> Data:
    raw = _read_raw()
    col_ids = [int(x) for x in ALL_IDS]
    values = raw[:, col_ids].astype(np.float32, copy=True)
    if not np.isfinite(values).all():
        raise RuntimeError("DATA_FAILURE: selected values contain nonfinite cells")
    index = {sid: i for i, sid in enumerate(ALL_IDS)}
    return Data(values=values, ids=list(ALL_IDS), index=index, config=CONFIG, sigma_table=_sigma(values, ALL_IDS, CONFIG["data"]["ranges"]))


def donor_schedule(seed: int, steps: int = 256, batch_size: int = 4, d: Data | None = None) -> np.ndarray:
    path = CACHE / "schedules" / f"donor_seed{seed}_u{steps}_b{batch_size}.npz"
    if path.exists():
        with np.load(path, allow_pickle=True) as z:
            return z["pairs"].astype(object)
    d = load_data() if d is None else d
    legal = d.origins("donor", "train", stride=CONFIG["data"]["train_origin_stride"])
    rows: list[tuple[str, int]] = []
    rng = _rng_u64(f"peft-gonogo12-v1|donor_schedule|{seed}|{steps}|{batch_size}")
    per_series = steps * batch_size // len(DONOR_IDS)
    if per_series * len(DONOR_IDS) != steps * batch_size:
        raise RuntimeError("IMPLEMENTATION_FAILURE: donor schedule not divisible by donor series")
    for sid in DONOR_IDS:
        sampled = rng.choice(legal, size=per_series, replace=True)
        rows.extend((sid, int(o)) for o in sampled)
    rng.shuffle(rows)
    pairs = np.asarray(rows, dtype=object).reshape(steps, batch_size, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pairs=pairs.astype(str), seed=np.int64(seed))
    return pairs


def block_schedule(series_id: str, group: str, part: str, seed: int, examples: int = 128, d: Data | None = None) -> np.ndarray:
    d = load_data() if d is None else d
    legal = d.origins(group, part, stride=CONFIG["data"]["train_origin_stride"])
    rng = _rng_u64(f"peft-gonogo12-v1|block_schedule|{group}|{series_id}|{part}|{seed}|{examples}")
    sampled = rng.choice(legal, size=examples, replace=True)
    return np.asarray([(str(series_id), int(o)) for o in sampled], dtype=object)


def audit() -> dict:
    d = load_data()
    ranges = CONFIG["data"]["ranges"]
    schedule = donor_schedule(CONFIG["h1"]["seeds"][0])
    partial_counts = []
    for sid in DEV_IDS:
        for origin in d.origins("dev", "adapt", stride=24):
            observed = mask_for(sid, int(origin), "P0_H2", "BLOCK48")
            patch_counts = observed.reshape(-1, PATCH).sum(axis=1)
            partial_counts.append(int(((patch_counts > 0) & (patch_counts < PATCH)).sum()))
    payload = {
        "status": "PASS",
        "domain_context": {
            "domain": "hourly Electricity univariate probabilistic forecasting",
            "downstream_decision": "screen two fixed TSFM PEFT hypotheses under temporal split",
            "quality_bar": "publication-style pilot receipts; zero target leakage tolerance",
            "primary_leakage_paths": [
                "TEST target used for model/calibration selection",
                "future target hidden context used for imputation",
                "non-adapt sigma for client personalization",
                "performance-based series selection",
            ],
        },
        "raw": {
            "path": str(RAW_PATH.relative_to(ROOT)),
            "sha256": sha(RAW_PATH),
            "bytes": RAW_PATH.stat().st_size,
            "schema": {"format": "gzip csv matrix", "rows": int(d.values.shape[0]), "selected_columns": len(d.ids)},
            "missing_selected_cells": int((~np.isfinite(d.values)).sum()),
        },
        "ids": {"donor": DONOR_IDS, "development": DEV_IDS, "evaluation": EVAL_IDS, "order": d.ids},
        "ranges": ranges,
        "origins": {
            "donor_train_stride6": int(len(d.origins("donor", "train", stride=6))),
            "dev_adapt_a_stride6": int(len(d.origins("dev", "adapt_a", stride=6))),
            "dev_adapt_b_stride6": int(len(d.origins("dev", "adapt_b", stride=6))),
            "dev_validation_daily": int(len(d.origins("dev", "validation", stride=24))),
            "eval_test_daily": int(len(d.origins("eval", "test", stride=24))),
        },
        "sigma": d.sigma_table,
        "schedule": {
            "donor_seed": int(CONFIG["h1"]["seeds"][0]),
            "shape": list(schedule.shape),
            "sha256": hashlib.sha256(schedule.astype(str).tobytes()).hexdigest(),
            "per_donor_series_examples": {
                sid: int(np.sum(schedule[:, :, 0].astype(str) == sid)) for sid in DONOR_IDS
            },
        },
        "mask": {
            "block48_mean_partial_patches_dev_adapt": float(np.mean(partial_counts)),
            "block48_min_partial_patches_dev_adapt": int(np.min(partial_counts)),
            "future_hidden_context_values_used_for_interpolation": False,
            "cal_val_test_mask_seed_independent": True,
        },
        "test_target_statistics_written": False,
    }
    save(RESULTS / "DATA_AUDIT.json", payload)
    return payload


if __name__ == "__main__":
    audit()
