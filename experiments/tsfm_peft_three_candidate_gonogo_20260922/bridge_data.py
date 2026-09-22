from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


NAME = "tsfm_peft_three_candidate_gonogo_20260922"
DATA_VERSION = "2026-09-22.t-bridge-v1"
CONTEXT = 512
HORIZON = 64
DEFAULT_SEEDS = (92201, 92202)

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / NAME
CACHE_DIR = ROOT / ".cache" / NAME / "bridge"
RESULT_DIR = ROOT / "results" / NAME
AUDIT_PATH = RESULT_DIR / "T_BRIDGE_ACCESS_AUDIT.json"


def _array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _packet_path(seed: int) -> Path:
    return CACHE_DIR / f"t_bridge_seed{int(seed)}.npz"


def _manifest_path(seed: int) -> Path:
    return CACHE_DIR / f"t_bridge_seed{int(seed)}.json"


def _load_data_module():
    data_path = EXP_DIR / "data.py"
    spec = importlib.util.spec_from_file_location("triage_full_data_for_bridge_build", data_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load data module from {data_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_full_audit() -> dict[str, Any]:
    path = RESULT_DIR / "DATA_AND_SPLIT_AUDIT.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build(seeds: tuple[int, ...] = DEFAULT_SEEDS) -> dict[str, Any]:
    """Create student-safe T bridge packets from the full data API."""
    data_module = _load_data_module()
    full = data_module.load()
    full_audit = _read_full_audit()
    split = full_audit["split"]
    offset = int(split["t_old_val_end"])
    bridge_end = int(split["t_bridge_end"])
    bridge_train_end = int(split["t_bridge_train_end"])
    bridge_val_start = int(split["t_bridge_val_start"])

    values = np.asarray(full["values"][offset:bridge_end], dtype=np.float32)
    sigma = np.asarray(full["sigma"], dtype=np.float64)
    ids = np.array(full["ids"], dtype="U64")
    val_origins = np.asarray(full["origins"]["T_BRIDGE_VAL"], dtype=np.int64)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if values.shape[0] != bridge_end - offset:
        raise RuntimeError("bridge slice length mismatch")
    if not np.isfinite(values).all():
        raise RuntimeError("bridge values contain non-finite entries")
    if not np.isfinite(sigma).all() or not np.all(sigma > 0):
        raise RuntimeError("sigma must be finite and positive")
    if val_origins.size == 0:
        raise RuntimeError("T_BRIDGE_VAL origins are empty")
    if not np.all(val_origins >= bridge_val_start):
        raise RuntimeError("validation origins start before BRIDGE_VAL")
    if not np.all(val_origins + HORIZON <= bridge_end):
        raise RuntimeError("validation targets exceed BRIDGE")
    if not np.all(val_origins - CONTEXT >= offset):
        raise RuntimeError("validation contexts access OLD rows")

    seed_manifests: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix=f"{NAME}_bridge_schedule_") as tmp_schedule_dir:
        data_module.SCHEDULE_DIR = Path(tmp_schedule_dir)
        schedules = {
            int(seed): np.asarray(data_module.schedule("T_BRIDGE_TRAIN", int(seed)), dtype=np.int64)
            for seed in seeds
        }

    for seed in seeds:
        schedule = schedules[int(seed)]
        if schedule.shape != (256, 8, 2):
            raise RuntimeError(f"unexpected schedule shape for seed {seed}: {schedule.shape}")
        origins = schedule[..., 1]
        if not np.all(schedule[..., 0] >= 0):
            raise RuntimeError("schedule series index below zero")
        if not np.all(schedule[..., 0] < values.shape[1]):
            raise RuntimeError("schedule series index exceeds bridge width")
        if not np.all(origins - CONTEXT >= offset):
            raise RuntimeError(f"seed {seed} schedule context accesses OLD rows")
        if not np.all(origins >= offset):
            raise RuntimeError(f"seed {seed} schedule target starts before BRIDGE")
        if not np.all(origins + HORIZON <= bridge_train_end):
            raise RuntimeError(f"seed {seed} schedule target exceeds BRIDGE_TRAIN")

        packet_path = _packet_path(int(seed))
        np.savez_compressed(
            packet_path,
            values=values,
            sigma=sigma,
            offset=np.array(offset, dtype=np.int64),
            origins=val_origins,
            schedule=schedule,
            ids=ids,
        )
        manifest = {
            "seed": int(seed),
            "packet": str(packet_path.relative_to(ROOT)),
            "packet_sha256": _file_sha256(packet_path),
            "values_shape": list(values.shape),
            "values_dtype": str(values.dtype),
            "values_global_start_inclusive": offset,
            "values_global_end_exclusive": bridge_end,
            "bridge_train_end_exclusive": bridge_train_end,
            "bridge_val_start_inclusive": bridge_val_start,
            "sigma_shape": list(sigma.shape),
            "sigma_dtype": str(sigma.dtype),
            "ids_count": int(ids.size),
            "origins_shape": list(val_origins.shape),
            "origins_dtype": str(val_origins.dtype),
            "schedule_shape": list(schedule.shape),
            "schedule_dtype": str(schedule.dtype),
            "hashes": {
                "values": _array_sha256(values),
                "sigma": _array_sha256(sigma),
                "ids": hashlib.sha256(
                    json.dumps(ids.tolist(), sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "origins": _array_sha256(val_origins),
                "schedule": _array_sha256(schedule),
            },
        }
        _write_json(_manifest_path(int(seed)), manifest)
        seed_manifests[str(int(seed))] = manifest

    audit = {
        "status": "PASS",
        "data_version": DATA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "CPU-only independent packet proving T student access can be restricted to BRIDGE rows.",
        "source_api_used_by_build_only": {
            "module": str((EXP_DIR / "data.py").relative_to(ROOT)),
            "calls": ["load()", "schedule('T_BRIDGE_TRAIN', seed)"],
        },
        "student_api_contract": {
            "function": "load_bridge(seed)",
            "allowed_reads": "one independent .cache/<NAME>/bridge/t_bridge_seed<seed>.npz packet",
            "forbidden_reads": [
                ".cache/<NAME>/data/electricity_selected_packet.npz",
                ".cache/<NAME>/data/values_float32.npy",
                ".cache/<NAME>/data/origins.npz",
                ".cache/<NAME>/data/sigma_float64.npy",
                ".cache/<NAME>/data/electricity.txt.gz",
                "experiments/<NAME>/data.py",
            ],
        },
        "global_rows": {
            "offset": offset,
            "bridge_end_exclusive": bridge_end,
            "bridge_train_end_exclusive": bridge_train_end,
            "bridge_val_start_inclusive": bridge_val_start,
            "old_rows_excluded_from_values": [0, offset],
        },
        "checks": {
            "values_are_exact_bridge_slice": True,
            "origins_use_global_rows": True,
            "train_schedule_contexts_begin_at_or_after_offset": True,
            "train_schedule_targets_wholly_in_bridge_train": True,
            "validation_targets_wholly_in_bridge_val": True,
            "validation_contexts_begin_at_or_after_offset": True,
            "no_interpolation": True,
            "cpu_only": True,
        },
        "seeds": seed_manifests,
    }
    _write_json(AUDIT_PATH, audit)
    return audit


def load_bridge(seed: int) -> dict[str, Any]:
    """Load a T-student bridge packet without importing or opening full data."""
    packet_path = _packet_path(int(seed))
    with np.load(packet_path, allow_pickle=False) as packet:
        values = packet["values"].astype(np.float32, copy=False)
        sigma = packet["sigma"].astype(np.float64, copy=False)
        offset = int(packet["offset"])
        origins = packet["origins"].astype(np.int64, copy=False)
        schedule = packet["schedule"].astype(np.int64, copy=False)
        ids = [str(x) for x in packet["ids"].tolist()]
    return {
        "values": values,
        "sigma": sigma,
        "offset": offset,
        "origins": origins,
        "schedule": schedule,
        "ids": ids,
    }


if __name__ == "__main__":
    result = build()
    print(json.dumps({"status": result["status"], "audit": str(AUDIT_PATH)}, sort_keys=True))
