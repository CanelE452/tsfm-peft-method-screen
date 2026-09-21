from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


NAME = "tsfm_peft_three_candidate_gonogo_20260922"
DATA_VERSION = "2026-09-22.v1"
AUTHOR_REPO_URL = "https://github.com/laiguokun/multivariate-time-series-data"
RAW_URL = (
    "https://raw.githubusercontent.com/laiguokun/"
    "multivariate-time-series-data/master/electricity/electricity.txt.gz"
)
REMOTE_REF = "refs/heads/master"
SELECT_SALT = "peft-triage-v1|"
CONTEXT = 512
HORIZON = 64
N_SERIES = 16
F_CLIENTS = 4
DEFAULT_STEPS = 256
F_CLIENT_STEPS = 64
PROBE_SEED = 92200
REPEAT_SEEDS = (92201, 92202)
ROLE_KEYS = (
    "Q_TRAIN",
    "Q_VAL",
    "Q_TEST",
    "T_OLD_TRAIN",
    "T_OLD_VAL",
    "T_BRIDGE_TRAIN",
    "T_BRIDGE_VAL",
    "T_TEST",
)

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".cache" / NAME
DATA_DIR = CACHE_DIR / "data"
SCHEDULE_DIR = CACHE_DIR / "schedules"
RESULT_DIR = ROOT / "results" / NAME
RAW_PATH = DATA_DIR / "electricity.txt.gz"
PACKET_PATH = DATA_DIR / "electricity_selected_packet.npz"
VALUES_PATH = DATA_DIR / "values_float32.npy"
SIGMA_PATH = DATA_DIR / "sigma_float64.npy"
ORIGINS_PATH = DATA_DIR / "origins.npz"
PROBE_PATH = DATA_DIR / f"probe_seed{PROBE_SEED}.npy"
SOURCE_PACKET_PATH = DATA_DIR / "source_packet.json"
AUDIT_PATH = RESULT_DIR / "DATA_AND_SPLIT_AUDIT.json"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _json_sha256(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def _stable_u32(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_revision() -> str | None:
    cmd = ["git", "ls-remote", AUTHOR_REPO_URL + ".git", REMOTE_REF]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return None
    fields = result.stdout.strip().split()
    if len(fields) >= 2 and fields[1] == REMOTE_REF:
        return fields[0]
    return None


def _download_raw() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RAW_PATH.with_suffix(".tmp")
    with urllib.request.urlopen(RAW_URL, timeout=120) as response:
        payload = response.read()
        source_headers = {
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "content_length": response.headers.get("Content-Length"),
        }
    tmp.write_bytes(payload)
    tmp.replace(RAW_PATH)
    return {
        "url": RAW_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "headers": source_headers,
        "bytes": RAW_PATH.stat().st_size,
        "sha256": _file_sha256(RAW_PATH),
    }


def _load_raw_matrix(path: Path) -> np.ndarray:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return np.loadtxt(f, delimiter=",", dtype=np.float64)


def _select_columns(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    train50_end = raw.shape[0] // 2
    train50 = raw[:train50_end]
    finite_by_col = np.isfinite(train50).all(axis=0)
    sigma_all = np.std(train50, axis=0, ddof=0, dtype=np.float64)
    eligible = np.flatnonzero(finite_by_col & (sigma_all > 0.0))
    ranked = sorted(
        (hashlib.sha256(f"{SELECT_SALT}{int(col)}".encode("utf-8")).hexdigest(), int(col))
        for col in eligible
    )
    selected = np.array([col for _, col in ranked[:N_SERIES]], dtype=np.int64)
    if selected.shape != (N_SERIES,):
        raise RuntimeError(f"expected {N_SERIES} eligible columns, found {selected.shape[0]}")
    sigma = sigma_all[selected].astype(np.float64, copy=True)
    ids = [f"electricity_col_{int(col)}" for col in selected]
    audit = {
        "train50_end": int(train50_end),
        "eligible_count": int(eligible.size),
        "selected_original_columns": selected.tolist(),
        "selection_salt": SELECT_SALT,
        "selection_rule": "eligible columns sorted by SHA256('peft-triage-v1|' + 0-based column index)",
        "sigma_ddof": 0,
        "sigma_hash": _array_sha256(sigma),
        "ids_hash": _json_sha256(ids),
        "selected_columns_hash": _array_sha256(selected),
    }
    return selected, sigma, ids, audit


def _splits(n_rows: int) -> dict[str, int]:
    q_train_end = int(n_rows * 0.60)
    q_val_end = int(n_rows * 0.80)
    t_old_train_end = int(n_rows * 0.50)
    t_old_val_end = int(n_rows * 0.60)
    t_bridge_end = int(n_rows * 0.80)
    bridge_train_end = t_old_val_end + int((t_bridge_end - t_old_val_end) * 0.75)
    return {
        "n_rows": int(n_rows),
        "q_train_start": 0,
        "q_train_end": q_train_end,
        "q_val_start": q_train_end,
        "q_val_end": q_val_end,
        "q_test_start": q_val_end,
        "q_test_end": int(n_rows),
        "t_old_train_start": 0,
        "t_old_train_end": t_old_train_end,
        "t_old_val_start": t_old_train_end,
        "t_old_val_end": t_old_val_end,
        "t_bridge_start": t_old_val_end,
        "t_bridge_train_end": bridge_train_end,
        "t_bridge_val_start": bridge_train_end,
        "t_bridge_end": t_bridge_end,
        "t_test_start": t_bridge_end,
        "t_test_end": int(n_rows),
    }


def _role_spec(splits: dict[str, int]) -> dict[str, dict[str, int | bool]]:
    return {
        "Q_TRAIN": {
            "target_start": splits["q_train_start"],
            "target_end": splits["q_train_end"],
            "context_min_start": 0,
            "mod6": False,
        },
        "Q_VAL": {
            "target_start": splits["q_val_start"],
            "target_end": splits["q_val_end"],
            "context_min_start": 0,
            "mod6": True,
        },
        "Q_TEST": {
            "target_start": splits["q_test_start"],
            "target_end": splits["q_test_end"],
            "context_min_start": 0,
            "mod6": True,
        },
        "T_OLD_TRAIN": {
            "target_start": splits["t_old_train_start"],
            "target_end": splits["t_old_train_end"],
            "context_min_start": 0,
            "mod6": False,
        },
        "T_OLD_VAL": {
            "target_start": splits["t_old_val_start"],
            "target_end": splits["t_old_val_end"],
            "context_min_start": 0,
            "mod6": True,
        },
        "T_BRIDGE_TRAIN": {
            "target_start": splits["t_bridge_start"],
            "target_end": splits["t_bridge_train_end"],
            "context_min_start": splits["t_bridge_start"],
            "mod6": False,
        },
        "T_BRIDGE_VAL": {
            "target_start": splits["t_bridge_val_start"],
            "target_end": splits["t_bridge_end"],
            "context_min_start": splits["t_bridge_start"],
            "mod6": True,
        },
        "T_TEST": {
            "target_start": splits["t_test_start"],
            "target_end": splits["t_test_end"],
            "context_min_start": splits["t_bridge_start"],
            "mod6": True,
        },
    }


def _legal_origins(spec: dict[str, int | bool]) -> np.ndarray:
    lo = max(int(spec["target_start"]), int(spec["context_min_start"]) + CONTEXT)
    hi = int(spec["target_end"]) - HORIZON
    if hi < lo:
        return np.empty((0,), dtype=np.int64)
    origins = np.arange(lo, hi + 1, dtype=np.int64)
    if bool(spec["mod6"]):
        origins = origins[(origins % 6) == 0]
    return origins


def _origin_audit(
    origins: dict[str, np.ndarray],
    specs: dict[str, dict[str, int | bool]],
    values: np.ndarray,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for role in ROLE_KEYS:
        arr = origins[role]
        spec = specs[role]
        context_starts = arr - CONTEXT
        target_ends = arr + HORIZON
        context_finite_failures = 0
        target_finite_failures = 0
        for origin in arr:
            x = values[int(origin) - CONTEXT : int(origin)]
            y = values[int(origin) : int(origin) + HORIZON]
            if not np.isfinite(x).all():
                context_finite_failures += 1
            if not np.isfinite(y).all():
                target_finite_failures += 1
        target_start = int(spec["target_start"])
        target_end = int(spec["target_end"])
        context_min_start = int(spec["context_min_start"])
        checks = {
            "nonempty": bool(arr.size > 0),
            "targets_start_in_role": bool(np.all(arr >= target_start)) if arr.size else False,
            "targets_end_in_role": bool(np.all(target_ends <= target_end)) if arr.size else False,
            "contexts_after_context_min_start": bool(np.all(context_starts >= context_min_start))
            if arr.size
            else False,
            "origin_mod6": bool(np.all((arr % 6) == 0)) if bool(spec["mod6"]) and arr.size else None,
            "context_finite_failures": int(context_finite_failures),
            "target_finite_failures": int(target_finite_failures),
        }
        report[role] = {
            "count": int(arr.size),
            "min_origin": int(arr[0]) if arr.size else None,
            "max_origin": int(arr[-1]) if arr.size else None,
            "target_start": target_start,
            "target_end_exclusive": target_end,
            "context_min_start": context_min_start,
            "context_start_min": int(context_starts.min()) if arr.size else None,
            "context_start_max": int(context_starts.max()) if arr.size else None,
            "target_end_min_exclusive": int(target_ends.min()) if arr.size else None,
            "target_end_max_exclusive": int(target_ends.max()) if arr.size else None,
            "hourly_train_or_mod6_eval": "mod6" if bool(spec["mod6"]) else "hourly",
            "context_crosses_role_start": bool(np.any(context_starts < target_start)) if arr.size else False,
            "hash": _array_sha256(arr),
            "checks": checks,
        }
    return report


def _validate_origin_report(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for role, item in report.items():
        checks = item["checks"]
        for key in (
            "nonempty",
            "targets_start_in_role",
            "targets_end_in_role",
            "contexts_after_context_min_start",
        ):
            if not checks[key]:
                failures.append(f"{role}:{key}")
        if checks["origin_mod6"] is False:
            failures.append(f"{role}:origin_mod6")
        if checks["context_finite_failures"] != 0:
            failures.append(f"{role}:context_finite_failures")
        if checks["target_finite_failures"] != 0:
            failures.append(f"{role}:target_finite_failures")
    return failures


def _load_packet() -> tuple[np.ndarray, np.ndarray, list[str], dict[str, np.ndarray]]:
    if not PACKET_PATH.exists():
        build_dataset(force=True)
    packet = np.load(PACKET_PATH, allow_pickle=False)
    values = packet["values"]
    sigma = packet["sigma"]
    ids = [str(x) for x in packet["ids"].tolist()]
    origins = {role: packet[f"origin_{role}"] for role in ROLE_KEYS}
    return values, sigma, ids, origins


def load() -> dict[str, Any]:
    """Load the audited Electricity panel and deterministic origin sets."""
    values, sigma, ids, origins = _load_packet()
    return {
        "values": values.astype(np.float32, copy=False),
        "sigma": sigma.astype(np.float64, copy=False),
        "ids": ids,
        "origins": {role: origins[role].astype(np.int64, copy=False) for role in ROLE_KEYS},
    }


def _schedule_rng(role: str, seed: int, client: int | None) -> np.random.Generator:
    marker = "client" if client is not None else "global"
    client_part = 0 if client is None else int(client) + 1
    seq = np.random.SeedSequence([int(seed), _stable_u32(role), _stable_u32(marker), client_part])
    return np.random.default_rng(seq)


def schedule(
    role: str,
    seed: int,
    steps: int = DEFAULT_STEPS,
    client: int | None = None,
) -> np.ndarray:
    """Return deterministic batches of (series_index, origin) pairs.

    Client schedules are the F workflow schedules: Q_TRAIN origins, one fixed
    first-four client series, and exactly 64 local updates.
    """
    if role not in ROLE_KEYS:
        raise ValueError(f"unknown role {role!r}")
    data = load()
    schedule_role = role
    if client is not None:
        if role != "Q_TRAIN":
            raise ValueError("client schedules are defined only on Q_TRAIN origins")
        if not 0 <= int(client) < F_CLIENTS:
            raise ValueError(f"client must be in [0,{F_CLIENTS - 1}]")
        if steps == DEFAULT_STEPS:
            steps = F_CLIENT_STEPS
        elif steps != F_CLIENT_STEPS:
            raise ValueError("client schedules use exactly 64 steps")
        schedule_role = f"{role}_CLIENT{int(client)}"
    if steps <= 0:
        raise ValueError("steps must be positive")

    origins = data["origins"][role]
    rng = _schedule_rng(role, int(seed), client)
    origin_idx = rng.integers(0, origins.size, size=(int(steps), 8), dtype=np.int64)
    sampled_origins = origins[origin_idx]
    if client is None:
        sampled_series = rng.integers(0, N_SERIES, size=(int(steps), 8), dtype=np.int64)
    else:
        sampled_series = np.full((int(steps), 8), int(client), dtype=np.int64)
    pairs = np.stack([sampled_series, sampled_origins], axis=-1).astype(np.int64, copy=False)

    SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
    file_stem = f"{schedule_role}_seed{int(seed)}_steps{int(steps)}"
    np.savez_compressed(SCHEDULE_DIR / f"{file_stem}.npz", pairs=pairs)
    _write_json(
        SCHEDULE_DIR / f"{file_stem}.json",
        {
            "role": role,
            "schedule_role": schedule_role,
            "seed": int(seed),
            "steps": int(steps),
            "batch": 8,
            "client": None if client is None else int(client),
            "shape": list(pairs.shape),
            "dtype": str(pairs.dtype),
            "hash": _array_sha256(pairs),
            "rng": "numpy SeedSequence([seed, stable(role), stable(global/client), client+1])",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return pairs


def probe() -> np.ndarray:
    """Return the fixed TRAIN-only context probe pairs for sensitivity work."""
    data = load()
    origins = data["origins"]["Q_TRAIN"]
    rng = np.random.default_rng(np.random.SeedSequence([PROBE_SEED, _stable_u32("probe")]))
    origin_idx = rng.integers(0, origins.size, size=(32,), dtype=np.int64)
    sampled_origins = origins[origin_idx]
    sampled_series = rng.integers(0, N_SERIES, size=(32,), dtype=np.int64)
    pairs = np.stack([sampled_series, sampled_origins], axis=-1).astype(np.int64, copy=False)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROBE_PATH, pairs)
    _write_json(
        DATA_DIR / f"probe_seed{PROBE_SEED}.json",
        {
            "seed": PROBE_SEED,
            "shape": list(pairs.shape),
            "dtype": str(pairs.dtype),
            "hash": _array_sha256(pairs),
            "source_origins": "Q_TRAIN",
            "target_access": "none",
        },
    )
    return pairs


def build_dataset(force: bool = False) -> dict[str, Any]:
    if PACKET_PATH.exists() and AUDIT_PATH.exists() and not force:
        return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    upstream_revision = _git_revision()
    source = _download_raw()
    raw_sha = source["sha256"]
    raw = _load_raw_matrix(RAW_PATH)
    if raw.ndim != 2:
        raise RuntimeError(f"expected 2D matrix, got shape {raw.shape}")
    raw_finite = np.isfinite(raw)
    if not raw_finite.all():
        raise RuntimeError("raw Electricity matrix contains non-finite values; no imputation allowed")
    if np.any(raw < 0):
        raise RuntimeError("raw Electricity matrix contains negative values")

    selected, sigma, ids, selection_audit = _select_columns(raw)
    values = raw[:, selected].astype(np.float32, copy=True)
    split = _splits(values.shape[0])
    specs = _role_spec(split)
    origins = {role: _legal_origins(specs[role]) for role in ROLE_KEYS}
    origin_report = _origin_audit(origins, specs, values)
    origin_failures = _validate_origin_report(origin_report)
    if origin_failures:
        raise RuntimeError("origin validation failed: " + ", ".join(origin_failures))

    packet_items: dict[str, Any] = {
        "values": values,
        "sigma": sigma,
        "ids": np.array(ids, dtype="U64"),
        "selected_original_columns": selected,
    }
    for role in ROLE_KEYS:
        packet_items[f"origin_{role}"] = origins[role]
    np.savez_compressed(PACKET_PATH, **packet_items)
    np.save(VALUES_PATH, values)
    np.save(SIGMA_PATH, sigma)
    np.savez_compressed(ORIGINS_PATH, **{role: origins[role] for role in ROLE_KEYS})

    finite_selected = np.isfinite(values)
    domain_context = {
        "domain": "hourly multivariate electricity consumption time series",
        "downstream_decision": "limited-budget PEFT candidate go/hold/no-go screen",
        "consumer": "Q/F/T training and audit code",
        "leakage_paths_controlled": [
            "train50-only sigma and column eligibility",
            "hash-only deterministic series selection independent of values after eligibility",
            "time-ordered splits",
            "Q/F validation and test origins fixed at origin % 6 == 0",
            "T bridge contexts start inside BRIDGE",
            "no interpolation or value-based origin filtering",
        ],
        "quality_bar": "publication-oriented local experiment audit",
    }
    hashes = {
        "raw_gzip_sha256": raw_sha,
        "values_float32_hash": _array_sha256(values),
        "sigma_float64_hash": _array_sha256(sigma),
        "ids_hash": _json_sha256(ids),
        "selected_columns_hash": _array_sha256(selected),
        "origins_hash": _json_sha256({role: origin_report[role]["hash"] for role in ROLE_KEYS}),
        "packet_sha256": _file_sha256(PACKET_PATH),
    }
    audit = {
        "status": "PASS",
        "data_version": DATA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "domain_context": domain_context,
        "source": {
            **source,
            "author_repo": AUTHOR_REPO_URL,
            "remote_ref": REMOTE_REF,
            "upstream_revision": upstream_revision,
        },
        "schema": {
            "raw_shape": [int(raw.shape[0]), int(raw.shape[1])],
            "values_shape": [int(values.shape[0]), int(values.shape[1])],
            "values_dtype": str(values.dtype),
            "sigma_shape": list(sigma.shape),
            "sigma_dtype": str(sigma.dtype),
            "ids_count": len(ids),
            "delimiter": ",",
            "has_header": False,
            "context": CONTEXT,
            "horizon": HORIZON,
            "native_horizon": HORIZON,
        },
        "selection": selection_audit,
        "f_clients": {
            "count": F_CLIENTS,
            "series_indices": list(range(F_CLIENTS)),
            "ids": ids[:F_CLIENTS],
        },
        "split": split,
        "finite": {
            "raw_all_finite": bool(raw_finite.all()),
            "raw_finite_count": int(raw_finite.sum()),
            "raw_total_count": int(raw.size),
            "selected_all_finite": bool(finite_selected.all()),
            "selected_finite_count": int(finite_selected.sum()),
            "selected_total_count": int(values.size),
            "selected_nonnegative": bool(np.all(values >= 0)),
        },
        "std": {
            "sigma_min": float(np.min(sigma)),
            "sigma_max": float(np.max(sigma)),
            "sigma_mean": float(np.mean(sigma)),
            "sigma_ddof": 0,
            "sigma_train50_only": True,
            "hash": hashes["sigma_float64_hash"],
        },
        "origins": origin_report,
        "hashes": hashes,
        "packets": {
            "raw": str(RAW_PATH.relative_to(ROOT)),
            "packet": str(PACKET_PATH.relative_to(ROOT)),
            "values": str(VALUES_PATH.relative_to(ROOT)),
            "sigma": str(SIGMA_PATH.relative_to(ROOT)),
            "origins": str(ORIGINS_PATH.relative_to(ROOT)),
            "source_packet": str(SOURCE_PACKET_PATH.relative_to(ROOT)),
        },
        "notes": [
            "Column IDs are anonymized original 0-based Electricity columns.",
            "Sigma uses population std (ddof=0) on the first 50% by time; contract did not specify ddof.",
            "Origin sets are defined only by split boundaries, context, horizon, and evaluation stride.",
        ],
    }
    _write_json(SOURCE_PACKET_PATH, audit["source"])
    _write_json(AUDIT_PATH, audit)
    return audit


def _seal_default_artifacts() -> dict[str, Any]:
    audit = build_dataset(force=True)
    probe_pairs = probe()
    schedule_hashes: dict[str, str] = {}
    for seed in REPEAT_SEEDS:
        for role in ROLE_KEYS:
            pairs = schedule(role, seed)
            schedule_hashes[f"{role}_seed{seed}"] = _array_sha256(pairs)
        for client in range(F_CLIENTS):
            pairs = schedule("Q_TRAIN", seed, client=client)
            schedule_hashes[f"Q_TRAIN_client{client}_seed{seed}"] = _array_sha256(pairs)
    audit["probe"] = {
        "path": str(PROBE_PATH.relative_to(ROOT)),
        "shape": list(probe_pairs.shape),
        "hash": _array_sha256(probe_pairs),
        "target_access": "none",
    }
    audit["canonical_schedules"] = schedule_hashes
    audit["hashes"]["canonical_schedules_hash"] = _json_sha256(schedule_hashes)
    _write_json(AUDIT_PATH, audit)
    return audit


if __name__ == "__main__":
    final_audit = _seal_default_artifacts()
    print(json.dumps({"status": final_audit["status"], "audit": str(AUDIT_PATH)}, sort_keys=True))
