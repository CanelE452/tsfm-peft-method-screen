from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, RESULTS, SEEDS, save, sha


PIN = "9d1799ede894606ad349eb66e335d8a814eb8acf"
BASE = f"https://raw.githubusercontent.com/KIT-IAI/EvaluatingEnsemblePostProcessing/{PIN}/"
FILES = {
    "README.md": "README.md",
    "create_correct_structure.R": "pre_processing/create_correct_structure.R",
    "workflow_train.R": "workflows/workflow_train.R",
    "workflow_predict.R": "workflows/workflow_predict.R",
    "ensembles": "pre_processing/Data/Sweden_Zone3_Ensembles.csv",
    "control": "pre_processing/Data/Sweden_Zone3_Control.csv",
    "power": "pre_processing/Data/Sweden_Zone3_Power.csv",
}
WEATHER = ["u100", "v100", "t2m", "sp", "speed"]
HORIZONS = [3, 6, 9, 12, 15, 18, 21, 24]
ROLES = {
    "TRAIN": ("1900-01-01", "2018-01-01"),
    "CAL": ("2018-01-01", "2018-07-01"),
    "VAL": ("2018-07-01", "2019-01-01"),
    "TEST": ("2019-01-01", "2019-09-01"),
}


def _raw_dir() -> Path:
    p = CACHE / "B" / "data" / "raw"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _raw_path(key: str) -> Path:
    return _raw_dir() / FILES[key].replace("/", "__")


def _download() -> dict[str, dict]:
    manifest: dict[str, dict] = {}
    for key, rel in FILES.items():
        url = BASE + rel
        path = _raw_path(key)
        if not path.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "codex-ab-data-audit"})
            with urllib.request.urlopen(req, timeout=180) as r:
                path.write_bytes(r.read())
        manifest[key] = {
            "url": url,
            "path": str(path),
            "sha256": sha(path),
            "bytes": path.stat().st_size,
        }
    return manifest


def _read_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    manifest = _download()
    ens = pd.read_csv(_raw_path("ensembles"), parse_dates=["time"])
    ctl = pd.read_csv(_raw_path("control"), parse_dates=["time"])
    power = pd.read_csv(_raw_path("power"), parse_dates=["time"])
    for frame in (ens, ctl, power):
        frame["issue"] = frame["time"] - pd.to_timedelta(frame["horizon"], unit="h")
    return ens, ctl, power, manifest


def _role_of_issue(issue: pd.Timestamp) -> str | None:
    for role, (lo, hi) in ROLES.items():
        if pd.Timestamp(lo) <= issue < pd.Timestamp(hi):
            if all(pd.Timestamp(lo) <= issue + pd.Timedelta(hours=h) < pd.Timestamp(hi) for h in HORIZONS):
                return role
    return None


def _make_power_series(power: pd.DataFrame) -> tuple[pd.Series, dict]:
    checks = power.groupby("time")["wind_power"].agg(["nunique", "size", "min", "max"])
    inconsistent = checks[checks["nunique"] > 1]
    if len(inconsistent):
        raise RuntimeError(f"BLOCKED_DATA: power duplicate valid times disagree: {len(inconsistent)}")
    series = power.sort_values("time").groupby("time", sort=True)["wind_power"].first().astype("float32")
    diffs = series.index.to_series().diff().dropna().value_counts()
    return series, {
        "duplicate_valid_times": int((checks["size"] > 1).sum()),
        "duplicate_values_identical": True,
        "valid_time_min": str(series.index.min()),
        "valid_time_max": str(series.index.max()),
        "unique_valid_times": int(len(series)),
        "diff_counts_hours": {str(k): int(v) for k, v in diffs.items()},
    }


def _complete_issues(ens: pd.DataFrame, ctl: pd.DataFrame, power: pd.DataFrame) -> tuple[list[pd.Timestamp], dict]:
    bad_member_keys = []
    for (issue, number), g in ens.groupby(["issue", "number"], sort=False):
        hs = sorted(g["horizon"].astype(int).tolist())
        if hs != HORIZONS or len(g) != 8:
            bad_member_keys.append({"issue": str(issue), "number": int(number), "horizons": hs, "rows": int(len(g))})
            if len(bad_member_keys) >= 10:
                break
    bad_control_keys = []
    for issue, g in ctl.groupby("issue", sort=False):
        hs = sorted(g["horizon"].astype(int).tolist())
        if hs != HORIZONS or len(g) != 8:
            bad_control_keys.append({"issue": str(issue), "horizons": hs, "rows": int(len(g))})
            if len(bad_control_keys) >= 10:
                break
    ens_counts = ens.groupby("issue").agg(rows=("time", "size"), members=("number", "nunique"), horizons=("horizon", "nunique"))
    ctl_counts = ctl.groupby("issue").agg(rows=("time", "size"), horizons=("horizon", "nunique"))
    pow_counts = power.groupby("issue").agg(rows=("time", "size"), horizons=("horizon", "nunique"))
    common = (
        set(ens_counts.query("rows == 400 and members == 50 and horizons == 8").index)
        & set(ctl_counts.query("rows == 8 and horizons == 8").index)
        & set(pow_counts.query("rows == 8 and horizons == 8").index)
    )
    issues = sorted(t for t in common if _role_of_issue(pd.Timestamp(t)) is not None)
    return issues, {
        "ensemble_issues": int(len(ens_counts)),
        "control_issues": int(len(ctl_counts)),
        "power_issues": int(len(pow_counts)),
        "complete_common_issues": int(len(common)),
        "role_eligible_issues": int(len(issues)),
        "first_complete_issue": str(min(common)),
        "last_complete_issue": str(max(common)),
        "bad_member_issue_number_keys_first10": bad_member_keys,
        "bad_control_issue_keys_first10": bad_control_keys,
        "exact_member_key_check_pass": not bad_member_keys,
        "exact_control_key_check_pass": not bad_control_keys,
    }


def _issue_tensor(
    issue: pd.Timestamp,
    ens: pd.DataFrame,
    ctl: pd.DataFrame,
    power_by_time: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ctx_times = [issue - pd.Timedelta(hours=3 * i) for i in range(128, 0, -1)]
    target_times = [issue + pd.Timedelta(hours=h) for h in HORIZONS]
    if any(t not in power_by_time.index for t in ctx_times + target_times):
        raise KeyError("missing context or target power")
    x = power_by_time.loc[ctx_times].to_numpy("float32")
    y = power_by_time.loc[target_times].to_numpy("float32")
    e = ens[ens["issue"].eq(issue)].sort_values(["number", "horizon"])
    c = ctl[ctl["issue"].eq(issue)].sort_values(["horizon"])
    if len(e) != 400 or len(c) != 8:
        raise KeyError("missing weather rows")
    if sorted(e["number"].unique().astype(int).tolist()) != list(range(1, 51)):
        raise KeyError("wrong member set")
    if sorted(e["horizon"].unique().astype(int).tolist()) != HORIZONS or c["horizon"].astype(int).tolist() != HORIZONS:
        raise KeyError("wrong horizons")
    weather = e[WEATHER].to_numpy("float32").reshape(50, 8, len(WEATHER))
    control = c[WEATHER].to_numpy("float32")
    return x, y, weather, control


def prepare() -> dict:
    ens, ctl, power, source_manifest = _read_sources()
    power_by_time, power_audit = _make_power_series(power)
    complete, coverage = _complete_issues(ens, ctl, power)
    rows = []
    dropped = []
    for issue in complete:
        role = _role_of_issue(pd.Timestamp(issue))
        try:
            x, y, weather, control = _issue_tensor(pd.Timestamp(issue), ens, ctl, power_by_time)
            rows.append((pd.Timestamp(issue), role, x, y, weather, control))
        except KeyError as exc:
            dropped.append({"issue": str(issue), "reason": str(exc)})
    if not rows:
        raise RuntimeError("BLOCKED_DATA: no usable Sweden Zone 3 issues after context/weather checks")
    role_counts = {role: sum(1 for r in rows if r[1] == role) for role in ROLES}
    if role_counts["TEST"] < 60:
        raise RuntimeError(f"BLOCKED_COVERAGE: TEST has {role_counts['TEST']} issues")
    x = np.stack([r[2] for r in rows]).astype("float32")
    y = np.stack([r[3] for r in rows]).astype("float32")
    weather_raw = np.stack([r[4] for r in rows]).astype("float32")
    control_raw = np.stack([r[5] for r in rows]).astype("float32")
    roles = np.asarray([r[1] for r in rows], dtype=object)
    issues = np.asarray([str(r[0]) for r in rows], dtype=object)
    train = roles == "TRAIN"
    sigma = float(np.std(y[train], ddof=0))
    if not np.isfinite(sigma) or sigma <= 0:
        raise RuntimeError("BLOCKED_DATA: nonpositive TRAIN sigma for wind power")
    w_mean = weather_raw[train].reshape(-1, len(WEATHER)).mean(0).astype("float32")
    w_std = weather_raw[train].reshape(-1, len(WEATHER)).std(0).astype("float32")
    if (w_std <= 0).any() or not np.isfinite(w_std).all():
        raise RuntimeError("BLOCKED_DATA: bad TRAIN weather scale")
    weather = (weather_raw - w_mean[None, None, None, :]) / w_std[None, None, None, :]
    control = (control_raw - w_mean[None, None, :]) / w_std[None, None, :]
    packet = CACHE / "B" / "data" / "Sweden_SE3_B_packet.npz"
    packet.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        packet,
        x=x,
        y=y,
        weather=weather.astype("float32"),
        control=control.astype("float32"),
        weather_raw=weather_raw,
        control_raw=control_raw,
        issue=issues,
        role=roles,
        sigma=np.asarray(sigma, dtype="float32"),
        weather_mean=w_mean,
        weather_std=w_std,
        weather_columns=np.asarray(WEATHER, dtype=object),
    )
    schedules = {}
    for seed in SEEDS:
        rng = np.random.default_rng(int(hashlib.sha256(f"B|schedule|{seed}".encode()).hexdigest()[:16], 16))
        train_idx = np.flatnonzero(train)
        tuples = rng.choice(train_idx, size=(256, 4), replace=True).astype("int64")
        sp = CACHE / "B" / f"schedule_{seed}.npz"
        np.savez_compressed(sp, indices=tuples)
        schedules[str(seed)] = {"path": str(sp), "sha256": sha(sp), "shape": list(tuples.shape)}
    schema = {
        "ensembles": {"shape_with_added_issue": list(ens.shape), "original_columns": [c for c in ens.columns.astype(str).tolist() if c != "issue"], "added_columns": ["issue"]},
        "control": {"shape_with_added_issue": list(ctl.shape), "original_columns": [c for c in ctl.columns.astype(str).tolist() if c != "issue"], "added_columns": ["issue"]},
        "power": {"shape_with_added_issue": list(power.shape), "original_columns": [c for c in power.columns.astype(str).tolist() if c != "issue"], "added_columns": ["issue"]},
        "weather_columns_used": WEATHER,
    }
    workflow_evidence = {
        "create_correct_structure_R": "w_parameters is a function argument used to select weather columns and to add ensemble/control/observation columns.",
        "workflow_train_R": "workflow_train passes caller-provided weather_params into create_correct_structure(..., w_parameters = weather_params).",
        "workflow_predict_R": "workflow_predict follows the same caller-provided weather_params pattern.",
        "fixed_list_limit": "No hard-coded weather_params call list was found in the pinned workflow files; this run fixes all common numeric forecast weather fields present in ensemble/control CSV: u100,v100,t2m,sp,speed.",
    }
    time_audit = {
        "power_time_interpretation": "valid time",
        "issue_formula": "issue = time - horizon_hours",
        "horizons_hours": HORIZONS,
        "timezone_evidence": "create_correct_structure.R uses as.POSIXct(filter_time, tz='UTC'); CSV timestamps are stored without offset and treated as archive UTC.",
        "release_time_verification": "RELEASE_TIME_UNVERIFIED: archive-defined issue experiment only, not an operational real-time backtest.",
        "role_counts": role_counts,
        "split_policy": ROLES,
        "contract_actual_range": "official actual Sweden power records cover valid times 2015-01-05 through 2019-11-01; this run uses issue roles whose targets stay before 2019-09-01, so TEST last issue is 2019-08-30.",
        "power_valid_time_audit": power_audit,
        "dropped_after_complete_issue_check": dropped[:25],
        "dropped_count": len(dropped),
        "missing_reason_counts": {
            "early_context_missing_power": int(sum("missing context" in d["reason"] for d in dropped)),
            "weather_key_missing": int(sum("weather" in d["reason"] for d in dropped)),
        },
    }
    member_audit = {
        "members": [1, 50],
        "member_count": int(ens["number"].nunique()),
        "complete_issue_requirements": "50 members x 8 horizons for ensemble, 8 horizons for control, 8 horizons for power",
        "coverage": coverage,
        "control_is_independent_path": True,
        "control_not_used_as_main_perturbed_member": True,
    }
    audit = {
        "status": "PASS",
        "domain_context": {
            "domain": "Sweden SE3 wind-power time series with same-issue ECMWF ensemble forecasts",
            "downstream_decision": "archive-condition PEFT method screen; not an operational release-time deployment claim",
            "primary_leakage_paths": [
                "using valid time as issue time",
                "future wind-power targets in context",
                "weather member selection by performance",
                "TEST-based model/checkpoint/calibration selection",
            ],
            "quality_bar": "publication-style pilot receipts with zero target leakage tolerance",
        },
        "source_manifest": source_manifest,
        "schema": schema,
        "time_mapping": time_audit,
        "member_audit": member_audit,
        "prepared_npz": {"path": str(packet), "sha256": sha(packet), "bytes": packet.stat().st_size},
        "arrays": {
            "x": list(x.shape),
            "y": list(y.shape),
            "weather": list(weather.shape),
            "control": list(control.shape),
            "weather_standardized": True,
            "x_y_raw_mw": True,
            "sigma_train_power_raw_mw_float64": float(np.std(y[train].astype("float64"), ddof=0)),
            "sigma_train_power_scope": "all TRAIN target y(issue+3h..+24h) raw MW values after common finite issue filtering",
            "weather_scale_train_forecasts_only": True,
            "all_arrays_finite_after_filter": bool(np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(weather).all() and np.isfinite(control).all()),
        },
        "workflow_w_parameters_evidence": workflow_evidence,
        "license_and_release_limits": {
            "repository_license_file_found": False,
            "raw_redistribution": "raw CSV and model weights are local cache only and not pushed",
            "readme_data_rights_note": "README states ECMWF data were obtained through an academic licence for research purposes.",
            "release_time": "actual operational forecast publication delay not verified",
        },
        "schedules": schedules,
    }
    r = RESULTS / "B"
    save(r / "DATA_AUDIT.json", audit)
    save(r / "CSV_SCHEMA.json", schema)
    save(r / "TIME_MAPPING_AUDIT.json", time_audit)
    save(r / "MEMBER_AUDIT.json", member_audit)
    save(
        r / "INFORMATION_CONTRACT.json",
        {
            "context": "128 actual wind_power observations strictly before issue; last context = issue - 3h",
            "target": "wind_power at issue+3h ... issue+24h",
            "weather": "same archive issue; 50 perturbed members, 8 leads, standardized by TRAIN forecast mean/std",
            "control": "same issue control path, standardized with same TRAIN forecast mean/std",
            "forbidden": ["future target in context", "future observed weather", "old forecast vintages as same-issue ensemble"],
            "release_time_limit": "actual forecast publication delay not verified",
        },
    )
    (r / "SOURCE_LIMITS.md").write_text(
        "# B source limits\n\n"
        "The pinned author repository provides same-issue archive forecast data and observed power. "
        "The issue/valid mapping is verified from CSV horizon fields, but actual operational publication delay "
        "was not verified. This is an archive-conditioned pilot, not a live deployment backtest.\n",
        encoding="utf-8",
    )
    return audit


def load_data() -> dict:
    packet = CACHE / "B" / "data" / "Sweden_SE3_B_packet.npz"
    if not packet.exists():
        prepare()
    with np.load(packet, allow_pickle=True) as z:
        return {
            "x": z["x"].astype("float32"),
            "y": z["y"].astype("float32"),
            "weather": z["weather"].astype("float32"),
            "control": z["control"].astype("float32"),
            "role": np.asarray(z["role"], dtype=object),
            "issue": np.asarray(z["issue"], dtype=object),
            "sigma": float(z["sigma"]),
            "weather_columns": [str(v) for v in z["weather_columns"].tolist()],
        }


def schedule(seed: int) -> np.ndarray:
    path = CACHE / "B" / f"schedule_{seed}.npz"
    if not path.exists():
        prepare()
    with np.load(path) as z:
        return z["indices"].astype("int64")


def batch(data: dict, indices: np.ndarray) -> dict:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    return {
        "x": data["x"][idx],
        "y": data["y"][idx],
        "weather": data["weather"][idx],
        "control": data["control"][idx],
        "sigma": np.full(len(idx), data["sigma"], dtype="float32"),
        "indices": idx,
    }


if __name__ == "__main__":
    prepare()
