"""CPU-only postrun audit for residual_feedback_peft_screen_v1_20260922.

This script checks data, clock, packet, reference, and issued-forecast ledgers.
It intentionally does not read score tables or parse model-selection contents.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import CACHE, EXP, RESULTS, ROOT, SEEDS, read, save, sha
from data import CONTEXT, HORIZON, Data


TRACKS = ("A", "B")
ROLES = ("DEV", "TEST")
ROLE_LOCAL = {
    "DONOR": set(range(0, 32)),
    "DEV": set(range(32, 40)),
    "TEST": set(range(40, 48)),
}
OFFSETS = {
    "A": [256, 192, 128, 64],
    "B": [64, 32, 16, 8],
}
VISIBLE_COUNTS = {
    "A": [64, 64, 64, 64],
    "B": [64, 32, 16, 8],
}


def _jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def _ok(name, details=None):
    return {"name": name, "status": "PASS", "details": details or {}}


def _fail(checks, failures, name, message, details=None):
    item = {"name": name, "status": "FAIL", "message": message, "details": details or {}}
    checks.append(item)
    failures.append(item)


def _require(checks, failures, name, condition, message, details=None):
    if condition:
        checks.append(_ok(name, details))
    else:
        _fail(checks, failures, name, message, details)


def _load_npy(path):
    with Path(path).open("rb") as handle:
        return np.load(handle, allow_pickle=False)


def _cpu_episode_from_values(values, series, issue, track):
    prefix = values[:issue, series].copy()
    support_issues = [issue - offset for offset in OFFSETS[track]]
    support = np.stack([prefix[s - CONTEXT : s] for s in support_issues]).astype(np.float32)
    y = np.full((4, HORIZON), np.nan, dtype=np.float32)
    mask = np.zeros((4, HORIZON), dtype=bool)
    for i, support_issue in enumerate(support_issues):
        visible = min(HORIZON, issue - support_issue)
        y[i, :visible] = prefix[support_issue : support_issue + visible]
        mask[i, :visible] = True
    return {
        "x": prefix[issue - CONTEXT : issue].copy(),
        "support": support,
        "y": y,
        "mask": mask,
        "issues": support_issues,
    }


def _masked_residual(y, q, mask, scale):
    out = np.zeros(HORIZON, dtype=np.float32)
    idx = np.flatnonzero(mask)
    out[idx] = (y[idx] - q[idx]) / np.float32(scale)
    return out


def _check_static_no_future_trace(checks, failures):
    tasks = (EXP / "tasks.py").read_text(encoding="utf-8")
    runner = (EXP / "runner.py").read_text(encoding="utf-8")
    details = {
        "tasks_episode_requires_prefix_length_equal_issue": "assert len(prefix)==t" in tasks,
        "tasks_episode_initializes_unreleased_y_as_nan": "np.full((4,64),np.nan" in tasks,
        "tasks_episode_uses_visible_count_min_64_t_minus_s": "n=min(64,t-s)" in tasks,
        "tasks_query_context_uses_prefix_before_issue": "prefix[t-256:t]" in tasks,
        "runner_uses_asof_for_fit_generator_support": "make_episode(d.asof(int(series),int(t)),int(t),arm[0])" in runner,
        "runner_uses_asof_for_fixed_eval": "prefix=d.asof(int(series),int(t));ep=make_episode(prefix,int(t),track)" in runner,
        "runner_uses_asof_for_online_b": "ep=make_episode(d.asof(int(series),int(t)),int(t),'B')" in runner,
        "issuer_prevents_overwrite": "assert not p.exists(),'Issued forecasts cannot be overwritten'" in tasks,
    }
    _require(
        checks,
        failures,
        "static_no_future_boundary_trace",
        all(details.values()),
        "Expected source-level no-future boundary patterns were not all found.",
        details,
    )


def _check_source_and_selection_seals(checks, failures):
    seal_path = RESULTS / "SOURCE_SEAL.json"
    if not seal_path.exists():
        _fail(checks, failures, "source_seal_exists", "SOURCE_SEAL.json is missing.")
        return
    seal = read(seal_path)
    source_file_results = {}
    for name, expected in seal.get("files", {}).items():
        path = EXP / name
        actual = sha(path) if path.exists() else None
        source_file_results[name] = {"expected": expected, "actual": actual, "match": actual == expected}
    data_file_results = {}
    for name, expected in seal.get("data", {}).items():
        path = RESULTS / name
        actual = sha(path) if path.exists() else None
        data_file_results[name] = {"expected": expected, "actual": actual, "match": actual == expected}
    _require(
        checks,
        failures,
        "source_seal_hashes",
        all(x["match"] for x in source_file_results.values()) and all(x["match"] for x in data_file_results.values()),
        "One or more source/data files no longer match SOURCE_SEAL.json.",
        {"source_files": source_file_results, "data_files": data_file_results},
    )

    for track in TRACKS:
        seal_file = RESULTS / track / "SELECTION_SEAL.json"
        if not seal_file.exists():
            _fail(checks, failures, f"{track}_selection_seal_exists", f"{track} SELECTION_SEAL.json is missing.")
            continue
        selection_seal = read(seal_file)
        source_match = selection_seal.get("source_seal") == sha(seal_path)
        file_results = {}
        for name, expected in selection_seal.get("files", {}).items():
            path = RESULTS / track / name
            actual = sha(path) if path.exists() else None
            file_results[name] = {"expected": expected, "actual": actual, "match": actual == expected}
        _require(
            checks,
            failures,
            f"{track}_selection_seal_hashes",
            source_match and all(x["match"] for x in file_results.values()),
            f"{track} selection seal does not match sealed files or SOURCE_SEAL.",
            {"source_seal_match": source_match, "files": file_results},
        )


def _check_data_and_packets(checks, failures, d):
    data_source = read(RESULTS / "DATA_SOURCE.json")
    series_split = read(RESULTS / "SERIES_SPLIT.json")
    time_split = read(RESULTS / "TIME_SPLIT.json")
    packet_manifest = read(RESULTS / "TRAIN_PACKET_MANIFEST.json")

    raw_path = ROOT / data_source["raw_path"]
    selected_path = ROOT / data_source["selected_cache"]["path"]
    raw_actual = sha(raw_path) if raw_path.exists() else None
    selected_actual = sha(selected_path) if selected_path.exists() else None
    _require(
        checks,
        failures,
        "raw_and_selected_cache_hashes",
        raw_actual == data_source["raw_sha256"] and selected_actual == data_source["selected_cache"]["sha256"],
        "Raw or selected Electricity cache hash changed.",
        {
            "raw_path": data_source["raw_path"],
            "raw_expected": data_source["raw_sha256"],
            "raw_actual": raw_actual,
            "selected_path": data_source["selected_cache"]["path"],
            "selected_expected": data_source["selected_cache"]["sha256"],
            "selected_actual": selected_actual,
        },
    )
    _require(
        checks,
        failures,
        "series_split_local_original_mapping",
        series_split["ids"] == d.ids and series_split["local_indices"] == d.role_indices,
        "SERIES_SPLIT ids/local_indices no longer match Data().",
        {"json_ids": series_split["ids"], "data_ids": d.ids, "local_indices": d.role_indices},
    )
    _require(
        checks,
        failures,
        "time_split_edges",
        time_split["edges"] == d.edges == [0, 7891, 15782, 21043, 26304],
        "TIME_SPLIT edges no longer match Data().",
        {"json_edges": time_split["edges"], "data_edges": d.edges},
    )

    schedule_details = {}
    for seed in SEEDS:
        seed_doc = packet_manifest["seed_schedules"][str(seed)]
        checks_for_seed = {}
        for kind, expected_array in (
            ("base_schedule", d.base_schedule(seed)),
            ("meta_schedule", d.meta_schedule(seed)),
        ):
            path = ROOT / seed_doc[kind]["path"]
            arr = _load_npy(path) if path.exists() else None
            series_ok = arr is not None and set(np.unique(arr[:, :, 0]).tolist()).issubset(ROLE_LOCAL["DONOR"])
            if kind == "base_schedule":
                issue_ok = arr is not None and int(arr[:, :, 1].min()) >= CONTEXT and int(arr[:, :, 1].max()) + HORIZON <= d.edges[1]
            else:
                issue_ok = arr is not None and int(arr[:, :, 1].min()) >= d.edges[1] + CONTEXT and int(arr[:, :, 1].max()) + HORIZON <= d.edges[2]
            checks_for_seed[kind] = {
                "path": seed_doc[kind]["path"],
                "sha_match": path.exists() and sha(path) == seed_doc[kind]["sha256"],
                "equals_Data_schedule": arr is not None and np.array_equal(arr, expected_array),
                "donor_series_only": bool(series_ok),
                "issue_window_ok": bool(issue_ok),
                "shape": None if arr is None else list(arr.shape),
            }
        schedule_details[str(seed)] = checks_for_seed
    _require(
        checks,
        failures,
        "training_schedules_unchanged_and_donor_only",
        all(
            all(v["sha_match"] and v["equals_Data_schedule"] and v["donor_series_only"] and v["issue_window_ok"] for v in per_seed.values())
            for per_seed in schedule_details.values()
        ),
        "One or more training schedules changed or violates DONOR/time constraints.",
        schedule_details,
    )

    episode_details = {}
    for track in TRACKS:
        episode_details[track] = {}
        for role in ROLES:
            doc = packet_manifest["episodes"][track][role]
            path = ROOT / doc["path"]
            arr = _load_npy(path) if path.exists() else None
            expected = d.episodes(track, role)
            role_series_ok = arr is not None and set(np.unique(arr[:, 0]).tolist()).issubset(ROLE_LOCAL[role])
            role_bounds = (d.edges[2], d.edges[3]) if role == "DEV" else (d.edges[3], d.edges[4])
            query_ok = arr is not None and bool((arr[:, 1] >= role_bounds[0]).all() and (arr[:, 1] + HORIZON <= role_bounds[1]).all())
            episode_details[track][role] = {
                "path": doc["path"],
                "sha_match": path.exists() and sha(path) == doc["sha256"],
                "equals_Data_episodes": arr is not None and np.array_equal(arr, expected),
                "local_series_role": role,
                "role_series_ok": bool(role_series_ok),
                "query_target_inside_role": bool(query_ok),
                "shape": None if arr is None else list(arr.shape),
            }
    _require(
        checks,
        failures,
        "episode_packets_unchanged_and_role_scoped",
        all(
            info["sha_match"] and info["equals_Data_episodes"] and info["role_series_ok"] and info["query_target_inside_role"]
            for per_track in episode_details.values()
            for info in per_track.values()
        ),
        "One or more episode packets changed or violates role/time constraints.",
        episode_details,
    )


def _check_support_clock(checks, failures, d):
    details = {}
    for scope in ["meta"] + [f"{track}_{role}" for track in TRACKS for role in ROLES]:
        if scope == "meta":
            pair_batches = [d.meta_schedule(seed).reshape(-1, 2) for seed in SEEDS]
            tracks = TRACKS
            role = "DONOR"
        else:
            track, role = scope.split("_")
            pair_batches = [d.episodes(track, role)]
            tracks = (track,)
        details[scope] = {}
        for track in tracks:
            support_min = None
            visible_count_ok = True
            targets_realized_ok = True
            context_ok = True
            for pairs in pair_batches:
                for series, issue in np.asarray(pairs, dtype=np.int64):
                    for offset, visible_expected in zip(OFFSETS[track], VISIBLE_COUNTS[track]):
                        support_issue = int(issue) - offset
                        support_min = support_issue if support_min is None else min(support_min, support_issue)
                        context_ok = context_ok and support_issue - CONTEXT >= 0
                        visible_count = min(HORIZON, int(issue) - support_issue)
                        visible_count_ok = visible_count_ok and visible_count == visible_expected
                        if track == "A":
                            targets_realized_ok = targets_realized_ok and support_issue + HORIZON <= int(issue)
            details[scope][track] = {
                "series_index_scope": "local48",
                "role": role,
                "minimum_support_issue": support_min,
                "all_support_issues_at_or_after_T1": support_min is not None and support_min >= d.edges[1],
                "support_contexts_have_256_history": bool(context_ok),
                "visible_counts_match_contract": bool(visible_count_ok),
                "A_support_targets_realized_before_query": bool(targets_realized_ok),
            }
    _require(
        checks,
        failures,
        "support_clock_contract",
        all(
            info["all_support_issues_at_or_after_T1"]
            and info["support_contexts_have_256_history"]
            and info["visible_counts_match_contract"]
            and info["A_support_targets_realized_before_query"]
            for per_scope in details.values()
            for info in per_scope.values()
        ),
        "Support clock constraints failed.",
        details,
    )


def _check_reference_forecasts(checks, failures, d):
    rows = _jsonl(RESULTS / "REFERENCE_FORECASTS.jsonl")
    seen = set()
    details = {"count": len(rows), "local_series_index": True, "bad_examples": []}
    ok = True
    for row in rows:
        key = (row.get("seed"), row.get("series"), row.get("issue"))
        if key in seen:
            ok = False
            details["bad_examples"].append({"reason": "duplicate_reference_key", "key": key})
        seen.add(key)
        series = int(row["series"])
        issue = int(row["issue"])
        path = ROOT / row["file"]
        file_ok = path.exists() and sha(path) == row["sha256"]
        issue_ok = issue >= d.edges[1]
        role_ok = 0 <= series < 48
        context_ok = False
        bank_ok = False
        if file_ok:
            with np.load(path, allow_pickle=False) as z:
                context = d.asof(series, issue)[-CONTEXT:]
                context_hash = hashlib.sha256(context.tobytes()).hexdigest()
                context_ok = context_hash == str(z["context_hash"]) == row["context_sha"]
                bank_ok = str(z["bank_hash"]) == row["bank_sha"]
        if not (file_ok and issue_ok and role_ok and context_ok and bank_ok):
            ok = False
            details["bad_examples"].append(
                {
                    "reason": "reference_forecast_mismatch",
                    "key": key,
                    "file_ok": file_ok,
                    "issue_at_or_after_T1": issue_ok,
                    "local_series_in_0_47": role_ok,
                    "context_hash_ok": context_ok,
                    "bank_hash_ok": bank_ok,
                }
            )
    _require(
        checks,
        failures,
        "reference_forecast_ledger",
        ok,
        "REFERENCE_FORECASTS ledger has duplicate keys, changed files, or invalid context hashes.",
        details,
    )


def _check_issued_forecasts(checks, failures, d):
    all_details = {}
    overall_ok = True
    for track in TRACKS:
        track_dir = RESULTS / track
        all_test = read(track_dir / "ALL_TEST_SAVED.json") if (track_dir / "ALL_TEST_SAVED.json").exists() else {}
        rows = _jsonl(track_dir / "ISSUED_FORECASTS.jsonl")
        seen_keys = set()
        seen_files = set()
        seen_hashes = set()
        bad = []
        role_counts = {}
        for row in rows:
            role = row.get("role")
            role_counts[role] = role_counts.get(role, 0) + 1
            key = (role, row.get("arm"), row.get("seed"), str(row.get("tag")), row.get("series"), row.get("issue"))
            path = ROOT / row["file"]
            duplicate_key = key in seen_keys
            duplicate_file = row["file"] in seen_files
            duplicate_hash = row["sha256"] in seen_hashes
            seen_keys.add(key)
            seen_files.add(row["file"])
            seen_hashes.add(row["sha256"])
            series = int(row["series"])
            issue = int(row["issue"])
            role_bounds = (d.edges[2], d.edges[3]) if role == "DEV" else (d.edges[3], d.edges[4])
            role_ok = role in ROLE_LOCAL and series in ROLE_LOCAL[role]
            clock_ok = row.get("target_start") == issue and row.get("visible_before") == issue
            window_ok = issue >= role_bounds[0] and issue + HORIZON <= role_bounds[1]
            file_ok = path.exists() and sha(path) == row["sha256"]
            npz_ok = False
            if file_ok:
                with np.load(path, allow_pickle=False) as z:
                    q = z["q"]
                    npz_ok = (
                        tuple(q.shape) == (HORIZON, 9)
                        and np.isfinite(q).all()
                        and int(z["series"]) == series
                        and int(z["issue"]) == issue
                    )
            row_ok = (
                not duplicate_key
                and not duplicate_file
                and role_ok
                and clock_ok
                and window_ok
                and file_ok
                and npz_ok
            )
            if not row_ok:
                bad.append(
                    {
                        "key": key,
                        "duplicate_key": duplicate_key,
                        "duplicate_file": duplicate_file,
                        "duplicate_hash": duplicate_hash,
                        "role_ok": role_ok,
                        "target_start_issue_visible_before_ok": clock_ok,
                        "query_window_inside_role": window_ok,
                        "file_hash_ok": file_ok,
                        "npz_payload_ok": npz_ok,
                    }
                )
        track_ok = all_test.get("status") == "COMPLETE" and rows and not bad
        overall_ok = overall_ok and track_ok
        all_details[track] = {
            "all_test_saved_status": all_test.get("status"),
            "issued_rows": len(rows),
            "unique_keys": len(seen_keys),
            "unique_files": len(seen_files),
            "unique_hashes": len(seen_hashes),
            "content_hash_uniqueness_required": False,
            "reason": "Identical forecasts across arms are legitimate, especially c=1 at checkpoint0; issuance keys and paths must be unique, and every file must match its recorded hash.",
            "role_counts": role_counts,
            "local_series_index": True,
            "original_test_ids": d.ids["TEST"],
            "bad_examples": bad[:20],
        }
    _require(
        checks,
        failures,
        "issued_forecast_ledgers",
        overall_ok,
        "Issued forecast ledger/file/hash/clock checks failed.",
        all_details,
    )


def _check_prefix_poisoning(checks, failures, d):
    series, issue = d.episodes("B", "TEST")[0]
    issue = int(issue)
    series = int(series)
    prefix = d.asof(series, issue)
    before = float(d.values[0, series])
    prefix[0] = before + 987654.0
    copy_ok = float(d.values[0, series]) == before

    values = np.array(d.values, copy=True)
    base = _cpu_episode_from_values(values, series, issue, "B")
    poisoned = np.array(values, copy=True)
    poisoned[issue : issue + HORIZON, series] = np.float32(1.0e20)
    poisoned_episode = _cpu_episode_from_values(poisoned, series, issue, "B")
    future_invariant = (
        np.array_equal(base["x"], poisoned_episode["x"])
        and np.array_equal(base["support"], poisoned_episode["support"])
        and np.array_equal(base["mask"], poisoned_episode["mask"])
        and np.allclose(base["y"], poisoned_episode["y"], equal_nan=True)
    )

    q = np.linspace(-1.0, 1.0, HORIZON, dtype=np.float32)
    y_nan = np.full(HORIZON, np.nan, dtype=np.float32)
    y_big = np.full(HORIZON, 1.0e20, dtype=np.float32)
    mask = np.zeros(HORIZON, dtype=bool)
    mask[:8] = True
    y_nan[:8] = np.arange(8, dtype=np.float32)
    y_big[:8] = np.arange(8, dtype=np.float32)
    residual_nan = _masked_residual(y_nan, q, mask, 3.0)
    residual_big = _masked_residual(y_big, q, mask, 3.0)
    masked_residual_ok = np.isfinite(residual_nan).all() and np.array_equal(residual_nan, residual_big)

    _require(
        checks,
        failures,
        "cpu_prefix_and_mask_poisoning",
        copy_ok and future_invariant and masked_residual_ok,
        "Prefix copy or masked residual no-future invariant failed.",
        {
            "series_local": series,
            "series_original": d.ids["TEST"][series - 40],
            "issue": issue,
            "asof_returns_copy": copy_ok,
            "future_poison_does_not_change_cpu_episode": future_invariant,
            "masked_residual_selects_observed_cells_before_math": masked_residual_ok,
        },
    )


def build_audit(require_complete=True):
    checks = []
    failures = []
    complete_path = RESULTS / "ALL_MAIN_COMPLETE.json"
    if not complete_path.exists() or read(complete_path).get("status") != "COMPLETE":
        status = "PENDING_ALL_MAIN_COMPLETE"
        if require_complete:
            _fail(checks, failures, "all_main_complete", "ALL_MAIN_COMPLETE.json is missing or not COMPLETE.")
            status = "FAIL"
        return {
            "status": status,
            "classification": "DATA_POSTRUN_AUDIT_NOT_READY" if status.startswith("PENDING") else "BLOCKED_IMPLEMENTATION",
            "all_main_complete": complete_path.exists(),
            "checks": checks,
            "failures": failures,
            "performance_files_read": False,
            "selection_contents_parsed": False,
        }

    d = Data()
    _require(checks, failures, "all_main_complete", True, "", read(complete_path))
    _check_static_no_future_trace(checks, failures)
    _check_source_and_selection_seals(checks, failures)
    _check_data_and_packets(checks, failures, d)
    _check_support_clock(checks, failures, d)
    _check_reference_forecasts(checks, failures, d)
    _check_issued_forecasts(checks, failures, d)
    _check_prefix_poisoning(checks, failures, d)

    status = "PASS" if not failures else "FAIL"
    return {
        "status": status,
        "classification": "DATA_CLOCK_LEDGER_AUDIT" if status == "PASS" else "BLOCKED_IMPLEMENTATION",
        "source": "Electricity",
        "series_index_convention": {
            "packet_and_issued_ledgers": "local48 indexes",
            "original_ids": d.ids,
            "local_roles": d.role_indices,
        },
        "time_edges": d.edges,
        "checks": checks,
        "failures": failures,
        "performance_files_read": False,
        "selection_contents_parsed": False,
        "optimizer_or_inference_rerun": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-pending", action="store_true", help="Do not fail when ALL_MAIN_COMPLETE is absent.")
    parser.add_argument("--no-save", action="store_true", help="Run audit without writing DATA_POSTRUN_AUDIT.json.")
    args = parser.parse_args()
    audit = build_audit(require_complete=not args.allow_pending)
    if not args.no_save:
        save(RESULTS / "DATA_POSTRUN_AUDIT.json", audit)
    print(json.dumps({"status": audit["status"], "failures": len(audit["failures"])}, ensure_ascii=False))
    if audit["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
