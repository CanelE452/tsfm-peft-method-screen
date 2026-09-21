"""Guarded, CPU-only final scoring of already sealed predictions."""

import hashlib
import gzip
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from common import CACHE, EXP, RESULTS, ROOT, event, read_json, save_json, sha
from evaluation import WINDOWS, apply_affine, bootstrap_effect, score_tables

SOURCES = ("Electricity", "ETTh1")
SEEDS = (92121, 92122)
ARMS = ("R_ROLLOUT_LORA", "S_STATE_ADAPTER", "U_UNCERTAINTY_ADAPTER")
FIXED = ("F0_MEDIAN", "F0_NATIVE", "CHRONOS2_DIRECT")
METHODS = (*FIXED, *ARMS, "R_NATIVE", "R_MC16")
VARIANTS = ("raw", "ordered", "affine")
COMPARISONS = (
    ("general_rollout_training", "F0_MEDIAN", ARMS[0]),
    ("generated_state_information", ARMS[0], ARMS[1]),
    ("additional_width_information", ARMS[1], ARMS[2]),
    ("total_U_vs_R", ARMS[0], ARMS[2]),
    ("U_vs_official_F0", "F0_NATIVE", ARMS[2]),
    ("U_vs_official_R", "R_NATIVE", ARMS[2]),
    ("U_vs_MC16", "R_MC16", ARMS[2]),
    ("U_vs_direct_long_model", "CHRONOS2_DIRECT", ARMS[2]),
    ("official_branching_vs_median", "F0_MEDIAN", "F0_NATIVE"),
    ("R_native_vs_R_median", ARMS[0], "R_NATIVE"),
)


def _expected():
    return {(source, f"{method}_seed{seed}") for source in SOURCES for method in METHODS
            for seed in ((0,) if method in FIXED else SEEDS)}


def _source_verification(training):
    original = training["source_hashes"]
    current = {str(f.relative_to(ROOT)): sha(f) for f in sorted(EXP.rglob("*.py"))}
    changed = {path for path in original.keys() | current.keys() if original.get(path) != current.get(path)}
    receipts = []
    if changed:
        amendment_path = RESULTS / "SOURCE_AMENDMENTS.json"
        if not amendment_path.exists():
            raise RuntimeError(f"Unrecorded code changes after training seal: {sorted(changed)}")
        amendments = read_json(amendment_path)["amendments"]
        for path in sorted(changed):
            matches = [r for r in amendments if r["path"] == path
                       and r["original_sha256"] == original.get(path)
                       and r["current_sha256"] == current.get(path)]
            assert len(matches) == 1, f"Missing unique amendment for {path}"
            record = matches[0]
            assert original.get(path) is not None, "Added/deleted execution files require manual audit."
            assert sha(ROOT / record["original_path"]) == original[path]
            diff = ROOT / record["diff_path"]
            assert diff.is_file() and diff.stat().st_size > 0
            receipts.append({**record, "diff_sha256": sha(diff)})
    return {"original_source_hashes": original, "current_source_hashes": current,
            "unchanged_since_training_seal": not changed, "amendments": receipts}


def _guard():
    """Do not load data or numerical TEST predictions before every receipt passes."""
    training = read_json(RESULTS / "TRAINING_CONTRACT_SEALED.json")
    assert training["master_sha256"] == sha(EXP / "contract/MASTER_CLI.txt")
    source_verification = _source_verification(training)
    seal = read_json(RESULTS / "ALL_SELECTIONS_SEALED.json")
    for key, filename in (("model_selection_sha256", "MODEL_SELECTION.json"),
                          ("lr_selection_sha256", "LR_SELECTION.json"),
                          ("calibration_sha256", "CALIBRATION_PARAMETERS.json")):
        assert seal[key] == sha(RESULTS / filename), f"Selection seal mismatch: {filename}"
    selections = read_json(RESULTS / "MODEL_SELECTION.json")
    expected_selections = {f"{s}/{a}/{seed}" for s in SOURCES for a in ARMS for seed in SEEDS}
    assert set(selections) == expected_selections
    for selection in selections.values():
        assert selection["seed"] in SEEDS and selection["selected_step"] in (0, 128, 256, 512)
        assert sha(ROOT / selection["checkpoint"]) == selection["checkpoint_sha256"]
    parameters = read_json(RESULTS / "CALIBRATION_PARAMETERS.json")
    assert set(parameters) == {f"{s}/{m}" for s, m in _expected()}
    for params in parameters.values():
        assert params["role_required"] == "CALIBRATION" and not params["feedback_changed"]
        assert len(params["blocks"]) == 4
        for block in params["blocks"]:
            assert len(block["grid_scores"]) == 35
            winner = min(block["grid_scores"], key=lambda r: (
                r["scaled_pinball"], (r["alpha"] - 1) ** 2 + r["beta"] ** 2,
                r["alpha"], abs(r["beta"]), r["beta"]))
            assert all(block[k] == winner[k] for k in ("alpha", "beta", "scaled_pinball"))
    manifest = read_json(RESULTS / "PREDICTIONS_MANIFEST.json")
    assert manifest["selection_seal_sha256"] == sha(RESULTS / "ALL_SELECTIONS_SEALED.json")
    assert manifest["all_test_predictions_saved_before_scoring"] is True
    tests = [r for r in manifest["files"] if r["role"] == "TEST"]
    cals = [r for r in manifest["files"] if r["role"] == "CALIBRATION"]
    assert len(tests) == 26 and {(r["source"], r["model"]) for r in tests} == _expected()
    assert len(cals) == 26 and {(r["source"], r["model"]) for r in cals} == _expected()
    for record in tests + cals:
        assert sha(ROOT / record["path"]) == record["sha256"], record["path"]
        receipt = read_json((ROOT / record["path"]).with_suffix(".json"))
        assert receipt == record, f"Manifest/receipt mismatch: {record['path']}"
    ledger = [json.loads(line) for line in (RESULTS / "UPDATE_LEDGER.jsonl").read_text().splitlines()]
    spent = {kind: {(r["fit_id"], r["step"]) for r in ledger
                    if r["kind"] == kind and r["status"].startswith("committed")}
             for kind in ("main", "smoke")}
    assert len(spent["main"]) == 12288 and len(spent["smoke"]) == 12
    fits = pd.read_csv(RESULTS / "FIT_LEDGER.csv")
    assert len(fits) == 24 and fits.main_updates.sum() == 12288
    assert fits.frozen_weights_buffers_unchanged.all()
    assert read_json(RESULTS / "MODEL_AND_SMOKE_AUDIT.json")["status"] == "PASS"
    resources = pd.read_csv(RESULTS / "RESOURCES.csv")
    expected_resources = {(s, m.rsplit("_seed", 1)[0], int(m.rsplit("_seed", 1)[1]), v, b)
                          for s, m in _expected() for v in ("ordered", "affine") for b in (1, 8)}
    assert len(resources) == 104
    assert set(resources[["source", "method", "seed", "variant", "batch_size"]].itertuples(index=False, name=None)) == expected_resources
    return tests, parameters, {"status": "PASS", "selection_seal_sha256": manifest["selection_seal_sha256"],
                             "prediction_manifest_sha256": sha(RESULTS / "PREDICTIONS_MANIFEST.json"),
                             "all_26_test_and_26_calibration_files_verified_before_scoring": True,
                             "test_scores_generated_after_selection_seal": True,
                             "main_updates": len(spent["main"]), "smoke_updates": len(spent["smoke"]),
                             "fit_count": len(fits), "selection_seed_excluded": True,
                             "resources_rows": len(resources), "resources_sha256": sha(RESULTS / "RESOURCES.csv"),
                             "source_provenance": source_verification}


def _scalar_check(y, q, sigma):
    """Independent scalar loops on a fixed small sample of actual predictions."""
    y, q, sigma = y[:3, :2], q[:3, :2], sigma[:2]
    losses, pces, coverages = [], [], []
    for s in range(y.shape[1]):
        total = 0.0
        count = 0
        hits = [0] * 9
        covered = 0
        for o in range(len(y)):
            for h in range(256):
                truth = float(y[o, s, h])
                count += 1
                covered += float(q[o, s, h, 0]) <= truth <= float(q[o, s, h, 8])
                for j in range(9):
                    tau = (j + 1) / 10
                    error = truth - float(q[o, s, h, j])
                    total += 2 * (tau * error if error >= 0 else (tau - 1) * error) / float(sigma[s])
                    hits[j] += truth <= float(q[o, s, h, j])
        losses.append(total / (count * 9))
        coverages.append(covered / count)
        pces.append(sum(abs(hits[j] / count - (j + 1) / 10) for j in range(9)) / 9)
    expected = {"scaled_pinball": float(np.mean(losses)), "pce": float(np.mean(pces)),
                "coverage80": float(np.mean(coverages))}
    frames = score_tables(y, q, sigma, "scalar_check", "scalar_check", 0, "input", range(len(y)), range(y.shape[1]))
    actual = frames["summary"].query("window == 'full_256'").iloc[0]
    errors = {k: abs(float(actual[k]) - value) for k, value in expected.items()}
    assert max(errors.values()) < 1e-10, errors
    return {"origins_checked": len(y), "series_checked": y.shape[1], "leads_checked": 256,
            "scalar_values": expected, "absolute_errors": errors}


def _bootstrap_seed(source, comparison, variant, window):
    return int(hashlib.sha256(f"rollout-v1|bootstrap|{source}|{comparison}|{variant}|{window}".encode()).hexdigest()[:8], 16)


def _effect_tables(origin):
    lookup = {(s, m, int(seed), v, w): group.sort_values("origin").set_index("origin").scaled_pinball
              for (s, m, seed, v, w), group in origin.groupby(["source", "method", "seed", "variant", "window"], sort=False)}
    seed_rows, effects = [], []
    for source in SOURCES:
        for variant in VARIANTS:
            for window in WINDOWS:
                jobs = [(label, baseline, candidate, variant, variant) for label, baseline, candidate in COMPARISONS]
                if variant == "affine":
                    jobs += [("affine_postprocessing", method, method, "ordered", "affine") for method in METHODS]
                for label, baseline, candidate, base_variant, cand_variant in jobs:
                    seeds = (0,) if baseline in FIXED and candidate in FIXED else SEEDS
                    pairs = []
                    metadata = {"source": source, "comparison": label, "baseline": baseline, "candidate": candidate,
                                "baseline_variant": base_variant, "candidate_variant": cand_variant,
                                "window": window, "metric": "scaled_pinball"}
                    boot_seed = _bootstrap_seed(source, label + baseline + candidate, variant, window)
                    for seed in seeds:
                        a = lookup[(source, baseline, 0 if baseline in FIXED else seed, base_variant, window)]
                        b = lookup[(source, candidate, 0 if candidate in FIXED else seed, cand_variant, window)]
                        assert a.index.equals(b.index)
                        assert (np.diff(a.index.to_numpy()) == 24).all(), "Bootstrap requires consecutive stride-24 origins."
                        av, bv = a.to_numpy(), b.to_numpy()
                        result = bootstrap_effect(av, bv, seed=boot_seed, reps=2000)
                        np.testing.assert_allclose(result["effect_a_minus_b"], av.mean() - bv.mean(), atol=1e-12, rtol=0)
                        seed_rows.append({**metadata, "seed": seed, **result})
                        pairs.append((av, bv))
                    av = np.mean([a for a, _ in pairs], axis=0)
                    bv = np.mean([b for _, b in pairs], axis=0)
                    result = bootstrap_effect(av, bv, seed=boot_seed, reps=2000)
                    effects.append({**metadata, "seed_summary": "fixed_model" if seeds == (0,) else "repeat_mean_92121_92122",
                                    "repeat_seed_count": 0 if seeds == (0,) else 2,
                                    "selection_seed_included": False, **result})
    return pd.DataFrame(seed_rows), pd.DataFrame(effects)


def _report(summary, effects, selections):
    full = effects.query("window == 'full_256' and baseline_variant == candidate_variant")
    rows = []
    for (source, method, variant, window), group in summary.groupby(["source", "method", "variant", "window"], sort=False):
        seeds = set(group.seed.astype(int))
        assert seeds == ({0} if method in FIXED else set(SEEDS))
        metric_cols = [c for c in ("scaled_pinball", "pce", "coverage80", "scaled_width80", "scaled_mae", "raw_mae", "scaled_rmse")]
        rows.append({"source": source, "method": method, "variant": variant, "window": window,
                     "seed_count": 1 if method in FIXED else 2,
                     **{c: float(group[c].mean()) for c in metric_cols}})
    means = pd.DataFrame(rows)
    means.to_csv(RESULTS / "SCORES_REPEAT_MEANS.csv", index=False)
    resources = pd.read_csv(RESULTS / "RESOURCES.csv")
    measured = resources.groupby(["source", "method", "variant", "batch_size"], as_index=False).agg(
        measured_wall_seconds=("median_wall_seconds", "mean"),
        peak_allocated_bytes=("peak_allocated_bytes", "max"),
        model_loading_seconds=("model_loading_seconds", "mean"))
    quality_cost = means.query("window == 'full_256' and variant != 'raw'").merge(
        measured, on=["source", "method", "variant"], validate="one_to_many")
    quality_cost.to_csv(RESULTS / "QUALITY_LATENCY.csv", index=False)
    text = ["# Rollout uncertainty PEFT 결과\n",
            "실행·채점 산출물입니다. 과학적 최종 범주는 검토 대기이며 자동 PASS 판정은 하지 않습니다. 선택 seed 92120은 모든 반복 평균에서 제외했습니다. 고정 F0/Chronos-2는 한 번 측정한 값이며 반복 seed로 복제해 평균하지 않았습니다.\n",
            "## 1. 현재 공식 branching\n",
            "현재 Chronos-Bolt 공식 장기 예측은 중앙값-only가 아니라 9개 분위수 경로와 9×9 축약을 사용합니다. 다음 표의 양수는 baseline−candidate 점수 개선입니다.\n"]
    questions = [
        ("official_branching_vs_median", "공식 branching과 F0 중앙값 경로"),
        ("general_rollout_training", "2. 일반 rollout LoRA 학습 효과"),
        ("generated_state_information", "3. 생성 상태 정보의 효과: S 대 R"),
        ("additional_width_information", "3. 예측 폭 정보의 추가 효과: U 대 S"),
        ("total_U_vs_R", "U 대 R 전체 차이"),
    ]
    columns = ["source", "baseline_variant", "effect_a_minus_b", "relative_gain_percent", "ci95_low", "ci95_high"]
    for label, title in questions:
        text += [f"### {title}\n", "```text\n" + full[full.comparison == label][columns].to_string(index=False, float_format=lambda x: f"{x:.6f}") + "\n```\n"]
    text += ["## 4. 단순 출력 보정\n", "각 모델에 동일한 CAL 전용 35점 grid를 적용했습니다. 보정값은 rollout에 되먹임하지 않았으며, conformal 보장을 뜻하지 않습니다. ordered 대비 affine 차이는 아래와 같습니다.\n"]
    affine = effects.query("window == 'full_256' and comparison == 'affine_postprocessing'")
    text += ["```text\n" + affine[["source", "candidate", "effect_a_minus_b", "ci95_low", "ci95_high"]].to_string(index=False, float_format=lambda x: f"{x:.6f}") + "\n```\n",
             "## 5. 공식 branching·MC16·직접 장기 예측 대비\n"]
    practical = full[full.comparison.isin(["U_vs_official_F0", "U_vs_official_R", "U_vs_MC16", "U_vs_direct_long_model"])]
    text += ["```text\n" + practical[["source", "baseline", "baseline_variant", "effect_a_minus_b", "relative_gain_percent", "ci95_low", "ci95_high"]].to_string(index=False, float_format=lambda x: f"{x:.6f}") + "\n```\n",
             "단일 경로/공식 branching/MC16의 조건부 context 계산량은 예제당 4/28/49회입니다. 시간 배수로 해석하지 않습니다. 아래는 실제 batch1/8 자원 수치입니다. 각 모델 3회 추론 중앙값을 구한 뒤, 학습 모델만 두 seed의 중앙값을 평균했습니다. peak는 seed 중 최대값입니다. loading은 추론 시간에 포함하지 않았습니다. MC16은 분위수 범위 밖 tail을 고정한 16개 유한 경로이며 완전한 joint sampling을 주장하지 않습니다. Chronos-2는 다른 크기·사전학습·구조의 직접 장기 모델입니다.\n",
             "```text\n" + quality_cost[["source", "method", "variant", "batch_size", "scaled_pinball", "measured_wall_seconds", "peak_allocated_bytes"]].to_string(index=False, float_format=lambda x: f"{x:.6f}") + "\n```\n",
             "## 6. 자료별·seed별 반례와 한계\n",
             "```text\n" + means.query("window == 'full_256' and variant != 'raw'").drop(columns=["window"]).to_string(index=False, float_format=lambda x: f"{x:.6f}") + "\n```\n",
             "probability score, PCE, coverage와 width는 서로 다른 지표입니다. coverage 상승만으로 개선을 선언하지 않습니다. full/tail_129_256의 point 지표는 SCORES_SUMMARY.csv에, 개별 seed 효과는 SEED_EFFECTS.csv에 보존했습니다.\n"]
    step_zero = [key for key, value in selections.items() if value["selected_step"] == 0]
    text += [f"선택 step0 모델 수: {len(step_zero)}/12. step0이면 추가 학습이 선택되지 않은 결과입니다. 해당 모델: {', '.join(step_zero) if step_zero else '없음'}.\n",
             "두 반복 seed는 고정되어 있으며 512 updates가 충분한 최적화였다고 주장하지 않습니다. CI는 7개 연속 원점 moving-block bootstrap 2,000회로, 고정 seed·선택 계열 아래 원점 불확실성만 반영합니다. horizon256과 stride24로 인접 target이 232시간 겹칩니다. 블록 길이7은 전체 256시간 의존을 완전히 제거한다는 보장이 없으며, 과거 탐색이나 optimizer 모집단 불확실성을 보정하지 않습니다.\n",
             "Electricity는 타임스탬프 없는 hourly slot이며 달력·UTC를 검증한 자료가 아닙니다. ETTh1은 온도 OT와 부하 6개입니다. 동일 24시간 시작 phase만 평가했습니다. 두 공개 자료는 개발·사전학습에 노출됐을 수 있으며 독립 확증이나 ETT 표준 12/4/4개월 재현이 아닙니다.\n",
             "## 7. 제한된 후속 투자 판단\n",
             "과학적 범주는 수치와 실측 비용을 검토해 FINAL_DECISION.md에 확정해야 합니다. 신규성·논문 PASS 또는 rollout PEFT 전체 반증은 선언하지 않습니다. 이번 예산 이후 자동 후속 실험은 0입니다.\n",
             "GitHub에는 코드·집계·검산·manifest를 남깁니다. 원자료, weights, 예측 cache는 로컬 보관이므로 저장소만으로 모든 수치를 재생할 수 있다고 주장하지 않습니다.\n"]
    (RESULTS / "REPORT_KO.md").write_text("\n".join(text), encoding="utf-8")
    decision = ["# 최종 판단 검토 초안\n", "상태: REVIEW_REQUIRED\n",
                "정해진 학습·선택·추론·채점은 완료됐습니다. 아래 관찰값으로 과학적 범주를 검토해야 하며 자동 METHOD_SIGNAL/PASS로 분류하지 않았습니다.\n"]
    for source in SOURCES:
        for variant in ("ordered", "affine"):
            selected = full[(full.source == source) & (full.baseline_variant == variant)
                            & full.comparison.isin(["additional_width_information", "total_U_vs_R"])]
            for _, row in selected.iterrows():
                decision.append(f"- {source} {variant}, {row['comparison']}: 개선 {row.effect_a_minus_b:.6f} ({row.relative_gain_percent:.3f}%), 원점 CI [{row.ci95_low:.6f}, {row.ci95_high:.6f}].")
    decision += ["\n선택용 seed 제외. 좋은 결과도 신규성/논문 PASS가 아니며, 음수 결과도 rollout PEFT 전체 반증이 아닙니다. 실측 자원과 seed별 반례를 함께 검토하십시오. 자동 후속 실험 없음.\n"]
    (RESULTS / "FINAL_DECISION.md").write_text("\n".join(decision), encoding="utf-8")


def evaluate_and_report():
    tests, parameters, verification = _guard()
    # Data and TEST target construction happen strictly after every gate above.
    from data import load
    audit = read_json(RESULTS / "DATA_AND_SPLIT_AUDIT.json")
    filenames = {"series_block": "SCORES_BY_SERIES_BLOCK.csv", "origin": "SCORES_BY_ORIGIN.csv",
                 "lead": "SCORES_BY_LEAD.csv", "summary": "SCORES_SUMMARY.csv"}
    written = set()
    summary_frames, origin_frames, scalar_checks = [], [], []
    begin = time.perf_counter()
    for source in SOURCES:
        d = load(source)
        expected = audit["sources"][source]
        assert d["ids"] == expected["ids"]
        np.testing.assert_array_equal(d["sigma"], [expected["train_sigma_population"][sid] for sid in d["ids"]])
        origins = d["origins"]["TEST"]
        assert (np.diff(origins) == 24).all()
        lo, hi = expected["split_edges"]["TEST"]["start"], expected["split_edges"]["TEST"]["end"]
        assert (origins >= lo).all() and (origins + 256 <= hi).all()
        y = np.stack([d["values"][o:o + 256].T for o in origins])
        for record in [r for r in tests if r["source"] == source]:
            method, seed_text = record["model"].rsplit("_seed", 1)
            seed = int(seed_text)
            raw = np.load(ROOT / record["path"], mmap_mode="r")
            assert list(raw.shape) == record["shape"] == [len(origins), len(d["ids"]), 256, 9]
            ordered = np.sort(raw, axis=-1)
            identity = apply_affine(ordered, d["sigma"], {"alpha": [1.0] * 4, "beta": [0.0] * 4})
            np.testing.assert_array_equal(identity, ordered)
            del identity
            for variant in VARIANTS:
                q = raw if variant == "raw" else (ordered if variant == "ordered" else apply_affine(ordered, d["sigma"], parameters[f"{source}/{record['model']}"]))
                tables = score_tables(y, q, d["sigma"], source, method, seed, variant, origins, d["ids"], raw_q=raw)
                scalar_checks.append({"source": source, "method": method, "seed": seed, "variant": variant,
                                      **_scalar_check(y, q, d["sigma"])})
                main = tables["summary"].query("window == 'full_256'").scaled_pinball.iloc[0]
                np.testing.assert_allclose(main, tables["origin"].query("window == 'full_256'").scaled_pinball.mean(), atol=1e-12, rtol=0)
                np.testing.assert_allclose(main, tables["lead"].scaled_pinball.mean(), atol=1e-12, rtol=0)
                for key, filename in filenames.items():
                    tables[key].to_csv(RESULTS / (filename + ".partial"), index=False, mode="a" if key in written else "w", header=key not in written)
                    written.add(key)
                summary_frames.append(tables["summary"])
                origin_frames.append(tables["origin"])
                del tables, q
            del raw, ordered
            event("model_test_scored", source=source, model=method, seed=seed)
        del y, d
    for filename in filenames.values():
        os.replace(RESULTS / (filename + ".partial"), RESULTS / filename)
    summary, origin = pd.concat(summary_frames, ignore_index=True), pd.concat(origin_frames, ignore_index=True)
    seed_effects, effects = _effect_tables(origin)
    seed_effects.to_csv(RESULTS / "SEED_EFFECTS.csv", index=False)
    effects.to_csv(RESULTS / "EFFECTS.csv", index=False)
    reread = pd.read_csv(RESULTS / "EFFECTS.csv")
    np.testing.assert_allclose(reread.a_mean - reread.b_mean, reread.effect_a_minus_b, atol=1e-12, rtol=0)
    _report(summary, effects, read_json(RESULTS / "MODEL_SELECTION.json"))
    verification.update(scalar_checks=scalar_checks, scalar_checks_count=len(scalar_checks),
                        affine_identity_verified_all_models=True, origin_lead_summary_pinball_agreement=True,
                        gain_recomputed_from_csv_baseline_and_candidate=True,
                        bootstrap_reps=2000, bootstrap_block_origins=7,
                        effect_rows=len(effects), seed_effect_rows=len(seed_effects),
                        fixed_baselines_not_duplicated_in_repeat_means=True,
                        scoring_seconds=time.perf_counter() - begin,
                        scored_variants=list(VARIANTS), scientific_decision="REVIEW_REQUIRED")
    artifacts = {}
    for name in [*filenames.values(), "SCORES_REPEAT_MEANS.csv", "SEED_EFFECTS.csv", "EFFECTS.csv", "QUALITY_LATENCY.csv"]:
        path = RESULTS / name
        original_bytes = path.stat().st_size
        if original_bytes >= 80_000_000:
            compressed = path.with_suffix(path.suffix + ".gz")
            with path.open("rb") as src, gzip.open(compressed, "wb") as dst:
                shutil.copyfileobj(src, dst)
            assert compressed.stat().st_size < 95_000_000
            path.unlink()
            path = compressed
        artifacts[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                           "bytes": path.stat().st_size, "uncompressed_bytes": original_bytes}
    verification["score_artifacts"] = artifacts
    save_json(RESULTS / "SCORE_ARTIFACTS_MANIFEST.json", artifacts)
    save_json(RESULTS / "VERIFICATION.json", verification)
    event("final_scoring_complete", variants=3, models=26, scientific_decision="REVIEW_REQUIRED")
    return verification


def synthetic_self_test():
    rng = np.random.default_rng(71)
    y = rng.normal(size=(7, 2, 256))
    q = np.sort(rng.normal(size=(*y.shape, 9)), axis=-1)
    check = _scalar_check(y, q, np.array([1.0, 2.0]))
    rows = []
    for source in SOURCES:
        for method in METHODS:
            for seed in ((0,) if method in FIXED else SEEDS):
                for variant in VARIANTS:
                    for window in WINDOWS:
                        for o in range(7):
                            rows.append({"source": source, "method": method, "seed": seed,
                                         "variant": variant, "window": window, "origin": o * 24,
                                         "scaled_pinball": 1 + o / 100 + METHODS.index(method) / 10})
    seed_rows, effects = _effect_tables(pd.DataFrame(rows))
    assert not (seed_rows.seed == 92120).any()
    assert set(effects.query("comparison == 'official_branching_vs_median'").repeat_seed_count) == {0}
    np.testing.assert_allclose(effects.a_mean - effects.b_mean, effects.effect_a_minus_b, atol=1e-12, rtol=0)
    print(json.dumps({"synthetic_only": True, "scalar_check": check,
                      "seed_effect_rows": len(seed_rows), "effect_rows": len(effects)}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-self-test", action="store_true", required=True)
    parser.parse_args()
    synthetic_self_test()
