# Candidate 02: DualClock-PEFT

[문제]
[확인] Problem gate: PASS. {}

[방법]
[확인] The recipe is documented in [CANDIDATE_02](../../docs/CANDIDATE_02.md). Proposed: DUALCLOCK_ADAPTER.

[강한 단순 baseline]
[확인] Strongest observed simple baseline: EVENT_SUMMARY_LORA. All prespecified baseline arms remain in the raw table.

[데이터]
[확인] Dataset: m5. Train/V/E manifests and input hashes are in screening_summary. E opened only after a saved selection seal, when executed.

[학습 파라미터]
[확인] Fits: 6; streams: 0. Contract: contract.json. Standard attention LoRA rank8/alpha16,96 projections; frozen native head. Candidate-specific additions are recorded in integrity.json.

[무결성]
[확인] {"fits": [{"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "STANDARD_LORA_3e-05", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "STANDARD_LORA_0.0001", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 12704, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "EVENT_SUMMARY_LORA_3e-05", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 12704, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "EVENT_SUMMARY_LORA_0.0001", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 16192, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "DUALCLOCK_ADAPTER_3e-05", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 16192, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "DUALCLOCK_ADAPTER_0.0001", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}], "metric_replay_max_abs": 5.551115123125783e-17, "no_future_context": true, "saved_predictions": 4, "selection_seal_before_e": true, "status": "PASS"}

[raw 결과]
| arm | variant | scaled_2pinball | median_mae | qmean_mse | interval80_coverage | interval80_width | rmsse | occurrence_brier | positive_demand_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | clean | 0.490207189618205 | 0.726555893691021 | 2.5192661077855787 | 0.7891845703125 | 2.2835986773764407 | 1.0580433443874329 | 0.17807775735855103 | 1.6833133697509766 |
| STANDARD_LORA | clean | 0.4842323298460305 | 0.7218637614454728 | 2.361271209691811 | 0.8491346571180556 | 2.158018675503524 | 1.0647921681350137 | 0.17676551640033722 | 1.7135552167892456 |
| EVENT_SUMMARY_LORA | clean | 0.4841542334368005 | 0.7219102167772942 | 2.361672864628515 | 0.8437364366319444 | 2.153335253797688 | 1.065341493616892 | 0.1767677515745163 | 1.7152010202407837 |
| DUALCLOCK_ADAPTER | clean | 0.4840854640646646 | 0.7218579620463567 | 2.359960939343229 | 0.8449978298611112 | 2.152562560911566 | 1.0649298457890966 | 0.17672273516654968 | 1.7141847610473633 |


[relative 결과]
[확인] Gain vs strongest simple baseline (% F0): 0.014028633931187822.
[확인] Diagnostics: {"gain_excluding_best10percent": -8.163931162569495e-06, "improved_series_fraction": 0.63671875, "median_series_effect": 5.476929583449197e-05, "zero_heavy_effect": 0.00015514523563879036}

[성공/실패 판정]
[판정] WEAK. Gain 0.0140% is below 1%; excluding the top10% improved series removes the gain.

[말할 수 없는 것]
[미검증] A one-seed development screen is not paper-level evidence, cross-dataset robustness or novelty certification. Some evaluation horizons overlap; no independent-sample significance claim is made. TSFM pretraining overlap is not excluded. See the candidate-specific scope in its protocol document. No unmeasured primary benefit is inferred from diagnostics.

[Round2 추천 여부]
[판정] Not recommended from this screen. Round2 not executed.
