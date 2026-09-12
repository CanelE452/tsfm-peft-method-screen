# Candidate 04: FR-LoRA

[문제]
[확인] Problem gate: PASS. {}

[방법]
[확인] The recipe is documented in [CANDIDATE_04](../../docs/CANDIDATE_04.md). Proposed: FR_LORA.

[강한 단순 baseline]
[확인] Strongest observed simple baseline: STANDARD_LORA. All prespecified baseline arms remain in the raw table.

[데이터]
[확인] Dataset: jena. Train/V/E manifests and input hashes are in screening_summary. E opened only after a saved selection seal, when executed.

[학습 파라미터]
[확인] Fits: 8; streams: 0. Contract: contract.json. Standard attention LoRA rank8/alpha16,96 projections; frozen native head. Candidate-specific additions are recorded in integrity.json.

[무결성]
[확인] {"fits": [{"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "STANDARD_LORA_3e-05", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "STANDARD_LORA_0.0001", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "RAW_STABILITY_LORA_3e-05", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "RAW_STABILITY_LORA_0.0001", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "F0_ANCHOR_LORA_3e-05", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "F0_ANCHOR_LORA_0.0001", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "FR_LORA_3e-05", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "FR_LORA_0.0001", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}], "metric_replay_max_abs": 0.0, "no_future_context": true, "saved_predictions": 5, "selection_seal_before_e": true, "status": "PASS"}

[raw 결과]
| arm | variant | scaled_2pinball | median_mae | qmean_mse | interval80_coverage | interval80_width |
| --- | --- | --- | --- | --- | --- | --- |
| F0 | clean | 0.5838571069542349 | 2.9852467222760124 | 16.890077483965634 | 0.7595486111111112 | 8.4159925305595 |
| STANDARD_LORA | clean | 0.5838571069542349 | 2.9852467222760124 | 16.890077483965634 | 0.7595486111111112 | 8.4159925305595 |
| RAW_STABILITY_LORA | clean | 0.5838571069542349 | 2.9852467222760124 | 16.890077483965634 | 0.7595486111111112 | 8.4159925305595 |
| F0_ANCHOR_LORA | clean | 0.5838571069542349 | 2.9852467222760124 | 16.890077483965634 | 0.7595486111111112 | 8.4159925305595 |
| FR_LORA | clean | 0.5838571069542349 | 2.9852467222760124 | 16.890077483965634 | 0.7595486111111112 | 8.4159925305595 |


[relative 결과]
[확인] Gain vs strongest simple baseline (% F0): 0.0.
[확인] Diagnostics: {"F0": {"correction_revision": 0.0, "raw_revision": 0.6537600978355921}, "F0_ANCHOR_LORA": {"correction_revision": 0.0, "raw_revision": 0.6537600978355921}, "FR_LORA": {"correction_revision": 0.0, "raw_revision": 0.6537600978355921}, "RAW_STABILITY_LORA": {"correction_revision": 0.0, "raw_revision": 0.6537600978355921}, "STANDARD_LORA": {"correction_revision": 0.0, "raw_revision": 0.6537600978355921}}

[성공/실패 판정]
[판정] FAIL. Every V-selected checkpoint is step0; no forecast improvement over any baseline or F0.

[말할 수 없는 것]
[미검증] A one-seed development screen is not paper-level evidence, cross-dataset robustness or novelty certification. Some evaluation horizons overlap; no independent-sample significance claim is made. TSFM pretraining overlap is not excluded. See the candidate-specific scope in its protocol document. No unmeasured primary benefit is inferred from diagnostics.

[Round2 추천 여부]
[판정] Not recommended from this screen. Round2 not executed.
