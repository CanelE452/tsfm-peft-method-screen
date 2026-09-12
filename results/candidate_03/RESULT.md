# Candidate 03: PatchPhase-PEFT

[문제]
[확인] Problem gate: PASS. {"construct": "Exact values, original time encodings, masks, context information, forecast origin and future tokens preserved; only masked padding changes patch membership", "loss": {"0": 0.3075066796881674, "12": 0.3369212596426045, "4": 0.35585428064558017, "8": 0.34975042040233084}, "loss_range_percent": 15.722455527288218, "normalized_discrepancy_percent": 25.716680321262835, "origins": 128, "status": "PASS"}

[방법]
[확인] The recipe is documented in [CANDIDATE_03](../../docs/CANDIDATE_03.md). Proposed: PHASE_CONDITIONED_ADAPTER.

[강한 단순 baseline]
[확인] Strongest observed simple baseline: PHASE_AUGMENTED_ADAPTER. All prespecified baseline arms remain in the raw table.

[데이터]
[확인] Dataset: ettm2. Train/V/E manifests and input hashes are in screening_summary. E opened only after a saved selection seal, when executed.

[학습 파라미터]
[확인] Fits: 6; streams: 0. Contract: contract.json. Standard attention LoRA rank8/alpha16,96 projections; frozen native head. Candidate-specific additions are recorded in integrity.json.

[무결성]
[확인] {"fits": [{"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "STANDARD_LORA_3e-05", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "STANDARD_LORA_0.0001", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 12288, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "PHASE_AUGMENTED_ADAPTER_3e-05", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 12288, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "PHASE_AUGMENTED_ADAPTER_0.0001", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 12312, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "PHASE_CONDITIONED_ADAPTER_3e-05", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 12312, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "PHASE_CONDITIONED_ADAPTER_0.0001", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}], "metric_replay_max_abs": 5.551115123125783e-17, "no_future_context": true, "saved_predictions": 12, "selection_seal_before_e": true, "status": "PASS"}

[raw 결과]
| arm | variant | scaled_2pinball | median_mae | qmean_mse | interval80_coverage | interval80_width |
| --- | --- | --- | --- | --- | --- | --- |
| F0 | 0 | 0.316376418669498 | 1.0904373169141186 | 3.128742685442962 | 0.769593253968254 | 3.247390838485906 |
| F0 | 4 | 0.4143365101308172 | 1.6361035785964868 | 6.411902164214483 | 0.7006448412698412 | 3.8755078169846136 |
| F0 | 12 | 0.33970282637839355 | 1.3381759868361913 | 4.343552824292911 | 0.7388392857142857 | 3.5110191592959583 |
| STANDARD_LORA | 0 | 0.3121450463831862 | 1.144005156792922 | 3.160091266997611 | 0.6989087301587301 | 2.9406839216881915 |
| STANDARD_LORA | 4 | 0.3510466860306255 | 1.3914428118557212 | 4.7526891594374 | 0.7414434523809524 | 3.6952780842973247 |
| STANDARD_LORA | 12 | 0.31786414696961696 | 1.2203048873075564 | 3.6172604108774338 | 0.7292906746031746 | 3.2558013380021005 |
| PHASE_AUGMENTED_ADAPTER | 0 | 0.312588620585931 | 1.1459566775103291 | 3.1654461205059254 | 0.6882440476190476 | 2.9444955236404127 |
| PHASE_AUGMENTED_ADAPTER | 4 | 0.35104837614049417 | 1.392267490338002 | 4.7676893009374215 | 0.736111111111111 | 3.687388608731063 |
| PHASE_AUGMENTED_ADAPTER | 12 | 0.31777551425783795 | 1.2203486648310597 | 3.616400420245674 | 0.722718253968254 | 3.253309361172825 |
| PHASE_CONDITIONED_ADAPTER | 0 | 0.31267840834583444 | 1.1460504715528619 | 3.165172838826742 | 0.6882440476190476 | 2.948452231248781 |
| PHASE_CONDITIONED_ADAPTER | 4 | 0.35099725935600884 | 1.3914689434108338 | 4.759022897582142 | 0.7368551587301588 | 3.6977549041159423 |
| PHASE_CONDITIONED_ADAPTER | 12 | 0.31793006220178766 | 1.220433000556474 | 3.613828840538036 | 0.7232142857142857 | 3.2633970955013494 |


[relative 결과]
[확인] Gain vs strongest simple baseline (% F0): -0.013716944787409558.
[확인] Diagnostics: {"baseline_phase_variance": 0.0474450232453152, "canonical_degradation_percent": 0.1685846135092086, "proposed_phase_variance": 0.04731412807978717, "variance_reduction_percent": 0.27588808388024016}

[성공/실패 판정]
[판정] FAIL. Phase augmentation baseline stronger; variance reduction 0.276% is below 30%.

[말할 수 없는 것]
[미검증] A one-seed development screen is not paper-level evidence, cross-dataset robustness or novelty certification. Some evaluation horizons overlap; no independent-sample significance claim is made. TSFM pretraining overlap is not excluded. See the candidate-specific scope in its protocol document. No unmeasured primary benefit is inferred from diagnostics.

[Round2 추천 여부]
[판정] Not recommended from this screen. Round2 not executed.
