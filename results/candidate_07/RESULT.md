# Candidate 07: Censor-Preserve LoRA

[문제]
[확인] Problem gate: PASS. {}

[방법]
[확인] The recipe is documented in [CANDIDATE_07](../../docs/CANDIDATE_07.md). Proposed: CENSOR_PRESERVE_LORA.

[강한 단순 baseline]
[확인] Strongest observed simple baseline: CENSORED_LOSS_LORA. All prespecified baseline arms remain in the raw table.

[데이터]
[확인] Dataset: m5. Train/V/E manifests and input hashes are in screening_summary. E opened only after a saved selection seal, when executed.

[학습 파라미터]
[확인] Fits: 8; streams: 0. Contract: contract.json. Standard attention LoRA rank8/alpha16,96 projections; frozen native head. Candidate-specific additions are recorded in integrity.json.

[무결성]
[확인] {"fits": [{"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "NAIVE_LORA_3e-05", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "NAIVE_LORA_0.0001", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "DROP_CENSORED_LORA_3e-05", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "DROP_CENSORED_LORA_0.0001", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "CENSORED_LOSS_LORA_3e-05", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "CENSORED_LOSS_LORA_0.0001", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "CENSOR_PRESERVE_LORA_3e-05", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "CENSOR_PRESERVE_LORA_0.0001", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}], "metric_replay_max_abs": 1.1102230246251565e-16, "no_future_context": true, "saved_predictions": 5, "selection_seal_before_e": true, "status": "PASS"}

[raw 결과]
| arm | variant | scaled_2pinball | median_mae | qmean_mse | interval80_coverage | interval80_width | rmsse | occurrence_brier | positive_demand_mae | censored_pinball | uncensored_pinball | censored_underprediction_bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | clean | 0.524918064010481 | 0.74157100937147 | 2.688611954360486 | 0.7801378038194444 | 1.4537481277786422 | 1.0719469598056663 | 0.17895342409610748 | 1.7210768461227417 | 3.279112615638583 | 0.30595381296755797 | 3.5888850688934326 |
| NAIVE_LORA | clean | 0.5210425642912357 | 0.7345395644339847 | 2.7031694555536143 | 0.7930636935763888 | 1.4735157484714199 | 1.0773069213224788 | 0.1773812472820282 | 1.746117115020752 | 3.293614370640089 | 0.2977932861599085 | 3.665320873260498 |
| DROP_CENSORED_LORA | clean | 0.5246834037366431 | 0.7390635957748857 | 2.703319829389714 | 0.7867024739583333 | 1.4245735996806637 | 1.079884942587654 | 0.17951858043670654 | 1.7513865232467651 | 3.31870601142159 | 0.30206612657925225 | 3.6499316692352295 |
| CENSORED_LOSS_LORA | clean | 0.52062265836078 | 0.7425889312292222 | 2.7004582558844783 | 0.8390570746527778 | 1.5845233097555738 | 1.0993704068367347 | 0.18219834566116333 | 1.8341747522354126 | 3.244757067937454 | 0.30163171007916395 | 3.8031907081604004 |
| CENSOR_PRESERVE_LORA | clean | 0.5207914630974182 | 0.7425393457771362 | 2.7007687454377507 | 0.8380398220486112 | 1.5779147165542597 | 1.0992742434762726 | 0.18210692703723907 | 1.8338583707809448 | 3.2583590628466186 | 0.30133555654157546 | 3.802339553833008 |


[relative 결과]
[확인] Gain vs strongest simple baseline (% F0): -0.03215830206879511.
[확인] Diagnostics: {"censored_bias_baseline": 3.8031907081604004, "censored_bias_proposed": 3.802339553833008, "censored_fraction": 0.0991482204861111, "censored_pinball_gain_percent": -0.4148071903445583, "uncensored_degradation_percent": -0.09679681214494222}

[성공/실패 판정]
[판정] FAIL. Censor-only baseline stronger on primary and censored-position loss (censored gain -0.415%). Tiny bias improvement is insufficient.

[말할 수 없는 것]
[미검증] A one-seed development screen is not paper-level evidence, cross-dataset robustness or novelty certification. Some evaluation horizons overlap; no independent-sample significance claim is made. TSFM pretraining overlap is not excluded. See the candidate-specific scope in its protocol document. No unmeasured primary benefit is inferred from diagnostics.

[Round2 추천 여부]
[판정] Not recommended from this screen. Round2 not executed.
