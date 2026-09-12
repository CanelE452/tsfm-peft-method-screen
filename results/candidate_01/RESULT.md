# Candidate 01: Freshness-Gated LoRA

[문제]
[확인] Problem gate: PASS. {"degradation_percent": {"block12": 26.795527039354294, "block6": 15.746194455578564, "refresh2": 4.560332413440988, "refresh4": 18.265769500042495, "stale": 95.36999202579737}, "loss": {"block12": 0.6473959855884119, "block6": 0.5909800084226919, "clean": 0.5105826685727453, "refresh2": 0.5338669355050801, "refresh4": 0.6038445219214088, "stale": 0.9975253188756759}, "mean_degradation_percent": 32.14756308684274, "status": "PASS"}

[방법]
[확인] The recipe is documented in [CANDIDATE_01](../../docs/CANDIDATE_01.md). Proposed: FRESHNESS_GATED_LORA.

[강한 단순 baseline]
[확인] Strongest observed simple baseline: FEATURE_LORA. All prespecified baseline arms remain in the raw table.

[데이터]
[확인] Dataset: jena. Train/V/E manifests and input hashes are in screening_summary. E opened only after a saved selection seal, when executed.

[학습 파라미터]
[확인] Fits: 6; streams: 0. Contract: contract.json. Standard attention LoRA rank8/alpha16,96 projections; frozen native head. Candidate-specific additions are recorded in integrity.json.

[무결성]
[확인] {"fits": [{"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "STANDARD_LORA_3e-05", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "STANDARD_LORA_0.0001", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1179648, "trainable_tensors": 192}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "FEATURE_LORA_3e-05", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1182720, "trainable_tensors": 384}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "FEATURE_LORA_0.0001", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1182720, "trainable_tensors": 384}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "FRESHNESS_GATED_LORA_3e-05", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1182720, "trainable_tensors": 384}, {"adapter_trainable_count": 0, "base_frozen": true, "checkpoint_replay_max_abs": 0.0, "fit": "FRESHNESS_GATED_LORA_0.0001", "frozen_unchanged": true, "head_frozen": true, "identity_max_abs": 0.0, "modules": 96, "trainable_count": 1182720, "trainable_tensors": 384}], "metric_replay_max_abs": 2.220446049250313e-16, "no_future_context": true, "saved_predictions": 20, "selection_seal_before_e": true, "status": "PASS"}

[raw 결과]
| arm | variant | scaled_2pinball | median_mae | qmean_mse | interval80_coverage | interval80_width |
| --- | --- | --- | --- | --- | --- | --- |
| F0 | block12 | 0.7405853897738189 | 3.905651258221931 | 25.8045380645096 | 0.7538194444444444 | 10.648772814042037 |
| F0 | block24 | 0.7959062243892927 | 4.141094773179955 | 29.5614064382358 | 0.7354166666666666 | 10.891163245671326 |
| F0 | refresh4 | 0.6873979981394296 | 3.5010070687366857 | 23.040002979639226 | 0.6694444444444445 | 8.425791066512465 |
| F0 | refresh8 | 0.7792485489963276 | 4.030504029740889 | 28.900877613443168 | 0.6845486111111111 | 9.797122941828436 |
| F0 | clean | 0.5838571069542349 | 2.9852467222760124 | 16.890077483965634 | 0.7595486111111112 | 8.4159925305595 |
| STANDARD_LORA | block12 | 0.7141776602511992 | 3.7578649647533897 | 24.596482141596134 | 0.7465277777777778 | 10.204298860952258 |
| STANDARD_LORA | block24 | 0.7563455642365099 | 3.9249823978791634 | 27.625296505857417 | 0.7335069444444445 | 10.417625018830101 |
| STANDARD_LORA | refresh4 | 0.6660890081724224 | 3.3823754272527164 | 21.66936815807966 | 0.6855902777777778 | 8.232488427021437 |
| STANDARD_LORA | refresh8 | 0.7814277722650866 | 4.069042103033927 | 28.876441345554323 | 0.7000000000000001 | 9.946180483367709 |
| STANDARD_LORA | clean | 0.5639840463736286 | 2.8744020020796195 | 15.798816863551366 | 0.7602430555555555 | 8.06425970296065 |
| FEATURE_LORA | block12 | 0.7127596731724416 | 3.753985986775822 | 24.60489135338187 | 0.7741319444444444 | 10.634448424561155 |
| FEATURE_LORA | block24 | 0.7525426775764553 | 3.9101926138417586 | 27.47396247341112 | 0.7598958333333333 | 10.819295350793334 |
| FEATURE_LORA | refresh4 | 0.6608378156179837 | 3.3886226107676825 | 21.49453705390065 | 0.7005208333333333 | 8.597901831898424 |
| FEATURE_LORA | refresh8 | 0.7699734855031657 | 4.014002874038286 | 28.267566861726717 | 0.7133680555555555 | 10.142570281566845 |
| FEATURE_LORA | clean | 0.5581533248334587 | 2.8569717460622392 | 15.53264329678046 | 0.778125 | 8.391140840161178 |
| FRESHNESS_GATED_LORA | block12 | 0.7141101430496384 | 3.7574069202360176 | 24.592731008018546 | 0.7467013888888888 | 10.199856493373712 |
| FRESHNESS_GATED_LORA | block24 | 0.7562345921675853 | 3.924256317690015 | 27.61926542035961 | 0.7336805555555556 | 10.413494487727682 |
| FRESHNESS_GATED_LORA | refresh4 | 0.6662507027605632 | 3.3825916985670723 | 21.676430245329406 | 0.6848958333333333 | 8.228553972848587 |
| FRESHNESS_GATED_LORA | refresh8 | 0.7815769628328701 | 4.069694964711864 | 28.885489562648388 | 0.6998263888888889 | 9.943867390561437 |
| FRESHNESS_GATED_LORA | clean | 0.5639261656993995 | 2.8739947272257673 | 15.795389310344925 | 0.7600694444444444 | 8.060971535162793 |


[relative 결과]
[확인] Gain vs strongest simple baseline (% F0): -0.7345232805096731.
[확인] Diagnostics: {"clean_degradation_percent": 0.9887420735624041, "corruption_worsening_percent": {"block12": 0.18235167690915768, "block24": 0.46386301275163244, "refresh4": 0.7874458693843255, "refresh8": 1.489059857044295}}

[성공/실패 판정]
[판정] FAIL. Feature baseline stronger; clean degradation 0.989% exceeds 0.5%.

[말할 수 없는 것]
[미검증] A one-seed development screen is not paper-level evidence, cross-dataset robustness or novelty certification. Some evaluation horizons overlap; no independent-sample significance claim is made. TSFM pretraining overlap is not excluded. See the candidate-specific scope in its protocol document. No unmeasured primary benefit is inferred from diagnostics.

[Round2 추천 여부]
[판정] Not recommended from this screen. Round2 not executed.
