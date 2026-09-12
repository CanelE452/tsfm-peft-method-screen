# Candidate 05: Maturity-PEFT

[문제]
[확인] Problem gate: ERROR. {}

[방법]
[확인] The recipe is documented in [CANDIDATE_05](../../docs/CANDIDATE_05.md). Proposed: MATURITY_PEFT.

[강한 단순 baseline]
[확인] Strongest observed simple baseline: not established (comparison incomplete / not executed). All prespecified baseline arms remain in the raw table.

[데이터]
[확인] Dataset: jena. Train/V/E manifests and input hashes are in screening_summary. E opened only after a saved selection seal, when executed.

[학습 파라미터]
[확인] Fits: 0; streams: 4. Contract: contract.json. Standard attention LoRA rank8/alpha16,96 projections; frozen native head. Candidate-specific additions are recorded in integrity.json.

[무결성]
[확인] {"completed_stream_checks": [{"arm": "F0", "checkpoint_sha256": "b9fa1a03d17400c7de52c991066347a277e420bb0aa7dca6fcb2c8857dd166d4", "issued_cache_matches_e_cache": true, "issued_count": 30, "issued_files_have_no_targets": true, "metric_replay_abs": 0.0}, {"arm": "IMMEDIATE_LORA", "checkpoint_sha256": "952c4f2d31ffa85f314cc3e3abf8c38f48646b2c890206dcba83233e6b130153", "issued_cache_matches_e_cache": true, "issued_count": 30, "issued_files_have_no_targets": true, "metric_replay_abs": 1.1102230246251565e-16}, {"arm": "WAIT_FULL", "checkpoint_sha256": "015c891948b525422a2459d2c3472e64f45d34bd292caafc351b339b5289398a", "issued_cache_matches_e_cache": true, "issued_count": 30, "issued_files_have_no_targets": true, "metric_replay_abs": 1.1102230246251565e-16}], "failed_gate": "finite gradient in TAFAS input GCM with partial targets", "maturity_not_run": true, "metric_replay_max_abs": 1.1102230246251565e-16, "scope": "Reconstruction of preserved receipts only; no new GPU inference, optimization, or E opening", "selection_seal_before_e": true, "status": "FAIL"}

[raw 결과]
| arm | variant | scaled_2pinball | median_mae | qmean_mse | interval80_coverage | interval80_width | worst5_origin_loss | receipt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | issued_completed_stream | 0.5838571069542349 | 2.9852467222760124 | 16.890077483965634 | 0.7595486111111112 | 8.4159925305595 | 0.98552917600794 | replayed from saved predictions after candidate abort |
| IMMEDIATE_LORA | issued_completed_stream | 0.6467702002324903 | 3.16504649395744 | 20.37034018968799 | 0.6546875 | 7.567794226606686 | 1.2403347962004347 | replayed from saved predictions after candidate abort |
| WAIT_FULL | issued_completed_stream | 0.6624827902964064 | 3.276563878564371 | 21.41763480550465 | 0.6810763888888889 | 8.172622834394376 | 1.2508906257406278 | replayed from saved predictions after candidate abort |


[relative 결과]
[확인] Gain vs strongest simple baseline (% F0): not evaluated.
[확인] Diagnostics: {"completed_streams": 3, "failed_streams": 1, "proposed_stream_executed": false}

[성공/실패 판정]
[판정] IMPLEMENTATION_BLOCKED. TAFAS-like partial-label gradient integrity failed; proposed stream not reached; no valid complete comparison

[말할 수 없는 것]
[미검증] A one-seed development screen is not paper-level evidence, cross-dataset robustness or novelty certification. Some evaluation horizons overlap; no independent-sample significance claim is made. TSFM pretraining overlap is not excluded. See the candidate-specific scope in its protocol document. No unmeasured primary benefit is inferred from diagnostics.

[수치 버그 수정]
[확인] 실행 종료 후 NaN label을 정규화 전에 처리하도록 수정했습니다. 회귀 테스트와 train-only GPU gradient smoke를 통과했으나 stream은 재실행하지 않았습니다. 원래 실행 코드와 결과는 보존했습니다. [상세](../../docs/POST_SCREEN_REPAIR.md).

[Round2 추천 여부]
[판정] Not recommended from this screen. Round2 not executed.
