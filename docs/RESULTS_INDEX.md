# 완료 작업과 검증 기록

2026-09-18 기준. 표의 링크는 완료된 실행 또는 검토 결과다. 과거 FAIL/STOP은 당시 고정 실험의 판정으로 보존한다. 최신 재검토의 후속 연구 우선순위는 방법론 성공이나 새로운 실험 실행을 뜻하지 않는다.

[통합 완료] 한국어 원고 v3에 기존 C3/MAG 대조와 최신 학습 요인·내부 기전·선행 검토를 본문으로 통합했다. 16쪽·표9개·그림8개·참고문헌12개, 기존 표1–5 및 음성 결과 보존, 새 학습·추론·bootstrap0회다. [PDF·DOCX·원문](../papers/persistence_adaptation/manuscript_v3_integrated_20260919/README.md), [검산](../papers/persistence_adaptation/manuscript_v3_integrated_20260919/AUDIT.json), [남은 제출 작업](../papers/persistence_adaptation/manuscript_v3_integrated_20260919/INTEGRATION_NOTES_KO.md). 원고 통합 완료를 신규성·독립 검증·게재 완료로 부르지 않는다.

[검토] 2026-09-19 논문 성립 가능성을 최신 근거와 공식 선행에 비춰 검토했다. 공개 근거143개 hash와 핵심 수치를 재검산했으며 새학습·추론0회다. PEFT 통제 실증 원고의 근거는 있지만 C3 새 방법 우위·충분한 신규성·제출 준비 완료는 확정하지 않았다. 초기화/기반 모델 전이 선행과의 겹침, 개발자료 재사용, 본문 미통합을 구분했다. [한국어 검토](../research/paper_viability_review_20260919/REVIEW_KO.md), [검산](../research/paper_viability_review_20260919/AUDIT.json).

[확인] C3 내부 기전 진단은 추가학습0회·1,196 autograd·160 checkpoint probe·192 E view(신규96/재사용96)를 완료했다. 두 B0의 입력 patch 표현은 같지만 초기 loss 신호와 Jacobian이 달랐으며, 전력 초기 gradient 분해에서 출력 신호 항 norm은 Jacobian 항의2.94배였다. 전력 전이 SHIFT8에서 다른 B0로 교환하면 C3 nMAE9.7476%·MAG10.1990% 악화해 두 방식의 B0 조합 의존성을 확인했다. pulse 및 일부 step의 반대 효과와 완전한 인과 매개 분석의 한계도 보존한다. [보고서](../results/c3_internal_mechanism_20260918/REPORT.md), [전체 조건](../results/c3_internal_mechanism_20260918/ALL_CONDITIONS.md), [독립 검산](../results/c3_internal_mechanism_20260918/INDEPENDENT_AUDIT.json), [논문 보충 PDF/DOCX](../papers/persistence_adaptation/internal_mechanism_v1_20260918/README.md). 새 방법·후속학습은 시작하지 않았다.

[확인] C3/MAG 학습 요인 분리는 32경로(신규24·재사용8), 24,576 본학습+12smoke updates, 192평가 view를 완료했다. 고정1,024 updates의 전력 전이 SHIFT8에서는 B0×초기값 상호작용(공통 분모−0.2872%)이 가장 큰 절대 점추정이고, V 선택에서는 B0 주효과(+0.4648%)가 가장 컸다. 두 수준·재사용 개발 E의 통제 실증으로 범위를 제한한다. [보고서](../results/c3_training_factorial_20260918/REPORT.md), [최종 판단](../results/c3_training_factorial_20260918/FINAL_DECISION.md), [독립 검산](../results/c3_training_factorial_20260918/INDEPENDENT_AUDIT.json), [한국어 논문 보충 PDF/DOCX](../papers/persistence_adaptation/training_factorial_v1_20260918/README.md). 추가 학습은 자동 시작하지 않았다.

[확인] C3/MAG 최대 영향 감사는 0학습·0추론으로 60조건을 검산했다. 전력 전이 SHIFT8 평균 MAG 이득0.2509%는 가장 큰 기여 seed를 제외하는 진단에서0.0557%로 줄었다. 같은 seed의 초기 조건은 일치하지만 seed 간 B0·초기값·순서가 묶여 최대 인과 요인은 미식별이다. [보고서](../research/c3_influence_audit_20260918/REPORT.md), [검산](../research/c3_influence_audit_20260918/AUDIT.json). 이 결과와 기존 대조를 반영한 [한국어 원고·PDF·DOCX](../papers/persistence_adaptation/manuscript_v2_20260918/README.md), [원고 검산](../papers/persistence_adaptation/manuscript_v2_20260918/AUDIT.json)을 작성했다. 원고 완성은 논문 PASS나 투고 완료가 아니다.



[확인] 저장 예측의 단순보정 대조를 신규학습0회로 완료했다. 같은 V 보정 뒤 VINTAGE는 STD 대비 최신2.326%·과거5.870% 개선했으나, 최신 월구간은0을 포함한다. 중심화 median 오차의 차이도 남았다. 기존 FR/PACE와의 겹침을 확인해 단순 consistency를 새후보로 재명명하지 않았다. [한국어 보고서](../results/covariate_calibration_control_20260916/REPORT.md), [해석·신규성 경계](../results/covariate_calibration_control_20260916/INTERPRETATION.md), [검산](../results/covariate_calibration_control_20260916/verification.json).

[확인] 실제 예보 LoRA 대조6/6fits·720updates와 평가를 완료했다. 과거예보 학습 VINTAGE는 최신입력에서 INIT0 대비4.590% 개선(두seed양성)했으나95%월구간은0을 포함하고, 과거입력에서는0.609% 악화했다. 알려진 학습대조의 양성 개발신호이며 새PEFT PASS는 아니다. [한국어 보고서](../results/covariate_lora_controls_20260916/REPORT.md), [구성요소·연구 판단](../results/covariate_lora_controls_20260916/INTERPRETATION.md), [검산](../results/covariate_lora_controls_20260916/verification.json).

[확인] 예보 신뢰도 사후 진단은 0fits·0추론으로 저장112개 경로를 검산했다. 예보 불일치가 큰 D13원점에서도 최신기상 평균손실0.148347이 F0 0.169632보다 낮아, 불일치만으로 기상을 차단하는 gate를 정당화하지 못했다. 오래된 예보의 손해와 기상오차/부하손해의 불일치를 구분했다. [한국어 보고서](../research/covariate_reliability_diagnostic_20260916/REPORT.md), [가까운 선행](../research/covariate_reliability_diagnostic_20260916/LITERATURE_BOUNDARY.md).

[확인] 예보 vintage 참조 비교는 0 neural fits·233개 예측·128호출을 완료했다. 최신 기상 입력이 F0보다17.743% 좋았고 원본 혼합은 최신경로보다1.113% 좋았으나, 사전 주 비교인 보정혼합은 최신경로보다0.214% 악화했다. teacher 신호 조건은 미충족이며 학생 학습은 실행하지 않았다. [한국어 보고서](../results/covariate_vintage_reference_20260916/REPORT.md), [양성 신호와 한계](../results/covariate_vintage_reference_20260916/INTERPRETATION.md).

[확인] 미래 공변량 제안의 데이터 감사는 0fits로 Liander 공식 예제 한 타깃을 검증했다. 333원점의 시간 선택은 일관됐지만 15원점에 기상 결측이 있고 과거 날씨 공개시각은 없었다. 실제 앙상블·예측 이득·신규성은 미확인이다. [한국어 보고서](../research/covariate_availability_audit_20260916/REPORT.md), [검증](../research/covariate_availability_audit_20260916/verification.json).

[확인] Attention prior 후속 감사는 신규학습0회로 LiSA의 공식 가산 보정 경로와 Tiny-Attention Adapter를 비교하고, 70개 공통 V 기록을 점검했다. 단일-head log-prior 보정의 연산 동치와 multi-head 확률 평균의 비동치를 확인했다. 단순 head 변형을 새 주제로 바로 학습할 근거는 부족하다. [한국어 보고서](../research/attention_prior_novelty_audit_20260916/REPORT.md).

[확인] 동결 attention prior 후보와 같은 용량 SIDE의 8/8fits·9143updates 및 평가를 완료했다. PRIOR는 SIDE 대비 평균 MSE를 전력1.007%·교통1.423% 개선했지만 두 seed의 방향이 달랐다. LoRA 대비 전력0.166% 개선/교통3.668% 악화, 학습 peak allocated 약60% 절감이다. 사전 개발 조건 미충족이며 독립 PASS·새 방법론 주제는 미확보다. [최종 판단](../results/channel_attention_prior_20260916/FINAL_DECISION.md).

[확인] 저장 LoRA4개에서 Q/K·V 유지/제거의16개 V 예측을 신규학습0회로 완료했다. 양쪽 제거 모두 악화하고 V 제거 비용이 더 컸다. 시간 관계만의 적응을 정답으로 선택할 근거는 없으며, 공동 적응한 checkpoint의 제거 진단이라는 한계가 있다. [한국어 보고서](../results/lora_projection_diagnostic_20260916/REPORT.md).

[확인] 동일 용량 마지막 adapter4/4fits·4632updates를 완료했다. 작은 adapter 대비 교통3.201% 개선했지만, LoRA 대비 전력3.623%·교통5.361% 악화했다. 학습 파라미터 수는 같고 peak allocated는 약73% 작았다. 알려진 용량 대조이며 새 방법론 주제 미확보다. [최종 판단](../results/channel_capacity_match_20260916/FINAL_DECISION.md).

[확인] 같은 시간대 입력 조건화 후보와 두 동일용량 대조12/12fits·15240updates를 완료했다. 후보는 일반 adapter 대비 전력0.039%/교통0.128% 악화했고, LH 대비 메모리약73% 절감과 MSE3.631%/8.984% 악화가 함께 나타났다. 실행은정상완료, 후보추가가치는미확인이다. [최종 판단](../results/channel_phase_transport_20260916/FINAL_DECISION.md).

[확인] 균형 표본 head-only4/4fits·5080updates와 평가를 완료했다. LoRA+head는head-only 대비 전력5.794%/교통11.520% 평균 MSE를 추가 개선했고 두seed 모두 개선했다. 표준 LoRA의 구성요소 가치 확인이며 새 방법론 PASS는 아니다. [보고서](../results/channel_head_only_20260916/REPORT.md).

[확인] 시작 위상 균형 LoRA 대조4/4 fits·4632updates를 완료했다. 같은updates에서 전력33.453%/교통58.954%, 각V 선택에서는40.345%/68.710% 평균 MSE 개선을 확인했다. 알려진 sampling 변경의 개발 결과이며 새 PEFT 방법론 PASS는 아니다. [보고서](../results/channel_phase_balance_20260916/REPORT.md), [원인·목표 구분](../results/channel_phase_balance_20260916/INTERPRETATION.md).

[확인] 최신 LH 잔차 분석은 신규학습0회로24시간대별 보정의 개발 이득(전력18~21%, 교통41~42%)과 train/V/E 예측 시작 위상 불일치를 확인했다. 추가 온라인 정답을 쓰는 통계 대조이며 독립 PEFT PASS가 아니다. [보고서](../results/channel_residual_evidence_20260916/REPORT.md), [원인·선행 검토](../results/channel_residual_evidence_20260916/INTERPRETATION.md).

[확인] 최신 표현 보존 진단은 신규4 fits·2030updates를 완료했다. 알려진 ReZero 대조가 기존 SHARED보다 네 조건 모두 좋았지만, 어느 원천에서도 두 seed 모두 LH를 넘지는 못했다. 재사용 개발 자료의 원인 진단이며 새 방법론 PASS가 아니다. [한국어 보고서](../results/channel_identity_diagnostic_20260916/REPORT.md), [선행·연구 검토](../results/channel_identity_diagnostic_20260916/RESEARCH_REVIEW.md).

[확인] 최신 짧은 이력 PEFT 직접 비교는120 fits·19680updates와 전체 검산을 완료했다. 세 방법 모두 ZERO 선택으로 후보의 추가 이득0%, 판정 NO_METHOD_TOPIC_THIS_RUN이다. STD fixed120은 F0보다 평균 primary5.811% 좋았지만95% 구간은0을 포함한다. [최종 판단](../results/building_peft_topic_decision_20260916/FINAL_DECISION.md), [한국어 보고서](../results/building_peft_topic_decision_20260916/REPORT.md).

[확인] 최신 새 건물 전이 실행은 57 fits·8,616 updates 후 STOP_NO_TRANSFER_SIGNAL로 종료됐다. POOLED는 tune에서 선택된 AFFINE 대비 dev 주지표30.340% 악화, 개선4/8이다. 표준 rank1 fixed120은 F0 대비 주지표7.978% 개선했지만 H3 및 raw RMSE는 악화했다. 조건부36 fits·기존 heldout은 미실행이다.

[확인] 이전 건물 cold-start coverage 실행은 recipe8+screen16 fits·2,880 updates 후 종료됐다. Coverage 상호작용 양의 건물2/4로 사전3/4 조건 미충족. Stage B/C는 미실행이며 이전 모든 결과를 보존한다.


| 완료 작업 | 결과 | 검증 기록 |
| --- | --- | --- |
| Attention prior 선행·학습량 감사:0fits, 가까운 기존 연산 확인 | [한국어 REPORT](../research/attention_prior_novelty_audit_20260916/REPORT.md), [공통 V 비교](../research/attention_prior_novelty_audit_20260916/validation_summary.csv) | [수식·gradient 검산](../research/attention_prior_novelty_audit_20260916/algebra.json), [입력·기존 결과 보존](../research/attention_prior_novelty_audit_20260916/verification.json) |
| 동결 attention prior:8/8fits·9143updates, 평균 양성·seed 일관성 미확보 | [한국어 REPORT](../results/channel_attention_prior_20260916/REPORT.md), [원점수](../results/channel_attention_prior_20260916/macro_scores.csv), [구성요소 해석](../results/channel_attention_prior_20260916/INTERPRETATION.md) | [200개 예측·8개 재생 검산](../results/channel_attention_prior_20260916/verification.json), [사전 프로토콜](../results/channel_attention_prior_20260916/PROTOCOL.md) |
| LoRA projection 진단:0fits·16개 V 예측, Q/K와V 공동 의존 | [한국어 REPORT](../results/lora_projection_diagnostic_20260916/REPORT.md), [원점수](../results/lora_projection_diagnostic_20260916/macro_scores.csv), [구성요소 분해](../results/lora_projection_diagnostic_20260916/effects.csv) | [16개 독립 검산·4개exact재생](../results/lora_projection_diagnostic_20260916/verification.json), [사전 프로토콜](../results/lora_projection_diagnostic_20260916/PROTOCOL.md) |
| 동일 용량 대조:4/4fits·4632updates, LoRA 대비 정확도·메모리 절충 | [한국어 REPORT](../results/channel_capacity_match_20260916/REPORT.md), [원점수](../results/channel_capacity_match_20260916/macro_comparisons.csv), [다음 연구 질문](../results/channel_capacity_match_20260916/INTERPRETATION.md) | [113개 예측·4개 checkpoint 검산](../results/channel_capacity_match_20260916/verification.json), [실행정책 재검산](../results/channel_capacity_match_20260916/completion_audit.json) |
| 같은위상 입력 조건화:12/12fits·15240updates, 고유 추가 가치 미확인 | [한국어 REPORT](../results/channel_phase_transport_20260916/REPORT.md), [최종 판단](../results/channel_phase_transport_20260916/FINAL_DECISION.md), [원점수](../results/channel_phase_transport_20260916/macro_scores.csv) | [304개 예측·12개 checkpoint 검산](../results/channel_phase_transport_20260916/verification.json), [용량 교란·원인 구분](../results/channel_phase_transport_20260916/INTERPRETATION.md) |
| 균형 표본 head-only 대조: 4/4fits·5080updates, LoRA의 추가 이득 확인 | [한국어 REPORT](../results/channel_head_only_20260916/REPORT.md), [원점수](../results/channel_head_only_20260916/comparisons.csv), [연구 판단](../results/channel_head_only_20260916/INTERPRETATION.md) | [104개 예측·4개 checkpoint 검산](../results/channel_head_only_20260916/verification.json), [사전 프로토콜](../results/channel_head_only_20260916/PROTOCOL.md) |
| 시작 위상 균형 LoRA: 4/4 fits·4632updates, 두 원천·두 seed 개선 | [한국어 REPORT](../results/channel_phase_balance_20260916/REPORT.md), [같은updates·선택 원점수](../results/channel_phase_balance_20260916/comparisons.csv) | [99개 예측 원점수·4개 checkpoint 검산](../results/channel_phase_balance_20260916/verification.json), [추가 정답 시점0](../results/channel_phase_balance_20260916/target_exposure_audit.json) |
| LH 잔차·시작 위상 감사: 신규학습0, 사후 개발 근거 | [한국어 REPORT](../results/channel_residual_evidence_20260916/REPORT.md), [판단 수정](../results/channel_residual_evidence_20260916/INTERPRETATION.md) | [32개 MSE·48개 인과성 검사](../results/channel_residual_evidence_20260916/verification.json), [시작 위상 manifest 감사](../results/channel_residual_evidence_20260916/phase_coverage_audit.json) |
| 표현 보존 채널 진단: 4/4 fits·2030updates, 알려진 대조의 개발 신호 | [한국어 REPORT](../results/channel_identity_diagnostic_20260916/REPORT.md), [원점수·비교](../results/channel_identity_diagnostic_20260916/comparisons.csv) | [62개 예측의 MSE/MAE·4개 checkpoint 검산](../results/channel_identity_diagnostic_20260916/verification.json), [공개 모듈 구조 검사](../results/channel_identity_diagnostic_20260916/structural_probe.json) |
| 짧은 이력 PEFT 직접 비교: 120/120 fits·19680updates, NO_METHOD_TOPIC_THIS_RUN | [한국어 REPORT](../results/building_peft_topic_decision_20260916/REPORT.md), [최종 결정](../results/building_peft_topic_decision_20260916/FINAL_DECISION.md), [추가 가치 해석](../results/building_peft_topic_decision_20260916/INTERPRETATION.md) | [3300개 원점수 검산](../results/building_peft_topic_decision_20260916/independent_cpu_verification.json), [600개 체크포인트·5100예측 복원](../results/building_peft_topic_decision_20260916/independent_gpu_verification.json), [후속 연결 완료](../results/building_peft_topic_decision_20260916/followup_connection.json) |
| 새 건물 전이: 57 fits·8,616 updates; STOP_NO_TRANSFER_SIGNAL, 후속36 fits 미실행 | [한국어 REPORT](../results/building_transfer_subspace_v1_20260915/REPORT.md), [선택 순위 역전 해석](../results/building_transfer_subspace_v1_20260915/INTERPRETATION.md), [원점수](../results/building_transfer_subspace_v1_20260915/scores.csv) | [240개 예측 검산](../results/building_transfer_subspace_v1_20260915/independent_verification.json), [실행 집계](../results/building_transfer_subspace_v1_20260915/execution_summary.json), [종료 후 미실행 분기 수정](../results/building_transfer_subspace_v1_20260915/post_run_patch.json) |
| 건물 cold-start: recipe8+screen16 fits·2,880 updates; STOP_NO_COVERAGE_PROBLEM_SIGNAL | [한국어 REPORT](../results/building_coldstart_coverage_v1_20260915/REPORT.md), [건물별 상호작용](../results/building_coldstart_coverage_v1_20260915/stageA_interaction.csv) | [128개 예측 검산](../results/building_coldstart_coverage_v1_20260915/independent_verification.json), [최종 감사](../results/building_coldstart_coverage_v1_20260915/publication_audit.json) |
| R1/R2 완료: R1 수치84updates·본학습0, R2 24/24 fits·고정 epoch 파일럿 | [한국어 REPORT](../research/peft_rank12_20260915/REPORT.md), [R1](../results/query_budget_repair_v2_20260915/REPORT.md), [R2](../results/channel_sharing_screen_v1_20260915/REPORT.md) | [완료 장부](../research/peft_rank12_20260915/completion.json), [R2 독립 검산](../results/channel_sharing_screen_v1_20260915/independent_verification.json) |
| Q/C 승인 재개 완료: Q 수치 중단180 updates, C24/24 fits·24,576 updates; 경량화 신호 관측·계수 이득 미충족 | [통합 한국어 REPORT](../results/priority12_resume_20260915/REPORT.md), [Q 진단](../results/query_budget_numeric_v2_resume_20260915/NUMERIC_RESULT.md), [C 원점수](../results/channel_basis_pilot_resume_20260915/metrics.csv) | [완료 검증](../results/priority12_resume_20260915/completion_verification.json), [172개 예측 검산](../results/channel_basis_pilot_resume_20260915/independent_verification.json) |
| Q v2 + 채널 공유 독립 파일럿: BLOCKED_GPU_BUSY / BLOCKED_GPU_BUSY; 신규 본학습 0 fits | [통합 한국어 보고서](../results/priority12_20260915/REPORT.md), [Q](../results/query_budget_numeric_v2_20260915/REPORT.md), [C](../results/channel_basis_pilot_20260915/REPORT.md) | [완료 검증](../results/priority12_20260915/completion_verification.json), [C 실제 CPU 모델 비교](../results/channel_basis_pilot_20260915/initial_cpu_parity.json) |
| Query 자원 제약 파일럿: A 폐기용 120 updates, B 0 fits; INCONCLUSIVE_NUMERICS | [수치 중단 보고](../results/query_budget_pilot_20260915/REPORT.md), [실제 측정표](../results/query_budget_pilot_20260915/resource_measurements.csv) | [종료 검증](../results/query_budget_pilot_20260915/completion_verification.json), [42개 비교 재계산](../results/query_budget_pilot_20260915/independent_verification.json), [117개 CPU 검사](../results/query_budget_pilot_20260915/cpu_tests.json) |
| Censor tail 통제 파일럿: 6 fits / 2,160 updates, smoke 6 updates | [최종 검토](../results/censor_tail_controlled_v1/FINAL_REVIEW.md), [전체 표/그림](../results/censor_tail_controlled_v1/REPORT.md) | [독립 재계산](../results/censor_tail_controlled_v1/verification.json), [완료 교차 검산](../results/censor_tail_controlled_v1/completion_audit.json) |
| 최신 Anchor / PatchPhase / Query / FR / Censor 재검토 | [최종 평가](../research/reopen_review_20260914/FINAL_ASSESSMENT.md), [방법별 판정](../research/reopen_review_20260914/METHOD_REOPEN_MATRIX.csv) | [978개 독립 검사](../research/reopen_review_20260914/verification.json) |
| PatchPhase v2: 12 fits / 8,640 updates | [결과 및 한계](../research/reopen_review_20260914/PATCHPHASE_V2_RESULT.md) | [예측 재생·체크포인트 증거](../research/reopen_review_20260914/patch_evidence.json) |
| Query: 18개 자원 측정점, 신규 forecasting fits 0 | [자원 비교](../research/reopen_review_20260914/QUERY_RESOURCE_FRONTIER.md), [그래프](../results/reopen_query_resource_20260914/quality_memory_time.png) | [수치 동등성](../results/reopen_query_resource_20260914/parity.json), [실행 기록](../results/reopen_query_resource_20260914/receipt.json) |
| FR 진입 조건 / Censor tail-gradient 진단: 신규 fits 0 | [FR](../research/reopen_review_20260914/FR_ENTRY_DIAGNOSTIC.md), [Censor](../research/reopen_review_20260914/CENSOR_TAIL_DIAGNOSTIC.md) | [통합 검증과 제한](../research/reopen_review_20260914/verification.json) |
| 미래 구간의 validation 선택 전달 비교: 신규 fits 0 | [최종 검토](../results/future_selection_transfer_v1/FINAL_REVIEW.md) | [최종 검증](../results/future_selection_transfer_v1/final_validation.json), [산출물 검증](../results/future_selection_transfer_v1/artifact_verification.json) |
| 시간적 전달 실패 A–C 분석 | [보고서](../results/temporal_transfer_diagnostic_v1/REPORT.md), [기존 gain 계산 정정](../results/temporal_transfer_diagnostic_v1/ERRATUM.md) | [산출물 검증](../results/temporal_transfer_diagnostic_v1/artifact_verification.json) |
| 시간적 전달 실패 D: 12개 상태의 gradient / perturbation 진단 | [최종 검토](../results/temporal_transfer_diagnostic_v1_D/FINAL_REVIEW.md) | [결과·검증 파일](../results/temporal_transfer_diagnostic_v1_D/) |
| Anchoring window-budget: 48 fits / 43,200 updates | [원래 보고서](../results/anchor_window_study_20260914/REPORT.md), [후속 효과 재검토](../research/reopen_review_20260914/ANCHOR_REVIEW.md) | [원래 검증](../results/anchor_window_study_20260914/verification.json), [재검토 증거](../research/reopen_review_20260914/anchor_evidence.json) |
| 더 이전의 전체 방법 재평가 | [전체 재평가](../research/full_reassessment_20260914/REPORT.md) | 해당 보고서의 비교·검증 링크 |
| 초기 7개 후보 및 이후 pilot / memory / calibration 연구 | [단계별 결과 안내](../README.md), [7개 후보 검토](../results/screening_summary/latest_review.md) | README의 각 결과 옆 verification 링크 |

## 이전 작업의 업로드 이력

- [`3750490`](https://github.com/CanelE452/tsfm-peft-method-screen/commit/3750490): 완료된 48-fit anchoring 연구.
- [`a45780b`](https://github.com/CanelE452/tsfm-peft-method-screen/commit/a45780b): 시간적 전달 진단과 미래 구간 비교를 포함한 이전 로컬 작업 보관, PatchPhase v2 사전 계획.
- [`f18d125`](https://github.com/CanelE452/tsfm-peft-method-screen/commit/f18d125): 완료된 12-fit PatchPhase v2 결과.
- [`b6dad71`](https://github.com/CanelE452/tsfm-peft-method-screen/commit/b6dad71): 최신 재검토, Query 자원 비교, FR / Censor 진단과 최종 검증.

2026-09-14 재검토 작업 종료 시 CPU 테스트 98개가 통과했다. 재검토 검증 기록에는 독립 검사 978개와 기존 결과 파일 1,091개의 해시 보존이 기록되어 있다. 이 수치는 당시 검증 범위이며, 이 색인을 작성하면서 학습을 다시 실행한 것은 아니다.

Query 자원 제약 파일럿은 117개 CPU 검사와 42개 저장 비교 재계산을 완료했고, 이전 결과 파일 1,217개가 보존됐다. 이는 기록된 수치 중단의 검증이며 예측 PASS가 아니다. B와 신규 V/E scoring은 실행하지 않았다.

## GitHub에서 확인할 수 있는 범위

코드, 고정 프로토콜, 결과 표, 보고서, 실행 기록, 검증 결과와 해시 manifest를 보관한다. `.gitignore`에 포함된 원시 데이터, 모델 가중치, `.cache`의 예측 배열과 체크포인트는 로컬에 남아 있다. 따라서 GitHub에서는 실행 근거와 검증 범위를 검토할 수 있고, 전체 수치 재생에는 해당 로컬 자료가 필요하다. 이후 완료 작업도 검증 후 commit/push하고 이 색인을 갱신한다.

## 예보 순서 대조 완료 (2026-09-16)

신규 4 fits·480 updates, smoke 4 updates. [REPORT](../results/covariate_order_control_20260916/REPORT.md), [해석](../results/covariate_order_control_20260916/INTERPRETATION.md), [4,032개 지표 및 16개 복원 검산](../results/covariate_order_control_20260916/verification.json). 동일 입력 멀티셋의 순서 효과는 작은 개발 차이이며 신규 방법 PASS가 아니다.

## MOMENT 고정 비교 완료 (2026-09-16)

36/36 경로 완료: 신규 28 fits·35,580 updates, 기존 8 fits·10,140 updates 재사용, smoke12·자원104 폐기 updates. [한국어 REPORT](../results/channel_controlled_improvement_v1_20260916/REPORT.md), [최종 결정 C](../results/channel_controlled_improvement_v1_20260916/FINAL_DECISION.md), [전체 원점수](../results/channel_controlled_improvement_v1_20260916/seed_scores.csv), [선택·구성요소 효과](../results/channel_controlled_improvement_v1_20260916/effects.csv).

PRIOR의 SIDE 대비 E_MIXED 이득은 전력 +0.1131%, 교통 −0.2384%로 제한된 구성요소 신호 기준 미충족. LH가 두 원천에서 가장 정확하다. LH native checkpoint는 자원 검사에서 peak62.48% 절감/step46~53% 증가, SIDE-fast보다도 낮은 peak를 기록했다. 동일 조건 2-update 검사는 bitwise 통과했으나 해당 옵션으로 전체20epoch를 새로 학습하지는 않았다. 독립 test·신규 방법 PASS가 아니며 자동 후속 연구는 종료한다.

[4,974개 독립 scalar·756개 checkpoint·40개 복원 검산](../results/channel_controlled_improvement_v1_20260916/independent_verification.json), [28개 완전 재개 상태·164개 타깃 정렬](../results/channel_controlled_improvement_v1_20260916/completion_audit.json), [실행 계약과 경로](CHANNEL_CONTROLLED_IMPROVEMENT_20260916.md), [재사용 감사](../results/channel_controlled_improvement_v1_20260916/REUSE_RECEIPT.md). 큰 예측·가중치는 로컬 cache에 남는다.

## 예보 경로 연결 직접 비교 — 48/48 완료

[한국어 REPORT](../results/forecast_path_structure_v1_20260916/REPORT.md), [최종 결정](../results/forecast_path_structure_v1_20260916/FINAL_DECISION.md), [실행·재현 문서](FORECAST_PATH_STRUCTURE_20260916.md).

세 타깃×네 군×두 LR×두 seed48fits·24,576updates, smoke24updates, 봉인 평가와151,200개 scalar 검산을 완료했다. GPU controller61.77분, 타깃당75 TEST원점이다. PATH 대 DROP raw+1.826%는 관측됐으나 핵심 POINT 대조는raw+0.033%·보정후+0.668%로 전체 구간이0을 포함했다. T0/T1의 양성과 T2 반전을 모두 보존하며 최종 추천은 **추가 근거 미확보** 하나다. PATH는 알려진 학습 규칙, POINT는 합성 통제이며 새PEFT 방법이나 새 독립 데이터 시험이라고 부르지 않는다. 추가 학습은 없다.

[지표·선택·보정 검산](../results/forecast_path_structure_v1_20260916/independent_verification.json), [48개 재개 상태·630개 집계 검산](../results/forecast_path_structure_v1_20260916/completion_audit.json), [출판 감사](../results/forecast_path_structure_v1_20260916/publication_audit.json), [감사 수정 이력](../results/forecast_path_structure_v1_20260916/AUDIT_CORRECTIONS.md). 기존 파일2,451개 hash 보존, 큰 원자료·가중치·예측은 로컬 cache에 있다.


## 아홉 조건 PEFT 비교 — 9/9 완료 (2026-09-17)

[전체 계약·현재 상태](../results/condition_studies_v1_20260916/MASTER_REPORT.md). 모든9명세를 먼저봉인; 기존48forecast경로는재학습하지않고R05/N06예측의존성으로만재사용한다. 모든 항목의 실행·평가·검산을 마쳤다. 본학습116/116fits·59,392updates, 폐기smoke58updates이며 필수 미실행 경로0개다. R05/N06에는 신규 신경망 학습이 없다. 성능 효과와 신규성을 별도로 판단했고 [최종 집중 문제는0개](../results/condition_studies_v1_20260916/FINAL_DECISION.md)다.

- [N01 ASYNC 완료](../results/condition_studies_v1_20260916/N01/REPORT.md): 16fits/8,192updates + smoke8. age항 A3는 A2보다0.175%, ridge A1보다4.017% 악화. 실제 학습·복원·선택·scalar 검산을 통과했고 뒤 후보도 독립 실행을 완료했다.

- [N02 ARCHIVE 완료](../results/condition_studies_v1_20260916/N02/REPORT.md): 16fits/8,192updates + smoke8. 추가 보정 B3는 검색 행 B2보다2.01% 악화(CI는0포함). 짧은 이력 B0의0.371624가 B3의0.387540보다 낮다. 실제 검색 비용·선택·완전 재개 상태 검산을 보존했다.
- [R05 VINTAGE 완료](../results/condition_studies_v1_20260916/R05/REPORT.md): 신규 신경망 학습0, CPU 정책225설정. 최신 점수 동일, E4의 지연 손실은 BINARY보다0.506% 개선했으나 LATEST 대비 CI는0포함. 이전 E 재분석이며 새 PEFT가 아니다.
- [N06 JOINT 완료](../results/condition_studies_v1_20260916/N06/REPORT.md): 신규 신경망 학습0, CPU 의존구조9fits. AR1은 독립 결합보다 합계 CRPS10.013% 개선했고 복잡한 두 방법은 AR1보다 악화했다. 주변분포 동일성 및 독립 scalar 검산 완료. 알려진 단순 결합의 충분성과 PEFT 신규성을 구분한다.

- [N03 CLOCK 완료](../results/condition_studies_v1_20260916/N03/REPORT.md): 16fits/8,192updates + smoke8. 고정 커널0.641412, 학습 커널0.648056으로 추가4계수는1.036% 악화. 512개 원점수·선택·복원 검산 완료. E 원점이23.75시간에 몰린 한계를 명시했다.

- [R04 STABILITY 완료](../results/condition_studies_v1_20260916/R04/REPORT.md): 16fits/8,192updates + smoke8. 모든 반복은 INIT 선택. 단순alpha=.25 평활화는 정확도0.178% 손해로 수정 RMS25% 감소; 추가 LoRA 적응 근거는 없다. 수정량·MAE/RMSE340개 추가 scalar와 CPU 선택 검산 완료.

- [N07 SPECTRAL 완료](../results/condition_studies_v1_20260916/N07/REPORT.md): 16fits/8,192updates + smoke8. PRED0.249438 대 SHUFFLE0.247272로0.876% 악화. 저에너지 성분의 작은 개선과 전체 시간영역 악화를 분리했다. 144개 원점수 및 선택·재개 검산 완료.

- [R08 LEAD 완료](../results/condition_studies_v1_20260916/R08/REPORT.md): 16fits/8,192updates + smoke8. 가용성 보정의 선형 horizon 대비 gain0.0588%는 CI가0을 포함한다. 동결0.375896 대 제안0.380829도 함께 보고했다. 176개 원점수·선택·재개·CPU 대조 검산 완료.

- [R09 MIXED 완료](../results/condition_studies_v1_20260916/R09/REPORT.md): 20fits/10,240updates + smoke10. NULL0.606694 대 MIXED0.538191로12.728% 악화했다. UNIFORM 대비0.171%는 불확실하다. 176개 주지표·66개 부가지표 scalar와 숨긴 상세값 권한·선택·resume·gradient 검산을 완료했다.

[최종 전체 보고서](../results/condition_studies_v1_20260916/MASTER_REPORT.md), [게시 감사](../results/condition_studies_v1_20260916/publication_audit.json), [116경로 검산](../results/condition_studies_v1_20260916/training_record_verification.json), [전체 forward 장부](../results/condition_studies_v1_20260916/FORWARD_LEDGER.csv). 이전2,516파일 보존, native forward115,034/200,000회, 외부 compute 오염0updates. 추가 학습·후속 후보 실행은 없다.

## 날짜 다양성 교정 — 적격2개 완료·사전4개 차단 (2026-09-17)

[표본 교정 계약·실행 기록](CONDITION_SAMPLING_REPAIR_20260917.md), [GPU 전 분산 감사](../results/condition_sampling_repair_v1_20260917/ORIGIN_REPAIR_AUDIT.md). N02/N03의 모든 역할은 날짜·phase 조건 통과. N01/R04/N07/R08은 phase count range3>2로 BLOCKED_DIVERSITY이며 성능 실패가 아니다. 적격32/32fits·16,384updates + smoke16과 양방향 교차 평가를 완료했다. N02/N03 모두 POSITIVE_UNCERTAIN + SAMPLING_SENSITIVE이며 재개 후보0개다. 기존R05/N06/R09는 재학습하지 않았다. [최종 보고서](../results/condition_sampling_repair_v1_20260917/MASTER_REPORT.md), [최종 결정](../results/condition_sampling_repair_v1_20260917/FINAL_DECISION.md), [검산](../results/condition_sampling_repair_v1_20260917/verification.json).


## 긴 이력 압축 PEFT — 20/20 신규 경로 완료 (2026-09-17)

[한국어 REPORT](../results/history_compression_v1_20260917/REPORT.md), [최종 결정](../results/history_compression_v1_20260917/FINAL_DECISION.md), [실행 문서](HISTORY_COMPRESSION_20260917.md). 기존 SHORT/LONG8fits를 검증 후 재사용하고 STATS_SHORT/POOL/POOL_KD/LEARN/LEARN_KD20fits·10,240updates와 smoke10updates를 완료했다. LEARN_KD대POOL_KD +0.0701%, 대LEARN +3.6627%. 구성요소 탐색 기준 충족=False, 정확도·자원절충 신호=False. 신규성은 미확보이며 재사용 개발 평가다. [수치 검산](../results/history_compression_v1_20260917/verification.json), [새 모델 E 복원](../results/history_compression_v1_20260917/independent_model_verification.json). 추가 학습은 없다.


## 입력 오류 강건성·지속 변화 보존 — 48/48 완료·현재 후보 종료 (2026-09-17)

[한국어 REPORT](../results/outlier_signal_peft_v1_20260917/REPORT.md), [최종 결정](../results/outlier_signal_peft_v1_20260917/FINAL_DECISION.md), [검산](../results/outlier_signal_peft_v1_20260917/verification.json). 참조 부품 직접 구현 후48/48fits·49,152main+24smoke updates, 축소 재현98series, E평가·5,120scalar 대조·선택24checkpoint 복원 완료. A5는 Electricity에서 A4 대비 지속 변화3.74% 개선을 보였지만 A1보다 지속 변화 오차가103.8% 컸고 ETTm1도79.6% 컸다. 제한된 추가 효과를 인정하면서 현재 고정 후보는 종료, 후속 집중0개로 결정했다. 실제 사건 레이블·정식 선행 검색·독립 source 검증은 남아 있으며 논문 PASS를 선언하지 않았다.

- [입력 오류 후속 v2 — 48/48 완료](../results/outlier_signal_followup_v2_20260917/REPORT.md): 49,152 main +24 smoke updates, optimizer0 진단, 전체 E저장·채점, 3,840 scalar 대조와24 checkpoint 복원 검산 완료. 기존 TRAIN shift 위치 clip2.93%/3.66% 대 E SHIFT8 87.00%/89.61%로 노출 차이를 확인했다. B5는 B4보다 SHIFT8 +0.41%/+0.24% 개선했지만 B0보다 오차60.18%/27.24% 높아 고정 후보를 종료했다. B3의 ETTm1 SHIFT_POINT +2.34% 부분 이득은 보존, 후속 투자0개. [최종 결정](../results/outlier_signal_followup_v2_20260917/FINAL_DECISION.md), [검산](../results/outlier_signal_followup_v2_20260917/verification.json), [진단](../results/outlier_signal_followup_v2_20260917/DIAGNOSTIC_REPORT.md).

- [B0 유지 + 추가 어댑터](../results/additive_b0_adapter_v1_20260917/REPORT.md): 완료·검산 통과. 추가24 fits / 본학습24,576 + smoke12 updates. Electricity SHIFT8에서 C3는 B0 대비3.963%, 일반 어댑터 C2 대비1.318% 개선했으나 FAULT와 ETTm1 SHIFT8에는 손해가 있어 범용 우월성은 미확인. 기존에 노출된 개발 E이며 논문 PASS 아님. [결정](../results/additive_b0_adapter_v1_20260917/FINAL_DECISION.md), [검산](../results/additive_b0_adapter_v1_20260917/verification.json), [공개 감사](../results/additive_b0_adapter_v1_20260917/publication_audit.json), [고정 계약](../results/additive_b0_adapter_v1_20260917/PROTOCOL.md).

- [지속성 어댑터 논문 후속 검증](../results/additive_persistence_validation_v1_20260917/REPORT.md): **완료·검산**. 58새fits·59,392main+36smoke updates,246평가view. 세seed 전력16계열 SHIFT8에서 C3/B0 +7.112%,C3/C2 +2.399%;C3/RECENCY +0.180%이나 세번째seed는−0.020%. ETTm1 C3/C2−1.847%,ETTm2−0.045%로 지점 일반성 미확보. 독립자료·논문 PASS 아님. [최종 판단](../results/additive_persistence_validation_v1_20260917/FINAL_DECISION.md), [주장–근거](../results/additive_persistence_validation_v1_20260917/PAPER_CLAIM_EVIDENCE.md), [논문 개요](../results/additive_persistence_validation_v1_20260917/PAPER_OUTLINE.md), [검산](../results/additive_persistence_validation_v1_20260917/VERIFICATION.json), [평가 전 통계 계약 수정](../results/additive_persistence_validation_v1_20260917/SEAL_AMENDMENT_01.json).

- [C3 추가 논문 근거 검증](../results/persistence_evidence_extension_20260918/REPORT.md): **완료·검산**, 새학습0회·111추가예측view. NESO2025 SHIFT8에서C3/B0 +5.459%,C3/C2 +1.312%이나C3/RECENCY−0.109%,세번째seed C3/C2−0.558%. 기존전력이득의fixed-weight분해는학습된가중치차이의역할을보였으며C3/RECENCY의LR·step동일성을감사했다. [논문 수정안](../results/persistence_evidence_extension_20260918/PAPER_REVISION.md), [최종 판단](../results/persistence_evidence_extension_20260918/FINAL_DECISION.md), [검산](../results/persistence_evidence_extension_20260918/VERIFICATION.json), [외부자료감사](../results/persistence_evidence_extension_20260918/EXTERNAL_SOURCE_AUDIT.json).

- [논문 준비 보강 실험](../results/paper_readiness_20260918/REPORT.md): **완료·검산**. 114개 비교(90개 새 평가·24개 재사용), 새 학습 0회. 동일 LR·1,024 updates의 C2/C3 학습·추론 교차, F0와 단순 기준선, NESO C3/RECENCY 교차를 수행했다. 전력16계열 SHIFT8에서 학습 가중치와 gate의 기여를 분리했고, NESO SHIFT8은 C3/RECENCY mask가 256입력 모두 같음을 확인했다. 원자료 손해도 보존했다. [논문용 그림9종·표·근거 묶음](../papers/persistence_adaptation/README.md), [한국어 검토 PDF](../papers/persistence_adaptation/EVIDENCE_BRIEF_KO.pdf), [검산](../results/paper_readiness_20260918/FINAL_ARTIFACT_AUDIT.json). 정식 선행 전체 우위나 투고 완료를 의미하지 않는다.

- [C3 약점 보완·시간 전이 검증](../results/c3_identifiability_temporal_20260918/REPORT.md): **완료·검산**, 기존 모델의60개 새 예측(56GPU·4CPU), 새 학습0회. 사전 봉인된 NESO2026 H1의128일에서 C3/B0 +3.580%, C3/C2 +1.984%, C3/RECENCY +0.061%(CI0포함). REFERENCE는 B0보다0.487% 악화. 기존/새5개 panel의 입력 mask별 기전 분해와3종 그림을 추가했다. 새 기간에서도 SHIFT8 mask가 C3/RECENCY 모두 같아 규칙 자체의 필요성은 미확립. [논문 보완 메모](../papers/persistence_adaptation/WEAKNESSES_AND_TEMPORAL_CHECK_KO.md), [최종 판단](../results/c3_identifiability_temporal_20260918/FINAL_DECISION.md), [검산](../results/c3_identifiability_temporal_20260918/VERIFICATION.json). 같은 제공자의 새 시기이며 독립 원천/논문 PASS가 아니다.

- [동일 계약 재사용 감사 — C3 논문 후속 검증](../results/additive_persistence_contract_reuse_20260918/REPORT.md): 다운로드 TXT와 기존 MASTER_CLI의 SHA-256 동일. 기존58fits(59,392main+36smoke updates)·246예측·필수 한국어4문서 완료를 확인하고 **추가학습0회로 재사용**. 이번에54개 실모델 복원,8,400 scalar항목,전체 원점 vector 재집계,LR10개·보정9개·효과660행을 다시 검산했다. 기존 결과348파일 불변. [재검증](../results/additive_persistence_contract_reuse_20260918/VERIFICATION.json). 다른 계약/후속 연구를 재개하지 않았다.

- [C3 약점 해결 설명 대조 3종](../results/c3_weakness_controls_20260918/REPORT.md): **완료·검산**, 새 30/30 fits·30,720 main+12 smoke updates·54개 평가 예측. C3/B0를 변경하지 않고 POS_ONLY/MAG_ONLY/OUTPUT_CONTEXT를 동일 학습 기회로 비교했다. 전력 16계열 SHIFT8에서 C3는 위치 대조보다 +0.6943%, 출력 보정보다 +2.8989% 좋지만 크기 대조보다 −0.2515% 나빴다. 기존 C3/C2 +2.3994%는 보존되며 연속성 고유의 추가 가치는 강화되지 않았다. ETTm1의 손해와 원자료·오류 절충도 유지했다. [한국어 해석](../results/c3_weakness_controls_20260918/INTERPRETATION_KO.md), [최종 판단](../results/c3_weakness_controls_20260918/FINAL_DECISION.md), [원점수](../results/c3_weakness_controls_20260918/RAW_SCORES.csv), [검산](../results/c3_weakness_controls_20260918/VERIFICATION.json), [재현 범위](../results/c3_weakness_controls_20260918/REPRODUCIBILITY.md). 반복 사용한 개발 E이며 독립 시험·논문 PASS가 아니다. 추가 자동 학습 0개.

- [C3–MAG_ONLY 영향 분해](../results/c3_magnitude_diagnostic_20260918/REPORT.md): **완료·독립 검산**, 새 학습0회·교차 예측36개·기존36개 재사용. 전력16계열 SHIFT8에서 C3 규칙은 고정 가중치 평균에서 유리했지만, MAG 학습 가중치 이득이 이를 상쇄했다. 공통 C3 오차 기준 가중치+0.6436%/규칙−0.3928%/합계MAG+0.2509%. 순이득85.4%는seed81552의signed기여이며 16계열 중13개MAG/3개C3 방향이다. 89,088입력에서 gate 차이는 same-sign extreme run의첫7관측에만 존재했다. [한국어 해석](../results/c3_magnitude_diagnostic_20260918/INTERPRETATION_KO.md), [최종 판단](../results/c3_magnitude_diagnostic_20260918/FINAL_DECISION.md), [독립 검산](../results/c3_magnitude_diagnostic_20260918/INDEPENDENT_AUDIT.json), [재현](../results/c3_magnitude_diagnostic_20260918/REPRODUCIBILITY.md). 사후 고정 함수 진단이며 새 방법 성공·독립 시험이 아니다.

- [시간 반응 보존 PEFT 방법 파일럿](../results/temporal_response_peft_20260919/PREFLIGHT_REPORT.md): **신규16/16 본학습 완료·학습 검산 통과, 평가 진행 단계**. 일반 어댑터 위에 B0의 유한 지속 변화 반응을 보존하는 손실을 구현하고 ANCHOR/SHUFFLE/IDEAL을 직접 대조한다. 기존 PLAIN4개 재사용, 신규 최대16fits·16,384main+16smoke. 결과·방법 우위·신규성은 아직 미확인이다. [고정 프로토콜](../experiments/temporal_response_peft_20260919/PROTOCOL.md).

  - [평가 재사용 검사 보완](../results/temporal_response_peft_20260919/AMENDMENT_01_KO.md): 새 E 평가 전 adapter hash와 기반 B0의 식별을 분리했다. 원래 봉인·5,152updates를 보존하고 정확한 epoch 경계에서 재개했으며 과학적 설정 변경은 없다.

  - [학습 독립 검산](../results/temporal_response_peft_20260919/TRAINING_AUDIT.json): 16,384main+16smoke, 중복 update0, 기존 PLAIN4개 및 총100개 checkpoint hash·V 선택 검증. 평가 결과와 논문 판단은 아직 대기 중이다.
