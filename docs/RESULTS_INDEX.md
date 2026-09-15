# 완료 작업과 검증 기록

2026-09-16 기준. 표의 링크는 완료된 실행 또는 검토 결과다. 과거 FAIL/STOP은 당시 고정 실험의 판정으로 보존한다. 최신 재검토의 후속 연구 우선순위는 방법론 성공이나 새로운 실험 실행을 뜻하지 않는다.


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
