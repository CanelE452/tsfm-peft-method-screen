# 완료 작업과 검증 기록

2026-09-16 기준. 표의 링크는 완료된 실행 또는 검토 결과다. 과거 FAIL/STOP은 당시 고정 실험의 판정으로 보존한다. 최신 재검토의 후속 연구 우선순위는 방법론 성공이나 새로운 실험 실행을 뜻하지 않는다.


[확인] 최신 짧은 이력 PEFT 직접 비교는120 fits·19680updates와 전체 검산을 완료했다. 세 방법 모두 ZERO 선택으로 후보의 추가 이득0%, 판정 NO_METHOD_TOPIC_THIS_RUN이다. STD fixed120은 F0보다 평균 primary5.811% 좋았지만95% 구간은0을 포함한다. [최종 판단](../results/building_peft_topic_decision_20260916/FINAL_DECISION.md), [한국어 보고서](../results/building_peft_topic_decision_20260916/REPORT.md).

[확인] 최신 새 건물 전이 실행은 57 fits·8,616 updates 후 STOP_NO_TRANSFER_SIGNAL로 종료됐다. POOLED는 tune에서 선택된 AFFINE 대비 dev 주지표30.340% 악화, 개선4/8이다. 표준 rank1 fixed120은 F0 대비 주지표7.978% 개선했지만 H3 및 raw RMSE는 악화했다. 조건부36 fits·기존 heldout은 미실행이다.

[확인] 이전 건물 cold-start coverage 실행은 recipe8+screen16 fits·2,880 updates 후 종료됐다. Coverage 상호작용 양의 건물2/4로 사전3/4 조건 미충족. Stage B/C는 미실행이며 이전 모든 결과를 보존한다.


| 완료 작업 | 결과 | 검증 기록 |
| --- | --- | --- |
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
