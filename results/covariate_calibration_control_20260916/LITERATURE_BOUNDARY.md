# 이번 후속의 선행 검토 범위

2026-09-16. 목적은 calibration과 기존 consistency를 새 방법으로 잘못 제안하지 않는 것이다. 검색 부재를 최초성 증명으로 사용하지 않는다.

| 자료 | 실제 확인 | 이번 판단에 사용하는 범위 |
| --- | --- | --- |
| PACE, Ni·Zhang·Koniusz, NeurIPS2024 | [공식 게재 페이지](https://proceedings.neurips.cc/paper_files/paper/2024/hash/70a06501001e1820fd1eb9ee821302d2-Abstract-Conference.html), [arXiv v1 §3.1–3.5](https://arxiv.org/html/2409.17137v1#S3), 실험설정의 일부 | 곱셈 adapter perturbation과 consistency가 알려진 PEFT 접근임을 확인. 본문 식5/7/11/12를 읽었다. 전체 증명 검증·공식코드 재현·Chronos 성능 비교는 미실행. 공식 최종본과v1 전체 일치도 미확인. |
| 저장소 FR_LORA | src/tsfm_peft_screen/candidates/fr_lora.py 전체, runners/common_eval.py의revision항, research/reopen_review_20260914/FR_ENTRY_DIAGNOSTIC.md | frozen forecast를 뺀 보정량의 쌍별 일관성은 저장소에 이미 있다. 예보 버전 쌍과 시간 origin 쌍은 다르므로 모든 구현이 같다고 하지는 않는다. 단순 재명명은 후보로 인정하지 않음. |
| 기존 building residual consistency | results/building_peft_topic_decision_20260916/MECHANISM_SPEC.md와LITERATURE_BOUNDARY.md | 레벨·형상 방향으로 다르게 가중한 frozen-output 보존항도 이미 실행한 후보. 이번 단순보정 이득으로 과거 판정을 소급 변경하지 않음. |

추가 검색에서 weather-error augmentation 관련 SONNET(OpenReview OMFssKwpyo)과 dissemination-aware weather forecast vintages(SSRN7353383)를 찾았다. 전자는 본문 열기가 브라우저 확인으로 막혔고 후자는 직접 열기에 오류가 있어, 검색 초록/발췌 이상의 상세 수식·실행·게재 상태를 검증하지 못했다. 이번 설계 판단의 수식 근거로 사용하지 않았다. 다른 검색 결과의 LLM calibration 논문도 제목만으로 TSFM 적용 가능성을 주장하지 않았다.

이전 [공변량 선행 검토](../../research/covariate_reliability_diagnostic_20260916/LITERATURE_BOUNDARY.md)는 유지한다. PACE·FR와 구별되는 필요성을 확인하기 전 consistency를 추가한 모델을 새 PEFT로 발표하지 않는다.
