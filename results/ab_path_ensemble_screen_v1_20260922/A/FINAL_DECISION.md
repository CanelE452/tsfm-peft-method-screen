# A 최종 판정

**GO_A_COMPRESSION_ONLY**

V 고정 기준선: FIXED3. 후보 CONTEXT3 CAL CRPS 개선 +0.0429%, seed별 +0.0719% / +0.0139%. median nMAE 개선 -0.1162%.

- `gain_at_least_1pct`: False
- `both_seeds_positive`: True
- `full9_crps_loss_at_most_1pct`: True
- `full9_pinball_loss_at_most_1pct`: True
- `latency_saving_at_least_15pct`: True
- `point_loss_at_most_1pct`: True
- `teacher_not_weaker_than_f0`: True

이는 일반 압축(A) 또는 기상정보(B)의 유용성을 인정하는 분류이며 새 모듈의 방법론 GO가 아닙니다. A는 CONTEXT3=GLOBAL3, B는 SCENARIO3가 SET·GBQR보다 불리합니다. 구현·자료·자원 block 없이 정해진 main 예산을 완료했습니다. B의 공개 지연 미검증과 학습 한계는 [보고서](REPORT_KO.md)에 별도로 표시했습니다.

추가 fit/LR/rank/seed/자료 탐색 0. 자동 후속 0. 논문 PASS/신규성 선언 없음.
