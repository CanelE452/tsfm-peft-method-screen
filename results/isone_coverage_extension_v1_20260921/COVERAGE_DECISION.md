# Coverage 판단

**REAL_SHIFT_CONDITION_TOO_RARE_FOR_PRIMARY_STUDY**

- 고정 E_CONFIRM: 2024-01-01 00:00 UTC 이상 ~ 2026-09-01 00:00 UTC 미만.
- S≥3: **12 origins / 3 unique UTC dates**.
- 요구조건: origins≥50 AND unique UTC dates≥14.
- optimizer 0 / model prediction 0 / target error scoring 0 / 자동 학습 0.

기간을 수년으로 확대해도 고정 primary 조건의 최소 원점 수 또는 날짜 수를 충족하지 못했다. 따라서 이 조건을 primary로 삼아 현재 MAG를 주력 방법론 연구로 계속하는 타당성이 낮다. 이 판단은 실제 조건의 희소성에 관한 것이며, MAG의 예측 성능이 나쁘다는 증거나 모든 robust PEFT의 반증이 아니다. threshold·source·gate를 바꾸어 주제를 살리거나 추가 학습을 시작하지 않고 종료한다.

기존 b3761e5의 2026년 7~8월 E 결과 9 origins / 2 days는 `INCONCLUSIVE_DATA_COVERAGE`로 보존한다. 이번 확대 결과와 원래 결과는 E 기간 및 TRAIN sigma가 다르므로 동일 표본의 성능 개선으로 표현하지 않는다. 새 source·MAG·threshold는 만들지 않았다.
