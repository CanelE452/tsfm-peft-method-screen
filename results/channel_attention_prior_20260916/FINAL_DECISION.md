# 최종 투자 판단

1. **새 방법론 주제는 아직 미확보다.** 동결 attention을 side 경로의 기준분포로 쓰는 후보에는 평균 개선 신호가 있었지만, 두 seed의 방향이 일치하지 않았고 강한 LoRA 대조를 안정적으로 넘지 못했다.
2. **변경과 선행의 차이:** 동일한 저차원 side attention에 동결 backbone의 층별 attention 평균분포를 logit 기준으로 추가했다. LST/LAST의 side 경로와 저차원 attention은 알려진 구성이다. 이번 차이의 최초성은 미확정이며 LAST 전체 재현도 아니다.
3. **핵심 결과:** 아래는 두 seed를 평균한 V 선택 모델의 재사용 E 점수다. 개선율은 양수가 좋다.

   | 원천 | PRIOR MSE | SIDE MSE | LoRA MSE | PRIOR 대 SIDE | PRIOR 대 LoRA |
   |---|---:|---:|---:|---:|---:|
   | 전력 | 0.279546 | 0.282389 | 0.280010 | +1.007% | +0.166% |
   | 교통 | 0.400725 | 0.406510 | 0.386548 | +1.423% | −3.668% |

   SIDE 대비 설명용 시간 블록 95% 구간은 전력 [−0.240, +2.292]%, 교통 [+0.614, +2.284]%다. 그러나 두 원천 모두 seed 41000에서 악화하고 41001에서 개선했다. LoRA 대비 구간은 전력 [−1.050, +1.542]%, 교통 [−4.834, −2.773]%다. 이 구간은 반복 노출된 E의 사후 개발 설명이며 독립 확증이 아니다. 학습 peak allocated는 약 426–428MiB로 LoRA보다 약 60% 작지만 학습 파라미터 수는 같다.
4. **실행:** 8/8 fits, 본학습 9,143 updates, 별도 smoke 8 updates, 재시도 0. 고정한 조기 종료에 따라 상한 10,160 중 1,017 updates를 쓰지 않았다. E 평가 16개 기록과 선택 체크포인트 8개 재생을 완료했다. 기존 대조 12 fits는 재학습하지 않았다. CPU의 학습 입력 관찰은 0 fits다. 미노출 평가, 설정 재탐색, 후속 후보 학습은 미실행이다.
5. **판정:** EXECUTION=COMPLETE; SIDE 대비 평균 효과=POSITIVE_BUT_UNCERTAIN; LoRA 대비 정확도·메모리 관계=TRADEOFF_ONLY; 사전 개발 조건=NOT_MET; NOVELTY=UNRESOLVED_PRIOR_COMPONENTS_KNOWN; 독립 SCREEN_PASS=NOT_EVALUATED; 새 방법론 주제=NOT_CONFIRMED.
6. **다음 결정: 보류.** 이번 기록으로 방법론 PASS를 선언하지 않는다. 전력만 남기거나 유리한 seed만 고르지 않는다. 같은 후보의 추가 튜닝이나 다른 후보의 학습을 이 실행에 자동 연결하지 않는다.

[한국어 상세 보고서](REPORT.md) · [구성요소별 해석](INTERPRETATION.md) · [원점수](scores.csv) · [검산](verification.json)
