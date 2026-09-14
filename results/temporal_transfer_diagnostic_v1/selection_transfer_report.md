# 같은 후보의 시간순 선택 진단

24 source×budget×seed×arm cell, 각각 LR 2개×step 4개에서 step0 중복만 합친 7개 후보. 총 192개 원본 예측·체크포인트 identity를 기록했다. 모든 cell의 서로 다른 후보가 확인됐고 누락 캐시는 없다.

S는 기존 V 앞 8 origins, D_diag는 뒤 8 origins다. 학습 타깃은 S보다 먼저 끝나며 S/D horizon 타깃은 겹치지 않는다. D_diag의 문맥에 S 관측이 들어가는 것은 예측 당시 과거 정보다. D_diag는 사후 재사용 V이며 새 holdout이 아니다.

표는 각 source/budget/arm의 두 seed에 대한 %F0 평균이다. 원손실·paired 차이·각 seed·앞4/뒤4·유효 타깃 수는 temporal_selection_comparisons.csv에 있다. 서로 다른 원천의 raw loss를 섞어 순위를 만들지 않았다.

| Source | Windows | Arm | FIXED %F0 | RECENT4 %F0 | SPREAD4 %F0 | ALL8 %F0 |
|---|---:|---|---:|---:|---:|---:|
| beijing | 32 | native | +0.4057 | +0.4057 | +0.4057 | +0.4057 |
| beijing | 32 | native_anchor | +0.8293 | -2.1591 | +0.8293 | -2.1591 |
| beijing | 233 | native | +2.5299 | +2.5103 | +4.0306 | +3.8136 |
| beijing | 233 | native_anchor | +1.7867 | +5.5525 | +5.5525 | +5.2500 |
| ettm2_later | 32 | native | +4.4324 | +0.0000 | +0.0000 | +0.0000 |
| ettm2_later | 32 | native_anchor | +3.9865 | +0.0000 | +0.0000 | +0.0000 |
| ettm2_later | 233 | native | +0.1788 | +0.0000 | +0.0000 | +0.0000 |
| ettm2_later | 233 | native_anchor | +0.1759 | +0.0000 | +0.0000 | +0.0000 |
| electricity_new | 32 | native | -0.2915 | -0.2915 | -2.5303 | -2.5303 |
| electricity_new | 32 | native_anchor | +0.0304 | +0.0304 | -2.4462 | -1.1153 |
| electricity_new | 233 | native | +0.4859 | +0.0954 | +0.7422 | -0.3657 |
| electricity_new | 233 | native_anchor | +0.3765 | +0.4441 | +0.6099 | +0.9007 |

| Source | RECENT4 S 개선/D 악화 | SPREAD4 | ALL8 | R2/R3 동일 예측 | 좋은 후보 있으나 ALL8 미선택 | 모든 후보 F0 이하 |
|---|---:|---:|---:|---:|---:|---:|
| beijing | 2 | 1 | 2 | 6 | 1 | 1 |
| ettm2_later | 0 | 0 | 0 | 8 | 8 | 0 |
| electricity_new | 3 | 3 | 6 | 2 | 4 | 2 |

각 원천의 분모는 8개의 서로 의존하는 budget/arm/seed cell이다. 24개를 독립 데이터셋으로 세지 않는다. 같은 예측 반환은 16/24이며 나머지 8/24에서 선택 창 배치에 따른 차이가 관찰된다.

RECENT4와 SPREAD4는 선택 origin 수가 4개로 같고, ALL8은 정보량도 더 많다. 선택 규칙에 따라 앞부분 이득/뒷부분 손해의 횟수는 5/4/8로 달라졌다. 이 횟수는 S 전체 8개와 D 전체 8개의 F0 대조이며, 규칙별 실제 선택 부분집합 점수는 별도 열이다.

ETTm2의 모든 S 기반 규칙은 F0를 선택했지만 D_diag에서는 F0보다 좋은 후보가 있었다. 따라서 "후보가 모두 같다"와 "선택된 결과가 F0 같다"를 구분한다. sparse FIXED의 두 seed 평균 이득은 native +4.4324%F0, anchor +3.9865%F0다. 이것은 새 test 이득이나 배포 가능한 사후 oracle 성능이 아니다.

각 후보의 S/D 순위 상관, ties, 사후 D 최선과 S 최선의 D 손실 차이는 selection_diagnostics.json에 모두 보존했다. 사후 최선은 낙관적 후보집합 진단이지 회수 가능한 이득의 확증이 아니다.

원점수는 origin별 scaled pinball 분자와 유효 타깃 수를 채널별로 합쳐 재구성했다. per_origin_scores.csv의 분자는 분위 평균과 train scale 나눗셈을 포함한다. 원점수는 mean_channel(sum_origin numerator / sum_origin count)이다. Beijing 결측 때문에 origin별 점수 평균을 대신 쓰지 않았다. 모든 비교 블록은 공통 4채널에 유효 타깃이 있으며 채널을 제거하지 않았다.

시간순 평가의 원칙은 [FPP3](https://otexts.com/fpp3/tscv.html), 유한한 검증 기준의 변동성과 선택 편향 구분은 [Cawley & Talbot (2010)](https://jmlr.org/papers/v11/cawley10a.html)를 참고했다. 이번 분석은 rolling-origin 재학습 CV나 공식 편향 보정 알고리즘이 아니다.
