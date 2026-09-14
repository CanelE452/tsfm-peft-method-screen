# 고정 보정 크기 진단

각 arm/cell의 RECENT4 checkpoint를 고정하고 정렬된 분위 예측 사이에서 alpha={0,.25,.5,.75,1}를 공통 적용했다. alpha는 S 전체 8개에서만 고르고 동점은 작은 alpha다. R2의 4개 선택 창 외에 S 전체와 5-alpha 탐색을 추가 사용하므로 R2와 선택 예산이 같지 않다.

| Source | Windows | Arm | Seed | S 선택 alpha | D %F0 | 원 R2 대비 개선 % |
|---|---:|---|---:|---:|---:|---:|
| beijing | 32 | native | 34000 | 1.0 | +1.0252 | +0.0000 |
| beijing | 32 | native_anchor | 34000 | 1.0 | +1.5854 | +0.0000 |
| beijing | 32 | native | 34001 | 1.0 | -0.2139 | +0.0000 |
| beijing | 32 | native_anchor | 34001 | 0.75 | -1.9866 | +3.6985 |
| beijing | 233 | native | 34000 | 1.0 | +1.9665 | +0.0000 |
| beijing | 233 | native_anchor | 34000 | 1.0 | +5.0393 | +0.0000 |
| beijing | 233 | native | 34001 | 0.5 | +6.4876 | +3.5417 |
| beijing | 233 | native_anchor | 34001 | 0.75 | +6.7843 | +0.7651 |
| ettm2_later | 32 | native | 34000 | 0.0 | +0.0000 | +0.0000 |
| ettm2_later | 32 | native_anchor | 34000 | 0.0 | +0.0000 | +0.0000 |
| ettm2_later | 32 | native | 34001 | 0.0 | +0.0000 | +0.0000 |
| ettm2_later | 32 | native_anchor | 34001 | 0.0 | +0.0000 | +0.0000 |
| ettm2_later | 233 | native | 34000 | 0.0 | +0.0000 | +0.0000 |
| ettm2_later | 233 | native_anchor | 34000 | 0.0 | +0.0000 | +0.0000 |
| ettm2_later | 233 | native | 34001 | 0.0 | +0.0000 | +0.0000 |
| ettm2_later | 233 | native_anchor | 34001 | 0.0 | +0.0000 | +0.0000 |
| electricity_new | 32 | native | 34000 | 1.0 | -0.6429 | +0.0000 |
| electricity_new | 32 | native_anchor | 34000 | 1.0 | -0.2177 | +0.0000 |
| electricity_new | 32 | native | 34001 | 1.0 | +0.0599 | +0.0000 |
| electricity_new | 32 | native_anchor | 34001 | 1.0 | +0.2785 | +0.0000 |
| electricity_new | 233 | native | 34000 | 1.0 | +0.4795 | +0.0000 |
| electricity_new | 233 | native_anchor | 34000 | 1.0 | +0.3701 | +0.0000 |
| electricity_new | 233 | native | 34001 | 0.5 | +0.8004 | +1.0860 |
| electricity_new | 233 | native_anchor | 34001 | 0.75 | +0.6562 | +0.1387 |

S-only interior alpha가 원모델과 F0를 모두 넘은 것은 4/24 cell이다. Beijing dense/native와 dense/anchor, Electricity dense/native와 dense/anchor의 seed34001에서 각각 관찰됐다. 각 쌍의 seed34000은 alpha1이 선택됐다. 두 원천에서 한 seed씩 나온 사례를 독립 4원천 재현으로 해석하지 않는다.

ETTm2 8개 cell은 R2 자체가 F0이므로 모든 alpha가 같은 예측이다. alpha0 동점 선택은 축소가 학습된 보정을 구조적으로 고쳤다는 근거가 아니다. 나머지 16개 적응된 R2 중 4개에서 interior 개선을 확인했지만 24개 전체를 함께 보고한다. Beijing sparse/anchor seed34001은 alpha.75로 원모델 손해를 일부 줄여도 F0보다 여전히 약1.99% 나쁘다.

correction_scale_curves.csv에 모든 S/D 곡선, correction_scale_selection.csv에 S 선택과 D 사후 최선이 있다. 사후 최선 alpha는 실행 가능한 모델 성능으로 쓰지 않는다. q는 분위 예측이며 조건부 평균·확률분포 mixture가 아니다. 출력 보간은 학습 중 anchoring이나 LoRA weight scaling과 일반적으로 다르다. alpha endpoint는 정확히 일치했다.

기존 mltimeseries hospital_shared_strength_v1에서도 F0–shared LoRA 출력 강도 선택을 이미 수행했다. 이번 국소 곡선을 새 PEFT 기법으로 이름 붙일 근거는 없다. 당시 GLOBAL은 alpha1, INDIVIDUAL은 GLOBAL보다 나빴다. 이번 결과는 그 실험의 새 재현이 아니라 다른 패널에서의 사후 진단이다.
