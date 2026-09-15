# 한 후보: 과거 잔차 일관성에 따른 레벨·형상 보존 LoRA

1. **문제/범위 [설계]** 3일·14일의 target-only 건물 전력 이력에서, 전체 예측 변화를 똑같이 억제하는 것보다 과거 잔차의 반복성에 따라 레벨/형상 보존을 나누는 것이 유용한가?

2. **관찰 [확인]** Genia H3의 기존 fixed120은 F0 대비 평균 편향3.879→60.099, 평균 제거 RMSE51.519→67.868로 둘 다 악화했다. Holly H14는 편향0.375→0.158, 평균 제거 RMSE0.195→0.150으로 둘 다 개선했다. 같은 변경을 항상 허용/차단할 근거가 없다. 새 STD16 fits는 ZERO를 선택했다(primary1.087340); 이것은 후보 비교를 금지하는 진입 gate가 아니다. 미래 잔차 분해는 원인 증명이나 배포 입력이 아니다.

3. **반대 설명 [추정]** 작은 과거2/13창의 일관성 자체가 다음날에 전달되지 않거나, 단순 uniform 보존/OLS만으로 충분할 수 있다. 특히 서로 다른 운영일이 섞이면 분산 추정이 불안정하다.

4. **유일한 변경 [설계]** native quantile loss에 frozen F0 대비 normalized 출력 보존항을 더한다. 노출된 학습 창에서 F0 median 잔차를 하루 평균b_i와 평균 제거s_i로 나눈다. 각 성분의 평균 제곱 S와 평균 추정의 표본 변동 N을 계산하고 u=(N+eps)/(S+N+eps)로 precision을 정한다. D=21×24에 대해 w_b=u_b/c, w_s=u_s/c, c=(u_b+(D−1)u_s)/D. 후보 항은 w_b·mean(delta)^2+w_s·mean((delta−mean(delta))²). delta=(q_theta−q_F0)/native input scale. 모든 학습 창의 초기 native loss L0와 median 잔차 MSE E0로 lambda=L0/(E0+eps)를 정하고 고정한다. eps=1e-6만 새 수치 상수다. 추론은 기존 LoRA의 native 출력이며 후처리 선택기/다른 입력/추가 trainable은 없다. 세부 식은 MECHANISM_SPEC에 고정한다.

5. **대조 [설계]** STD=native rank1 LoRA. SIMPLE=같은 lambda와 같은 F0 통계 계산을 사용하되 w_b=w_s=1인 uniform 출력 보존 LoRA. 모두147,456 trainables, 같은 LR2개·budget5개·seed·순열. F0/AFFINE/SEASONAL24와 각방법 fixed120도 감사한다. 동일 평균 precision이므로 단순히 전체 penalty를 약하게 만든 비교를 피한다.

6. **선행 경계 [확인/추정]** L2-SP/Fisher는 초기 parameter 거리와 중요도 보존이며, TILDE-Q는 label과의 이동/형태 불변 task loss다. 여기서는 native task loss를 보존하고 target residual로 forecast의 두 subspace precision을 정한다. 비등방 정규화·편향/형상 분해 자체는 알려진 원리다. 정확히 같은 조합의 부재나 충분한 방법론 신규성은 아직 입증하지 못했으므로 사전 NOVELTY=UNRESOLVED. 성능이 좋아도 이를 자동 변경하지 않는다.

7. **예측/반증 [설계]** 반복성 있는 성분은 덜 억제하고 일관성 없는 성분은 더 보존해 uniform 대비 미래 오차를 줄일 것으로 기대한다. weights가 같으면 uniform과 동치다. 과거 잔차 신뢰도가 미래에 전달되지 않거나 SIMPLE이 같거나 좋으면 이번 변경의 추가 가치를 인정하지 않는다. 후보 교체0회, 결과 기반 상수 변경0회.

8. **예산/판정 [설계]** STD16 후 SIMPLE/CANDIDATE32, 선택 봉인 후 LOCKED6×H2×방법3×seed2=72, 전체120 fits/19,680updates. Smoke합계24 이내. 후보의 개발 부호와 무관하게 유효한 비교는 LOCKED로 진행한다. 계약의 primary, paired building bootstrap2000(seed61690), seed/H 조건·최강 고정 대조 및1% 기준을 그대로 적용한다. 실행/예측/신규성 상태를 분리한다.
