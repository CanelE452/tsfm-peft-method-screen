# N06 JOINT


10. N06 JOINT — 주변분포를 동일하게 유지한 하루 전체 위험
======================================================================

neural fits=0. 기존 예보 실험의 FROZEN_WEATHER S0 raw quantiles를 고정.
point MSE로 학습한 이번다른track의모델을확률모델로사용하지않는다.
타깃T0/T1/T2; V_CALIBRATE8월로 dependence만 fitting; E는 재사용개발평가.
목표: 같은시점별분포로도 다른시간결합이 하루총량/연속위험 예측에 영향을 주는가?

선행: TACTiS-2(ICLR2024),
Efficiently Generating Correlated Sample Paths from Multi-step TSFMs
(NeurIPS2025 시계열 foundation model workshop; 메인학회 아님).
copula·잔차rank 재배열 자체는 기존방법이다. 이실험은 알려진 대조의 유용성 확인.

각origin H24, sampleN=256.
각lead에 공통 u_j=(j+.5)/256을 원래 quantile함수 선형보간으로 매핑한 sorted256값을 만든다.
u가 .01보다작거나 .99보다큰부분은같은공통끝값으로clamp.
이는공통finite-tail 가정이다. 극단꼬리추정성과/무한지지확률의정확성을주장하지않음.
모든coupling은 각lead의 이256값의 순열만 바꾼다.
따라서 경험적주변분포와각lead의ensemble CRPS는정확히같아야한다.
독립적marginal quantile을개선했다고결론내릴수없는설계다.

V_CAL에서 진짜값의quantile CDF 위치u를 구함:
동일q값은대응tau의중간값으로병합,단조보간, u clip [.01,.99].
z=Phi_inverse(u). calibration 16경로뿐이므로 covariance불확실성 큼.

4 coupling:
F0 INDEPENDENT: lead별독립permutation.
F1 AR1: V_CAL z의인접lead lag1 correlation rho를 pooled추정,
   clip rho[-.95,.95], R_ij=rho^|i-j|.
F2 SHRUNK: V_CAL z의24x24상관행렬 S, R=.5I+.5S; 대각1재정규화.
   symmetrize+고정eigen floor1e-6후재정규화. 더좋은shrinkage탐색금지.
F3 EMPIRICAL: V_CAL의24차원PIT rank경로16개를반복표집해256template만들고
   각lead template순위로공통256예측값을재배열(Schaake-style대조).
F1/F2도 Gaussian R 표본 256개의 각 lead 순위로 공통 예측값을 재배열.
V_CAL에서 상수인 lead는 상관 추정이 정의되지 않으므로 해당 off-diagonal을 0,
diagonal을 1로 두고 그 위치를 보고한다. 상수/결측을 임의 작은 noise로 숨기지 않는다.
고정 seed 73260과 동일한 tie-break를 기록한다. empirical template의 동점은
별도 고정 난수 순위로 해소하되, E 정답을 이용하지 않는다. 256 표본이라는 Monte Carlo
근사의 한계도 보고하고, seed를 바꿔 유리한 샘플 경로를 고르지 않는다.

primary: 하루24시간합계의sample CRPS/(24*sigma_TRAIN).
CRPS_N=mean|s_i-y_sum| - (1/(2N^2))*sum_ij|s_i-s_j|.
부가: 전체경로energy score, p=.5 variogram score, 연속6h평균 최대값의초과확률 Brier.
임계값은TRAIN의rolling6h평균90분위수로만정함. E에서좋은threshold선택금지.
에너지는W의시간평균 x1h의합이면Wh이며 원자료단위를다시확인.
원래각시간의pinball/CRPS동일성표를필수로붙임.
전체 energy pair합은 256x256, 큰 배열은 origin 단위로 계산해 메모리를 제한한다.
energy score = mean_i ||s_i-y||_2 - .5*mean_ij ||s_i-s_j||_2 (각 값은 TRAIN sigma로 표준화).
variogram score = mean_{h<k} (|y_h-y_k|^.5 - mean_i|s_ih-s_ik|^.5)^2,
모든 h<k에 같은 가중치. Brier=(예측 사건확률-실제 사건0/1)^2.
이것들은 보조 지표이며 sum-CRPS보다 유리하게 나왔다고 primary를 교체하지 않는다.

필수검사: 각lead sorted samples exact동일, correlation PSD, quantile순서,
V_CAL/E분리, 두시간공동/교대toy에서동일주변분포·다른합계위험을재현.
분포결합의효과와marginal misspecification/finite-tail한계를구분.
N06는새PEFT를이미설계한것이아니다. 기존간단copula로충분하면그것을채택할근거다.


[설계] 전체 계약의 공통 조항도 적용. 실행/효과/단순 대안/신규성을 분리한다.
