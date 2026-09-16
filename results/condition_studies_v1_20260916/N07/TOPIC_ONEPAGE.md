# N07 SPECTRAL


11. N07 SPECTRAL — 작은 크기와 예측 가능성을 분리한 학습
======================================================================

데이터: Electricity4채널, C336/H48.
문제: 큰 패턴에 비해 에너지는 작아도 과거로 예측 가능한 주파수성분을 적응이 놓치는가?
선행: Fredformer(KDD2024), MSFT(NeurIPS2025).
주파수강조/다중스케일학습 자체가 새 기여가 아니며 아래 고정가중 loss도 [미검증]이다.

채널별 TRAIN64 원점에서 H48 future의 unitary full FFT Z_k를 계산.
E_k=mean|Z_k|^2 (신호는TRAIN mu/sigma로표준화).
과거48/직전48의FFT에서 각k의실수·허수와상수5개로 future Z_k를ridge예측.
TRAIN64를시간순4block으로나누고 앞1block->다음block,앞2->다음,앞3->다음의
forward-chaining 예측을 모아 skill_k=1-MSE_pred/MSE_train_mean 계산.
각 fold 정규화와 평균은 그fold 앞부분만. 겹친label/context에 대한48slot purge 적용.
유효한 fold가부족하면 해당skill을0으로놓고그이유기록. E/V로보완하지않음.
skill은[0,1]로clip. DC와Nyquist는재가중대상에서제외.

가중치:
w_BASE=1.
w_ENERGY: positive-frequency E_k^-1/2을 mean비로정규화후[.25,4]clip.
w_PRED: E_k가positive frequency중앙값이하인bin에서
   u_k=skill_k*min(4,median(E)/(E_k+eps)), 그외u_k=0; w=1+u.
w_SHUFFLE: w_PRED의positive-frequency순서를고정seed로섞음.
각가중치를negative frequency로mirror,DC/Nyquist 원값1,
마지막에전체H개의mean이1이되도록정규화. real-valued signal에대칭가중치.
eps=1e-8*max(mean(E),1). 값과공식은TRAIN에서봉인.

4개 학습군:
G0 BASE / G1 ENERGY / G2 PRED / G3 SHUFFLE.
L(w)=.5*mean(e^2)+.5*mean_k(w_k*|FFT_ortho(e)_k|^2), e=(yhat-y)/sigma.
G0는Parseval에의해그냥MSE와동일하다. 'uniform Fourier loss'를별도5번째fit으로세지않음.
G2와G3는가중치multiset 동일; 어떤frequency에놓는지가다름.
가중치가모두1이면G2/G3=G0로alias. 가짜고유후보fit을실행하지않음.

주지표: 원래시간영역normalizedRMSE 전체.
부가: TRAIN에서고정한저에너지/예측가능bin의오차,큰성분손해,MAPE사용안함.
개선 주장은bin몇개오차만좋아진것으로성립하지않음.
G2의직접상대는G1/G3및G0.
추가CPU대조: 위ridge성분예측을시간영역으로역변환한예측; 사용정보동일.

강한정식선행 Fredformer/MSFT는이번결과와직접재현비교하지않음.
후속주제로남으면그누락을우선보고하고, 주파수bias전체가원인이라고확정하지않음.
필수검사: FFT Parseval, 실수역변환,weight평균/대칭,TRAIN-only skill,
E label변경으로weights불변, 일정한weight의별도학습미실행.


[설계] 전체 계약의 공통 조항도 적용. 실행/효과/단순 대안/신규성을 분리한다.
