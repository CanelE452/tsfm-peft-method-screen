# R08 LEAD


12. R08 LEAD — 다른 채널의 선행 정보가 관측된 예측 구간
======================================================================

데이터: Traffic4채널, C336/H48.
문제: 선행채널의관측을직접쓸수있는초기구간과,그채널도예측해야하는뒷구간을
다르게조정하는것이단순시차정렬·일반horizon조건보다유용한가?
선행: LIFT(ICLR2024). 동적선행지표/지연정렬을처음제안한다고쓰지않는다.
이전Gaussian정렬실험의단순회귀우위는보존. 실제Traffic으로바꾼새조건이다.

TRAIN만으로:
타깃c마다다른채널j, lag l=1..24의corr(z_c[t],z_j[t-l])를계산.
같은donor의최대절대corr lag를선정하고,서로다른상위2donor선정.
동률donor index작은것/lag작은것. TRAIN finite공통시점만.
고정donor/lag에ridge(상수포함) 회귀 g_c=b_c+sum_j beta_cj*z_j[t-l_j].
alpha .1/1/10의TRAIN forward-chaining fold만사용. 시험용합성비선형추가금지.

미래h=0..47에서:
s_j(h)=이미관측한 z_j[o+h-l_j] (h<l_j),
       현재model이예측한 donor j의 zhat_j[h-l_j] (h>=l_j).
절대시각으로조건검사. donor의실제미래정답을사용하지않는다.
g_c(h)=b_c+sum beta*s_j(h).
a_c(h)=sum |beta|*1[h<lag]/max(sum|beta|,1e-8).
base point zhat_c에 delta_c=g_c-zhat_c를더하는방식으로정의.
순환참조 방지: 모든g는보정전base zhat에서동시에계산,다른보정g를입력으로사용하지않음.

4학습군(모두동일LoRA기본point MSE):
H0 BASE: 보정0.
H1 STATIC: zhat+tanh(theta_c)*delta. theta0 초기.
H2 LINEAR_H: zhat+[(1-h/47)tanh(u_c)+(h/47)tanh(v_c)]*delta.
H3 LEAD_H: zhat+[a_c(h)tanh(u_c)+(1-a_c(h))tanh(v_c)]*delta.
H2/H3는동일8개scalar; 차이는선행정보의실제가용성반영여부.
H0/H1도같은donor관측과시간을접근가능하게한다. 표준model은원래모든4채널입력.
추가CPU기준: frozen모델과고정ridge로g만만든예측, H0/g의전역blend(V에서alpha선택).

주지표: 전체H48normalizedRMSE. h<lag정보가용비율구간은부가표.
기대: H3가H2와단순blend를넘을때 '실제가용구간에따른조정'의개발근거.
반례: H1/단순ridge/LIFT형구성으로충분 -> 새PEFT기여미확보.
선행정보가약한4채널에서결과가없으면그자료의한계로기록, E보고donor채널교체금지.
후속정식비교LIFT가남아있음을표시. H3를LIFT보다새롭다고전제하지않음.

필수검사: 모든관측donor timestamp<o, predicted donor index>=0,<H,
g루프의순환없음,초기모든학습군=base, h가시간조건을가진toy의정답정렬,
학습된g제거진단만으로처음부터g가없는H0보다좋다고결론내리지않음.


[설계] 전체 계약의 공통 조항도 적용. 실행/효과/단순 대안/신규성을 분리한다.
