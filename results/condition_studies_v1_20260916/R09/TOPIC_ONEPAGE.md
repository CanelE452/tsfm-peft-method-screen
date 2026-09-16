# R09 MIXED


13. R09 MIXED — 상세/집계 레이블이 섞일 때의 보존
======================================================================

데이터: Electricity4채널. C336h/H24h. origin은완료된24h block경계(index mod24=0).
이주제는월합전용옛실험의재현이아니다. 24h집계+일부상세라는식별가능성을보완한새모사다.
목표: 집계자료를추가활용하면서상세예측패턴손해를줄이는가?
선행: Temporal Disaggregation of Time Series(The R Journal2013), Denton/Chow-Lin 계열.
관측null space보존/투영이라는수학자체를신규성으로주장하지않음.

가장중요한정보계약:
숨긴하루의시간별값이,뒤training origin의context에그대로다시나오면안된다.
관측권한은window label별이아니라 전체절대시간block별로정의한다.

1) 전체시간축을24h block으로구분.
2) TRAIN64 forecast origin을값과무관하게선정하고,그64 target blocks중
   hash가작은16개만detailed,48개는aggregate-only로고정.
3) 그외TRAIN/V/E과거로사용할block은hash modulo4==0인1/4만detailed.
   단V/E의미래ground truth는scorer가보유하며모델에미리공개하지않음.
4) detailed block은완료시시간별값을사용가능.
   aggregate-only block은완료시24개합계또는평균만사용가능.
   입력에서는완료된그block의관측평균을24번반복하고observed-resolution mask추가.
5) role선정과통계는이허용관측에만의존. 원시세부값의std/주파수를selection에쓰지않음.
6) TRAIN sigma는허용된상세값+집계평균복원sequence에서계산.
7) V_SELECT는설계상상세validation labels32일을모델선택기에허용하는것이며,
   '상세검증정답도전혀없는환경'을해결했다고주장하지않음.
8) 미래의집계량을추론때주지않는다. 알고있는합계로TEST예측을맞춰주는행위금지.

관측loss:
상세target이면 L_fine=mean_h[((yhat-y)/sigma)^2].
aggregate-only이면 L_agg=((mean(yhat)-observed_mean)/sigma)^2.
관측평균계산후rawfine label은training packet에서삭제.
q0는같은허용context를받은F0 point prediction.
delta=(yhat-q0)/sigma.
P(delta)=각24h블록평균을repeat, N(delta)=delta-P(delta).
N은집계평균으로관측되지않는상세변화다.

5개학습군:
I0 FINE_ONLY: 관측상세16target만반복해512updates.
   I1..I4와정보량/고유label수가다르므로이비교는집계정보의추가효과가섞임.
   loss를0으로만들어512updates인척하지않고실제상세16개반복횟수를기록.
I1 MIXED: 위관측loss,64target각8회.
I2 IMPUTE: aggregate target의pseudo detail=q0+(observed_mean-mean(q0)).
   detailedtarget은실제label. 이pseudo값으로MSE. 원형preserving disaggregation 대조.
   pseudo는TRAIN한정, 정답이라고부르지않음. 새예측때aggregate를받지않음.
I3 UNIFORM: MIXED + .1*mean(delta^2), aggregate-only block에만벌점.
I4 NULL: MIXED + .1*mean(N(delta)^2), aggregate-only block에만벌점.
I3/I4의동일계수는동일gradient총량을보장하지않음. loss/grad기여를기록한다.
입력resolution mask는모든방법이같이받는다. 상세target의추가보존벌점없음.

[수학적 동일성 — 구현 전 반드시 확인]
I2의 pseudo detail을 y_tilde=q0+(observed_mean-mean(q0))로 쓰면,
  mean[((yhat-y_tilde)/sigma)^2] = L_agg + mean[N(delta)^2]
가 정확히 성립한다. 즉 I2는 I4의 NULL 계수를 1로 둔 것과 같은 목적함수다.
현재 I4의 계수 .1과 차이가 있으므로 학습은 구분되지만, I4가 I2보다 좋아진 것만으로
새로운 보존 원리가 생겼다고 해석하지 않는다. 같은 계열의 보존 강도 차이일 수 있다.
I4 대 I3는 관측 가능한 평균 변화까지 억제하는지의 대조이고,
이 역시 알려진 projection/shrinkage 원리의 제한된 비교다.
공통 데이터 eligibility에서 원시 TRAIN 표준편차를 확인하는 일반 규칙은 R09에서 적용하지 않는다.
R09는 관측 지도부터 만든 뒤 허용된 상세/집계 값만으로 eligibility와 통계를 계산한다.

직접비교: I4 vs I3/I2/I1. I0는기본관측정보기준.
primary=전체실제시간별normalizedRMSE.
필수secondary=24h평균/합계RMSE,상세패턴오차(mean제거),허용context상태별결과.
기대: aggregate학습의유용한수준변화는허용하면서세부패턴손해를줄임.
반례: I2의단순보존복원으로충분하면특별한loss필요성약함.
aggregate오차만개선되고시간별오차가나빠지면기본목표달성아님.

필수검사:
숨긴fine target에합계0인perturbation을주어도입력/통계/optimizer label/pseudo불변,
미래test aggregate가입력에포함되지않음,
N^2=N, P^2=P, mean(Ndelta)=0, ||delta||²=||Pdelta||²+||Ndelta||²,
상수offset은NULL벌점0이지만UNIFORM벌점>0,
Fine-only의반복횟수와집계군의unique labels차이를정직하게보고.


[설계] 전체 계약의 공통 조항도 적용. 실행/효과/단순 대안/신규성을 분리한다.
