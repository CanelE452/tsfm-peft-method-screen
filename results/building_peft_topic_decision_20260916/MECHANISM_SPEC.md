# 완전 명세 — target residual consistency로 보존 기하만 변경

[설계] 이 명세와 EXECUTION_CONTRACT를 함께 봉인한다. 후보1개, 상수 grid 추가0, native backbone/head/양 행렬 LR/rank/quantile task loss 변경0.

## 입력·통계

H는72 또는336개 유한한 관측치다. stride24, 각 x_i24→y_i24가 모두 H 안에 있는 n=2/13창만 사용한다. m.instance_norm(x_i)에서 얻은 **입력만의 native scale** sigma_i>0를 사용한다. y_i는 과거 supervision이며 평가 target이 아니다. F0의 raw native21분위수×24 예측 Q0_i, native loss ell0_i를 no-grad/LoRA disabled로 한 번 계산한다. 실제 config에서 tau=.5 위치를 찾는다.

r_ih=(Q0_i,.5,h−y_ih)/sigma_i, b_i=mean_h r_ih, s_ih=r_ih−b_i.

S_b=mean_i(b_i)^2; N_b=sum_i(b_i−mean_i b_i)^2/[n(n−1)].
S_s=mean_h(mean_i s_ih)^2; N_s=mean_h sum_i(s_ih−mean_i s_ih)^2/[n(n−1)].

n>=2이므로 sample variance가 정의된다. N은 독립 일별 표본 가정의 평균 분산식에서 착안한 operational 통계다. 자기상관·비정상 시계열에서 올바른 posterior variance/신뢰구간이라고 주장하지 않는다. Float64로 계산하고 모든 통계·계수는 detach/fixed다.

새 상수 eps=1e-6. 두 성분에 u_k=(N_k+eps)/(S_k+N_k+eps). eps는 두 신호가0인 합법 입력에서도 양의 유한 precision이 되게 하는 수치 바닥이며, train-only 무차원 standardized residual에 적용한다. D=Q×24=504, c=[u_b+(D−1)u_s]/D, w_b=u_b/c, w_s=u_s/c. trace([w_b P+w_s(I−P)])/D=1; P=11^T/D. 추가 clipping/요일 feature/learned gate 없음.

L0=mean_i ell0_i, E0=mean_ih r_ih², lambda=L0/(E0+eps). 초기 과거 예측의 오차 에너지 크기만큼 모델이 움직일 때 보존항이 대략 초기 native loss 크기가 되게 하는 **train-only 척도 보정**이다. 성능을 본 grid나 source 추정치가 아니다. SIMPLE/CANDIDATE에 같은 lambda를 적용한다. gain 지표의0분모에 eps를 넣는 정책과 무관하다.

## 학습 식

공통: theta는 rank1 LoRA A/B192 tensors,147456 scalars. B=0 초기화, alpha2, backbone/head frozen, AdamW/FP32 등 계약 동일. delta_i=(Qtheta_i−Q0_i)/sigma_i, Q는 **정렬 전 raw**21×24; c_i=mean_{q,h} delta_i.

STD: J_i=ell_native_i(theta).

SIMPLE: J_i=ell_native_i(theta)+lambda·mean_{q,h}(delta_i²).

CANDIDATE: J_i=ell_native_i(theta)+lambda·[w_b c_i²+w_s mean_{q,h}(delta_i−c_i)²].

세 군 모두 같은 창 stream을 반복한다. task loss는 모델이 반환하는 native loss 그대로(32위치 중 첫24 supervised mask, horizon mean/quantile sum 유지). 부가 보존항은 첫24 raw forecasts에만 적용한다. raw 모든분위수의 산술평균 c_i는 상수 방향 projector의 좌표일 뿐이며 **조건부 평균 예측**이라고 부르지 않는다. w_b는 공통 위치 변화, w_s는 나머지 시간형태·분위수 간 모양 변화를 제어한다.

## 동치와 반례

w_b=w_s=1이면 후보는 SIMPLE과 수식상 같다. lambda=0이면 두 보존군은 STD와 같다. 그 경우 결과에서 별도의 추가 가치를 주장하지 않고 동일설정 fit 재사용 여부를 기록한다. delta가 상수이고 w_b!=1이면 SIMPLE과 후보의 값·gradient는 달라진다. 초기 delta=0일 때는 둘 다 penalty/gradient0이므로 첫 step의 같음이 오류가 아니다.

## 추론·선택

추론 input은직전24h뿐이다. 96 LoRA가 붙은 native 모델의21×24 raw 출력, 공통 increasing quantile 정렬, 실제 .5 위치 median을 사용한다. 후보를 위한 외부 source/model/추론 gate/출력 mixture/OLS 추가 없음. Scalar 통계는 학습 중 고정하며 추론에 다시 맞추지 않는다.

각 방법의 LR{3e-5,1e-4}×budget{ZERO,EPOCH1,EPOCH4,EPOCH16,FIXED120}를 DISCOVERY seed61680에서 동일 기회로 선택한다. 全H/건물 global pair1개, tie=적은updates→작은LR. LOCKED는 각방법 선택LR에서 공통 최대trajectory를 실행하고 선택 checkpoint와 fixed120을 함께 저장한다. seed61681도 동일 recipe다. 선택 seal은 LOCKED target을 열기 전에 만든다. ZERO=F0로 학습 효과를 주장할 수 없다.

## 검사·비용

FP64 P행렬 식과 간단한 집계 구현의 출력/gradient atol1e-12/rtol1e-12. 합법적인 일정 delta 반례, 동일 precision 동치, trace1, zero gradient 검사. 실모델은 기존 STD_PROTOCOL의 동일-shape FP32 오차 기준, frozen/finite/parameter count/step0 identity/새 모델 복원/합성 평가-target poison 반복 확인. Native task loss에서 A 첫 gradient0은 정상이다.

실제 후보 smoke: SIMPLE/CANDIDATE 각2updates×dummy future2조건=8updates; STD smoke2와 합계10, 전체24 이내. Dummy future는 H와 별도이며 학습 함수에 전달되지 않는다. 같은 H/seed에서 final trainable hash와 forecast가 같아야 한다. LOCKED target은 smoke에서 읽지 않는다.

F0 통계 계산은 추가 neural fit이 아니지만 inference/통계 비용을 fit wall과 calibration 기록에 포함한다. trainable 수가 같아도 속도/메모리는 실제 측정한다. 학습 뒤 새로운 상수·precision 규칙을 추가하지 않는다.
