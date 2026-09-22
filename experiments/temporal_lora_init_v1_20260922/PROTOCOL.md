# 시간 의존성 기반 LoRA 초기화: 실행 전 고정 프로토콜

사용자 승인: 앞서 설명한 시간 의존성 초기화 후보를 실제 실험한다. 목적은 내일까지 후속 투자할 TSFM PEFT 후보의 선택이다. 신규성·논문 PASS를 선언하지 않는다. 기존 실험은 수정·재개하지 않는다.

## 가설과 선택 근거

시간차 공분산의 방향으로 LoRA를 초기화하면 무작위 또는 단순 분산 방향보다 짧은 적응 예산에서 유리할 수 있다. 추가 모듈을 붙이지 않고 초기화만 바꿔 원인을 분리한다. 초기화 준비 비용도 계산한다.

- Backbone: amazon/chronos-bolt-small revision 772f3d25d38aec6d914c8949dab4462e2d46f5d8. 검증된 Python3.11/CUDA 환경과 가중치만 재사용한다.
- 자료: 공식 원자료와 검증 캐시의 SHA를 다시 확인한다. Electricity는 이전 TRAIN-only hash 순서32열 중 첫8열, ETTh1은7열 전체. 기존에 사용한 자료이므로 독립 확증이 아닌 개발 screen이다. 성능에 따라 자료·계열을 바꾸지 않는다.
- Context256 / horizon64: native 단일 예측으로 rollout의 영향을 배제한다. TRAIN/CAL/VAL/TEST는 시간순60/10/10/20%. TRAIN은 모든 합법 hourly origin, 다른 구간은64슬롯 간격으로 target 비중첩. 모든 target이 해당 구간 안에 있어야 한다. sigma는 TRAIN만 사용한다.
- Seeds92341,92342. AdamW lr1e-4, betas(.9,.999), eps1e-8, wd0, clip1, batch8. 모든 q/v LoRA rank8, alpha16, dropout0. FP32,TF32off, eval dropout. 학습 파라미터294912개 동일.
- 예산: 2자료×4군×2seed=16 main fits, 각512 updates, 총8192. 실제모델 smoke는4군×2=8 updates 상한. 실패 update도 포함하며 자동 재학습·LR/rank/seed/lag 탐색은0이다.
- 손실/지표: .1부터 .9까지 native quantile의 mean twice-pinball / TRAIN sigma. 계열 동일가중. median nMAE, raw MAE,80%포함률/폭, quantile crossing도 보고한다. exact CRPS라고 부르지 않는다.

## 초기화: RANDOM / PCA / TEMP / SHUFFLE

RANDOM은 표준 PEFT Kaiming A와 zero B이다. PCA는 분산 방향, TEMP는 시간차 방향, SHUFFLE는 시간 짝만 섞은 대조이다. 모든 군은 이후 A와 B를 모두 학습한다.

TRAIN 입력256contexts를 계열별 균형 추출한다. 같은 계열의 초기화 context끼리는 겹치지 않도록256간격 block을 사용한다. target은 초기화에 사용하지 않는다. Frozen F0 encoder6층 SelfAttention.q의 입력 activation을 수집하고 q/v 입력 일치를 실제 확인한다. 16개 non-overlapping16시간 patch만 사용하며 마지막 REG토큰은 제외한다.

각 window에서 token 평균을 빼 window 공통 수준을 제거한다. PCA는 C0=평균 Z_t Z_t^T, TEMP는 C1=평균(Z_t Z_(t+1)^T + 전치)/2의 상위8 대수적 고유벡터를 사용한다. SHUFFLE는 각 window의15개 right endpoint 순서만 고정 RNG92340으로 섞어 같은 C1을 계산한다. 입력량과 endpoint 주변분포는 유지한다. 인접16시간 patch 관계를 시험하는 한 가지 구현이며 계절성 전체를 모델링한다는 주장은 하지 않는다. Encoder가 양방향인 공통 특성도 남아 있으므로 시간 관계의 인과 해석은 제한적이다.

Encoder 각층 q/v의 A만 해당 basis로 초기화한다. 각 row norm은 같은 seed RANDOM A의 norm을 복사한다. B=0으로 초기 예측은 native와 일치해야 한다. Decoder와 cross-attention 초기 LoRA는 모든 군에 같은 seed RANDOM을 유지한다. 단일 decoder query에는 시간 lag가 정의되지 않기 때문이다. PCA는 EVA 전체 구현이나 rank 재배분 재현이 아닌 matched fixed-rank 대조이다. 고유벡터 부호는 최대 절대 좌표가 양수가 되게 고정한다.

## 학습·봉인·보정

동일 seed의 네 군은 같은 사전 생성 training packet을 사용한다. Checkpoint0/32/128/256/512를 저장하고 모든 fit512까지 수행한다. 짧은 학습의 primary는 고정128이며 selected는 raw VAL 최소, 동점 이른 checkpoint이다. 모델·CAL·baseline을 TEST 전에 봉인하고 모든 TEST 예측을 저장한 다음 채점한다.

자료별 primary baseline은 RANDOM/PCA의 두 seed 평균 raw VAL128이 작은 군이다. selected checkpoint에만 공통 CAL을 제공한다: median + beta*sigma + alpha*(q-median). alpha=[.5,.75,1,1.25,1.5,2,3], beta=[-.5,-.25,0,.25,.5]. CAL pinball 최소, 동점 identity squared distance→alpha→abs(beta)→signed beta. F0도 같은 보정 기회를 받는다. 보정 selected는 보조 비교이며 raw128 primary를 대체하지 않는다.

## 비용과 판정

초기화 activation 수집·전송, 공분산/eigensolve,A설정 시간을 기록한다. 공통 activation을 한 번 수집해 실험을 절약하더라도 PCA/TEMP/SHUFFLE의 개별 사용 비용에는 전체 수집 시간을 각각 청구한다. RANDOM에는 필요 없는 비용을 넣지 않는다. 각 방법에 필요한 공분산만 계산한다. 공통 자료 읽기·모델 로딩·VAL·checkpoint/장부 I/O는 적응 compute와 분리하고 wall time도 공개한다.

각 seed RANDOM512 raw VAL score×1.01에 처음 도달한 checkpoint 시간을 비교한다. step0부터 이미 통과하면 NO_ADAPTATION_HEADROOM으로 속도 우위를 주장하지 않는다. 미도달은 censored이며 목표를 바꾸지 않는다. 적응 시간은 초기화+입력전송/forward/backward/optimizer의 누적 시간이다. 준비 비용을 제외한 update 수 우위만으로 효율 성공을 선언하지 않는다.

두 자료 각각 TEST128에서 TEMP가 V고정 baseline과 SHUFFLE보다 평균0.5% 이상 좋고 두 seed 모두 양성이면 mechanism signal이다. 여기에 자료별 두 seed 평균 time-to-target이 RANDOM보다20% 이상 짧고 selected+CAL TEST가 baseline보다1% 넘게 나쁘지 않으면 GO_EFFICIENCY_SCREEN이다. 품질 조건만 통과하면 MECHANISM_ONLY_COST_NOT_MET, 기준선/SHUFFLE 대비 평균효과가 비양수이면 NO_GO_CURRENT_RECIPE, 작은 양수·seed 불일치는 HOLD이다. 숫자는 이 개발 screen의 투자 기준이며 보편적인 학술 기준은 아니다.

개선율은 두 seed 원점수 평균의 비율로 계산하며 자료 간 임의 합산은 하지 않는다. Bootstrap은2000회, 비순환 연속7origins(64h간격),계열과 두 seed를 공동 보존,RNG92343이다. seed 모집단 불확실성 추정으로 포장하지 않는다.

## 검증·종료·선행

native parity,zero-B,gradient,저장/복원,frozen backbone/buffer,공통 sampling,시간 짝 조작을 main 전에 검증한다. main 전 source/hash/자료/설정을 봉인한다. 단일 executor와 intent/commit 장부를 사용하고 partialfit 자동 replay는 금지한다. 자료/구현/resource 문제와 과학적 부정 결과는 구분한다. 결과표·학습곡선·비용·그림·REPORT_KO·판정·manifest·검산을 scoped commit/push하고 자동 후속 없이 종료한다.

EVA https://arxiv.org/abs/2410.07170 (NeurIPS2025)는 activation 분산 초기화, CorDA https://arxiv.org/abs/2406.05223 (NeurIPS2024)는 context covariance의 선행 근거다. 시간차 방향이 더 좋다는 것과 신규성은 미확인이다. 별도 후보인 sampling-rate adapter는 시간축 검증 부담이 커 이번에 실행하지 않는다. 현재 구현이 실패해도 모든 시간 기반 PEFT의 반증으로 쓰지 않는다.
