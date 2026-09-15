# 같은 시간대 표현 집계의 입력 조건화 — 유한12-fit 개발 비교

2026-09-16. 기준980f9056eb41edf071eac0e185dbf77f2e8ba854. 원래 건물120-fit 계약과 최근 각4-fit 대조는 종료 상태로 보존한다. 사용자 활성 목표의 실패 후 후보 탐색 요청에 따라, 이번에는 한 adapter 변경과 두 단순 대조를 비교한다. 완료한 실험을 재실행하지 않는다.

## 질문·관찰·반례

균형 표본에서 LoRA+head는 head-only보다 전력5.794%/교통11.520% 평균 MSE가 낮았다. 교통의 추가 이득에는 하루 반복 잔차 감소가 포함됐다. 이는 표현 적응의 가치에 대한 개발 관찰이며 아래 연산이 효과적이라는 증거가 아니다. 단순 MLP adapter로도 충분하거나, 같은 시간대 평균이 충분하거나, 서로 다른 날의 집계가 이상 패턴을 지워 악화할 수 있다는 반례를 함께 둔다.

## 완전한 후보 명세

입력x는 기존 MOMENT normalizer를 거친96시간 context, p_i는8시간 비중첩 patch(i=0..11), h_i는 동결된 MOMENT 마지막 encoder의512차원 표현이다. A는16×512, B는512×16, bias없음. z_i=A h_i. 모든arm의 출력은 h'_i=h_i+B GELU(sum_j T_ij z_j). 기존 동일6144→96 head로 예측하고 동일 normalizer로 복원한다. A는PyTorch Linear 기본초기화(seed+202 독립 RNG), B=0. 초기 함수는 기존 head-only/LH와 정확히 같다. head와A/B만 학습(589920+16384=606304개), 기존 LoRA encoder는B=0인 초기값으로 전체 동결한다. 고정 LoRA forward의 비용은 남는다.

세 arm은 학습 파라미터·초기값·손실·모델·head가 같고 T만 다르다:
- POINTWISE: T=I. 일반 병렬 bottleneck residual adapter.
- UNIFORM: j-i가3의 배수일 때T_ij=1/4, 그 외0. 같은 상대 하루 시간대의4개 patch 평균.
- CONDITIONED(유일 후보): 같은mask에서 T_ij=exp(-d_ij)/sum_k exp(-d_ik), d_ij=mean_8((p_i-p_j)^2). softmax는FP32, FP64검사에서는FP64. 온도1, 하루24h(3patch), rank16을 사전 고정하며 튜닝하지 않는다. self가 포함돼 안전하게 정의된다. extra trainable gate없음. 반복되는 동일patch라면UNIFORM과 같고, 다른날 패턴이 다르면 가중치가 달라진다.

전체context가 예측 시점 이전이므로 context 내 양방향 집계는 미래정답을 쓰지 않는다. 채널마다독립적이며 ID·다른채널·달력시간·추가source·온라인E정답을 쓰지 않는다. 현재context 밖의 memory bank가 아니다. gaussian kernel attention/graph aggregation·residual bottleneck 자체는 알려진 원리다. 구체적 차이는 원시 동일위상 patch 유사도로 frozen 표현의 adapter branch를 조건화하는 배치이며 수학적 최초성을 선언하지 않는다.

## 고정 데이터·학습·예산

직전 balanced sampler와 모든96→96 창·split70/10/20·train 통계·32채널·MOMENT-small revision·원자료를 그대로 쓴다. electricity/traffic×seed41000/41001×3arms=최대12fits, 최대15240optimizerupdates. smoke2updates×2원천×3arms=최대12별도. 기존LP/LH 결과 재사용. 실패도attempt에 포함하고 재시도·후보교체·데이터교체·하이퍼파라미터 탐색없음.

AdamW lr.001 betas(.9,.999),eps1e-8,weight_decay0,clip1; StepLR5epochs gamma.5; BF16 batch8; cap20epochs; Vpatience5,min_delta1e-4; INIT포함V최저MSE. arm 간 epoch순열동일. headdropout.1과기존 train/eval 정책유지. 모델 smoke와경계검사통과시 성능입장gate없이12fits를실행한다.

12개V선택을 먼저봉인하고 노출된E를 평가한다. 주비교는각V선택, 추가비교는사전고정epoch(전력41000=1/41001=3, 교통둘=1)의동일updates다. fixedepoch가실제trajectory에없으면미실행기록하고학습연장없음. MSE/MAE/rawMAE·채널별원점수·실제파라미터·학습시간·메모리모두공개. 같은trajectorycheckpoint는fit중복집계없음.

## 사전 판단 정책

이E는DISCOVERY_REUSED_E이며 새로운독립SCREEN_PASS로판정하지않는다. 후보가POINTWISE와UNIFORM 각대조보다두source macro를각각1%이상개선하고두seed방향이모두양수이면구성요소개발신호로기록한다. 자원목표는LH대비trainables20%이상/실측학습peak50%이상감소와두source MSE악화1%이내를동시에만족하는지따로기록한다. 1%는이번후속투자용사전실용폭이며통계적동등성의증명이아니다. 결과를보고폭을넓히지않는다. 미충족은실패원인·불확실성·tradeoff로나누며기존FAIL재분류없음. 개발신호가있어도신규성검토·독립평가없이는전체새방법론목표완료가아니다. 추가후속학습분기없음.

## 검사·안전

FP64명시식출력/gradient최대차1e-10,초기identityexact,후보/대조가합법입력에서다를수있음,반복patch에서는uniform동치,채널순열등변검사. 실제BF16INIT V8예측과기존LHcacheexact. 2smoke후head/down/up실제변경·frozen/buffer보존·finite검사. 미래값poison학습창불변,모델/데이터/소스/과거결과hash보존,새모델선택checkpoint재생exact,전예측scalar MSE/MAE atol/rtol1e-12,원장/선택/정규화/원점교차검산. 기준을올려통과시키지않는다.

GPU하나/worker하나/기존lock. RustDesk만승인예외. 시작30초안정/free4GiB이상,실행중1GiB미만/외부compute는경계대기. 누적대기600초/controller7200초상한. GPU수치/환경오류는성능FAIL과분리한다. 준비오류수정은최대1회기록하며학습설정변경·오류fit교체는없다. 큰cache/weights/원자료는로컬에남기고코드·보고서·원점수·검산을검증후push한다.
