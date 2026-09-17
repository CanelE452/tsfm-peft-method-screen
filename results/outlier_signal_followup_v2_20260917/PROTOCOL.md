입력 오류 강건성 후속 — 학습 노출 진단 + 지속성 인식 적응의 제한 비교
버전: outlier_signal_followup_v2_20260917
대상 저장소: CanelE452/tsfm-peft-method-screen
작성 기준 commit: e2b1ea4a050021740a64c58c50bc8d02b156b06e

======================================================================
0. 이번 작업의 목적
======================================================================

이번 작업은 outlier_signal_peft_v1의 A5를 성능이 나올 때까지 튜닝하는 작업이 아니다.

직전 결과에서 확인된 사실:
- A1 RAW/AUG LoRA는 Electricity에서 A0 대비 FAULT 평균 +1.2523%,
  REFERENCE +0.5019% 개선을 보였고 SHIFT에는 -0.5415% 손해가 있었다.
- A5 CLIP_RESIDUAL은 같은 크기의 A4 CLIP_GENERIC보다 일부 조건에서 개선했다.
  Electricity SHIFT 평균 +3.7382%, ETTm1 FAULT +0.0454%, SHIFT +0.0362%.
- 그러나 A5는 A1 대비 SHIFT 평균 오차가
  Electricity +103.7594%, ETTm1 +79.5626% 컸다.
- 큰 손해는 SHIFT8에서 특히 컸다.
- 기존 TRAIN의 persistent shift amplitude는 4,
  평가에는 shift amplitude 4와 8이 있다.
- hard clip은 observed median ±6*robust_scale을 사용한다.
- A5는 hard clip 후 제거된 차이를 bounded embedding residual로 전달한다.

따라서 이번의 질문은 두 개다.

Q1. 기존 A5가 실패한 원인의 하나가
    "TRAIN에서 clip되어 복원해야 하는 persistent shift를 충분히 경험하지 못한 것"인가?

Q2. 큰 값을 무조건 clip한 뒤 복원하는 대신,
    "짧게 튀는 극단값과 일정 기간 지속되는 변화"를 관측된 과거의 지속성으로 구분하여
    필요한 경우에만 원래 값을 보존하는 작은 적응이
    raw augmentation LoRA보다 추가 가치를 주는가?

최종 목적:
- failure mechanism을 확인한다.
- 하나의 새 후보만 직접 비교한다.
- 단순 대안으로 충분하면 그 사실을 인정하고 종료한다.

이번에 하지 않는 것:
- A5 rank 확대, adapter 폭 확대, LR 무제한 탐색.
- clip threshold grid search.
- 새로운 backbone/dataset을 성능이 나올 때까지 추가.
- rollout, sensor delay, retrieval, PRIOR, history compression 등 다른 주제 재개.
- 기존 E 결과에 맞춰 방법 수식 변경.
- 실제 오류/사건 레이블이 있다고 주장.
- 성공할 때까지 v3/v4 자동 생성.

======================================================================
1. 저장소와 기존 결과 감사
======================================================================

시작 즉시 최신 HEAD/dirty state를 확인한다.

기준:
e2b1ea4a050021740a64c58c50bc8d02b156b06e

반드시 읽고 hash를 기록:
- results/outlier_signal_peft_v1_20260917/REPORT.md
- results/outlier_signal_peft_v1_20260917/FINAL_DECISION.md
- results/outlier_signal_peft_v1_20260917/verification.json
- results/outlier_signal_peft_v1_20260917/origin_audit.json
- results/outlier_signal_peft_v1_20260917/MODEL_SELECTION.json
- results/outlier_signal_peft_v1_20260917/LR_SELECTION.json
- results/outlier_signal_peft_v1_20260917/scores_by_condition.csv
- results/outlier_signal_peft_v1_20260917/scores_by_origin.csv
- experiments/outlier_signal_peft_v1_20260917/reference_core.py
- experiments/outlier_signal_peft_v1_20260917/model.py
- experiments/outlier_signal_peft_v1_20260917/train.py
- experiments/outlier_signal_peft_v1_20260917/evaluate.py

기존 결과는 read-only.
덮어쓰지 않는다.

새 경로:
experiments/outlier_signal_followup_v2_20260917/
results/outlier_signal_followup_v2_20260917/
.cache/outlier_signal_followup_v2_20260917/
scripts/run_outlier_signal_followup.py

======================================================================
2. 연구 주장과 해석 경계
======================================================================

사용할 수 있는 표현:
- synthetic measurement fault
- synthetic persistent shift
- unmodified reference
- persistence-aware attenuation
- development evidence

사용 금지:
- 실제 센서 고장
- 실제 regime change
- 실제 사건 보호
- anomaly detector
- TATO를 이겼다
- 새로운 독립 test
- 논문 PASS
- 모든 outlier robustness의 해결

이번 synthetic 조건에서
measurement fault와 persistent shift는 generator가 알고 있지만
generator metadata는 모델 입력으로 주지 않는다.

모델은 오직 관측된 과거 x_obs만 사용한다.

======================================================================
3. Phase A — 새 학습 0회의 failure-mechanism 진단
======================================================================

가장 먼저 기존 v1의 실제 TRAIN/V/E 입력과 selected A5 checkpoint를 사용해
zero-training 진단을 수행한다.

optimizer update = 0.

3.1 기존 TRAIN에서 실제 clip exposure 측정

source별, state별로 아래를 계산한다.

TRAIN state:
REFERENCE
POINT
BURST
SHIFT

단, generator audit를 이용해 POINT/BURST severity와 SHIFT duration/delta를 분리 보고한다.

각 example에서:
- robust median
- robust scale r_observed
- clip lower/upper
- number of clipped points
- fraction clipped
- clipped absolute mass = sum |x_obs-x_clip|
- normalized clipped mass / (512*r_observed)
- maximum discarded magnitude / r_observed
- longest consecutive clipped run
- number of sign-consistent clipped points in last 64 slots

SHIFT에 대해서만 generator audit를 분석용으로 이용해:
- shift-affected positions count
- 그중 clipped count
- shift positions clipped fraction
- nonzero residual example fraction
- residual mass / |true synthetic delta|
를 계산한다.

generator의 delta/상태는 이 진단 파일에만 존재.
model packet에 추가하지 않는다.

3.2 V/E severity별 clip exposure

기존 V/E의:
POINT4/8/16
BURST4/8/16
SHIFT4/8
SHIFT_POINT
를 동일 방식으로 분석.

특히:
- TRAIN SHIFT와 E SHIFT4/SHIFT8의 clip exposure 비교
- TRAIN measurement fault와 E fault의 residual magnitude 비교
- SHIFT4와 SHIFT8에서 A5가 실제로 받은 discarded feature norm 비교

3.3 기존 selected A5의 실제 adapter 사용량

selected A5 checkpoints에서 V만 사용해:
- adapter delta RMS / native patch embedding RMS
- patch별 adapter norm
- scenario별 adapter norm
- discarded feature norm
- adapter norm vs discarded norm correlation

E에서는 이미 본 개발 결과를 descriptive audit로만 사용할 수 있다.
새 방법의 선택에 E를 쓰지 않는다.

기존 residual-zero 진단과 연결해:
- A5가 residual feature를 사용했는가?
- 사용량이 SHIFT8에서 커졌는가?
- TRAIN에서 같은 범위를 경험했는가?
를 보고한다.

3.4 진단 태그

성능 gate가 아니다.
후속 비교는 정상 구현이면 모두 실행한다.

아래 descriptive tag만 기록:
TRAIN_EXPOSURE_GAP:
- TRAIN persistent shift에서 nonzero residual exposure가
  E SHIFT8보다 명확히 작거나,
- TRAIN에서 clipped persistent-shift example이 희소함.

NO_EXPOSURE_GAP:
- TRAIN에서도 persistent shift residual이 충분히 존재.

HARD_CLIP_INFORMATION_LOSS:
- SHIFT에서 clipped mass가 크고,
  raw A1 대비 clip 계열의 손해와 방향이 일치.

ADAPTER_UNDERUTILIZED:
- residual feature는 큰데 adapter output이 거의 0.

태그 문턱을 결과를 보고 새로 만들지 않는다.
정량값을 우선 보고하고 태그는 설명용이다.

산출물:
DIAGNOSTIC_REPORT.md
train_clip_exposure.csv
eval_clip_exposure.csv
adapter_usage.csv
failure_hypothesis.json

======================================================================
4. Phase B — 학습 분포를 맞춘 고정 비교
======================================================================

이번 새 학습에서는 "큰 값의 크기"만으로
fault와 persistent shift를 쉽게 구분하지 못하게 한다.

모든 arm이 같은 base origins, same labels, same synthetic draws를 사용.

source:
- electricity
- ETTm1

backbone:
- amazon/chronos-bolt-small
- 기존 v1과 동일 pinned snapshot/revision
- FP32
- TF32 off
- dropout 0

input:
context 512
horizon 64

channels:
v1과 동일 source별 첫 4개 적격 채널.
성능을 보고 채널 교체 금지.

origins:
v1의 sealed origin 목록을 그대로 재사용.
- TRAIN 256 distinct days
- V 64 distinct days
- E 128 distinct days
새 origin 재선정 금지.

E는 재사용 development period임을 명시.

======================================================================
5. 새 TRAIN augmentation — magnitude overlap을 명시적으로 만든다
======================================================================

32 epochs.
각 base example은 4 state를 각 8회 경험.

state schedule:
REFERENCE
POINT
BURST
SHIFT

각 state의 8회 exposure 안에서 severity는 deterministic cycle.

POINT:
amplitude [4,8,16,4,8,16,4,8] * r0
count [1,2,4]도 deterministic/hash fixed cycle.

BURST:
amplitude [4,8,16,4,8,16,4,8] * r0
duration [2,4] deterministic balanced.

SHIFT:
amplitude [4,8,4,8,4,8,4,8] * r0
duration [24,48,24,48,48,24,48,24]
sign fixed by predeclared hash RNG.

REFERENCE:
unchanged.

중요:
- measurement fault는 x만 변형, y unchanged.
- persistent shift는 recent x와 future y에 같은 delta.
- 동일 example/state/exposure에서 모든 arms에 정확히 같은 transform.
- A0 같은 no-fault arm을 이번에는 두지 않는다.
  이번 질문은 augmentation 필요성 자체가 아니라
  matched training에서 preprocessing/candidate의 추가 가치이기 때문이다.
- 기존 A1/A0 결과는 역사적 참고만 하고 새 실험 원점수와 합치지 않는다.

TRAIN generator key/sequence를 protocol에서 먼저 seal.
E/V generator는 기존 v1과 같은 sealed arrays를 재사용한다.
평가 조건을 바꾸지 않는다.

======================================================================
6. 비교군 — 6개만 고정
======================================================================

B0 RAW_AUG
- hard clip 없음.
- 위 matched TRAIN augmentation 전체.
- 표준 rank8 q/v LoRA.
- 가장 강한 기본 기준.

B1 HARD_CLIP
- 기존 clip6.
- matched TRAIN augmentation.
- 표준 LoRA.
- hard clip 자체의 손익.

B2 CLIP_RESIDUAL
- 기존 A5 구조를 그대로 사용.
- clip6 + discarded residual bounded adapter.
- 단, matched TRAIN augmentation만 달라짐.
- "기존 실패가 training exposure 문제였는가" 확인.

B3 PERSIST_FIXED
- 학습 파라미터는 B0/B1과 같은 LoRA만.
- 아래 persistence-aware soft attenuation은 고정 수식.
- 단순 방법으로 충분한지 확인.

B4 PERSIST_MAG
- B3와 같은 soft attenuation 구조.
- restore gate가 magnitude 정보만 사용.
- persistence feature는 사용하지 않음.
- B5의 일반 learnable gate 대조.

B5 PERSIST_LEARNED
- 이번 유일한 새 후보.
- B4와 같은 trainable parameter count.
- magnitude + temporal persistence 정보를 이용해
  raw와 clipped 사이를 soft하게 선택.
- LoRA는 나머지와 동일.

새 arm 자동 추가 금지.

======================================================================
7. persistence-aware soft attenuation의 정확한 정의
======================================================================

목표:
극단값을 무조건 삭제하지 않고,
관측 과거에서 "같은 방향의 큰 변화가 일정 기간 지속되는가"를 이용해
raw 정보 보존량을 조절한다.

7.1 공통 robust coordinates

각 series x_obs:
m = median(x_obs)
r = max(1.4826*median(|x_obs-m|), 0.1*sigma_train)

hard clipped:
x_clip = clamp(x_obs, m-6r, m+6r)

excess:
e_t = |x_obs_t-x_clip_t| / r

signed deviation:
d_t = (x_obs_t-m)/r

7.2 persistence feature

오직 과거 context 안에서 계산.
generator metadata 사용 금지.

extreme indicator:
I_t = 1[ |d_t| > 3 ]

same-sign indicator relative to t:
S_{t,k} = 1[ I_k=1 and sign(d_k)=sign(d_t) ]

trailing window W=8 slots.

p_t =
if I_t=0: 0
else mean_{k=max(0,t-7)..t} S_{t,k}

즉:
- 고립된 extreme이면 p가 작음.
- 같은 방향 extreme가 최근 여러 점 이어지면 p가 큼.

이 feature의 물리 시간은 source마다 다르므로
"8시간 persistence"라고 부르지 않는다.
8 observation slots라고 보고.

7.3 B3 PERSIST_FIXED

restore fraction:
g_t = p_t

input:
x_eff = x_clip + g_t*(x_obs-x_clip)

따라서:
- p=0 -> hard clip
- p=1 -> raw value
- 중간 -> soft restoration

trainable preprocessing parameter 0개.

7.4 B4 PERSIST_MAG

persistence를 쓰지 않는 parameter-matched learnable control.

feature:
z1 = log1p(e_t)
z2 = I_t
z3 = 1

logit restore:
L_t = a0 + a1*z1 + a2*z2

g_t = sigmoid(L_t)

parameters a0,a1,a2 = 3 scalars per source/model.
초기화:
a0=-4, a1=0, a2=0
초기에는 대부분 hard clip에 가깝게 시작.

7.5 B5 PERSIST_LEARNED

feature:
z1 = log1p(e_t)
z2 = p_t
z3 = I_t

L_t = b0 + b1*z1 + b2*z2 + b3*(z1*z2)

g_t = sigmoid(L_t)

parameters b0,b1,b2,b3 = 4 scalars.

parameter count가 B4와 다르므로
B4에 dummy scalar 1개를 trainable-but-zero-effect로 추가하지 않는다.
가짜 parameter matching 금지.

대신 보고서에:
B4 extra=3,
B5 extra=4라고 정확히 기록.

B4/B5의 차이는 단순 용량 1 parameter도 포함하므로,
B5>B4만으로 persistence의 순수 효과라고 주장하지 않는다.

이를 보완하는 zero-training ablation:
선택된 B5의 V에서 p_t를 0 또는 deterministic permutation으로 변경.
optimizer0.
이 진단은 의존성 근거이며 B4보다 우월성 증거가 아니다.

7.6 왜 embedding residual이 아니라 input soft restoration인가

이번 후속의 설계상 핵심:
삭제한 정보를 작은 embedding에서 다시 복원하지 않고
원래 수치 좌표에서 필요한 만큼 보존한다.

native Chronos-Bolt instance normalization은 x_eff에서 계산.
prediction inverse도 동일 loc/scale 사용.

즉 transform/inverse pair의 내부 일관성을 유지한다.

======================================================================
8. 정보 권한과 leakage 검사
======================================================================

B3/B4/B5 preprocessing은 다음만 볼 수 있다:
- 현재 example의 observed context x_obs
- TRAIN population sigma

볼 수 없음:
- generator state 이름
- true shift amplitude
- injected fault mask
- original clean x0
- future y
- future offset
- evaluation condition

동일 x_obs에 서로 다른 hypothetical y를 넣어도
x_eff/gate/model output이 bitwise 또는 사전 허용오차 내 동일해야 함.

persistent shift의 ground-truth type을 gate label로 지도하지 않는다.
gate는 end-to-end forecasting loss로만 학습.

======================================================================
9. 학습 파라미터
======================================================================

모든 arm:
Chronos-Bolt-small q/v LoRA:
- encoder self-attention
- decoder self-attention
- decoder cross-attention
- q,v only
- rank8
- alpha16
- dropout0
- bias train 없음

예상 LoRA parameter:
294,912
실제 모듈 이름/shape/count seal.

B0/B1/B3:
LoRA only.

B2:
LoRA + 기존 residual adapter 4,872.

B4:
LoRA + 3 gate scalars.

B5:
LoRA + 4 gate scalars.

원래 backbone/head/input embedding은 동결.

======================================================================
10. loss와 optimizer
======================================================================

기존 v1과 동일:
TRAIN raw population sigma로 normalized 2-pinball.

J =
mean 2*rho_q(y-pred) / sigma_train

native q=.1...9.
prediction은 원단위 inverse 후 loss 계산.

AdamW:
betas(.9,.999)
eps1e-8
weight_decay0
clip_norm1
scheduler 없음.

LR 후보:
1e-4
3e-4

gate scalar에도 같은 optimizer/LR 적용.
별도 gate LR 추가 금지.

======================================================================
11. 학습 budget
======================================================================

source2 × arm6.

선택 seed:
81550
arm마다 LR2개.

반복 seed:
81551, 81552
선택된 LR만.

fit 수:
2 source × 6 arm × (2 selection LR + 2 repeat seeds)
= 48 fits

각 fit:
32 epochs
1024 updates

main updates:
49,152

smoke:
2 source × 6 arm × 2 updates
=24 updates

hard cap:
49,176 optimizer updates

새 seed/LR/epoch 추가 금지.

checkpoint:
0,256,512,768,1024

======================================================================
12. validation selection
======================================================================

이번에는 TRAIN이 SHIFT4와 SHIFT8을 모두 포함하므로
V selection도 큰 지속 변화의 손해를 숨기지 않는다.

V objective:
동일 비중의 5개 condition
REFERENCE
POINT8
BURST8
SHIFT4
SHIFT8

각 condition nMAE -> 평균.

실제 배포 빈도라고 주장하지 않는다.
tradeoff development objective다.

tie:
smaller LR
earlier checkpoint

반복 seed에서도 같은 V objective로 checkpoint 선택.

모든 LR/checkpoint selection seal 후 E scoring.

E를 보고 selection 변경 금지.

======================================================================
13. evaluation
======================================================================

기존 v1 E origins와 synthetic condition arrays를 그대로 재사용.

조건:
REFERENCE
POINT4/8/16
BURST4/8/16
SHIFT4/8
SHIFT_POINT

E synthetic draws도 그대로.

주 지표:
nMAE = |median prediction-y| / TRAIN sigma

보조:
raw MAE
channel NRMSE
TRAIN-scale 2-pinball
quantile crossing
resource

패널:
FAULT = POINT/BURST six conditions equal weight
REFERENCE
SHIFT4
SHIFT8
SHIFT = mean SHIFT4/8
SHIFT_POINT
HISTORY_TRIGGERED_SUBSET는 기존 방식으로 참고만.

======================================================================
14. 필수 직접 비교
======================================================================

가장 중요:
B5 vs B0
- 새 persistence-aware learnable transform이
  raw augmentation LoRA보다 정말 필요한가?

B5 vs B3
- learnable gate가 단순 persistence rule보다 필요한가?

B5 vs B4
- persistence feature를 추가한 learnable gate의 가치가 있는가?
  단, extra scalar 1개 차이 명시.

B5 vs B2
- input-space selective restoration이
  기존 embedding residual보다 좋은가?

B3 vs B0
- 단순 persistence-aware preprocessing만으로 충분한가?

B2 vs B0
- matched severity가 기존 A5 구조를 살렸는가?

B1 vs B0
- hard clip 자체의 손익.

추가:
기존 v1 A1/A5 결과는 historical reference.
새 TRAIN distribution이 다르므로 새 결과와 단순 전후 개선율로 합치지 않는다.

======================================================================
15. 실패 가설별 기대
======================================================================

H1 TRAIN_EXPOSURE_MISMATCH가 주원인이라면:
- B2(CLIP_RESIDUAL matched)가 기존 A5보다 SHIFT8에서 크게 개선될 가능성.
- 그러나 B0보다 좋아야 새 경로 필요성 근거.

H2 HARD_CLIP 자체가 주원인이라면:
- B1/B2가 B0보다 SHIFT에서 계속 크게 나쁨.
- B3/B5처럼 raw를 선택적으로 보존하는 방법이 차이를 줄여야 함.

H3 TEMPORAL_PERSISTENCE가 유용하다면:
- B3가 B1보다 좋아질 수 있고,
- B5가 B4보다 일관된 추가 이득을 보여야 함.

H4 단순 augmentation이면 충분하다면:
- B0가 전체 tradeoff에서 B3/B5와 같거나 더 좋음.
- 이 경우 새 PEFT 방법 개발 근거 없음.

======================================================================
16. 불확실성
======================================================================

같은 origin의 조건/채널/synthetic seed를 독립 날짜로 세지 않는다.

paired time-block bootstrap 2000.
Electricity 7*24 slots.
ETTm1 7*96 slots.

optimizer seed2는 draw 안에서 평균.
두 seed가 optimizer 모집단 전체 CI는 아님.

source별 결과 유지.
p-value 합치지 않는다.

보고:
- mean
- each seed sign
- 95% block interval
- number of origins/blocks

======================================================================
17. correctness / smoke
======================================================================

GPU main training 전 반드시 통과:

1) existing v1 source/model hashes.
2) existing origins exact reuse.
3) train matched augmentation severity counts exact.
4) all arms same label stream hash.
5) all arms same synthetic draws.
6) future y 변경 -> preprocessing/gate unchanged.
7) generator metadata absent from model packet.
8) B3:
   p=0이면 x_eff==x_clip.
   p=1이면 x_eff==x_obs.
9) B4/B5 gate 0<=g<=1.
10) no NaN/Inf.
11) B0 raw path native wrapper parity.
12) B1 hard clip reproduces v1 clip implementation for same x.
13) B2 reproduces v1 A5 architecture for same input/checkpoint initialization.
14) common LoRA initialization.
15) B4/B5 gate gradient finite after backward.
16) frozen backbone/head unchanged.
17) checkpoint exact restore.
18) batch permutation invariance within FP32 tolerance.
19) same past/different hypothetical future -> same model output.
20) scalar pinball independent replay.
21) 2 updates each of 12 source/arm groups =24 smoke.
22) selected model restore on evaluation packet.

gate parameter initial values, final values, distribution of g를 기록.

======================================================================
18. 자원/재개
======================================================================

one GPU / one worker.
기존 사용자의 GPU 작업 종료 금지.

기존 Watch/guard 사용:
startup free >=4GiB
update-boundary free >=1GiB
disk free >=10GiB
cache <=50GiB
hard safety <=24h

optimizer journal first.
epoch boundary exact resume.
ambiguous update 자동 replay 금지.

main 49,152
smoke 24
hard cap 49,176.

resource:
train optimizer seconds
validation seconds
checkpoint IO
peak allocated/reserved
E inference seconds
inference peak

======================================================================
19. 산출물
======================================================================

PROTOCOL.md
MACHINE_CONTRACT.json
SOURCE_MANIFEST.json
DIAGNOSTIC_REPORT.md
train_clip_exposure.csv
eval_clip_exposure.csv
adapter_usage.csv
failure_hypothesis.json
DATA_MANIFEST.json
AUGMENTATION_MANIFEST.json
PARAMETER_RECEIPT.json
smoke_checks.json
FIT_LEDGER.csv
optimizer.jsonl
LR_SELECTION.json
MODEL_SELECTION.json
GLOBAL_EVALUATION_SEAL.json
predictions_manifest.json
scores_by_origin.csv
scores_by_condition.csv
paired_effects.csv
gate_behavior.csv
resources.csv
verification.json
REPORT.md
FINAL_DECISION.md

필수 그림:
1) TRAIN vs E의 state별 clipped fraction / residual mass.
2) FAULT / REFERENCE / SHIFT4 / SHIFT8 tradeoff.
3) B3/B4/B5 gate restore fraction distribution.
4) 같은 predeclared origin에서 raw/clip/persist transform 예시.
best/worst case 성능으로 사례 선택 금지.

======================================================================
20. 최종 판정
======================================================================

실행 판정:
COMPLETE
PARTIAL_RESOURCE
BLOCKED_DATA
INVALID_IMPLEMENTATION

과학적 판정은 자동 1% threshold로 하지 않는다.

다음 표현 중 근거에 맞게 선택:

TRAINING_EXPOSURE_EXPLAINS_PART:
- matched training으로 B2의 기존 큰 SHIFT 손해가 크게 줄었으나
  B0 대비 우위는 없음.

PERSISTENCE_PREPROCESSING_SUFFICIENT:
- B3가 B0 수준 또는 그 이상이고
  B5의 learnable gate 추가 가치가 없음.

PERSISTENCE_METHOD_SIGNAL:
최소한:
- B5 vs B0가 FAULT/REFERENCE/SHIFT의 주요 tradeoff에서
  단순히 한 패널을 희생해 얻은 개선이 아니고,
- 두 seed 방향이 대체로 일관되며,
- B5 vs B4/B3에서 persistence-aware learned component의
  제한된 추가 가치가 관찰됨.
이 태그도 논문 PASS가 아니다.

RAW_AUGMENTATION_SUFFICIENT:
- B0가 가장 단순하고 tradeoff도 같거나 더 좋음.

NO_ADDED_METHOD_EVIDENCE:
- B5가 직접 대조를 넘지 못함.

HARD_CLIP_MISMATCH:
- clip 계열이 raw 계열 대비 큰 persistent shift 손해를 계속 보임.

중요:
작은 양수는 그대로 보고한다.
CI가 0 포함 -> equivalence라고 쓰지 않는다.
B5가 좋다고 실제 센서 오류/실제 regime change 해결 주장 금지.

======================================================================
21. 다음 단계 규칙
======================================================================

이번 비교 후 자동 후속 학습 금지.

B5에 METHOD_SIGNAL이 있어도:
다음 필수 단계는
- 정식 TATO 또는 가까운 강건 선행 비교,
- 실제 품질/이벤트 레이블이 있는 자료,
- 독립 source/backbone 중 무엇이 필요한지
보고서에만 제안.

B5가 실패하면:
- rank 증가
- threshold search
- W window grid
- new gate features
- seed 추가
자동 실행 금지.

======================================================================
22. CLI에 그대로 전달할 실행 문장
======================================================================

이 문서 하나를 outlier-signal 후속 실험의 유일한 실행 계약으로 사용해.

목표는 기존 A5를 튜닝해서 살리는 것이 아니다.
먼저 optimizer0 진단으로 기존 TRAIN에서
clip되어 복원해야 하는 persistent-shift 정보가 실제로 얼마나 있었는지 계산해.
TRAIN/V/E의 state별 clipped fraction, discarded mass, consecutive run,
A5 adapter 사용량을 보고해.

그 다음 기존 electricity/ETTm1의 sealed origins와 Chronos-Bolt-small을 그대로 사용해
새 matched augmentation으로 B0~B5 여섯 군을 고정 비교해.

matched TRAIN에서는 measurement fault와 persistent shift의 magnitude 범위를 겹치게 하고,
특히 SHIFT4와 SHIFT8을 둘 다 학습하게 해.
모든 군에 동일한 synthetic draws와 labels를 제공해.

새 후보는 B5 PERSIST_LEARNED 하나다.
hard clip 후 embedding에서 복원하는 기존 A5를 자동 변형하지 말고,
관측 과거의 magnitude와 temporal persistence만 사용해
raw와 clipped 값 사이를 input space에서 soft하게 선택하도록 구현해.

B3 fixed persistence rule,
B4 magnitude-only learnable gate,
B0 raw augmentation LoRA,
B1 hard clip,
B2 기존 residual 구조를 반드시 함께 비교해.
B5가 B0를 못 넘으면 특별한 방법의 필요성이 약하다고 결론낼 수 있어야 한다.

generator state, fault mask, clean x0, true delta, future y는 gate/model input으로 사용하지 마.
같은 observed past에서 hypothetical future만 바꿔도 transform/output은 같아야 한다.

LR/seed/updates/checkpoint/V objective/E conditions는 문서대로 먼저 seal하고
결과를 보고 변경하지 마.
최대48fits / 49,152 main updates +24 smoke를 넘지 마.

모든 선택을 봉인한 뒤 E prediction 전체를 저장하고 정답을 채점해.
결과는 FAULT, REFERENCE, SHIFT4, SHIFT8, SHIFT_POINT로 분리해서 보고해.
오류 성능만 좋아지고 persistent shift가 크게 나빠지는 방법을 우승자로 만들지 마.

한국어 REPORT.md와 FINAL_DECISION.md를 작성해.
실행 완료, failure mechanism, 단순 대안의 충분성,
새 persistence-aware component의 추가 가치, 신규성 한계를 각각 분리해.
좋든 나쁘든 다른 후보나 v3를 자동 시작하지 마.

검산 후 승인된 범위 안에서 scoped commit/push해.
