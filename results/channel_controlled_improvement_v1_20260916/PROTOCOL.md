교정된 표본 위의 PEFT 성능 개선·검증 — 단일 CLI 실행 지시문
작성일: 2026-09-16 KST
대상: CanelE452/tsfm-peft-method-screen
읽기 기준 commit: fa5be2988562249d45bd8609b84ecf14d0f8690c
새 작업명(제안): channel_controlled_improvement_v1_20260916

이 문서 하나만 이번 실행의 계약으로 사용한다. 과거 지시문과 합쳐 실행하지 않는다.
이 문서는 저장소 수정·GPU 학습의 실행 결과가 아니라, 사용자가 CLI에 전달할 설계다.
확인된 기존 사실은 [확인], 실행 선택은 [설계], 원인·기대는 [추정]으로 구분한다.
성능 향상이나 논문 PASS를 보장하지 않는다. 불리한 결과를 없애기 위한 재튜닝도 하지 않는다.

======================================================================
0. 이번에 개선할 것과 하지 않을 것
======================================================================

최상위 목표:
“일별 예측 시작점 편향을 교정한 다변량 시계열 예측에서, 기존 LoRA+HEAD / SIDE / PRIOR를
공정하게 학습·선택했을 때 얻는 정확도와 비용을 개선하고, PRIOR의 추가 가치가 남는지 판단한다.”
소비처: 사용자가 어떤 구현을 남기고 어떤 연구 후보를 중단할지 결정한다.

이번 연구 범위는 MOMENT-small, Electricity/Traffic, 32채널, 과거96→미래96 예측이다.
건물 cold-start, 기상예보 vintage, 불확실한 미래 공변량, Chronos Query의 수치 복구와 별개다.

[확인 S1] 마지막 검토 뒤 fa5be29에는 기상예보 vintage 참조 비교까지 추가됐다.
그 결과를 읽고 중복 여부만 기록한다. 본 실험의 방법·데이터·교사로 가져오지 않는다.
다른 로컬 프로젝트가 이 별도 주제를 수행하고 있다면 방해하거나 종료하지 않는다.

[확인 S2-S7]
- 학습 시작점 위상을 다양화한 LoRA는 옛 동일위상 학습보다 성능이 좋아졌다.
- 교정된 학습 표본 뒤에도 LoRA+HEAD는 HEAD-only와 같은 용량의 출력 어댑터보다 유리했다.
- PRIOR는 SIDE 대비 작은 평균 개선이 있지만 두 seed의 방향이 다르다.
- PRIOR와 SIDE는 모두 761,952개 학습 파라미터를 사용한다.
- attention prior 자체의 신규성은 미확정이고 가까운 기존 연구가 있다.
- 최신 비교는 재사용된 개발 평가 구간이다. 별도 확증 자료가 아니다.

[설계] 이번 변경은 세 가지뿐이다.
1) 이미 교정된 TRAIN 시작점은 그대로 유지한다.
2) 세 기존 방법 모두 학습률 2개, 같은 20 epochs, 같은 세 seed를 받는다.
3) 검증의 예측 시작점도 분산한 패널을 추가하고, 같은 학습 기록에서 선택 효과를 비교한다.

새 head, attention head 수/정렬, PRIOR 온도·gate·lambda, basis, 새로운 loss,
증류, teacher mixture, source transfer, 새 백본은 만들지 않는다.
효과가 없으면 다른 후보를 자동으로 추가하지 않는다.
이번은 새로운 PEFT 발명이 아니라 기존 방법의 최적화·선택과 추가 가치의 유한 검증이다.

======================================================================
1. 실행 전에 목적과 예상 결과를 고정한다
======================================================================

행동 / 필요한 이유 / 실행 조건
A. 기존 수정과 범위 감사
   - 이미 해결된 표본 문제를 되살리지 않기 위해.
   - 서로 다른 주제·설정의 결과를 하나의 개선으로 합치지 않기 위해.
   -> 교정 TRAIN 원점과 모델 계약을 hash로 고정. 기존 결과 변경 금지.
B. 동일 기회의 학습률 비교
   - 단일 학습률에 유리한 방법을 구조적으로 우월하다고 오인하지 않기 위해.
   - 기존 구현 자체의 미래 예측 개선 가능성을 실제로 확인하기 위해.
   -> 모든 방법에 동일한 두 후보 LR, 동일한 epoch·검증 기회 제공.
C. 분산된 검증 시작점으로 선택
   - 한 시작위상에만 맞는 checkpoint를 선택하는 문제의 가능성을 확인하기 위해.
   - 동일 학습 결과를 고정하고 선택 방식만 바꾼 효과를 측정하기 위해.
   -> 두 V 패널의 원점 수 동일, 같은 시간 분할 내부, E로 선택 금지.
D. SIDE와 PRIOR의 직접 비교
   - side 경로 자체의 메모리 이득과 PRIOR 고유의 예측 이득을 분리하기 위해.
   - 가까운 단순 대조를 개선하는지를 확인하기 위해.
   -> 동일 side 초기값·폭·head, 두 방법의 차이는 기존 log prior 추가 하나.

실행 전 예상 결과 / 지지하는 것 / 최상위 연결 / 독자 질문:
- LR/선택 수정으로 세 방법 모두 좋아짐 / 학습·평가 체계 개선 / 실용 모델 선택에 연결 /
  “PRIOR만의 이득은?” -> SIDE와 직접 비교.
- PRIOR가 튜닝한 SIDE보다도 반복적으로 좋아짐 / 제한된 구성요소 이득 / 후속 검토에 연결 /
  “재사용 평가에 맞춘 것 아닌가?” -> 독립 확증은 아직 없다고 명시.
- SIDE가 PRIOR와 같거나 좋음 / prior 필요성 미확보 / PRIOR 확장 중단 /
  “그냥 SIDE면 되지?” -> 그렇다는 결과도 정식으로 보고.
- LH가 여전히 가장 정확하고 side가 덜 메모리 사용 / 정확도-메모리 절충 /
  추가 구조가 아니라 사용 목적별 선택 / 동일하지 않은 품질을 동등하다고 하지 않음.
- 선택 패널 변경에 이득이 없음 / 이번 선택 변경은 근거 부족 / 원래 정책과 나란히 보존 /
  다른 시작점을 성능에 맞춰 다시 만들지 않음.

가지치기:
옛 불균형 TRAIN을 다시 학습하는 2×2 대형실험은 제외한다. 기존 수정 효과가 이미 기록돼 있다.
Tiny-Attention/LST/LiSA 전체 재현은 이번 범위를 넘는다. 직접 선행 비교 미실행으로 보고한다.
주제가 실제 신규 데이터에서 검증됐다고 주장하기 위한 임의 “새 test” 생성도 제외한다.

======================================================================
2. 실제 파일을 읽고 연결한다 — 이름을 추측하지 않는다
======================================================================

기준 commit에서 확인한 기존 경로:
- docs/RESULTS_INDEX.md
- results/channel_phase_balance_20260916/PROTOCOL.md
- results/channel_phase_balance_20260916/seal.json
- results/channel_phase_balance_20260916/sampler_audit.json
- results/channel_phase_balance_20260916/schedules.json
- results/channel_attention_prior_20260916/PROTOCOL.md
- results/channel_attention_prior_20260916/seal.json
- research/attention_prior_novelty_audit_20260916/REPORT.md
- experiments/channel_phase_balance_20260916/run.py
- experiments/channel_attention_prior_20260916/model.py
- experiments/channel_attention_prior_20260916/run.py
- experiments/peft_rank12_20260915/rank2_model.py
- experiments/peft_rank12_20260915/rank2_data.py
- experiments/peft_rank12_20260915/run_rank2.py

직접 읽은 재사용 지점:
- attention model.py: SideBlock, Model, make(arm, ids, seed, device), ARMS=['SIDE','PRIOR'].
- 그 파일의 original_make는 rank2_model의 make를 가리킨다.
- rank2_data.py: load, batch, loss, metrics, independent.
- run_rank2.py: configure, step. step은 BF16/effective batch의 기존 계산을 포함한다.
- common.py의 실제 위치·상태 저장·GPU guard·RNG API는 연결 전에 읽는다.
모듈 검색 경로와 같은 이름의 model/common 충돌을 피한다. 별도 프로세스에서 import 경로를 고정한다.

새 경로(이 문서가 제안하는 이름; 현재 존재한다고 전제하지 않음):
  experiments/channel_controlled_improvement_v1_20260916/
  results/channel_controlled_improvement_v1_20260916/
  .cache/channel_controlled_improvement_v1_20260916/
  docs/CHANNEL_CONTROLLED_IMPROVEMENT_20260916.md
  scripts/run_channel_controlled_improvement.py

실제 로컬 commit·branch·dirty files·실행 중인 worker를 먼저 기록한다.
기준 이후 이 비교가 이미 수행됐으면 결과부터 확인하고 중복하지 않는다.
user 파일을 reset/stash/clean하지 않는다. .cache/model/raw를 정리하지 않는다.
기존 완료 결과의 hash를 보존한다. 이번 새 폴더 및 의도적인 index 추가만 제외한다.
이름이 비슷한 옛 run을 덮어쓰지 않는다.

======================================================================
3. 모델·학습 파라미터·정보 계약
======================================================================

세 방법:
LH: 원래 rank2_model의 LH. MOMENT encoder의 LoRA와 forecasting HEAD 학습.
SIDE: 현재 단일-head·폭10·8층 side attention과 HEAD. 동결 prior를 사용하지 않음.
PRIOR: SIDE와 같고 attention score에 기존 log P0만 더함.

[확인 S4-S7] 원래 LH는 LoRA q/k/v, rank8, alpha32 계약이다.
이번 런에서 rank·모듈 집합을 다시 실제 named_parameters로 확인한다.
SIDE/PRIOR는 encoder 내부 원래 LoRA를 초기 0 상태로 고정하고 encoder 전체를 동결한다.
모두 forecasting HEAD는 학습한다. pretrained 모델 자체가 forecasting INIT를 보증하지 않는다.
무학습 baseline은 seasonal-naive(직전24시간 패턴을 미래96까지 반복)와 last-value를 보고한다.
무작위 초기 forecasting HEAD의 INIT를 Chronos식 유효한 F0로 이름 바꾸지 않는다.

공통:
- MOMENT-small, hidden512, 8층, patch length8.
- L96/H96/C32. 기존 model revision과 weight files hash 그대로.
- 입력·출력 mask/정규화·head 구조·dropout 정책은 기존 구현을 유지한다.
- BF16 forward + 기존 FP32 loss/Adam 상태, effective batch8, microbatch8 고정.
- 모든 trainable 수는 761,952개인지 실측.
- 서로 다른 모델의 frozen 여부 차이는 원래 방법 정의다. 몰래 encoder를 풀지 않는다.
- 같은 source/seed에서 HEAD 초기값을 공통으로 복제한다.
- SIDE/PRIOR의 side 초기 파라미터도 같게 한다.
- LoRA B나 side up=0 초기화에서 일부 첫 gradient가0인 것은 정상일 수 있다.

효율성 변경은 모델 성능 변경과 별도다.
본학습에서 SIDE는 기존처럼 prior도 추출하되 사용하지 않는 경로를 유지한다.
이렇게 얻은 SIDE 비용은 최적 SIDE 비용이 아니다. 최종 보고서에 반드시 쓴다.
10절에서 사용하지 않는 prior 추출을 제거한 자원 전용 SIDE도 가능하면 따로 측정한다.

======================================================================
4. 데이터 — TRAIN은 교정 상태 그대로, V/E만 두 패널로 기록
======================================================================

[확인 S7] 기존 staged data:
  .cache/channel_sharing_screen_v1_20260915/
  원자료 경로·hash·선택32개 열·mean/std는 기존 contract/seal에서 읽는다.
  Electricity T=26304, t1=18412, t2=21043.
  Traffic     T=17544, t1=12280, t2=14035.
  mean/std는 [0,t1)에서 계산한 기존 값, 동일32개 채널 유지.
  학습 target [0,t1), V target [t1,t2), E target [t2,T).
  x=[o-96,o), y=[o,o+96).
기존 64채널 결과를 복제하거나 첫32개를 다른 열로 바꾸지 않는다.

TRAIN:
results/channel_phase_balance_20260916/seal.json의 교정 train origins를 그대로 사용.
새 sampling 방법을 발명하거나 full stride1로 늘리지 않는다.
Electricity512개, Traffic504개. 각 epoch에 전부 한 번씩 사용한다.
41000/41001의 epoch 순서는 기존 balanced schedules와 같게 유지한다.
41002는 np.random.default_rng(41002)의 permutation을20회 만들어 봉인한다.
같은 source/seed의 모든 방법·LR은 같은 epoch stream을 사용한다.
32채널은 각 origin과 함께 입력; 임의의 다른 시점 채널을 섞지 않는다.

V_FIXED, E_FIXED:
기존 contract의 validation/evaluation origin 목록 그대로.
N_V=27/18, N_E=54/36 (Electricity/Traffic).

V_MIXED, E_MIXED [설계]:
같은 분할 내부에서 원점 수를 유지하면서 행 번호 modulo24를 분산한다.
행 기준 위상이지, 데이터의 현지시각 의미까지 검증했다는 뜻은 아니다.
새 timestamp 값이나 달력 feature를 모델에 추가하지 않는다.

아래 규칙은 값·예측을 읽지 않고 origin·분할 경계만으로 생성한다.
seed: V는 Electricity62016/Traffic62017, E는63016/63017.
N개 원점에 floor(24*j/N), j=0..N-1의 위상을 배정한 뒤 위 seed로 순열한다.
각 기존 원점을 배정 위상으로 최대23행 이동한다. 경계를 넘으면24행 뒤로 옮긴다.
아래 부록A의 함수를 그대로 사용하거나 동등한 구현을 CPU검사로 입증한다.

검사:
- 두 패널의 원점 개수 동일; 원점 중복 없음; 시간순 유지.
- 모든 V target은[t1,t2), 모든 E target은[t2,T)에 완전히 들어감.
- 원점별 이동량<=23; 채널과 H/L 불변.
- 24위상 원점 수의 최댓값-최솟값<=1.
  Traffic V는18원점뿐이므로24위상을 모두 볼 수 없다. 18위상만 포함됨을 기록한다.
- 서로 다른 origin의 target이 겹칠 수 있다. 이를 독립 관측으로 세지 않는다.
- V_FIXED와V_MIXED의 정답 집합은 일부 다르다.
  “순수 위상 변화만의 인과효과”나 “동일한 정답 묶음”이라고 하지 않는다.
- 실제 target timestamp union, 중복 수, phase 분포, 이동량을 두 패널 모두 저장.
- time phase가 고르게 된다고 실제 배포 origin 분포를 확인한 것은 아니다.

노출 표시:
이 원천과 평가 기간은 이미 여러 방법 개발에 사용됐다.
MIXED origin을 새로 만들었거나 seed41002를 추가해도 독립 test가 되지 않는다.
모든 결과에 DISCOVERY_REUSED_PERIODS를 표시한다.
이번 작업은 본래 데이터 선택이 옳았는지 전체 검증하는 연구가 아니며,
독립 원천·새 기간의 replication은 별도 계획/승인 대상으로 남긴다.

입력 무결성:
TRAIN worker에는 train prefix, validation은 development prefix만 전달한다.
E 배열은 모든 학습·정책 선택 봉인 뒤 scorer만 읽는다.
기존 raw를 staging하기 위해 기계적으로 읽은 것과 성능 선택에서 사용한 것을 구분한다.
결측처리·valid loss 분모는 rank2_data의 기존 규칙 유지.
E target을 바꿔도 train/V sample·정규화·모델 선택에 변화가 없어야 한다.

======================================================================
5. 세 방법에 같은 최적화 기회를 준다
======================================================================

[설계] grid:
  datasets={Electricity,Traffic}
  methods={LH,SIDE,PRIOR}
  initial LR={1e-3,3e-4}
  seeds={41000,41001,41002}
총36개의 full training trajectories(동일 계약 재사용분은 신규 학습과 별도 집계).

LR은 모든 학습 파라미터에 동일하게 적용. HEAD만 다른 LR을 주지 않는다.
두 LR은 유한한 개발 선택이며 각 방법의 최적값을 증명한 것이 아니다.
모두 AdamW beta(.9,.999), eps1e-8, wd0, clip1.
StepLR(step_size5,gamma0.5)을각 epoch의동일한위치에서호출한다.
정밀도/TF32/dropout/스레드 수는 기존 configure를 사용하고 실제 값을 기록한다.

최대이자 정상 완료목표20epochs.
이번에는 V patience로 학습 경로를 중간 종료하지 않는다.
동일한20epoch 후보 집합을 얻은 뒤 각 선택 정책을 비교하기 위한 변경이다.
INIT와1..20을모두저장. 최종epoch와V선택epoch의성능은구분한다.
20epoch에서도V가개선중이면BUDGET_LIMITED 표시하고자동연장하지않는다.
nonfinite/자원위반을정상early stop으로바꾸지않는다.

새로 학습하는 모든 trajectory는 원래 초기 상태에서 시작한다.
기존 completed trajectory가 아래를 전부 만족하면 재학습 대신 명시적 참조를 허용한다:
- 같은 model revision/모듈 코드/초기 parameter hash, 같은 seed와 전체20epoch 표본 순서.
- LR/Adam/StepLR/정밀도/dropout/batch와 loss reduction이 정확히 같음.
- INIT부터20epoch까지 모두 완료했고 각 checkpoint/hash/실제 업데이트 기록이 보존됨.
- 옛 V 평가가 학습 RNG를 보존했음을 코드·기록으로 확인.
- 기존 score가 아니라 해당 checkpoint로 이번 V_FIXED/V_MIXED를 새로 평가해 선택함.
이 재사용은 별도 REUSE_RECEIPT에 저장한다. 옛 fits를 이번 신규 fits로 세지 않는다.
서로 다른 결과 파일의 best checkpoints를 이어 붙여 전체 trajectory라고 부르면 안 된다.
중간 종료된 옛 trajectory는 새로 학습하거나 완전한 재개 상태가 있을 때만 이어간다.
과거 best checkpoint만으로 Adam/scheduler/RNG를 복구한 것처럼 이어 학습하지 않는다.
초기 HEAD/LoRA 공유와 학습한 checkpoint warm-start를 구분한다.
기존 체크포인트에 정확한 optimizer·scheduler·RNG·epoch stream 상태가 없는 한 재학습은 필수다.
기존 결과는 새 fixed20/LR/선택 계약의 성능으로 재명명하지 않는다.

학습 예산:
- Electricity: ceil(512/8)=64 updates/epoch.
- Traffic: ceil(504/8)=63 updates/epoch.
- 각 원천18fits(3방법×2LR×3seed),20epochs.
- 본학습 optimizer 최대45,720 updates,36 fit IDs.
- 실제 모델 smoke: 2원천×3방법×2updates=12updates. 별도.
- 선택된 모델의 resource 검사: 최대120 폐기 updates. 별도.
학습시간/평가/저장/초기화/재생/대기 비용을 따로 장부에 기록한다.
같은 epoch 비교이지 동일시간 비교가 아니다.

실행 순서는 seed64016으로36개 cell을 순열해 봉인한다.
paired seed는 실험 난수변동을 통제하는 장치다. 서로 다른 구조의 모든 dropout경로가 같다고 하지 않는다.
V패널평가는 preserve_rng 및 eval/no_grad로 감싸 학습 난수 상태를 바꾸지 않는다.
전체 패널을 평가한 scalar로 비교. 마지막 batch 크기가작을때 batch평균을다시단순평균하지않는다.

======================================================================
6. 선택 정책 — E를 보기 전에 모두 고정
======================================================================

주 선택은 V_MIXED다. source/method별 하나의 LR을 선택하고 세 seed가 공유한다.

1) 각 LR·seed에서 epoch0..20 중 V_MIXED MSE 최소를 찾는다.
   정확한 동률은 더 이른 epoch.
2) 각 LR의 세 seed 최소 V 점수를 평균한다.
3) 평균이낮은LR을 source/method의 주 LR로고정.
   정확한동률이면3e-4.
4) 선택 LR에서 seed별 가장좋은V_MIXED epoch를실제반환모델로선택.

PRIOR만E에좋은LR이나seed를고르지않는다.
LH/SIDE/PRIOR 모두같은선택기회.
V사용에따른낙관편향은남으므로이V점수를일반화점수로말하지않는다.

같은 학습 기록에서 다음 보조 정책도 사전에 지정한다. 추가 optimizer fit 없음.
P_MAIN: 위 주 선택(두LR 허용,V_MIXED).
P_VFIXED: P_MAIN과같은LR을고정하고epoch만V_FIXED 최소로선택.
          -> trainedtrajectory/LR를유지한채검증패널의선택차이를봄.
P_LR_FIXED: 초기LR=1e-3만허용하고epoch는V_MIXED로선택.
          -> V패널을유지한채두LR선택을허용한차이를봄.
P_LAST20: P_MAIN과같은LR의epoch20.
          -> 서로다른반환시점으로인한차이와고정학습끝점의차이를구분.

이정책의결과를본후가장좋은정책을주정책으로승격하지않는다.
P_MAIN의시점과LR,나머지정책의시점,모든hash를selection_seal에기록.
불리한방법을모델선택단계에서지워3방법비교에서제외하지않는다.

36개예정학습이완료되고선택을봉인한뒤에만E점수계산.
실행중일부cell이막히면정상완료부분은보존한다.
부족한cell이있는경우주3seed비교를성공으로부르지않고INCOMPLETE_EXECUTION으로종료.
seed를교체해서결측cell을채우지않는다.

======================================================================
7. 평가와 원인 해석
======================================================================

평가 패널:
- E_MIXED: 이번의주평가,재사용기간.
- E_FIXED: 역사적배포시작점조건을유지한보조평가.
두패널모두위네선택정책을평가하되같은checkpoint는한번예측후참조한다.
INIT와seasonal-naive/last-value도같은정보로기록. INIT를성능좋은F0라고하지않음.

Primary: Train 표준화 공간 equal-channel MSE. 낮을수록좋음.
Secondary: 같은공간MAE,채널별원점수,raw MAE,origin phase별오차.
다른원천의raw MSE를그대로평균하지않는다.
PRIOR의b대비gain=100*(MSE_b-MSE_PRIOR)/MSE_b.
원천별3seed원점수를먼저평균한뒤gain을계산하고seed별gain도모두보고.
LH와SIDE를각각별도대조군으로표시.

보고할 대비:
A. P_MAIN vs P_LR_FIXED, 동일방법/평가패널: 제한된LR탐색의개선여부.
B. P_MAIN vs P_VFIXED, 동일방법/LR/평가패널: 검증패널선택의차이.
C. PRIOR vs SIDE, 모두P_MAIN: prior추가의가치.
D. PRIOR/SIDE vs LH, 모두P_MAIN: 내부LoRA대비품질·자원절충.
E. P_LAST20끼리: 같은최대updates에서의단순비교.
F. E_FIXED vs E_MIXED: 평가시작점분포민감도. 서로다른정답묶음임을명시.

시작점분산+LR+학습지속을함께바꾼결과전체를prior의기여라고합치지않는다.
학습이개선됐다고새메커니즘이발명된것처럼이름붙이지않는다.
기존표와직접비교할때는패널/epoch선택/seed/LR/정밀도/손실분모를나란히적는다.
전력의0.2%개선과교통의3%손해를평균하나로숨기지않는다.

불확실성:
각E패널의origin을시간순으로8개contiguous blocks로분할(array_split).
2000회paired block bootstrap(seed65016/65017).
한resample에서모든방법/선택정책/seed/채널에같은origin block index를적용한다.
길이가다른block을N개origin으로얻도록이어붙였다잘라낼때계수·분모를명시한다.
3seed를독립source3개로세지않는다. seed별효과도표로보인다.
시간블록CI는개발자료의서술적구간이다. 반복한방법선택의편향까지제거하지않으며
3seed만으로초기화모집단의분산을정확하게추정했다고하지않는다.
E의targetoverlap과phase-시간상관의한계도기록한다.

======================================================================
8. 저장된 PRIOR에 대한 최소 구성요소 검사 — 새 모델을 만들지 않음
======================================================================

P_MAIN이 선택한 PRIOR 체크포인트에서 V_MIXED만 사용.
- 원래prior 사용
- prior항만0: 같은학습가중치에서softmax(QK)로변경
두예측을비교. source2×seed3×2조건의최대12논리적예측,추가fit0.
기존예측은중복계산하지않아도된다.

이 검사는학습된PRIOR가prior경로에의존하는지만볼수있다.
prior를제거한성능손해는그원리의우월성증명이아니다.
처음부터SIDE로학습한대조가주증거다.
이진단점수로새temperature/head/gate를선택하지않는다.
PRIOR가SIDE를못이겨도제거시손해가클수있다. 공동적응의존성과학습이득을분리한다.

======================================================================
9. 실행·모델 수치 검사 — 문제가 아닌 동등성 검사로 멈추지 않게 한다
======================================================================

필수CPU:
- 부록A origin경계/phase/count검사와train/V/E poison검사.
- 동일입력/정답mask/분모의vectorized loss와scalar loss검산.
- SIDE/PRIOR 동일초기side·HEAD,INITIALup0의공통eval경로확인.
- 모델의trainablecount와frozen list를파일로기록.

실모델smoke:
모든source/arm2updates;같은precision/batchshape/정규화정책사용.
intended parameter가실제로바뀌고frozen weights/buffers가유지되는지검사.
첫step에서A/q/k가0gradient일수있는초기화를오류로취급하지않는다.
출력·loss·gradient·update가유한한지,HEAD와전체adapter가변하는지확인.

복원:
selected checkpoint를새모델에복원하고같은입력/batchshape/precision/eval에서재예측.
동일조건의bitwise일치는우선기록한다.
다르면standardized forecast기준FP32 export에rtol1e-5,atol1e-6검사와불일치위치를보고.
통과범위를결과에맞춰늘리지않는다. 기준초과는해당복원의무결성오류로수정·재검사한다.

하지않을검사:
BF16과FP32가같아야한다거나,다른microbatch형태의Adam업데이트가비트단위로같아야한다는
Query의옛진입gate를가져오지않는다. 이번main run의microbatch는모두8로고정한다.
GPU메모리부족이면한방법만batch를줄이지말고현실적인공통계약변경이필요하다고기록한다.

재실행정책:
성능이나seed를바꿔재시도금지. 의미버그·I/O중단은재현근거와패치기록을남긴다.
완전state(학습파라미터·Adam·scheduler·RNG·epoch/batch위치)가있으면동일fit재개가능.
없으면그fit을실행오류로보존하고새attempt자동대체금지.
20epochs의trainable checkpoint뿐아니라최소최신resume state를별도원자적저장한다.
오류를수정했으면옛예측을조용히새버전으로덮어쓰지않는다.

======================================================================
10. 자원 측정 — 정확도와 분리해 보고한다
======================================================================

모든main fit에서:
- step peak allocated/reserved, 모델로드/V평가/재로드전체peak 별도.
- active training seconds, epoch별updates, gradient/clipnorm, totalwall.
- 선택모델까지의updates와실제로수행한전체20epochs+두LR탐색비용을분리.
- 두비중첩단계의메모리최댓값을더하지않는다.
- 학습파라미터수감소를같은비율의속도·메모리감소로해석하지않는다.

별도의선택상태resource측정:
각source의seed41000 P_MAIN을사용,평가정답을사용하지않는고정Train배치만사용.
옵션4개: LH-current, LH-native-checkpoint(지원될때), SIDE-fast, PRIOR-current.
SIDE-fast는사용하지않는q/k snapshot/P0 재계산을제거한동일SIDE다.폭/정밀도/출력변경없음.
실제수정코드는읽은model.py를복제한명시적new option으로구현.
기존SIDE를덮어쓰지않는다. same-shapedforward/gradient/update동일성검사후자원표에만사용.
LH-checkpoint는로컬T5/MOMENT/PyTorch의지원API를읽고사용한다.지원되지않으면NOT_MEASURED.
checkpoint를현재후보의예측성능에유리하게만허용하지않는다.

각option: optimizer state를생성하는2warmupupdates +3blocks×3timedupdates.
모든옵션은동일arm의동일savedparameter/Adam/RNG로복구후시작.
측정state의업데이트는폐기한다.본학습결과를교체하지않는다.
2sources×4options×11updates=88,동등성검사등포함절대상한120.
옵션순서는고정seed66016으로순열.전체forward+loss+backward+clip+step+synchronize시간.
배치형태8 origins×32channels,같은정밀도,같은Train배치.

SIDE-fast나LH-checkpoint의동일성을확보하지못하면그 옵션의 자원 주장만 보류한다.
메모리토너먼트로새가설/새fit을발생시키지않는다.
당장주장가능한메모리절감은실제로측정한대조설정에한정한다.
학습peak,추론peak,NVML 전체VRAM은서로다른수치다.

======================================================================
11. 보고 판정 — 실패·절충·개선의 의미를 분리
======================================================================

옛PROTOCOL의STOP/FAIL은그대로둔다.이번새프로토콜로옛결과를소급PASS로바꾸지않는다.
주비교의연속수치가먼저고,상태코드는그요약이다.

EXECUTION_COMPLETE:
재사용을 포함한 36개 유효 trajectory와 선택/예측/검산을 완료했는지.성능좋음을뜻하지않음.
PIPELINE_IMPROVEMENT:
P_MAIN이P_LR_FIXED/P_VFIXED보다나은지source/method별로보고.
없는곳은없다고쓰고개선된case만추리지않는다.
PRIOR_COMPONENT_SIGNAL:
적어도한source의E_MIXED에서P_MAIN PRIOR가P_MAIN SIDE보다
3seed평균MSE를1%이상줄이고3seed모두gain>0이며,paired시간블록95%구간하한>0일때
“제한된개발신호”로표시한다.이1%는추가투자용문턱이며논문표준/보편적합격선아님.
다른source의손해와E_FIXED민감도는항상보고.두source전승을강제하지않음.
평균만양수이거나CI가0을포함하면WEAK_OR_UNCERTAIN_SIGNAL.
SIDE/LH와품질·시간·메모리의손익이섞이면TRADEOFF_ONLY.

prior가없어도side경로가저메모리인것은선행에서알려진이점이다.
SIDE와PRIOR가모두메모리를줄였다고PRIOR의새기여로더하지않는다.
LH가가장정확하면그사실을그대로결론으로쓴다.
신규성은별도칸: 알려진원리 / 가까운선행과의차이 / 직접재현미실행 / 후속증거필요.
어떤수치신호가나와도이번기간재사용결과를논문최종PASS로부르지않는다.

최종결정은후속1개만:
A) PRIOR에제한적추가근거있음 -> 방법/recipe고정후별도독립자료검증을제안.자동실행안함.
B) SIDE로충분 -> prior확장중단;유효한저메모리대조군으로SIDE보존.
C) LH가충분 -> 교정된LH를작동기준으로보존;새attention후보자동생성안함.
D) 실행판정불가 -> 미완료cell/원인을보고;새seed로메우지않음.

======================================================================
12. 자원 안전·권한·정리
======================================================================

한GPU·한worker,검증된프로젝트lock/guard사용.다른프로젝트PID종료금지.
동일프로젝트에서현재도유효한사용자승인이확인된예외만적용하고목록을저장.
GUI과외부학습compute를혼동하지않되오염된timing을깨끗한측정으로세지않는다.
시작30초안정,freeVRAM>=4GiB,실행경계free>=1GiB.
누적GPU대기600초.총GPU작업wall4시간.이는안전상한이지완료시간약속이아님.
디스크실제여유와예상checkpoint용량을prepare에서계산;10GiB미만이면부족내역보고.
기존자료를삭제해공간을만들지않는다.전역CUDA/driver/package변경금지.

중단시:
상태·완료부분·최근완전resume point·미실행범위를저장.
다른worker와경합하며무한재시도하거나4시간뒤새run으로자동재개하지않음.

======================================================================
13. 산출물과 CLI 실행 흐름
======================================================================

prepare -> smoke -> train(36 trajectories) -> select/seal -> evaluate ->
resource-check -> verify -> report -> authorized scoped commit/push.
명세만작성하고종료하지않는다.핵심입력/안전/구현조건이충족되면정한범위까지실행한다.
낮은중간성능으로같은고정배치의나머지방법실행을막지않는다.

새runner명령예시(새로구현할인터페이스,존재한다고가정하지않음):
  prepare / smoke / train / select / evaluate / resources / verify / status / resume
resume은같은fit의완전state재개만;새후보·새LR·seed대체가아님.
호출전--help와단계상태검사를확인한다.기존runner의main을실행해옛경로에쓰지않는다.

필수파일:
PROTOCOL.md, SCOPE.md, contracts/model/data/source hashes,
phase_panels.csv, exposure_ledger.md, schedules.json,
fit_manifest.csv, fit_attempts.csv, training_curves.csv,
V_FIXED/V_MIXED 원점수와예측manifest, selection_seal.json,
E_FIXED/E_MIXED 원점수, effects.csv, uncertainty.csv,
prior_dependency.csv, resources.csv, independent_verification.json,
REPORT.md, FINAL_DECISION.md.
사용자에게JSON만주지않는다.원시모델/큰예측cache/인증정보는push하지않는다.

REPORT 질문순서:
1. 왜이비교를했나? -> 교정된시계열학습기반에서기존3방법의최적화·선택과prior추가가치.
2. 무엇이같고달라졌나? -> 모델/입력고정,LR와V선택비교,재사용기간.
3. 실제몇개를완료했나? -> main/smoke/resource/resume/update장부분리.
4. 학습·선택수정이좋아졌나? -> 같은패널의정책별원점수.
5. PRIOR가SIDE를넘었나? -> source/seed별gain,CI,최종20시점.
6. LH보다나은선택인가? -> 정확도/메모리/시간을함께보고.
7. 알려진원리와남은차이는? -> 신규성성공으로격상하지않음.
8. 한계와다음결정하나. -> 독립자료평가는미실행;자동후속없음.

최소그림:
- source별V_MIXED 학습곡선(같은LR/epoch범위를명시).
- 3seed의PRIOR-SIDE gain과LH대비gain(음수포함).
- source별정확도-실측학습peak/시간.서로다른실험수치를묶어새측정처럼그리지않음.
- origin phase분포(T/V_FIXED/V_MIXED/E_FIXED/E_MIXED).

원점수scalar재계산은float64,rtol1e-10/atol1e-10.
저장model/path/hash/selection규칙도별도함수로재계산.
이미승인된scoped commit/push가확인될때이번새파일만올리고RESULTS_INDEX를추가갱신.
권한확인이안되거나push가실패하면로컬보고서위치와미업로드상태만정확히보고.

======================================================================
부록 A. 값·예측을 읽지 않는 phase-panel 참조 함수
======================================================================

아래코드는원점배치의명세다.실제raw데이터검사는CLI에서별도로수행해야한다.
작성단계에서공개된행수/분할경계로만CPU산술검산을했다.

import numpy as np

def phase_panel(origins, start, stop, horizon, seed):
    base = np.asarray(origins, dtype=np.int64)
    n = len(base)
    assert n > 0 and np.all(np.diff(base) > 0)
    phases = np.floor(24 * np.arange(n) / n).astype(np.int64)
    phases = np.random.default_rng(seed).permutation(phases)
    out = []
    for b, p in zip(base, phases):
        o = int(b + (p - b) % 24)
        if o + horizon > stop:
            o -= 24
        if o < start:
            o += 24
        assert start <= o and o + horizon <= stop
        assert o % 24 == p and abs(o - b) <= 23
        out.append(o)
    out = np.asarray(out, dtype=np.int64)
    assert len(np.unique(out)) == n and np.all(np.diff(out) > 0)
    counts = np.bincount(out % 24, minlength=24)
    assert counts.max() - counts.min() <= 1
    return out, counts

CPU산술검산결과(실데이터검증이아님):
Electricity: train512,V27,E54;Vphase최소1/최대2,E최소2/최대3.
Traffic: train504,V18,E36;Vphase최소0/최대1,E최소1/최대2.
모든생성원점경계·유일성·이동량검사통과.
본학습예산36fits·45,720updates검산.
8*(4*512*10+2*512)+589920=761952는수식검산이며실제model.numel검사를대체하지않는다.

======================================================================
출처와 적용 경계
======================================================================

Repo의S1-S7은위기준commit으로고정해읽었다.로컬최신변경은실행전에확인한다.
S1 docs/RESULTS_INDEX.md
S2 experiments/channel_phase_balance_20260916/run.py
S3 research/attention_prior_novelty_audit_20260916/REPORT.md
S4 results/channel_attention_prior_20260916/PROTOCOL.md
S5 experiments/channel_attention_prior_20260916/model.py
S6 experiments/channel_attention_prior_20260916/run.py
S7 experiments/peft_rank12_20260915/{rank2_data.py,run_rank2.py}
Repo: https://github.com/CanelE452/tsfm-peft-method-screen/tree/fa5be2988562249d45bd8609b84ecf14d0f8690c

외부원자료:
- Tiny-Attention Adapter: Contexts Are More Important Than the Number of Parameters(2022/EMNLP)
  https://aclanthology.org/2022.emnlp-main.444/
  작은attention adapter의원리선행.현SIDE/PRIOR를공식완전재현이라고하지않음.
- LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning(2022/NeurIPS)
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/54801e196796134a2b0ae5e8adef502f-Abstract-Conference.html
  백본역전파를줄이는side경로의선행.이메모리원리가PRIOR고유기여는아님.
- On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation(2010/JMLR)
  https://www.jmlr.org/papers/v11/cawley10a.html
  유한한검증자료에대한반복선택과평가편향의근거.이번결과의원인을증명한논문은아님.
- PyTorch Numerical accuracy
  https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
  배치/정밀도/연산순서의차이.현재설치버전은별도기록.

두LR·3seed·20epoch·36fits·origin생성·1%신호선은이번개발실험의설계선택이다.
문헌의보편적합격선이아니며,결과를보고완화하지않는다.
