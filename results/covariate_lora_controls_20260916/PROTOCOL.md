# 실제 예보 입력의 LoRA 학습 대조 — 최대 6 fits

2026-09-16, 기준134d60f. 완료된 건물120-fit/예보 참조/신뢰도 진단의 원점수와 판정은 보존한다. 새로운 구조가 아닌 알려진 학습 대조를 통해, 현재 문제에 학습 가능한 추가 이득이 있는지 확인한다. 모든 결과는 동일 원천·동일 타깃의 DISCOVERY이며 독립 PASS/새 방법론으로 판정하지 않는다.

## 데이터·학습 분리

앞서 봉인한 covariate_vintage_reference의 입력/target 파일과 모델 receipt를 그대로 사용한다. OS Gorredijk, Liander revision dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044, simulated available_at. 원시 기상 정답은 읽지 않는다.

TRAIN=3·4월 1·8·15·22일08UTC, 총8원점. V=5월 같은4원점. D=6~9월 같은16원점. 기존 C12를 시간순으로 train8/V4로 나누는 새 개발 학습이며, 이 원점의 과거 zero-shot 결과가 이미 노출됐음을 명시한다. 10~12월 점수 계산 없음. 추가 원점·타깃·날짜·경로를 만들지 않는다.

context336h/horizon24h. 부하 context·label은 그대로, 공변량3개는 TRAIN8의 past arrays만 모아 계산한 공통 feature mean/std(floor1e-6)로 표준화한다. 모든 방법·V/D에 같은 변환을 적용하며 모델의 native instance normalization은 그대로다. 스케일 계산에 V/D·기상 정답·미래 부하를 사용하지 않는다. 과거 입력은 동일, 미래 경로 k0~k3는 기존 simulated-as-of 선택을 그대로 쓴다. 새 dropout mask는 표준화 후 적용한다.

## 정확히 세 학습군 × 두 seed

모두 rank1/alpha2, 기존96 attention q/k/v/o projections(12layers×time/group), 147456개 LoRA 파라미터, 동결 native backbone/head, FP32. seed={61710,61711}. 새로운 gate·head·학습상수 네트워크 없음.

- STD: 최신 미래 예보 k0.
- EXODROP: k0 + 각 feature 전체 과거·미래 구간에 같은 Bernoulli(0.7) mask /0.7. 독립 채널3개, train-only. 알려진 Exogenous Dropout의 채널 masking 비교이며 논문 전체 재현/새 PEFT가 아니다.
- VINTAGE: 미래 경로 k=(epoch_index mod4)를 교대로 사용. 과거 입력은 그대로. 15epochs 동안 각 원점의 k0/1/2 노출4회, k3노출3회. known data augmentation 대조이며 이전 예보라는 추가 학습 정보가 있다. 같은 계산량이 같은 정보량은 아님.

공통 한 원점/step(모델 배치에는 target1+covariate3행), 120updates=15epochs, AdamW lr1e-4 betas(.9,.999)eps1e-8 wd0, clip1, scheduler없음, dropout0/eval-mode backbone/TF32off. epoch마다 같은 seed+epoch permutation으로8개 원점을 순환하며 augmentation RNG는 별도다. 같은seed는 같은초기LoRA·원점순서. 모든 method가120까지 실행, 조기종료 없음.

체크포인트{0,40,80,120}. V4의 최신 경로 primary 평균 최소로 선택, 동률 적은updates. 모든6fit의 선택을 봉인한 뒤 D를 채점한다. 각 D에서 k0와k3를 각각 평가해 정상 최신예보 이득과 과거예보 품질변화 견고성을 분리한다. selected/fixed120 둘 다 보관하며 D에서 바꾸지 않는다. k3는 실제 지난 예보를 쓰는 스트레스 조건으로, 미래 기상 정답을 오염시킨 합성 실험은 아니다.

## 지표·연구 판정

Primary=기존 raw21분위수 정렬 후 mean2pinball / 부하contextstd. 보조=raw2pinball, medianRMSE/MAE, 포함률80/폭. latest와oldest의 평균을 사후 새 primary로 합치지 않는다. 각 방법/seed/조건/checkpoint 원점수를 모두 공개한다. 기존 zero-shot PATH k0/k3 및 F0를 재사용할 때 input 변환/초기화 수치 차이와 원본 출처를 명시한다. 직접공통 대조는 이번 INIT0 예측이다.

LoRA의 이득, dropout의 추가가치, 실제vintage augmentation의 추가가치를 각각 보고한다. 알려진 대조가 좋으면 KNOWN_CONTROL_USEFUL, 불안정하면 POSITIVE_BUT_UNCERTAIN/TRADEOFF, 모두비우세이면 NO_ADAPTATION_GAIN_THIS_BUDGET. 방법론 신규성은 KNOWN_METHOD_CONTROLS_ONLY로 고정. 어떤 결과도 새PEFT PASS로 올리지 않는다. 상대이득·월별방향 및 월블록bootstrap2000(seed61712)을 기술 통계로 제시한다. V4, D16, 타깃1, 원천1, rank1/lr1개라는 한계가 있다.

## 실행·검증 한도

본학습 최대6fits/720updates, smoke3arms×2updates=6 별도. 오류fit도6에서차감, 대체seed/학습률/기간없음. 초기화/native pipeline/동일shape 재생의 normalized maxabs<=1e-5, FP64 LoRA explicit 출력/gradient<=1e-12, primary scalar abs/rel1e-10. 검사 뒤 허용오차 변경하지 않는다. 준비API오류는 기록 후1회수정 가능, nonfinite/OOM에 설정변경 재시도하지 않는다.

사전검사: input provenance와 train-only scaling; 미래 target은 native forward의 loss 계산 이외 입력에 사용되지 않음; forward 예측이 label poison에 불변; scalar/native loss의 변수·mask 정책 확인; intended trainables/update/frozen/buffers; A첫gradient0 허용; 저장weights 새모델재생; native21taus와median확인; 모든결과 독립검산.

단일GPU/worker, 기존lock/Watch/RustDesk예외만. startup30초/free4GiB, runtimefree1GiB, 외부compute 안전경계대기600초, controllerwall3600초. fit별 checkpoint와status/업데이트를 저장한다. 현재job observationtimeout은 재시작 사유가 아니다. 후속별도후보 자동학습없음. 코드·protocol·원점수·한국어REPORT·검증을push하고 원자료/weights/cache는 로컬에 유지한다.
