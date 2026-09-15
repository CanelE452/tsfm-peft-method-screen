PEFT 1·2순위 실행 지시문 — Query 복구 + 채널 변환 공유 비교
작성일: 2026-09-15
대상 저장소: CanelE452/tsfm-peft-method-screen
확인 기준 commit: 9051d03cebe23ae35d9689e7af576c63c4d5edd5
확인한 Time-PEFT upstream commit: ea4e7e1887bb35587bab7ea93e2af3685ac55852

이 문서는 두 작업을 실제로 구현·검사·조건부 학습·평가하도록 하는 CLI 지시문이다.
계획 파일만 작성하고 종료하지 않는다. 단, 무결성·입력·자원 조건을 임의로 우회하지 않는다.
이 문서를 작성한 대화에서는 저장소 수정이나 실제 foundation model 학습을 실행하지 않았다.

======================================================================
0. 범위, 실행 순서, 상한
======================================================================

사용자가 승인한 대상은 다음 두 개다.
R1: 미완료 Query의 수치 중단을 한 차례 복구하고, 원래 자원 비교와 조건부 12 fits를 완료한다.
R2: 채널마다 큰 변환을 따로 두는 대신 공통 변환과 작은 채널별 계수를 쓰는 비교를 실행한다.
    R2는 신규성이 입증된 신기법이 아니라 구조·예산 비교 파일럿이다.

순서: 공통 저장소 감사 → R1 → R1 결과 보존·GPU 해제 → R2 → 통합 보고.
R1의 예측/속도 성공은 R2의 선행 조건이 아니다. R1이 STOP이면 그 상태를 보존하고 R2로 간다.
R1의 전역 환경 손상·실제 GPU 장애가 있으면 R2도 안전 복구 전 GPU 실행하지 않는다.
R2에서만 필요한 패키지가 없다는 이유로 R1 환경을 바꾸지 않는다.
두 트랙을 동시에 GPU에서 실행하지 않는다.

새 산출물 경로(아직 존재한다고 가정하지 말고 이번에 만들 이름):
  research/peft_rank12_20260915/
  results/query_budget_repair_v2_20260915/       (R1)
  results/channel_sharing_screen_v1_20260915/   (R2)
  .cache/query_budget_repair_v2_20260915/
  .cache/channel_sharing_screen_v1_20260915/
  experiments/peft_rank12_20260915/             (새 실행기·설정·검사)

상한:
- R1 신규 본학습: 최대 12 fit attempts. 기존 실행의 본학습 0회와 구분.
- R1 신규 폐기용 검사 updates: 수치 진단 최대 96 + 자원 검사 최대 480 = 최대 576.
- R2 신규 본학습: 6 arms × 2 sources × 2 seeds = 최대 24 fit attempts.
- R2 실제 모델 smoke: 최대 24 optimizer updates. 본학습 초기 상태에 이어 쓰지 않는다.
- 두 트랙 합계: 최대 36 본학습 attempts. 모든 조건을 통과했을 때의 상한이지 의무 횟수가 아니다.
- 오류 attempt를 새 이름으로 대체하지 않는다. warmup/smoke/profile를 본학습 fits로 세지 않는다.
- Censor·FR·Freshness·DualClock·PatchPhase 재튜닝, 추가 LR/seed/데이터 탐색은 금지.
- R1 GPU 단계 wall cap 4시간, R2 GPU 단계 wall cap 12시간. 이는 안전 상한이지 소요시간 예측이 아니다.
- 외부 GPU 대기는 트랙별 누적 600초 상한. 무한 대기 금지.
- 각 트랙 구현·수치 문제 수정은 원인 있는 1차 수정까지. 같은 결과를 보고 반복 수정하지 않는다.

실행 여부, 수치 유효성, 예측 이득, 자원 이득, 신규성은 별도 칸으로 보고한다.
PAPER_PASS 같은 상태는 만들지 않는다. 아래의 SIGNAL은 제한된 후속 투자 신호다.
JSON은 내부 manifest로만 사용해도 된다. 사용자 주 산출물은 한국어 REPORT.md와 읽을 수 있는 표다.

======================================================================
1. Phase 0 — 목적 트리와 가지치기
======================================================================

최상위 목적:
“아직 답하지 않은 두 가지 제한된 PEFT 비교를 끝내고, 후속 방법 연구에 투자할 근거가 남는지 판단한다.”
소비처: 사용자·지도교수가 다음 방법 연구를 계속할지 판단한다.

R1의 이유 두 가지:
1) 수치 검사 중단을 예측 성능 실패와 분리한다.
2) Standard에 강한 메모리 절약 옵션을 줘도 Query의 실용적 이점이 남는지 확인한다.
설계 조건: 의미가 같은 배치·손실·업데이트를 비교; 단위·정밀도를 고려한 검사;
           가장 빠른 유효 Standard 옵션 유지; frozen encoder 비용 포함; 같은 시간의 예측 비교.

R2의 이유 두 가지:
1) 제한된 학습 파라미터를 모든 채널에 공유할지, 채널별로 배분할지의 효과를 직접 비교한다.
2) 채널별 큰 모델을 줄여도 정확도가 유지되는지, 단순 공유/저랭크 대안으로 이미 충분한지 확인한다.
설계 조건: 같은 backbone/HEAD/학습 정보; 공통 구성과 채널 블록 예산 분리;
           공유·독립·저랭크·basis 대조; 알려진 방법을 새 방법으로 포장하지 않음.

예상 결과 / 목적 지지 / 최상위 연결 / 독자 첫 질문:
- R1 계산 의미가 같고 사전 오차 내 → 부분 지지 → 자원 비교 → “실제로 빨라지는가?”
- R1 Standard의 단순 최적화 후 이점 소멸 → 현재 가설 불지지 → 학습 전 종료 → “Query를 왜 쓰나?”
- R1 같은 제약에서 예측 이득 → 지지 → 후속 검토 → “다른 길이/장비에서도 되나?”
- R2 BASIS가 동일 예산 단순 대조에 우세 → 부분 지지 → 구조 연구 검토 → “선행과 무엇이 다른가?”
- R2 큰 INDIV_REF와 비슷하고 총 학습 파라미터 감소 → 부분 지지 → 압축/공유 연구 검토.
- R2 완전 공유/일반 저랭크로 충분 → 특정 basis 필요성 불지지 → 현재 basis 후보 종료.
- 오류·상한 중단 → 목적 판정 불가 → INCONCLUSIVE, 과학적 FAIL로 바꾸지 않음.

삭제한 작업:
- 모든 V→E 반전 원인의 완전 규명; 다음 결정을 바꾸는 검사까지만 한다.
- 수요 검열 손실 수정; R2에 occurrence/hurdle/anchoring을 섞지 않는다.
- 새로운 동적 router, 채널 clustering, basis 수 탐색; 우선 하나의 고정 구조만 비교한다.
- 전체 PEFT/백본 벤치마크. R2는 1 backbone, 1 horizon, 32 fixed channels의 개발 비교다.

======================================================================
2. 공통 감사·보호 규칙
======================================================================

[확인 R1-S1] 기준 commit의 Query 최신 실행은 INCONCLUSIVE_NUMERICS,
A 폐기 updates 120, B fits 0이다. 기존 기록을 PASS나 새 결과로 덮어쓰지 않는다.

먼저 git HEAD/branch/status, 기존 GPU 작업, 원자료/모델/cache 경로, 디스크 여유를 기록한다.
기준 commit 뒤의 관련 변경과 결과가 이미 있으면 읽고 중복 실행을 방지한다.
보여 주지 않은 변수·컬럼·함수·CLI 옵션은 존재한다고 추정하지 않는다.
이 문서에 ‘새로 만들’이라고 한 경로·arm·상태 이름만 새 인터페이스로 구현한다.

기존에 확인한 파일:
  docs/RESULTS_INDEX.md
  results/query_budget_pilot_20260915/REPORT.md
  results/query_budget_pilot_20260915/PROTOCOL.md
  results/query_budget_pilot_20260915/contract.json
  src/tsfm_peft_screen/forecast_query/budget.py
  src/tsfm_peft_screen/forecast_query/equal_time.py
  src/tsfm_peft_screen/forecast_query/model.py
  scripts/run_query_budget_pilot.py
  scripts/finalize_query_budget_pilot.py
  tests/test_query_budget_pilot.py
  configs/query_budget_pilot_20260915.json
  src/tsfm_peft_screen/backbone.py
  src/tsfm_peft_screen/lora.py
  src/tsfm_peft_screen/metrics.py
  research/overnight_20260913/data_receipt.json

실행기를 그대로 재호출해 종료 run을 덮어쓰지 않는다. 새 wrapper/모듈로 경로를 분리한다.
기존 결과·연구 파일의 SHA256을 봉인한다. 새 폴더와 명시적 RESULTS_INDEX 추가만 예외다.
사용자 dirty change를 stash/reset/clean하지 않는다. 새 작업만 변경하고 기록한다.
무관한 환경/드라이버/torch를 업그레이드·다운그레이드하지 않는다.
원자료·전체 모델·대형 예측 cache·토큰을 GitHub에 올리지 않는다.

GPU 안전:
- 기존 .cache/gpu.lock을 확인해 단일 worker/프로젝트 lock 사용.
- 자기 PID와 자식, 다른 compute PID를 구분. 일반 display process를 학습으로 오인하지 않음.
- 시작 전 외부 compute 없음 30초, free>=4GiB, RAM available>=2GiB.
- 실행 중 외부 compute 발생 또는 free<1GiB이면 안전 경계에서 대기. 다른 프로세스 종료 금지.
- 타이밍 구간 오염은 별도 기록하고 좋은 구간만 골라 대체하지 않음.
- OOM 후 몰래 batch/context/precision 바꾸지 않는다. 사전 dry-fit 탐색만 허용.
- timeout·환경 오류에는 실제 진행량과 재개 가능한 checkpoint를 보존하고 종료.

두 트랙에서 원점수의 단위/분모가 다르다. Chronos pinball과 MOMENT MSE를 직접 합산하지 않는다.
모든 결과는 원점수와 G=100*(baseline-candidate)/baseline을 같이 제공한다.
baseline 0이면 상대 개선율은 NA이며 절대 차이만 보고한다.
두 seed는 optimizer 반복이지 두 개의 독립 데이터셋이 아니다.

======================================================================
R1. Query의 수치 중단 복구 → 자원 측정 → 조건부 12 fits
======================================================================

R1.1 바꾸는 것과 유지하는 것

[확인 R1-S1,S2] 기존 check_parity는 FP32 microbatch 비교에 원 단위 raw 최대 절대 차이와
relative L2<=1e-5를 함께 요구했다. batch1/2는 수학적으로 같은 목적식을 의도하지만
부동소수점 연산 순서와 pinball 분기 때문에 비트 동일성을 보장할 수 없다.
이것이 이번 차이의 원인이라고 단정하지 않는다.

이번 변경은 아래로 제한한다.
- 배치 분할이 정말 같은 학습 의미인지 검사한다.
- 수치 검사를 same-shape 재계산과 different-shape microbatch로 분리한다.
- 원 단위 출력 차이는 기록만 하고, 허용 판정은 train scale로 나눈 무차원 차이도 사용한다.
- 아래 새 허용치는 이미 본 중단 기록 이후 제안한 변경이다. 기존 실행을 소급 PASS 처리하지 않는다.
- 새로운 검사 배치/수치 규칙을 이번 실행 전 봉인하고 그 뒤 값을 보고 완화하지 않는다.

유지: 모델/출력층 고정, rank, 손실, LR, seed, 데이터 구간, context4096/H48,
1/2/4/8GiB, 5% speed 진입선, 시간120초, 본학습12회 상한.
R1을 통과시키려고 Query만 다른 정밀도/배치/초기화로 실행하지 않는다.

R1.2 N0 — CPU 고정밀 의미 검사

실제 micro_origins/batch/손실 reduction을 읽고 아래를 CPU float64 작은 모형으로 검사한다.
- effective batch는 2 origins × 각4채널. origin 내부 채널 그룹 유지.
- 분할 두 손실에 각각 1/2; 둘의 gradient 누적 뒤 clip과 optimizer.step을 딱1회.
- 그룹 ID 재번호화가 그룹 membership을 바꾸지 않음.
- 첫 origin만 바꿨을 때 둘째 그룹 출력이 변하지 않음. 실제 모델에서도 입력 격리 검사.
- 입력/target/스케일/마스크의 순서와 padding이 동일.
- 누락 target이 있는 toy에서도 기존 native loss의 분모를 그대로 재현.
- 검증/평가 target은 입력·normalizer·batch 옵션 결정에 사용하지 않음.
- loss/gradient/SGD 및 Adam 단일 update에 float64 rtol1e-9, atol1e-11 사용.
- gradient가 0인 tensor는 절대 오차로 검사; LoRA B=0일 때 첫 A-gradient=0을 실패로 세지 않음.

float64 toy에는 attention의 그룹 격리, normalize, native 형태 pinball, 누적/clip을 포함하되
이를 전체 Chronos 실행의 고정밀 증명이라고 쓰지 않는다.
확정적인 loss scaling/group/update 오류가 나오면 그 한 원인을 수정하고 같은 CPU 검사를 다시 한다.
원인을 모른 채 학습 코드를 재작성하거나 모델의 문제로 돌리지 않는다.

R1.3 N1 — 실제 모델, 새로운 고정 Train 배치의 수치 검사

N0 통과 뒤 6개 상태(2 sources×3 arms), seed39000을 사용한다.
상태당 2회 폐기 warmup으로 LoRA/Adam이 움직인 상태를 저장한 뒤 매 비교마다 동일하게 복구한다.
과거 중단 때 사용한 원점만 재검사하지 말고 다음 두 Train 쌍을 새 검사 배치로 고정한다.
  [16640,17664], [20992,22016]
각 x는 [origin-4096,origin), y는 [origin,origin+48).
추가 probe 정답은 모두 Train 구간 안이며 모델 선택 성능에 사용하지 않는다.

FP32 reference:
- autocast off, FP32 params, TF32 off, torch의 float32 matmul highest precision 지원 확인.
- deterministic 설정과 attention/backend 상태를 기록한다. 새 API를 외워서 쓰지 말고 설치 버전 확인.
- 같은 kernel 선택을 비교 양쪽에 적용한다. backend를 arm별로 다르게 하지 않는다.
- dropout/RNG, model state, optimizer state, gradient zero 상태를 복구한다.
- pinball residual이 0 근처인지, 두 계산에서 부호가 바뀐 개수를 진단으로 기록한다.
  그 위치를 손실에서 제거하거나 임의 smoothing하지 않는다.

내보낼 것:
normalized z; raw; raw/train_std; task scalar; clip 전/후 모든 학습 gradient;
실제 parameter delta; Adam exp_avg/exp_avg_sq 각각; frozen hash; RNG.
parameter delta는 before/after tensor를 float64로 옮긴 뒤 뺀다.
전체 parameter와의 상대 차이가 아니라 update끼리의 차이를 계산한다.

새 수치 수락 규칙 — 설정 실행 전 고정, 한 번 실패하면 더 늘리지 않음:
A) same microbatch, checkpoint on/off:
   z, raw/train_std, loss, raw/clipped gradients, update, Adam 각각
   RMS(error) <= 1e-7 + 1e-5*RMS(reference).
   z와 scaled raw의 max-abs error <=1e-4. RNG/frozen hashes는 exact.
   원단위 raw max-abs와 tensor별 extrema는 진단에 보존한다.
B) FP32 microbatch1 대2:
   z 및 raw/train_std: RMS(error)<=1e-6+1e-4*RMS(reference), max-abs<=1e-4.
   scalar task: abs(diff)<=1e-6+1e-4*abs(reference).
   raw/clipped gradient와 update: RMS(error)<=1e-8+1e-4*RMS(reference).
   Adam은 exp_avg와 exp_avg_sq를 섞지 않고 각각 기록;
     RMS(error)<=1e-10+2e-4*RMS(reference).
   전체 벡터 기준 외 tensor별 수치도 기록; 유한한 새 tensor와 같은 params 집합 확인.
   개별 tensor의 norm이 극소인 경우 큰 relative 값만으로 다시 판정하지 않는다.
   step 후 같은 고정 probe 입력의 z/scaled raw도 위 출력 규칙을 만족해야 한다.
C) BF16 microbatch1 대2 (실제 학습 경로):
   기존 의미 유지: z, raw/train_std, raw gradient, update의 relative-L2<=1e-2.
   참조 norm<1e-12이면 max-abs<=1e-6.
   scaled raw max-abs<=2e-2. loss/Adam은 유한성과 진단 수치도 기록.
   같은 입력·파라미터의 two layouts를 비교하며 FP32와 BF16 자체의 완전 동일성을 요구하지 않음.

이 값들은 PyTorch가 보장하는 보편 오차 한계도, 모델 정확도의 허용 손해도 아니다.
의미 검사 + 실제 forward/gradient/update의 제한적 근접성을 수락하는 새 운영 계약이다.
통과해도 모든 학습 궤적이 같거나 원인이 순수 반올림이라고 증명했다고 쓰지 않는다.
기존 1e-5 기준의 결과도 나란히 계산해 변경 효과를 공개한다.

FP32 기준 미충족 시:
- 가장 큰 차이의 tensor/연산/마스크/손실 분기 위치를 한 차례 추적.
- 필요하면 해당 작은 연산 블록을 float64로 재계산해 차이 감소 여부를 기록.
- 전체 모델을 강제로 FP64로 돌려 메모리를 넘기거나 무한한 원인 분석을 하지 않음.
- 의미 버그면 증거와 패치 기록 후 N0/N1을 남은 예산 안에서 한 번 다시 검사.
- 여전히 실패하면 R1은 INCONCLUSIVE_NUMERICS_V2. Standard만 무효 옵션으로 빼지 않는다.
  R2 환경이 안전하다면 R2는 별도 수행한다.

N1과 제한 재검사/초기 warmup을 합친 실제 폐기 optimizer updates<=96.
한 번의 의미 없는 체크가 넘어갔다고 GPU 본학습 완료라고 쓰지 않는다.

R1.4 A — 자원 프로파일

N1이 통과한 경우 기존 실행기의 profile을 새 경로에서 완료한다.
기존 run의 결과가 이미 존재해도 새 정밀도 정책에서 재사용 가능한지 hash/옵션/precision이
모두 일치하는 경우만 참조한다. 과거 cold/warmup 숫자를 새 반복 속도 결과로 대체하지 않는다.

모델:
amazon/chronos-2, revision29ec3766d36d6f73f0696f85560a422f50e8498c.
standard/side/query 각1,179,648 trainable params; frozen native output head.
context4096/H48; 2 origins×4channels; BF16 autocast와 FP32 parameters/Adam.
LR Standard/Query1e-4, Side1e-3; AdamW beta(.9,.999), eps1e-8, wd0; clip1.
원래 Query/Side의 frozen full-context 계산·cache 생성/해제 비용도 매 step 포함.

데이터(0-based, 양끝 포함인 origins):
train16384..22528 stride32; V23040..23712 stride96; E24576..26016 stride96.
train_std=[16384,22576)의 채널별 float64 std, floor1e-6.
원천은 기존 ETTm2/Electricity 첫4채널. 정확한 ID와 raw/processed hash는 기존 contract에서 읽는다.
기존 E를 다시 사용하는 개발 비교다. 새로운 test라고 이름만 바꾸지 않는다.

36옵션:
Standard CP0/3/6/9/12 × micro origins1/2 ×2 sources.
Side, Query 각각 CP0/12 × micro origins1/2 ×2 sources.
CP block indices는 현재 block_indices(k)의 linspace 규칙을 그대로 쓴다.
동일 arm의 동일 warm parameters/Adam/RNG에서 각 옵션을 시작한다.

측정:
기존처럼 3 timing blocks, 각 block3 complete updates.
원점쌍 순서 [17408,18432], [19456,20480], [17408,18432].
옵션 순서를 seed59000+block_index로 고정한다.
block1로 저장/배치 옵션 선택; block2/3으로 속도 확인; 확인 결과로 재선택하지 않음.
cold optimizer state 생성과 warm peak 모두 측정. 전체 모델 allocation을 빼지 않음.
완전 step 시간=zero_grad+입력 전송+frozen encode/cache+forward/loss+backward+
유한성 검사+clip+Adam step+CUDA synchronize.
파일 저장, gradient export, 모델 재생성은 별도 wall 장부; 같은 제외 원칙 적용.
A의 모든 warmup/cold/parity/timing/실패 updates<=480. N1의96과 별도 장부.

B* 선택:
예산{1,2,4,8}GiB. peak allocated<=0.95*budget인 유효 옵션을 feasible로 간주.
reserved/NVML/물리 free도 별도 기록. simulated tensor budget이지 해당 크기 GPU 실행 증명이 아님.
두 sources의 세 arms 모두 적어도 한 옵션을 가지는 가장 작은 공통 budget을 B*로 선택.
예측 점수와 속도 gain으로 B*를 고르지 않는다.
각 arm의 옵션은 B* 안에서 block1 time 최소;2%내 동률이면 낮은 peak→작은CP→micro2.
Train에서 origin1 평가 forward도 B* 안에 들어오는지 확인.

본학습 진입:
1) 의미/수치/자원 검사 유효.
2) 적어도 한 source에서 Standard의 무제약 최속 옵션이 B*에서는 불가능.
3) 선택 Query가 같은 source의 선택 Standard보다 block2와3 각각>=5% 빠름.
   speed_gain=100*(t_Standard-t_Query)/t_Standard.
모든 source에서 이길 필요 없음. Side보다 빠를 필요도 없음. 모든 수치를 보고.
충족=RESOURCE_SIGNAL → B; 미충족=STOP_RESOURCE_SCREEN → R1 B0회, R2 진행.

R1.5 B — 최대12 fits

{ETTm2,Electricity}×{39000,39001}×{Standard,Side,Query}.
동일 seed는 같은8192 effective-batch stream의 prefix. 빠른 arm의 추가 updates는 허용하고 횟수 공개.
active training120초/fit; V checkpoints0/30/60/120초; 각 경계의 첫 완전 update 종료 시 저장.
overshoot<=1초, update cap8192. A에서 step>1초면 INVALID_TIMING_PREFLIGHT로 중단.
시간·메모리 위반을 본 뒤 budget이나 설정을 바꾸지 않는다.
학습은 새 초기 상태로 시작. N1/A에서 적응한 state를 넘기지 않는다.
각 fit에서 best V 상태를 실제 파일로 저장하고 재로드해 동일 입력 예측 재생.
0단계도 선택 가능; 의미상 F0와 같은지는 arm별 초기 경로 차이로 확인.
V 최소 primary, 정확한 동률은 더 이른 checkpoint.

12 fits 선택을 봉인한 뒤에만 E를 계산한다. 누락 fit이 있으면 핵심 비교는 INCONCLUSIVE_EXECUTION.
F0, 각 arm 자체초기상태, selected 모델 예측을 같은 input/precision으로 저장.
Primary=기존 equal-channel train-std-scaled raw 2-pinball; 낮을수록 좋음.
G=100*(L_baseline-L_query)/L_baseline; Standard/Side/F0를 각각 보고.
16E origins를 시간순4개씩4blocks로 묶은 paired bootstrap1000회, seed59100.
이는 기존 개발자료의 기술적 구간이며 독립확증/다중탐색 보정이 아님.

QUALITY_SIGNAL:
적어도 한 source에서 Query 두 seed 평균이 Standard와 Side 모두보다 낮고,
각 seed에서 Standard 대비 gain>0. effect size와 다른source 손해는 그대로 기록.
자원 우위만 있으면 TRADEOFF. 예측이 나쁘면 같다고 바꾸어 말하지 않음.
F0보다 나쁜데 Standard보다 좋으면 적응손해 완화이지 유용한 적응 확보라고 하지 않음.
새 LR/seed/context 탐색 없이 R1 종료.

R1 산출물:
NUMERICS_REVIEW.md (기존/신규 기준, 의미와 수치 진단 분리)
원래 실패 유지, 신규 contract, 모든 profile rows, B* 선택 근거, 실제 fits/updates,
train/V curves, 선택 seal, prediction cache hashes, 원점수, 자원·정확도 별도 REPORT.md.
같은 정의를 벡터화한 지표와 별도의 scalar-loop 지표로 각각 재계산해 abs 차이 <=1e-10인지 검사한다.

======================================================================
R2. 채널 변환의 공유·독립·저랭크·basis 비교 — 최대 24 fits
======================================================================

R2.1 정확한 연구 질문과 선행 경계

[설계] “32개 채널에 같은 학습 파라미터 예산을 줄 때, 채널별 변환을 소수의 공통 affine
변환과 작은 채널별 계수로 표현하는 것이 완전 공유·개별 변환·일반 저랭크보다 유리한가?”
부차 질문은 큰 채널별 어댑터의 정확도를 더 적은 총 학습 파라미터로 유지하는가이다.
첫째는 동일 예산 비교, 둘째는 파라미터 절감 비교다. 두 주장을 섞지 않는다.
affine 변환은 행렬을 곱하고 bias를 더하는 변환이며, basis는 공유할 기본 변환을 뜻한다.

[확인 R2-S1] Time-PEFT 공개 run.py는 MOMENT를 사용하고 LoRA·주파수 어댑터·채널 어댑터·HEAD를 학습한다.
채널 블록은 공통 down projection(2D→h), ReLU, dropout 0.1, 채널별 up(h→D), 공통 LayerNorm이다.
[확인 R2-S2] C-LoRA(CIKM 2024)도 채널별 저랭크와 공유 행렬을 사용한다.
[확인 R2-S3] MoLA 공개본도 LoRA의 부분 파라미터 공유를 다루지만, 예측 step별 공유가 핵심이다.
따라서 “공통 변환 + 작은 계수” 자체를 새 개념이라고 전제하지 않는다.

BASIS는 아래 수식으로 정의한 [미검증] 실험 후보다.
성능 SIGNAL과 novelty status(차이 있음/가까움/동치/미확인)를 별도로 보고한다.
선행과 큰 원리가 비슷하다는 이유만으로 GPU를 막지는 않는다. 다만 같은 모델·적용 위치·수식·
학습/평가 설정의 결과가 이미 현재 저장소에 있으면 중복 실행하지 않는다.
정확히 기존 재매개변수화와 동치라면 KNOWN_PARAMETERIZATION으로 표시하고 “새 방법 발견” 주장을 금지한다.
그 경우 실험의 용도는 알려진 구조가 이번 예산에서 유리한지 확인하는 것이다.
그 비교도 이미 답해졌으면 R2는 DUPLICATE_STUDY로 종료한다.

진입 전 선행 표에는 최소 Time-PEFT / Channel-Aware Low-Rank Adaptation / 일반 공유행렬
factorization을 포함한다. 필요하면 MoLA·채널/전문가 basis 공유 논문을 최대 3개 추가한다.
수식·삽입 위치·학습 백본 여부·채널/예측 step 구분·파라미터 수·공식 구현 확인 깊이를 나란히 적는다.
문헌조사는 새 후보를 무한히 발명하는 작업이 아니다. 이번 고정 비교의 해석 경계에만 사용한다.

R2.2 모델과 환경 — R1과 독립

MOMENT-small: AutonLab/MOMENT-1-small, D=512.
upstream run.py의 build_model/TimePEFTPipeline/FrequencyAdapter/ChannelAdapter를 먼저 읽는다.
실제 모델 revision을 prepare에서 확인·pin하고 weight hashes를 봉인한다.
로컬 cache가 있으면 재사용한다. 없으면 공개 checkpoint만 프로젝트 cache에 받아 고정한다.
권한·네트워크·모델 불일치가 있으면 BLOCKED_MODEL이다. 다른 backbone으로 바꾸지 않는다.

환경은 R1의 Chronos .venv와 분리한다. 필요하면 전용 venv를 새로 만든다.
공식 requirements와 현재 설치 API를 확인한 뒤 dependency lock을 작성한다.
기존 프로젝트의 torch/CUDA/드라이버를 변경하지 않는다. 오래된 torch 버전을 전역 설치하지 않는다.
호환성 수정은 전용 venv에서 기록한다. R1의 monkeypatch 상태가 전달되지 않게 별도 프로세스로 실행한다.

공통 모델 계약:
- L=96, H=96, C=32. MOMENT patch length=8은 실제 config로 검증한다.
- 사전학습 encoder/embedder/normalizer의 기존 파라미터는 고정한다.
- attention LoRA rank=8, alpha=32, target=q/k/v로 고정한다.
- forecasting HEAD는 모든 arm에서 학습한다. 실제 shape와 학습 파라미터 목록을 기록한다.
- pretrained weights는 공통이다. source/seed별 최초 HEAD와 LoRA state도 공통이다.
- 새 forecasting HEAD의 초기 상태를 “유효한 zero-shot F0”라고 부르지 않는다.
  pretrained forecasting HEAD 존재가 확인되지 않으면 초기 상태의 이름은 INIT로 쓴다.
- 무학습 기준선은 seasonal-naive, 즉 시간별 자료의 직전 24시간 패턴 반복이다. fit으로 세지 않는다.
- LH 참고군과 동일 예산 4개 arm의 주 비교를 구분한다.

다섯 channel-adapter arm에는 FrequencyAdapter(top_k=3, D=512)를 동일하게 유지한다.
학습 경로는 다음과 같다.
  MOMENT embeddings [batch, channel, patch, D]
  → frequency branch
  → concat(backbone, frequency)
  → 각 channel adapter
  → 공통 HEAD
공유한다는 것은 가중치를 공유한다는 뜻이다. 다른 채널의 현재 입력을 attention으로 읽는다는 뜻이 아니다.
이 실험을 channel-interaction의 인과 검증으로 주장하지 않는다.

[확인 R2-S1] 현재 upstream train()은 best_loss 숫자를 갱신하지만, 읽은 코드에는 best weights를
저장·복구하는 경로가 보이지 않고 마지막 model을 반환한다.
새 runner에서는 모든 arm에 실제 best-checkpoint 저장/복구를 똑같이 구현한다.
이 수정, small 모델, 32채널, 20 epoch 상한과 자체 split 때문에 공식 논문 표의 완전 재현이 아니다.
Time-PEFT-inspired controlled pilot라고 표시한다. 논문의 최대 개선율을 목표값으로 사용하지 않는다.

R2.3 여섯 arms — LoRA/HEAD 정책과 학습 예산 고정

LH:
  LoRA + trainable forecasting HEAD. Frequency/channel adapter는 없다.
  더 적은 parameter를 쓰는 기본 적응 참고군이다. 동일 예산 상대라고 표시하지 않는다.
INDIV_REF:
  Time-PEFT-style channel block, h=256(공개 ratio=2), 채널별 full up-projection.
  큰 원형의 참조군이다. 동일 예산 그룹에 넣지 않는다.
INDIV_BUDGET:
  같은 구조를 h=16으로 줄인다. 채널별 full up-projection을 유지한다.
SHARED_BUDGET:
  모든 채널이 up-projection 하나를 공유한다. h=191.
FACTOR_BUDGET:
  채널별 left factor와 공통 right factor를 사용한다. h=95, q=51.
BASIS_BUDGET:
  공통 affine up + 3개 residual affine basis와 32×3개 계수를 사용한다. h=95.

주 비교는 INDIV_BUDGET / SHARED_BUDGET / FACTOR_BUDGET / BASIS_BUDGET이다.
이 네 arm을 모두 같은 합성 이름의 새로운 PEFT라고 부르지 않고, 비교를 위해 정의한 매개변수화로 구분한다.
다섯 channel arm의 frequency module, attention LoRA, HEAD, LayerNorm 정책은 동일하다.
LoRA rank나 HEAD를 한 arm에서만 키워 budget을 채우지 않는다. 사용하지 않는 dummy parameter도 금지한다.

수학적 정의 — 아래 기호는 이번에 구현할 명세이지 기존 변수명이 아니다.
H_c,F_c ∈ R^(patch×D), X_c=concat(H_c,F_c).
Z_c=Dropout_0.1(ReLU(X_c W_down^T + b_down)), W_down ∈ R^(h×2D).
최종 출력은 각 up 결과에 같은 구조의 trainable LayerNorm(D)를 적용한다.
LayerNorm eps와 affine 여부는 공개 모듈 설정을 기록해 모든 arm에서 통일한다.

INDIV:
  Z_c U_c^T + b_c, U_c ∈ R^(D×h), b_c ∈ R^D.
SHARED:
  Z_c U^T + b, 모든 채널이 동일한 U,b를 사용한다.
FACTOR:
  (Z_c A_c) V + b_c, A_c ∈ R^(h×q), V ∈ R^(q×D).
  factor 중간에 추가 ReLU나 layer를 넣지 않는다.
  C-LoRA의 공식 삽입·concat을 재현한 것이 아니라 가장 단순한 공유-factor 대조다.
  C-LoRA 원문 수식과의 차이를 명시한다.
BASIS:
  Z_c U_0^T + b_0 + sum(k=1..3) a_ck * (Z_c U_k^T + b_k).
  a_ck는 학습 가능한 실수이며 시간이나 미래 target에 의존하지 않는다.
  softmax/router/clustering을 추가하지 않는다.
  정적 계수이므로 evaluation에서 U_c=U_0+sum a_ck U_k로 합칠 수 있다.
  이 성질만으로 신규성이나 속도 이득을 주장하지 않는다.

파라미터 계산 — 채널 블록만, bias와 LayerNorm을 포함한다.
P_indiv(h) = (2D+1+CD)h + CD + 2D.
P_shared(h) = (3D+1)h + 3D.
P_factor(h,q) = (2D+1)h + C*h*q + q*D + CD + 2D.
P_basis(h,K) = (2D+1+(K+1)D)h + (K+1)D + CK + 2D; K=3.

D=512, C=32의 검산값:
  INDIV_REF(h=256)           4,474,112
  INDIV_BUDGET(h=16)           295,952
  SHARED_BUDGET(h=191)         295,103
  FACTOR_BUDGET(h=95,q=51)     295,935
  BASIS_BUDGET(h=95,K=3)       295,103

동일 예산 그룹은 목표 295,952개 대비 차이가 0.5% 이하여야 한다.
실제 named_parameters/numel로 모두 재계산한다. 공통 LoRA+frequency+HEAD를 더한 총수도 별도 표에 넣는다.
채널 블록의 절감률을 전체 trainable 절감률이라고 쓰지 않는다.
이 비교는 폭과 예산 배분까지 바꾸므로 “같은 폭에서 basis만 바꾼 인과효과”라고 부르지 않는다.
실제 D가 512와 다르면 폭을 몰래 바꾸지 말고 CONTRACT_MISMATCH로 중단한다.

초기화:
- frozen weights, LoRA, HEAD, frequency의 공통 state는 source/seed별로 exact clone한다.
- down-projection의 폭은 다르므로 같은 tensor를 억지로 복제하지 않는다. fan-in 초기화 규칙을 통일한다.
- INDIV arm 안에서는 공통 초기 U를 모든 채널에 복사하고 학습 중 독립 parameter로 둔다.
- SHARED는 U 하나를 공유한다. up-projection bias는 모든 channel arm에서 0으로 초기화한다.
- BASIS의 U0는 일반 Linear fan-in 초기화, residual Uk,bk는 0이다.
  계수 a는 고정 seed의 normal(std=0.1)로 초기화한다. 모든 계수를 같은 값으로 초기화하지 않는다.
- FACTOR의 A_c는 평균 0, 분산 1/h인 정규분포에서 뽑은 공통 A를 채널마다 복사한다.
  V는 평균 0, 분산 1/(3q)인 정규분포에서 뽑는다. PyTorch의 std에는 각각 제곱근을 넣는다. 이 곱의 초기 유효 weight 분산은 1/(3h)이다.
  초기화 시드와 수식을 기록하고 결과를 보고 바꾸지 않는다.
- 서로 다른 폭의 arm들이 INIT에서 완전히 같은 예측을 해야 한다고 요구하지 않는다.
- 첫 step의 LoRA A-gradient 또는 BASIS 계수 gradient가 0일 수 있다. 그 사실만으로 실패하지 않는다.
- main training에서 BASIS weight를 detach하거나 window 사이에 cache하지 않는다.
  static merge는 inference 진단에만 적용한다.

R2.4 코드·데이터 검사 — 성능 학습 전

필수 CPU 검사:
1) 위 parameter 수가 일치하고 공통 학습 parameter 집합에 차이가 없는지 확인한다.
2) BASIS의 explicit sum과 merged up의 fp64 출력·gradient를 비교한다(rtol=1e-9, atol=1e-11).
3) 계수를 0으로 둔 BASIS가 같은 폭의 shared U0와 일치하는지 확인한다.
4) dropout을 끈 eval 모드에서 채널과 해당 parameter/계수를 함께 순열 변경하면 출력도 같이 순열 변경되는지 확인한다.
5) channel parameter가 원본 32개 ID에 일관되게 연결되는지 확인한다.
6) loss/mask/마지막 작은 batch의 가중치를 독립 scalar 계산으로 검증한다.
7) 선택 checkpoint 저장→새 모델 구성→restore 후 같은 입력의 예측을 재생한다.
8) INDIV_REF의 forward를 vendored Time-PEFT의 동일 state/FP32/eval과 비교한다.
   수치 일치가 공식 논문의 성능 재현을 의미하지는 않는다.

실제 모델 smoke는 6 arms × 2 sources × 2 updates = 최대 24 updates다.
같은 Train 배치에서 loss/gradient/optimizer/frozen hash/HEAD update를 확인한다.
첫 gradient가 0일 수 있는 초기화와 학습 경로가 완전히 끊긴 오류를 구분한다.
smoke 상태를 main training에 넘기지 않는다.

원자료:
- Electricity: 기존 data/raw/electricity.txt.gz 및 해당 manifest를 먼저 확인한다.
- Traffic: data/raw/overnight_20260913/traffic.txt.gz.
- Traffic raw SHA256:
  c7be5a00519d344a5ec0eabdbfec5ea0c7dd1eed5f9b1a3843a93bb88086a56d.
- Electricity hash는 기존 processed/계약에서 읽어 확인한다. 값을 만들어 넣지 않는다.
- 기존 4채널 processed 파일을 복제해 32채널로 늘리지 않는다. 원시 전체 행렬에서 선택한다.

시간 분할:
원자료 행수 T에 대해 t1=floor(0.7T), t2=floor(0.8T).
Train targets=[0,t1), V targets=[t1,t2), E targets=[t2,T).
기존에 본 기간·채널을 exposure ledger에 표시한다. 새 원천이나 미노출 test라고 주장하지 않는다.
예상 행수는 Electricity 26304, Traffic 17544다. 다르면 원인을 보고하고 자동 절단·대체하지 않는다.

채널 선택:
원본 열 순서에서 Train[0,t1)에 finite하고 std>1e-6인 첫 32개를 선택한다.
V/E 성능이나 정답 요약으로 채널을 선택하지 않는다. 32개 미만이면 BLOCKED_DATA다.
표준화 mean/std는 선택 채널의 Train에서만 float64로 계산한다. V/E에는 같은 값을 적용한다.
모델의 normalizer는 유지한다. 정규화를 바꿔 성능을 맞추지 않는다.
0을 임의로 결측으로 취급하지 않는다. 진짜 입력 NaN은 같은 window의 과거 값으로 forward-fill하고,
처음부터 결측이면 train mean을 사용해 채운다. 미래 정답은 impute하지 않고 공통 mask로 제외한다.
V/E 정답이 모두 결측인 채널은 선택에서 제거하지 않고 N/A와 유효 분모를 기록한다.

원점(0-based):
  Train grid = range(96,t1-96+1,24).
  512개보다 많으면 linspace로 양끝을 포함하는 512개를 선택한다. 이하면 전체를 사용한다.
  Train 128개 미만이면 BLOCKED_DATA다.
  V = range(t1,t2-96+1,96), E = range(t2,T-96+1,96). 각각 최소 8개 원점이 필요하다.
  x=[o-96,o), y=[o,o+96).
Train/V/E의 target은 겹치면 안 된다. V/E의 입력이 앞선 기간을 포함하는 것은 허용한다.
예상 원점 수는 Electricity 512/27/54, Traffic 504/18/36이다. 실제 값을 다시 계산한다.
전체 원점·ID·행 범위·hash를 성능 계산 전에 봉인한다.

R2.5 학습 — 실제 최대 24 fits

sources={Electricity,Traffic}, seeds={41000,41001}, 위의 여섯 arms.
실행 순서는 seed 61510으로 고정 shuffle한다. 먼저 좋은 원천만 보고 다음 원천을 고르지 않는다.
source/seed별 epoch의 Train 원점 순서를 공유한다. 학습 중 새 window나 channel clustering을 추가하지 않는다.

고정 recipe:
- 최대 20 epochs, early stopping patience=5, min_delta=1e-4(표준화 V MSE).
- AdamW lr=1e-3, betas=(.9,.999), eps=1e-8, weight_decay=0, gradient clip=1.
- StepLR: 매 5 epochs gamma=0.5.
- HEAD/LoRA/frequency/channel에 같은 LR을 사용한다. 별도 LR 탐색은 하지 않는다.
- 동일 시간 비교가 아니라 같은 epoch·데이터 노출 상한 비교다. 실제 시간도 함께 기록한다.
- 공개 Time-PEFT의 최대 100 epochs보다 작은 개발 상한이다. 충분한 수렴을 주장하지 않는다.

Batch와 precision:
- effective batch = 8 origins × 32 channels.
- R2 전용 BF16 autocast, FP32 parameters/Adam. FFT는 FP32로 수행한다.
- 배치를 나눌 때 origin 단위로만 나누고 32채널을 함께 유지한다.
- 유효 target 수를 반영한 올바른 loss weighting으로 누적한 뒤 clipping과 step을 한 번 적용한다.
- 사전 dry-fit에서 micro-origins {8,4,2,1}을 큰 순서로 검사한다.
  두 원천의 모든 arm에서 안전하게 가능한 가장 큰 공통 값을 선택한다. 예측 점수는 사용하지 않는다.
- 이 탐색은 forward/backward만 하는 검사로 제한한다. optimizer state를 제외하고 가능하다고
  결론 내리지 말고, 상태 크기를 산정한 뒤 선택된 값의 실제 24-update smoke에서 확인한다.
- 사전 목록의 다음 작은 값으로 가는 것은 허용하지만 main 학습 중 자동 batch 축소는 금지한다.
- BF16 검사 실패 또는 micro=1도 불가능하면 BLOCKED_RESOURCE다. 별도 precision으로 몰래 재시도하지 않는다.
- 모든 arm에 TF32/backend 설정을 통일한다. 새 compile/offload/양자화는 추가하지 않는다.

Validation과 선택:
매 epoch마다 같은 mask와 배치 순서로 V를 평가한다. INIT도 기록하고 선택 후보에 포함한다.
V 집계는 배치 평균들의 단순 평균이 아니라 원소별·채널별 유효 분모를 사용한다.
best V의 weights를 저장하고 종료 시 실제 best state를 복구한다.
작은 개선도 실제 최소 V checkpoint 기록에는 반영한다. min_delta는 patience를 리셋하는 기준이다.
따라서 early stopping의 “의미 있는 개선”과 실제 최저 V checkpoint 선택을 분리해 구현한다.

24개 cell의 선택이 완성된 뒤 E를 한 번 연다.
오류 fit이나 wall cap으로 불완전하면 안전한 나머지 예정 cell까지만 수행하고 핵심 E는 열지 않는다.
상태는 INCONCLUSIVE_EXECUTION이다. 시간 상한 중단을 과학적 FAIL로 기록하지 않는다.
fit별 최대 update 수는 Electricity 1280, Traffic 1260으로 예상되며 실제 grid로 검증한다.
20 epoch 끝에서도 V가 개선되는 경우 BUDGET_LIMITED를 표시한다. 추가 epoch는 자동 실행하지 않는다.

R2.6 지표·대조·판정

Primary는 Train 표준화 단위의 equal-channel MSE다.
채널별 유효 origin×horizon의 제곱 오차를 평균한 뒤 채널 평균을 구한다.
보조 지표는 표준화 MAE, 원단위 채널별 MAE, updates, 학습 파라미터 수, active/wall 시간,
peak allocated/reserved, NVML, INIT/LH/seasonal-naive 대비 결과다.
R1의 scaled pinball 숫자와 직접 비교하지 않는다.

주 비교:
1) 동일 예산 네 arm을 각각 비교하고 채널 예산·전체 trainable을 둘 다 표시한다.
2) source/seed별로 V만 사용해 INDIV_BUDGET, SHARED_BUDGET, FACTOR_BUDGET 중
   가장 좋은 대조군을 선택하고 봉인한다. 선택 비용도 보고한다.
   E에서 사후 고른 최선 대조군을 실제 선택 정책 성능이라고 부르지 않는다.
3) BASIS 대 INDIV_REF는 동일 예산이 아닌 압축 절충 비교다.
4) LH는 간단한 LoRA+HEAD 참고군이다. 채널 공유의 추가 이득과 전체 pipeline 이득을 구분한다.

후속 투자 신호 — 논문 합격·신규성 기준이 아니다.
BUDGET_SIGNAL:
- 적어도 한 원천에서 두 seed 각각 BASIS의 오차가 V-선택 동일 예산 대조군보다 낮다.
- 해당 원천에서 seed 평균 원점수로 계산한 상대 감소율이 0.5% 이상이다.
- 해당 원천의 BASIS 평균이 LH 평균보다도 낮다.
다른 원천의 손해를 숨기지 않는다. 모든 원천에서 승리할 필요는 없다.
0.5%는 이번 pilot의 후속 투자 기준이다. 그보다 작은 양성 차이도 원점수와 함께 보존한다.

COMPRESSION_SIGNAL(보조):
BASIS의 전체 trainable이 INDIV_REF보다 20% 이상 적고, 해당 원천의 seed 평균 MSE 악화가 0.5% 이내다.
원천별로 판정하고 충족 범위만 보고한다. 이 범위는 통계적 비열등성 증명이 아니다.
BASIS가 큰 REF보다 더 좋더라도 파라미터 수·최적화·정규화의 경쟁 설명을 남긴다.

음성 결과의 해석:
- SHARED가 충분하다 → 작은 채널별 계수의 필요성이 확보되지 않았다.
- FACTOR가 충분하다 → basis bank의 추가 가치가 확보되지 않았다.
- 큰 REF만 좋다 → 현재 예산에서 정확도 유지에 실패했다.
- 모든 channel arm이 LH보다 나쁘다 → 해당 설정에서 channel pipeline 추가 가치가 없다.
- INIT/seasonal-naive가 좋다 → 최적화·학습 상한도 살펴보되 그 자체로 구현 오류를 단정하지 않는다.

불확실성:
원천별 시간순 E 원점을 연속 최대 8개 블록으로 나누어 paired block bootstrap 1000회(seed 61520).
같은 resample을 모든 arm/seed/channel에 적용한다. 두 seed로 데이터 표본 수를 두 배로 세지 않는다.
이미 사용한 원천·기간이므로 개발 기술통계다. 좋은 채널·시간만 골라 주 평가를 바꾸지 않는다.

메커니즘 진단 — 추가 학습 0회:
선택된 BASIS의 V에서만 계수의 채널 순서를 고정 seed로 섞고,
계수를 채널 평균으로 대체하는 두 개입을 수행한다. 선택·threshold 변경에는 사용하지 않는다.
원래 결과와 거의 같으면 채널별 계수가 이득을 만드는지 의문으로 보고한다.
이 개입은 학습 분포 밖 입력일 수 있으므로 고유한 인과 기여 증명으로 해석하지 않는다.
static merge의 정확성 검사 후 inference 비용은 별도 측정할 수 있지만,
훈련 속도에 merge의 비용 절감을 몰래 적용하지 않는다.

R2.7 산출물

PROTOCOL.md, NOVELTY_BOUNDARY.md, UPSTREAM_DIFF.md, PARAMETER_BUDGET.csv,
MODEL/DATA/ENV_RECEIPTS, FIT_MANIFEST, CPU/smoke 결과, 모든 Train/V 곡선과 원점수,
best checkpoint seals, E 예측 cache hashes, 원천·seed·채널별 점수,
resources.csv, 독립 float64 지표 재계산, 한국어 REPORT.md를 남긴다.

보고서는 다음 질문 순서로 쓴다.
어떤 파라미터를 공유했나 → 왜 필요한가 → 단순 공유/저랭크와 무엇이 다른가 →
비교 예산과 HEAD 정책은 같은가 → 정확도·파라미터 이득이 있나 → 신규성·평가의 한계는 무엇인가.
PASS 하나로 묶지 말고 EXECUTION_VALID, BUDGET_SIGNAL, COMPRESSION_SIGNAL,
REFERENCE_EFFECT_REPRODUCED_OR_NOT, NOVELTY_STATUS를 따로 표시한다.

======================================================================
3. CLI 실행 인터페이스·최종 통합 보고
======================================================================

아래는 기존 명령이라고 가정하지 말고 이번에 만들 인터페이스다.
experiments/peft_rank12_20260915/run_rank1.py:
  prepare / numerical-check / profile / train-evaluate / verify / status
experiments/peft_rank12_20260915/run_rank2.py:
  prepare / smoke / train-evaluate / verify / status
experiments/peft_rank12_20260915/run_all.py:
  R1 → R2를 순차 수행한다. 각 트랙의 조건 미충족은 이유를 저장하고 다음 독립 트랙으로 진행한다.
  단순 exception 하나로 둘 다 실행 완료라고 쓰지 않고, 환경 고장인지 연구 중단인지 구분한다.

구현과 --help/CPU tests를 확인한 뒤 실행한다. 원래 runner의 전역 RUN/OUT/CACHE만 거칠게 바꿔
source hash·선택 seal·자원 장부가 다른 run을 가리키는 오류를 만들지 않는다.
같은 미완료 run의 유효 checkpoint로 재개할 수 있어도 이번 시도 계약에 명시한 경우에만 재개한다.
다른 PID의 동일 작업이 실행 중이면 그 상태를 읽고 중복 실행하지 않는다.

최종 research/peft_rank12_20260915/REPORT.md에는 다음을 담는다.
1) R1 실제 진행량(수치 진단/profile/본학습)과 R2 실제 진행량을 나란히 표시한다.
2) 기존 INCONCLUSIVE와 새 판정의 차이: 무엇을 바꾸었고 무엇을 유지했는가?
3) R1의 자원 신호와 예측 신호를 분리한다. 미실행 B의 오차는 0이 아니라 NOT_RUN이다.
4) R2의 동일 예산 4개 arm/큰 참조군/LH의 원점수와 파라미터 수를 기록한다. 원단위·표준화를 명시한다.
5) 관찰과 추정을 분리하고, 실패 또는 미확정 원인을 근거와 함께 설명한다.
6) 두 연구의 지표를 직접 합치지 말고 NEXT_ACTION을 각각 제시한다.
7) 후속 제안은 트랙당 최대 1개다. 별도 후속 학습은 자동 실행하지 않는다.

원점수 CSV와 hash도 남기되 사용자에게 JSON만 제시하지 않는다.
그림은 실제 결과가 있는 경우에만 만들고 한 그림에 한 주장을 담는다. 빈 실험의 가상 그래프는 금지한다.
그림보다 원점수·분모·반복 수·선택 근거 확인이 먼저다.

기존에 승인된 scoped commit/push 권한을 현재 프로젝트 기록에서 확인한 경우에만
새 코드·설계·표·보고서·검증 기록을 커밋/푸시하고 RESULTS_INDEX에 항목을 추가한다.
권한이 확인되지 않으면 로컬 결과를 보존하고 “업로드 미실행”이라고 보고한다.
새 공개 저장소를 무단 생성하지 않는다. 상류 참조 코드의 라이선스와 수정 사항 표기를 보존한다.

======================================================================
출처와 확인 범위
======================================================================

[R1-S1] 기존 Query 수치 중단과 미실행 B:
https://github.com/CanelE452/tsfm-peft-method-screen/blob/9051d03cebe23ae35d9689e7af576c63c4d5edd5/results/query_budget_pilot_20260915/REPORT.md
[R1-S2] 실제 비교/메모리 wrapper:
https://github.com/CanelE452/tsfm-peft-method-screen/blob/9051d03cebe23ae35d9689e7af576c63c4d5edd5/src/tsfm_peft_screen/forecast_query/budget.py
[R1-S3] 실행기:
https://github.com/CanelE452/tsfm-peft-method-screen/blob/9051d03cebe23ae35d9689e7af576c63c4d5edd5/scripts/run_query_budget_pilot.py
[R1-S4] PyTorch 공식 numerical accuracy: batch와 slice의 비트 동일성을 보장하지 않음.
https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
[R1-S5] 공식 근접성 검사: 절대/상대 허용오차의 의미.
https://docs.pytorch.org/docs/stable/testing.html
[R1-S6] 중간값 재계산의 메모리/계산 절충:
https://pytorch.org/blog/activation-checkpointing-techniques/

[R2-S1] Time-PEFT: Temporal and Multichannel Complexity-Based Fine-Tuning for
Time-Series Foundation Models(2026/ICML), 저자 공개 코드.
https://github.com/kaist-dmlab/TimePEFT/blob/ea4e7e1887bb35587bab7ea93e2af3685ac55852/run.py
https://github.com/kaist-dmlab/TimePEFT/blob/ea4e7e1887bb35587bab7ea93e2af3685ac55852/README.md
이번 작성에서 공개 코드를 읽었다. R2는 공식 논문 전체 수치 재현이 아니다.

[R2-S2] Channel-Aware Low-Rank Adaptation in Time Series Forecasting(2024/CIKM):
https://arxiv.org/html/2407.17246v1
https://doi.org/10.1145/3627673.3679884
https://github.com/tongnie/C-LoRA
arXiv에서 읽은 저자본이며 CIKM 2024 출판 정보를 확인했다. Continual C-LoRA 등 비슷한 이름의 다른 방법과 혼동하지 않는다.

[R2-S3] Mixture of Low Rank Adaptation with Partial Parameter Sharing for Time
Series Forecasting(2025 공개; 이번 확인은 arXiv 공개본, 정식 게재 상태 미확인):
https://arxiv.org/abs/2505.17872
예측 step별 mixture와 정적인 채널별 basis를 구분한다. 부분 공유라는 큰 아이디어의 최초성을 주장하지 않는다.

[R2-S4] Traffic 원자료 영수증:
https://github.com/CanelE452/tsfm-peft-method-screen/blob/9051d03cebe23ae35d9689e7af576c63c4d5edd5/research/overnight_20260913/data_receipt.json

R1의 새 수치 허용치, R2의 32채널·MOMENT-small·20 epochs·0.5%·24 fits·공유 basis 3개는
이번 검토용 [설계] 선택이다. 선행이 보장한 성능·성공 확률·보편적 논문 PASS 기준이 아니다.

======================================================================
작성 단계에서 수행한 로컬 검산 (실제 모델 학습 결과 아님)
======================================================================

- R2 채널 블록의 수식과 PyTorch로 구성한 실제 모듈의 파라미터 개수가 일치함.
- 동일 예산 4개 arm의 차이 <=0.5%: INDIV 295,952 / SHARED 295,103 / FACTOR 295,935 / BASIS 295,103.
- 작은 float64 텐서에서 BASIS의 explicit sum과 static merge 출력의 최대 차이 약 1.11e-15.
- 예상 행수에서 Train/V/E 원점 개수와 target 비중첩 계산을 확인함.
- 전체 MOMENT/Chronos 모델 실행, 실제 원자료 로드, GPU 자원 검사, 예측 학습은 미실행.
CLI는 이 수치 검산을 실제 모델·실제 데이터의 preflight 대체물로 쓰지 않는다.
