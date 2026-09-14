Query 자원 제약 파일럿 — CLI 실행 지시문
작성일: 2026-09-15 (KST)
대상: CanelE452/tsfm-peft-method-screen
확인 기준 commit: 6a8bf6224a94fb790d26a96a2b9cb7505f24b70d
제안 run ID: query_budget_pilot_20260915

이 문서는 JSON 검산 자료가 아니라, 구현·검사·조건부 학습·평가까지 수행하는 지시문이다.
아래 새 경로와 상태명은 이번 작업에서 만들도록 지정한 이름이며 기존 파일이라고 가정하지 않는다.
이 문서를 작성한 대화에서는 저장소 수정이나 GPU 학습을 실행하지 않았다.

======================================================================
0. 실행 범위와 첫 결정
======================================================================

사용자의 요청은 직전 합의한 두 번째 Query 후보를 실제로 시험하는 것이다.
Censor 재튜닝, 새로운 어댑터 발명, 다른 연구 주제 탐색은 이번 작업에 포함하지 않는다.
계획서만 쓰고 종료하지 말고, 실행 전 검사를 통과하면 A를 수행하고,
A의 사전 조건을 충족하면 별도 확인 없이 B의 최대 12 fits까지 수행하라.
조건 미충족이나 환경 오류는 이유·근거를 저장하고 종료한다. 무한 대기하지 않는다.

A: 기존 모델 구조를 유지한 자원·수치 비교. 본 예측 학습 fits 0.
   단, 비용을 실제 optimizer.step까지 측정하기 위한 폐기용 업데이트는 수행한다.
B: A가 지지할 때만 Standard / Side / Query × 2원천 × 2seed = 12 fit attempts.
   각 fit의 active training budget은 120초이며 0/30/60/120초에 V를 평가한다.
   학습 예산이지 전체 실행 소요시간의 예측이나 보장은 아니다.
마지막: 독립 수치 검증, 원점수·자원·선택 기록, 한국어 결과 보고서 작성.
이후 새 학습, 추가 seed/LR, 다른 budget 학습은 자동으로 실행하지 않는다.

총 제한:
- A의 폐기용 optimizer updates: 최대 480회. warmup·parity·실패 시도 포함.
- B: 최대 12 fit attempts. 실패한 attempt를 새 fit으로 대체하지 않는다.
- B의 fit별 안전 update 상한: 8192회. 시간 예산 전에 도달하면 INVALID_TIMING.
- 외부 GPU compute 대기: 시작·중간을 합쳐 최대 600초.
- GPU 단계 A+B+평가 전체 wall cap: 4시간. 초과 시 부분 결과를 보존하고 종료.
- GPU 동시 실행은 1개 worker만. 다른 프로젝트의 프로세스를 종료하지 않는다.

======================================================================
1. Phase 0 — 목적과 이 실험으로 말할 수 있는 것
======================================================================

최상위 목적:
"동일한 긴 과거 입력, 학습 파라미터 수, 텐서 메모리 예산, 학습시간을 줄 때,
과거 표현을 고정하고 미래 query 경로만 적응하는 기존 Query 구현을 선택할 이유가 있는가?"
소비처는 사용자·지도교수의 후속 투자 결정이다. 이번 결과를 논문 채택 판정으로 쓰지 않는다.

[확인: S1-S5] 현재 Query는 긴 과거를 계산하지 않는 모델이 아니다.
고정 백본으로 과거 전체를 계산하고 층별 정보를 얻은 뒤 4개 query 토큰의 경로를 학습한다.
이 중 최종 예측에는 3개 미래 토큰을 사용한다. 모든 고정 계산·cache 비용을 포함해야 한다.

[확인: S1-S2] 기존 동일시간 실험은 context4096, horizon48, allocated8GiB였으며,
Standard도 빠른 checkpoint-off 설정을 사용할 수 있었다.
이후 자원 표는 새 고정 상태의 메모리·시간과 이전 학습의 정확도를 함께 보여 준다.
그 표를 새 자원 설정에서 다시 학습한 성능표로 취급하지 않는다.

[설계] 이번 차이는 Standard에 부분 재계산과 origin 단위 microbatch를 함께 허용하고,
공통 제약 안의 실제 학습시간을 예측 성능으로 연결하는 것이다.
구조·손실·rank·문맥 길이까지 한꺼번에 바꾸지 않는다.

상류 제한을 먼저 명시한다:
- 사용자가 실제 배포 장비에서 요구하는 VRAM 상한은 이번 대화에서 지정하지 않았다.
- 따라서 기본 실험은 1/2/4/8 GiB의 사전 정의된 텐서 예산 시뮬레이션이다.
- 이 결과만으로 "1GiB GPU에서 실행된다", "현재 GPU에 필수다"라고 주장하지 않는다.
- 단일 context4096 조건으로 문맥이 길어질수록 이점이 커진다고 주장하지 않는다.
- 기존 개발 데이터와 E를 재사용한다. 새로운 독립 test가 아니다.
- Query가 기존 LoRA보다 정확하다는 결론을 미리 두지 않는다.

행동의 이유와 그로부터 나온 실행 조건:
(1) 자원 비교: 학습 투자 가치 확인 + 단순 재계산/배치 분할로 장점이 사라지는지 확인.
    -> Standard의 실행 가능한 빠른 설정을 제외하지 말 것. 손실로 자원 설정을 고르지 말 것.
(2) 동일시간 학습: 단계별 속도를 실제 예측으로 연결 + Side와의 품질 절충 확인.
    -> 동일 시간·동일 V 기회, 고정 백본 계산을 시간에 포함할 것.
(3) 수치 검사: 효율 변경의 의미 보존 + 잘못된 그룹 분할/누출 방지.
    -> 채널 그룹 유지, 누적 gradient 뒤 clipping/step 1회, 저장 방식 동등성 검사.
(4) 검증 및 보고: 계산 실수 방지 + 기존 개발 결과를 독립 확증으로 오인하지 않게 함.
    -> 평가 전 선택 봉인, 원점수 재계산, 이전 결과 불변, 재사용 범위 표시.

역주행 판정:
예상 결과 / 목적 지지 / 최상위 도달 / 독자 질문
- A에서 공정한 Standard 대비 반복되는 속도 이점이 남음 / 부분 / B로 연결 / 예측도 좋아지나?
- A에서 배치 분할·재계산으로 이점이 없어짐 / 현재 속도 가설 불지지 / 종료 / Query가 왜 필요한가?
- B에서 같은 제약·시간의 예측까지 좋아짐 / 지지 / 후속 검토 / 다른 조건과 선행 대비 차이는?
- B에서 메모리만 줄고 예측·속도는 불리 / 부분 또는 불지지 / 주장 제한 / 실제 얻는 이득은?
- 실행 오류 / 판정 불가 / 안 닿음 / 어떤 계산을 확인하지 못했나?

가지치기:
- Censor 추가 실험과 HEAD fits는 제외한다. 이번 핵심 상대는 Standard와 Side다.
- 새 context/horizon 탐색, rank 탐색, 손실 변경, CPU offload, 양자화, torch.compile,
  새로운 activation compression은 제외한다. 미실행 범위라고 보고한다.
- A의 속도 신호가 없더라도 Query 전체를 반증했다고 쓰지 않는다.
  이번에 선택한 "같은 제약에서 처리량을 예측 이득으로 연결"하는 가설을 닫는 것이다.

======================================================================
2. 선행·기존 구현 확인 및 저장소 보호
======================================================================

[확인: S6] LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning
(2022/NeurIPS)은 고정 백본의 표현을 읽는 별도 경량 학습 경로를 이미 다룬다.
Side는 이런 원리를 반영하는 현재 저장소의 대조군이지 공식 LST의 완전 재현은 아니다.
Query의 큰 아이디어 자체를 최초라고 부르지 않는다.

[확인: S7-S9] Activation checkpointing은 중간값 저장을 재계산으로 바꾸며,
부분 적용으로 속도와 메모리의 절충점을 바꿀 수 있다.
현재 설치된 PyTorch API와 사용 중인 wrapper를 확인해 재사용한다. 패키지를 업그레이드하지 않는다.

먼저 아래 실제 파일을 읽고 manifest에 파일 hash를 기록한다:
- docs/FORECAST_QUERY_EQUAL_TIME_PLAN.md
- docs/FORECAST_QUERY_EQUAL_TIME_EXECUTION.md
- configs/forecast_query_equal_time_plan.json
- research/reopen_review_20260914/QUERY_RESOURCE_FRONTIER.md
- src/tsfm_peft_screen/forecast_query/model.py
- src/tsfm_peft_screen/forecast_query/equal_time.py
- src/tsfm_peft_screen/forecast_query/checkpoint_diagnostic.py
- scripts/run_forecast_query_equal_time.py
- src/tsfm_peft_screen/backbone.py
- src/tsfm_peft_screen/lora.py
- src/tsfm_peft_screen/metrics.py
- docs/RESULTS_INDEX.md

partial checkpoint 자원 실험의 실제 실행 코드는 저장소에서 찾아 읽은 뒤 사용한다.
읽지 않은 함수명·열 이름·config key·CLI 옵션을 추측해서 호출하지 않는다.
기존 runner는 결과 경로·선택 개수·시간 경계가 고정돼 있으므로 그대로 재실행하지 않는다.

[확인: S3-S5] 재사용 가능한 이름:
ForecastModel / EqualTimeModel / TrainingClock / preserve_rng.
EqualTimeModel은 기존 전체 on/off 중심이며, choose_storage도 bool on/off 선택이다.
이번에는 부분 재계산·microbatch 조합을 지원하는 새 wrapper/selector가 필요하다.
기존 전역 설정을 무작정 바꿔 옛 실험의 의미를 변경하지 않는다.

저장소 확인:
- 현재 commit, branch, dirty files, 실행 중인 동일 프로젝트 작업을 기록한다.
- 기준 commit 이후 관련 결과가 이미 있다면 먼저 읽는다. 같은 실험이 완료됐으면 중복하지 않는다.
- 사용자 변경사항을 stash/reset/clean하지 않는다. 원시 데이터·cache·체크포인트를 삭제하지 않는다.
- 기존 results/research의 기록은 hash manifest로 보호한다.
- 새 결과 폴더가 있으면 상태를 확인한다. 완료/실패 기록을 덮어쓰거나 run ID만 바꿔 재시도하지 않는다.
- 코드·검사 준비를 마친 뒤 실행 contract를 봉인한다. 이후 성능을 보고 설정을 바꾸지 않는다.

새로 만들 경로의 제안:
  docs/QUERY_BUDGET_PILOT_20260915.md
  configs/query_budget_pilot_20260915.json
  scripts/run_query_budget_pilot.py
  scripts/finalize_query_budget_pilot.py
  results/query_budget_pilot_20260915/
  .cache/query_budget_pilot_20260915/
  tests/ 아래 이번 wrapper·clock·seal·metric 검사
JSON은 CLI 내부 manifest 용도로 사용해도 된다. 사용자에게는 REPORT.md가 주 결과다.

======================================================================
3. 공통 모델·데이터 계약
======================================================================

[확인: S1,S3,S5] 유지할 모델 조건:
- model: amazon/chronos-2
- revision: 29ec3766d36d6f73f0696f85560a422f50e8498c
- arm: standard / side / query, F0는 무학습 기준선
- 각 학습 arm trainable parameters: 1,179,648개. 실제 파라미터 목록으로 재계산한다.
- Standard/Query: 기존 attention rank8, alpha16, dropout0.
- Side: 기존 12층 width64 lateral network. 새로 더 약한 대조군을 만들지 않는다.
- frozen native output head 및 그 외 원래 백본 가중치 고정.
- context4096, horizon48, 원천별 기존 첫4채널.
- effective batch: 2 origins × 각 origin의 4채널 = 8개 계열.
- BF16 autocast, FP32 parameters와 Adam state. autocast cache 정책을 모두 동일하게 유지.
- AdamW, weight_decay0, gradient clipping1. 기존 native asinh loss를 유지한다.
- Standard/Query LR=1e-4, Side LR=1e-3로 고정한다.
  이 값들은 이전 계획의 탐색 범위 안에서 고른 단일 recipe다.
  새 LR 탐색은 없고, 각 방법의 최적 LR를 증명한 실험도 아니다.
- 새 본학습 seed: 39000,39001. seed 변경은 새 독립 데이터 확보가 아니다.

동일 정보 조건:
- Query/Side의 frozen encode·KV/feature cache는 각 forward마다 계산한다.
- 외부 cross-window 또는 epoch 간 사전 추출 cache는 사용하지 않는다.
- 미래 target을 입력·정규화·특징에 사용하지 않는다.
- 다른 원점의 4채널 그룹 사이에는 attention 정보가 흐르지 않아야 한다.

데이터는 이전 동일시간 비교의 ETTm2와 Electricity를 그대로 쓴다.
0-based 원본 시점 기준:
  Train origins: 16384..22528 inclusive, stride32 (193개)
  V origins: 23040..23712 inclusive, stride96 (8개)
  E origins: 24576..26016 inclusive, stride96 (16개)
  train scale 계산 범위: [16384,22576), 채널별 float64 std, floor1e-6
  x=[o-4096,o), y=[o,o+48)
원자료 hash와 실제 네 채널 ID를 읽어 기록한다. 순서가 바뀌면 자동 대체하지 않는다.

[확인: S1] ETTm2의 48 step은 12시간, Electricity의 48 step은 48시간이다.
같은 step 수를 같은 물리적 시간 범위라고 쓰지 않는다.

이 split은 이미 이전 실험에서 평가된 개발 자료다.
- 결과 제목과 결론에 "기존 평가 구간을 재사용한 개발 비교"를 명시한다.
- 기존 E 점수로 LR·budget·저장 옵션을 고르지 않는다.
- 동일 구간 재사용은 통제 비교를 위한 것이며 새 독립 성능 검증이 아니다.
- 기존 E였던 구간을 새 test라고 이름만 바꾸지 않는다.
- 두 seed·네 채널·여러 horizon을 독립 데이터셋으로 세지 않는다.

staging:
기존 manifest의 파일 hash를 확인하고 development와 evaluation 배열을 새 run에 분리한다.
원자료를 기계적으로 읽어 분리한 사실은 기록하되 새 E scoring은 선택 봉인 뒤에만 수행한다.
A에서는 Train만 읽는다. B에서는 Train/V만 읽고 모든12 선택을 봉인한 뒤 E를 연다.
파일 누락 시 경로를 추정하거나 더 좋은 데이터로 교체하지 말고 BLOCKED_INPUT으로 종료한다.

======================================================================
4. 텐서 메모리 예산 — 실제 GPU 용량과 혼동하지 말 것
======================================================================

[설계] 이번 그리드는 B∈{1,2,4,8} GiB이며 1GiB=2^30 bytes다.
현재 관측된 684/733MiB 등의 차이 사이에 새 budget을 끼워 넣지 않는다.
이 네 예산의 결과를 모두 표로 남긴다. Query가 유리한 예산만 골라 보여주지 않는다.
실제 장비 요구를 주장하는 실험이 아니라 사전 고정한 텐서 예산 곡선이다.

[확인: S8] max_memory_allocated는 tensor에 할당된 CUDA memory의 peak다.
GPU 전체 VRAM 사용량, allocator reserved memory, CUDA context/NVML 수치와 같지 않다.
따라서 아래를 따로 기록한다:
- 전체 모델/학습 상태를 포함한 peak allocated. 이전 baseline allocation을 빼지 않는다.
- peak reserved, step 시작 allocated/reserved.
- GPU 전체/가능하면 프로세스별 NVML 사용량, 실제 free VRAM.
- 로딩, frozen encode/cache, forward/backward, optimizer step, 평가의 peak.

각 옵션의 feasibility는 optimizer state 초기 생성과 warm state,
측정·검사한 모든 실행의 allocated peak가 0.95×B 이하일 때 인정한다.
5% 여유는 이번의 운영상 안전 마진이며 문헌의 성능 기준이 아니다.
B에서 학습할 때는 실제 모든 update의 peak <= B를 추가 검사한다.
넘으면 MEMORY_BUDGET_VIOLATION. 같은 fit에서 budget 확대/설정 교체/재시작 금지.

검사 과정은 여유 있는 동일 GPU에서 실행한 후 tensor peak로 각 B의 가능 여부를 판정한다.
실제 작은 GPU를 하드웨어적으로 재현했다는 뜻이 아니다.
set_per_process_memory_fraction을 allocated B와 같은 제약이라고 취급하지 않는다.
일부러 큰 dummy tensor로 GPU를 채우거나 시스템 설정을 바꾸지 않는다.
실제 장비에서는 reserved/context/다른 라이브러리의 메모리를 추가로 고려해야 한다.

======================================================================
5. A — 자원 비교와 수치 검증
======================================================================

5.1 저장·배치 옵션

Standard:
- 재계산 block 개수 0,3,6,9,12.
- 기존 부분 checkpoint 진단과 같은 block 선택 순서를 사용하고 실제 index를 기록한다.
  확인할 기존 순서가 없으면 prefix blocks[0:k]로 고정하고 성능을 본 뒤 바꾸지 않는다.
Query:
- 기존 query 경로 재계산 off/on(0/12).
Side:
- 기존 trainable side 경로 재계산 off/on(0/12).
각 옵션에 microbatch origins∈{1,2}를 허용한다.
재계산 wrapper는 기존 구현처럼 use_reentrant=False를 명시한다.
재계산 함수가 다른 layer/index를 참조하는 늦은 closure binding이나 mutable state 변경을 막는다.
총 18옵션/원천, 2원천에서 36옵션이다.

microbatch origins=1의 정확한 의미:
1. effective batch의 첫 origin 4채널 전체를 함께 forward/backward.
2. 두 번째 origin 4채널 전체를 함께 forward/backward.
3. 각 micro loss는 전체 effective batch와 같은 reduction이 되도록 1/2 가중.
4. 두 gradient를 더한 뒤 전체 trainable parameters에 clipping을 1회 적용.
5. optimizer.step도 1회만 수행.
채널을 1개씩 분리하지 않는다. micro마다 clip/Adam step을 하지 않는다.
loss masking/reduction은 기존 native loss와 동일하게 보존한다.

5.2 측정 상태·반복

- resource seed39000. 원천/arm별로 2회 warmup한 parameter/Adam/RNG를 CPU에 저장한다.
- 동일 arm의 옵션은 그 동일 상태에서 시작한다. 다른 구조끼리 동일 tensor라고 부르지 않는다.
- kernel warmup은 옵션당1 update이며 폐기한다.
- 각 옵션에서 3개의 timing block을 측정한다. block마다 같은 상태로 복구 후3 updates.
- 측정의 고정 batch origin쌍:
    block 내 update1: [17408,18432]
    update2: [19456,20480]
    update3: [17408,18432]
  모두 Train 안에 있고 첫4채널을 함께 사용한다.
- block마다 36옵션의 실행 순서를 seed59000+block_index의 permutation으로 고정한다.
- block1은 저장 옵션 선택용, block2/3은 선택 뒤 속도 재확인용이다.
  2/3의 속도 결과로 다시 옵션을 선택하지 않는다.
- 새 모델/옵션은 가능한 별도 프로세스로 실행하고 이전 tensor/cache를 남기지 않는다.
- 각 block 직전에 상태를 복원하되 복구/I/O 시간은 측정 step clock 밖에 기록한다.
- 모든 측정에서 CUDA synchronize와 같은 backend/precision/thread 설정을 사용한다.
- step 시간은 zero_grad, batch 구성·전송, frozen encode/cache, 손실, backward,
  gradient 유한성 검사, clipping, Adam step, 마지막 synchronize를 전부 포함한다.
- 로그의 CPU state export·검산 I/O는 clock 밖에서 동일하게 처리하고 wall 장부에 남긴다.
- 실제 gradient와 optimizer update가 일어나는 비용을 측정한다. forward만 재지 않는다.

업데이트 계산:
  상태 warmup: 6상태×2=12
  옵션 kernel warmup: 36×1=36
  BF16 측정: 36×3blocks×3updates=324
  FP32 동등성 참조/옵션 비교: 최대72
  기본 합계 상한444. 모든 추가 점검·실패 시도를 포함한 절대 상한480.
폐기용 A 업데이트를 forecasting fits로 세지 않고, 업데이트가 0이라고도 쓰지 않는다.

5.3 수치 검사

필수:
- 실제 trainable count, frozen parameters 불변, finite loss/gradient/update.
- 초기 LoRA B=0이면 첫 step의 A-gradient가0일 수 있다.
  모든 tensor가 매 step 반드시 nonzero이어야 한다는 잘못된 검사를 만들지 않는다.
- 실제 학습 상태의 총 update가0이 아닌지, 지정한 parameter만 변하는지 검사한다.
- 같은 microbatch 크기의 checkpoint 옵션끼리:
  출력·loss·raw/clipped gradient·update·Adam·RNG를 비교.
  기존 기준 FP32 relative-L2/max-abs1e-5, BF16 1e-4를 유지한다.
- microbatch1 대2:
  우선 FP32에서 동일 effective-batch 목적식의 출력/gradient/update를 비교한다.
  FP32 기준은1e-5. 그룹 mask, loss 분모, 누적 뒤 clipping을 확인한다.
  BF16은 batch shape가 달라져 bitwise equality를 보장할 수 없다.
  normalized output, raw output/train scale, 전체 gradient, update의 relative-L2를 기록한다.
  이번 허용 상한은1e-2; 원소별 최대 차이도 기록한다. 이는 새 수치 허용 규칙이지 기존 결과가 아니다.
  참조 norm<1e-12인 tensor는 relative 대신 max-abs<=1e-6을 적용한다.
  BF16 raw max error는 train scale 단위로 별도 보고하고2e-2를 넘으면 차이를 조사한다.
- 검사 실패를 "Standard는 불가능"으로 처리해 Query를 유리하게 만들지 않는다.
  이번 조합 비교의 무결성을 확보하지 못한 것으로 A를 종료한다.
- 허용오차를 결과에 맞춰 늘리지 않는다. 필요하면 실패 근거를 남기고 이번 실행 종료.
- 학습 전 Query와 native F0의 precision별 차이도 기록한다.
  B에서는 arm 자체 step0 점수도 저장해 초기 경로의 반올림 차이를 학습 이득으로 세지 않는다.

FP32 검산 자체는 학습용 BF16 budget feasibility와 구분한다.
필요시 순차 실행하되 안전한 실제 VRAM 범위 안에서만 수행한다.

5.4 B* 및 설정 선택 — 사전 고정 규칙

모든 검증된 옵션의 peak를 사용해 각 budget의 feasibility 표를 만든다.
B*는 {1,2,4,8}GiB 중 두 원천 모두에서 세 arm 전부 적어도 한 옵션이 가능한 가장 작은 값이다.
선택에 정확도나 속도 이득의 크기를 사용하지 않는다.
모든 옵션의 forward-only 평가가 origin1로 같은 B* 안에 들어오는지도 Train 입력으로 확인한다.
없으면 NO_COMMON_FEASIBLE_BUDGET으로 종료한다. budget grid를 추가하지 않는다.
더 낮은 B에서 Query만 가능했던 기록은 자원 관찰로 남기되 임의의 품질 점수를 만들지 않는다.

각 원천/arm의 설정은 B* 안의 유효 옵션 중 block1의 complete-step 중앙값이 최소인 것으로 고른다.
최소시간의2% 안이면 더 낮은 peak, 그다음 더 적은 CP blocks,
그다음 microbatch origins=2를 우선하는 deterministic tie-break를 쓴다.
block2/3으로 설정을 바꾸지 않는다.

메모리 제약이 실제로 활성화됐는지 확인한다:
적어도 한 원천에서 Standard의 budget 제약 없는 최속 옵션이 B*에서 infeasible이어야 한다.
그렇지 않으면 NO_BINDING_MEMORY_CONSTRAINT. 기존 넉넉한 메모리 비교를 반복하지 않는다.

B 진행 조건:
- 위 무결성과 세 arm의 공통 feasibility를 만족한다.
- 선택된 Query가 선택된 Standard보다 block2와3 각각에서 complete-step 중앙값이
  최소5% 짧은 원천이 적어도 하나 있다.
  speed_gain=100×(t_standard-t_query)/t_standard.
- 5%는 짧은 자원 측정의 미세한 차이로 추가 학습을 늘리지 않기 위한 투자 기준이다.
  통계적 유의성·논문 합격선이 아니며 연속 수치와 모든 반복을 그대로 보고한다.
- 다른 원천도 숨기지 않는다. 두 원천 모두 좋아야 한다고 요구하지 않는다.
- Query가 Side보다 빠를 필요는 없다. Side의 속도와 Query의 품질은 B에서 비교할 대상이다.

미충족: STOP_RESOURCE_SCREEN, B는0 fits.
충족: RESOURCE_SIGNAL. B*와 원천별 설정·전체 자원 표를 봉인하고 B12fits 진행.
Query가 유리한 다른 예산을 다시 찾거나 input 길이를 바꾸지 않는다.

======================================================================
6. B — 조건부 최대12 fits
======================================================================

A가 통과했을 때만 다음 전체를 수행한다:
  {ettm2,electricity} × {39000,39001} × {standard,side,query}
공통 B*, 원천별 사전 선택 설정, 단일 LR recipe를 유지한다.
A에서 쓰던 parameter/Adam state를 넘기지 않고 각 fit을 새 초기 상태에서 시작한다.

sampling:
- 원천/seed마다 8192개의 effective batch origin쌍을 Train193개에서 생성해 봉인한다.
- 동일 seed의 세 arm은 같은 무한 표본 stream의 prefix를 사용한다.
- microbatch 분할은 같은 effective batch 내부에서만 한다.
- 빠른 arm은 더 많은 updates/examples를 처리할 수 있다. 그것이 이번 비교의 일부다.
- 동일 업데이트·동일 데이터 노출 횟수의 비교라고 부르지 않는다.

시간:
- active training budget120초/fit, 경계0/30/60/120초.
- 경계를 넘긴 첫 complete effective update를 마친 상태를 저장한다.
- 단계 시간 clock은 A와 동일한 범위를 포함한다. partial microbatch에서 멈추지 않는다.
- overshoot<=1.0초. 이를 넘으면 INVALID_TIMING. 초과 업데이트를 지우거나 몰래 제외하지 않는다.
- A에서 complete update가1초보다 길거나 clock을 안정적으로 잴 수 없으면
  본학습 전에 INVALID_TIMING_PREFLIGHT로 종료한다. 경계를 사후 늘리지 않는다.
- 학습 첫 Adam state 생성도 비용에 포함한다.
- optimizer 초기화 이전과 이후 peak를 기록하고, update마다 peak<=B*를 확인한다.
- V 평가·checkpoint I/O·모델 로딩·A·GPU 대기는 active clock과 구분해 wall 장부에 전부 기록한다.
- end-to-end 총비용도 별도 표에 넣는다. active-time 이득을 전체 비용 이득으로 바꾸어 말하지 않는다.

checkpoint·V:
- 0/30/60/120초 네 상태를 모두 별도 파일로 저장한다.
- V evaluation은 모든 arm에서 origin1×4채널, 같은 순서·precision으로 수행한다.
- V/I/O가 training RNG를 바꾸지 않도록 preserve_rng와 상태 복원을 적용한다.
- 평가에서 gradient 계산/parameter update를 하지 않는다.
- V 최소 scaled_2pinball 선택, 동률은 더 이른 시간.
- 학습 전 상태도 선택 가능하다. 마지막 step만 강제로 쓰지 않는다.
- 학습 실행 순서는 seed59010으로12개 cell을 permutation해 봉인한다.
- 같은 B* 안에서 재계산/배치 옵션을 V 점수로 다시 고르지 않는다.

모든12 fits가 완료돼야 E를 연다.
12개 선택의 arm/dataset/seed/실제시간/step/원본checkpoint hash/V score/B*/설정/LR을 봉인한다.
디스크에서 선택 모델을 다시 불러 V 예측 재생을 확인한다.
모든 source/data/config hash가 그대로임을 확인한 후 E를 한 번 평가한다.

일부 fit이 실행 오류로 끝나면 나머지 자원 안전이 보장될 때만 예정된 나머지 cell을 수행한다.
그러나12개 선택이 완성되지 않으면 주 E 비교는 열지 말고 INCONCLUSIVE_EXECUTION으로 종료한다.
오류 fit을 성능 실패로 표시하거나 추가fit으로 채우지 않는다.

E 예측:
- F0 공통 기준선,12 selected 모델, arm 자체 step0를 동일 입력·precision으로 저장한다.
- F0는 동결 백본의 기준선이고 학습 fit에 포함하지 않는다.
- Query/Side의 frozen encode/cache 및 origin1 평가도 B*를 준수한다.
- 이전 예측 cache를 새 예측으로 재명명하지 않는다.
- 초기 경로가 F0와 같다는 수치 확인 없이 gain을 모두 학습 효과라고 해석하지 않는다.

======================================================================
7. 지표·결과 판정 — 부등호 혼동 금지
======================================================================

Primary = 기존 equal-channel train-std-scaled raw 2-pinball.
낮을수록 좋다. 모델이 예측한21개 분위수와 실제 미래 값의 오차다.
F0의 실제 loss는1이 아니다. ratio를 보고 싶으면 loss/F0_loss라고 명시한다.

원점수:
  L[d,s,a] = 원천d, seed s, 방법a의 E primary.

Query의 비교군b 대비 개선율:
  G[d,s;b] =100×(L[d,s,b]-L[d,s,query])/L[d,s,b].
양수는 Query가 좋음. 100×Query/b는 개선율이 아니라 손실 비율이다.

원천별 주 표:
  두 seed의 raw loss 평균을 먼저 계산한 뒤 같은 원천 baseline 평균으로 나눈 개선율.
  seed별 raw loss와 seed별 gain도 함께 제시한다.
  baseline은 Standard, Side, F0를 각각 나눠 쓴다.
  qmean은 분위수의 단순 평균이지 확정된 조건부 평균이 아니다. qmean MSE는 주 지표로 쓰지 않는다.

보조 지표:
median MAE, 80% interval coverage/width, 실제 updates/examples,
active seconds, selected checkpoint time, 총 fit/search wall time,
peak allocated/reserved, NVML/free-memory, 초기 경로 대비 개선.
원천별 native loss 단위가 다르므로 두 원천 raw loss를 직접 평균하지 않는다.
전체 macro gain이 필요하면 원천별 상대 gain을 같은 가중치로 평균한다.

불확실성:
16 E origins를 시간순4개씩 묶은4블록으로 두고1000회 paired bootstrap(seed59100).
모든 arm/seed/channel/horizon에 같은 블록 resample을 적용한다.
seed별/원천별 결과를 보고하고 2seed를4개 독립 데이터로 세지 않는다.
블록4개뿐인 기술적 구간이며 검정력·독립 확증을 주장하지 않는다.
기존 자료 반복사용/조건 선택의 편향까지 보정한 confidence interval이라고 부르지 않는다.

B의 결과는 하나의 억지 PASS로 합치지 않는다:
- 예측 개선 신호:
  적어도 한 원천에서 Query의 두 seed 평균 오차가 Standard와 Side 모두보다 낮고,
  각 seed에서도 Standard 대비 gain이 양수인 경우.
  크기·불확실성·다른 원천 손해를 그대로 보고한다. 아주 작은 차이를 확증이라고 하지 않는다.
- tradeoff:
  Query가 속도나 메모리에서 유리하지만 품질에서는 일부 손해가 있는 경우.
  손해율을 그대로 쓰고 이번 뒤늦은 허용오차를 만들어 동등성이라고 하지 않는다.
- 추가 가치 미확보:
  A의 이점이 본학습에서 사라지거나, 품질·시간·메모리 조합에서 단순 대조가 충분한 경우.
- 실행 판정 불가:
  코드/수치/자원/선택 무결성 실패. 과학적 실패와 구분한다.

F0를 모든 cell에서 이겨야 다음 비교를 허용하는 규칙은 만들지 않는다.
다만 Query가 Standard보다 좋지만 F0보다 나쁘면 "적응 손해 완화"라고 명확하게 쓴다.
두 원천 전부의 승리를 요구하지 않는다. 한 원천 신호를 범용 우위로 확대하지도 않는다.
모든 B 결과를 보고하면 종료한다. 보고서가 후속 투자를 권해도 자동 학습하지 않는다.

======================================================================
8. 자원·무결성 안전장치
======================================================================

- 기존 GPU lock을 사용하되 자기 worker/자식 PID만 구분한다.
- 과거 Censor의 별도 GPU 공존 승인을 이번 타이밍 비교에 자동 재사용하지 않는다.
- 시작 전30초 동안 외부 compute 없음, free VRAM>=4GiB를 확인한다.
- Xorg/일반 디스플레이 graphics PID를 compute 학습 작업으로 오인하지 않는다.
- 외부 compute가 생기거나 실제 free<1GiB이면 안전 경계에서 중지/대기한다.
- 대기 시간은 누적600초 상한. 상한 도달 시 BLOCKED_GPU_BUSY와 해당 PID·메모리를 기록한다.
- 외부 compute가 timed update 도중 끼어들면 해당 구간을 깨끗한 timing으로 세지 않는다.
  같은 run에서 유리한 재측정만 골라 대체하지 말고 오염 범위와 실행 제한을 보고한다.
- query cache 크기만 재거나 optimizer state를 빼는 불공정한 메모리 측정 금지.
- fit 중 per-step empty_cache, arm별 다른 allocator/backend, 비공개 CPU offload 금지.
- 자동 OOM 재시도/배치 축소 금지. 계획된 다른 옵션의 독립 측정과 실패 옵션 재시도를 구분한다.
- CUDA/패키지/드라이버는 기존 검증 환경 유지. 시스템 라이브러리 교체 금지.
- 물리 자원 부족은 시뮬레이션 budget의 알고리즘 불가능과 다르다. 구분해 보고한다.

======================================================================
9. 산출물과 사용자 보고
======================================================================

반드시 남길 산출물(새 경로):
- PROTOCOL.md: 목적, 위 모든 numeric setting, 자료 재사용, 선행 대비 범위.
- contract/manifest: git/source/model/data/config hashes, 버전, GPU, 실행 순서.
- resource_measurements.csv:36옵션의 모든 반복, peak, 시간, 유효성/실패 사유.
- resource_budget_table.csv:1/2/4/8GiB 전부, B* 결정 과정, binding 여부.
- resource_selection.json: block1 선택과block2/3 검증의 분리.
- parity 결과: CP, microbatch, 초기F0 차이, frozen weight, 실제 update.
- fit_attempts.csv: A폐기updates와 Bfits/updates를 구분.
- trajectories.csv: 각V 기회, active time/step, rawV점수, loss.
- selection seal, 예측 cache 목록/hash, per-origin/per-seed raw metrics.
- metrics.csv와 resources.csv: 다른 시점에서 가져온 수치를 새 joint measurement처럼 섞지 않음.
- independent_verification: 별도 scalar metric 구현과 vectorized 구현의 차이<=1e-10.
- REPORT.md: 사용자가 먼저 읽는 한국어 결과 보고서.

보고서 흐름:
1. 무엇을 비교했고 왜 Censor 대신 Query인가? (Censor 결과를 재해석해 PASS로 바꾸지 않음)
2. 어떤 예산에서 어떤 옵션이 가능한가? simulated tensor budget임을 제목에 표시.
3. Standard에 부분 재계산/배치 분할을 줘도 자원 이점이 남았나?
4. B를 실행했나? 실행했다면 동일시간에 예측도 좋아졌나?
5. 원천별 원점수와 tradeoff. 값이 같으면 같다고 쓰고 부등호로 의미를 바꾸지 않음.
6. 한계: 이전 개발데이터,2seed, 단일길이4096, 제한된옵션/LR, 실제VRAM미검증, 신규성미확정.
7. 종료 판정과 다음 한 가지 판단. 자동 추가학습은 없음.

그림은 필요 최소3개:
- budget별 Standard/Side/Query feasible 설정 및 시간 (측정한 모든budget 표시).
- B가 있으면 원천별 error vs 실제 active training time.
- B가 있으면 실제 학습에서 측정한 품질/peak/time 표 또는 그림.
A에서 종료하면 빈 B 그림이나 가상의 정확도를 만들지 않는다.

코드·자료 검증 후 기존에 승인된 commit/push 워크플로가 확인되면 새 작업만 scoped commit/push한다.
그 권한이 확인되지 않거나 push가 실패하면 로컬 결과와 미업로드 상태를 보고한다.
원시 데이터·모델 weights·큰 prediction cache·토큰/자격증명은 GitHub에 올리지 않는다.
RESULTS_INDEX는 기존 링크와 옛 판정을 보존하면서 새 항목만 추가한다.
과거 결과의 immutable hash 검사는 새 폴더와 의도한 index 변경을 명시적으로 제외한다.

최종 사용자 메시지에는 JSON만 던지지 말고 다음을 먼저 한국어로 써라:
- A/B 실행 여부, 실제 fits/updates, 종료 상태.
- Query의 자원 이점과 실제 예측 이득을 분리한 결론.
- 가장 강한 대조군과 원점수, 반복 결과, 안 된 경우 구체 근거.
- REPORT.md 위치와 검증/업로드 상태.

======================================================================
10. 실행기 인터페이스 — 아래는 새로 구현할 인터페이스
======================================================================

run_query_budget_pilot.py에 아래 단계를 명시적으로 구현한다:
  prepare        : 저장소/자료/계약 확인, CPU tests. GPU 결과를 완료라고 표시하지 않는다.
  profile        : A 전체와 자원 선택 봉인. STOP이면 그 이유와 보고서 생성.
  train-evaluate : RESOURCE_SIGNAL일 때만 B12fits -> V선택 봉인 -> E평가.
  status         : fit/현재phase/대기이유/실패상태/결과경로 표시.
finalize_query_budget_pilot.py:
  --verify-only  : 새 fits 없이 수치·hash·선택·산출물을 독립 검사.

이 옵션들은 지금 존재한다고 가정하지 말고 구현·--help·tests 확인 뒤 사용한다.
기존 scripts/with_cuda.sh와 .venv가 현재 환경에서 유효한지 확인해 사용한다.
새 runner 구현 후 실행 순서는 prepare -> profile -> 조건부 train-evaluate -> finalize다.
명령어만 적고 실제 요청된 실행을 생략하지 않는다.
다만 외부 자원·입력·무결성이 막혔을 때는 설정을 멋대로 바꾸지 말고 근거와 함께 종료한다.

======================================================================
출처 — 실행기가 다시 확인할 원자료
======================================================================

아래 repo 경로는 기준 commit에서 읽은 자료다. 최신 revision이 다르면 delta를 먼저 검토한다.

S1. 기존 동일시간 설계:
https://github.com/CanelE452/tsfm-peft-method-screen/blob/6a8bf6224a94fb790d26a96a2b9cb7505f24b70d/docs/FORECAST_QUERY_EQUAL_TIME_PLAN.md
S2. 최신 Query 자원 표:
https://github.com/CanelE452/tsfm-peft-method-screen/blob/6a8bf6224a94fb790d26a96a2b9cb7505f24b70d/research/reopen_review_20260914/QUERY_RESOURCE_FRONTIER.md
S3. 기존 모델:
https://github.com/CanelE452/tsfm-peft-method-screen/blob/6a8bf6224a94fb790d26a96a2b9cb7505f24b70d/src/tsfm_peft_screen/forecast_query/model.py
S4. 기존 clock·wrapper:
https://github.com/CanelE452/tsfm-peft-method-screen/blob/6a8bf6224a94fb790d26a96a2b9cb7505f24b70d/src/tsfm_peft_screen/forecast_query/equal_time.py
S5. 기존 runner 및 config:
https://github.com/CanelE452/tsfm-peft-method-screen/blob/6a8bf6224a94fb790d26a96a2b9cb7505f24b70d/scripts/run_forecast_query_equal_time.py
https://github.com/CanelE452/tsfm-peft-method-screen/blob/6a8bf6224a94fb790d26a96a2b9cb7505f24b70d/configs/forecast_query_equal_time_plan.json
S6. LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning(2022/NeurIPS):
https://proceedings.neurips.cc/paper_files/paper/2022/hash/54801e196796134a2b0ae5e8adef502f-Abstract-Conference.html
S7. PyTorch activation checkpointing 설명:
https://pytorch.org/blog/activation-checkpointing-techniques/
S8. PyTorch CUDA tensor peak memory API:
https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html
S9. PyTorch checkpoint API:
https://docs.pytorch.org/docs/stable/checkpoint.html

이 계획의1/2/4/8GiB,5% 안전여유/속도진입선,120초,12fits,480폐기업데이트,
단일LR,seed39000/39001은 이번 제한 실험용 설계 선택이다. 문헌의 보편적 PASS 기준이 아니다.
외부 문서는 개념과 API 근거이며 현재 로컬 설치 버전의 동작 검사를 대체하지 않는다.

구현 전 고정 보충: cold Adam 생성36 + kernel warmup36 + 공유 warmup12 + FP32옵션36 + BF16 timing324 = A 최대444 updates. FP32 참조를 옵션 간 재사용하여72 상한 내에서36만 사용하며, 절약한36회를 옵션별 cold Adam 생성 peak 검사에 사용한다. 절대 상한480은 그대로다. 3개 timing block은 지시문의 block1/2/3을 내부 index0/1/2로 대응하여 seed59000/59001/59002 순서를 사용한다. CPU gradient export를 동기화된 clock 구간 밖에서 수행하며 모든 timing 옵션에 같은 절차를 적용한다. 한 worker 안에서 각 옵션의 모델/optimizer 수명을 끝내고 다음 옵션 전에만 gc/empty_cache를 수행한다.
