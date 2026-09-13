# Side를 포함한 제한된 동일 학습 시간 비교 설계

상태: **설계 고정, 미실행**. 이 문서는 사용자가 요청한 다음 비교의 구체적 설계다.
실행기는 아직 구현하지 않았다. 신규 GPU 작업, optimizer update, fit, V/E 접근은 0이다.
과거 6225a99까지의 FAIL/STOP/TRADEOFF_ONLY는 그대로 보존한다.

## 연구 질문과 이번 비교의 역할

같은 시계열 관측 정보, 학습 데이터 풀, GPU 메모리 한도, 명목 학습 시간을 줄 때
forecast-query를 Standard LoRA나 단순 Head/Side 대신 선택할 예측 품질상의 이유가 있는가?

현재 문제는 Query의 학습 가능 여부가 아니다. 기존 파일럿의 작은 품질 차이와
고정 상태의 자원 측정을 실제 시간 예산 내 학습 결과로 연결하지 못했다는 점이다.
이번 비교는 이를 판단하는 개발 실험이다. 새로운 PEFT 모듈이나 신규성 증명이 아니다.
checkpoint-on끼리만 비교하면 실행 가능한 빠른 Standard-off를 부당하게 제외할 수 있다.
따라서 각 방식이 공통 메모리 한도 내에서 빠른 저장 방식을 고르도록 한다.

## 비교 대상과 고정 요소

- F0: 동결된 Chronos-2. 최적화 없이 공통 기준 예측만 제공.
- Standard LoRA: attention q/k/v/o, rank8, alpha16, dropout0.
- Head: 기존 zero-output residual head.
- Side: 기존 12층 width64 lateral side network.
- Query: 기존 frozen-past / 4-query-token 구조.
- 네 학습 방식 모두 trainable1,179,648개, 동일 frozen native output head,
  context4096, horizon48, 첫4채널, 2origins/batch = 8시계열.
- seed30000/30001, 같은 seed의 Standard/Query 초기 LoRA는 동일하게 생성.
  서로 다른 구조의 Head/Side를 동일 tensor나 동일 Adam 상태라고 표현하지 않는다.
- AdamW, weight_decay0, clip1, BF16 autocast + FP32 params/Adam.
  기존 native asinh loss를 그대로 사용한다. 이번에 loss나 architecture를 함께 바꾸지 않는다.
- 모든 방식은 동일 train origin sampling stream의 prefix를 사용한다.
  빠른 방식은 같은 시간에 더 많은 업데이트와 예제를 처리할 수 있다.
  이 차이가 이번 질문의 대상이며, 동일 업데이트 실험이라고 부르지 않는다.
- raw covariate, context 밖 추가 정보, 미래 정답, 외부 cross-window cache를 제공하지 않는다.
  frozen feature/KV 추출은 매 학습 forward 비용에 포함한다.

## 데이터 역할과 노출 범위

인덱스는 원본 processed panel의 0-based native timestep이다.

| 역할 | origins | 개수 | target 범위, half-open |
|---|---|---:|---|
| Train | 16384..22528, stride32 | 193 | [16384,22576) 내부 |
| V | 23040..23712, stride96 | 8 | [23040,23760) 내부 |
| E | 24576..26016, stride96 | 16 | [24576,26064) 내부 |

Train과 V에는 과거 block-shape에서 평가한 데이터가 일부 포함된다.
이미 알려진 개발 데이터를 재사용한다고 명시한다. 이를 새 증거라고 세지 않는다.
현재 기록상 이전 최신 평가 target의 마지막 exclusive index는23504이다.
이번 E target은 그 뒤에 위치하고, 두 패널 모두26064행 이상이다.
원본 파일이 이전에 기계적으로 읽히거나 staging된 사실과
해당 target으로 모델 점수를 계산한 사실은 구분한다.
메타데이터 및 저장소 실행 기록에 근거한 제한된 미평가 구간이며
외부 독립 데이터나 foundation pretraining 미노출 구간이라고 주장하지 않는다.

스케일은 Train의 [16384,22576)에서 채널별 std를 계산하여 1e-6으로 하한을 둔다.
context는 각 origin의 [o-4096,o), target은 [o,o+48)이다.
E 안에서도 나중 origin은 그 시점까지 도착한 과거 관측을 context로 사용할 수 있다.
E 정답으로 파라미터를 업데이트하거나 선택을 수정하지 않는다.
targets는 origin 간 중첩되지 않지만 context 중첩과 시계열 의존성은 남는다.

ETTm2 horizon48은12시간, Electricity는48시간이다.
E 첫/마지막 origin 간 길이도 각각15일/60일이다. 같은 index 폭을 같은 시간척도로
해석하거나 서로 다른 패널의 raw loss를 직접 평균하지 않는다.

실행 전 별도 staging 단계가 development와 E 파일을 분리한다.
학습기는 development 파일만 읽을 수 있어야 한다. 원본 compressed tail을
staging 과정에서 읽는 것은 기록하되 E 통계, 그림, scoring을 금지한다.
모든16개의 V 선택과 checkpoint hash를 봉인한 뒤 E target 접근을 허용한다.

## 저장 방식 선택: 손실을 보지 않는 공통 preflight

메모리 한도는 기존 GPU 안전 한도인 allocated8GiB를 사용한다.
이는 학습 알고리즘 자체가 8GiB를 필요로 한다는 뜻도, VRAM8GiB 기기에서
동일하게 실행된다는 뜻도 아니다. 실제 GPU free/RAM guard가 추가 적용된다.

4방법 × checkpoint off/on × 2고정 train batch × BF16 3회 = 48측정.
고정 origins17408/18432는 두 데이터셋 모두 Train에 포함된다.
각 방식마다 ETTm2 고정 batch의2 updates로 warmed Adam/parameter state를 만들고,
그 방식의 on/off/반복에서는 같은 parameter/Adam/Python/NumPy/CPU/CUDA RNG를 복원한다.
구조가 다른 방식 간에는 파라미터 값이 동일하다고 주장하지 않는다.

추가 FP32 동등성16 updates, state warmup8 + variant kernel warmup8 updates를
별도 카운트한다. preflight 총 상한80 optimizer updates.
warmup LR은 각 방식의 작은 LR이며 loss로 storage option을 고르지 않는다.
기존 진단의38 updates와 별도다. 본학습 fit에는 warmup 학습 상태를 넘기지 않는다.

각 dataset/arm에서 메모리 한도를 충족하고 수치 동등성이 검증된 옵션 중
BF16 전체 step 중앙값이 가장 짧은 것을 선택한다.
시간 차이가2% 이내이면 peak가 작은 쪽, 완전 동일하면 off를 택한다.
표준에도 off를 허용하고 Side/Head에도 의미 있는 trainable block checkpoint를 제공한다.
동일 autocast policy, frozen encode 포함 timing, 완전한 optimizer.step 포함 peak.
on/off 출력, loss, gradient, update, Adam, RNG 동등성을 두 precision별로 확인한다.
tolerance는 직전 진단과 같다. 선택 결과를 fit 전에 봉인하고 V/E로 바꾸지 않는다.

## 동일 시간의 정확한 의미

한 recipe당 **명목 active training30초**, 2recipe이므로 dataset/seed/arm당60초다.
2dataset × 2seed × 4arm × 2LR = **최대32 fit attempts**.
명목 active training 합계960초(16분). wall time 예측치는 아니다.

각 training step의 host batch 구성과 device transfer부터 forward, frozen encode/cache,
loss, backward, gradient finite check/clipping, Adam update 및 CUDA synchronize까지
누적한 시간을 active training clock으로 쓴다.
학습 첫 optimizer state 생성도 포함한다. V/E scoring, checkpoint I/O, GPU 대기,
model loading, 공통 preflight는 이 clock 밖이며 별도 wall-time 장부에 반드시 기록한다.
따라서 결과는 동등한 최적화 시간의 품질이며 end-to-end 총비용 우위와 다르다.
총 탐색 wall time도 공개해 느린 validation/checkpoint 비용을 숨기지 않는다.

checkpoint 기회는0/10/20/30초의 네 번으로 고정한다.
경계를 넘긴 첫 업데이트를 끝내고 checkpoint를 저장하며 실제 누적 초를 기록한다.
모든 경계의 초과 허용은0.5초다. 마지막 시간은 [30,30.5]초 범위여야 한다.
정확히 같은 밀리초의 학습이라고 표현하지 않는다.
초과>0.5초 또는4096-update 안전 상한에30초보다 먼저 도달하면
INVALID_TIMING으로 기록하고 E 선택/평가를 진행하지 않는다.
시간 초과 업데이트를 몰래 제외하거나 optimizer 상태를 부분 롤백하지 않는다.

preflight 이후 모델/Adam/RNG를 새 초기 상태로 재설정한다.
동일 V 기회와 동일2LR 검색 예산을 보장한다.
checkpoint 저장과 V 평가가 학습 RNG stream을 바꾸지 않도록 상태를 복원한다.
실행 순서는 seed별로 deterministic permutation을 사전 manifest에 저장하여
한 방식만 계속 GPU가 차갑거나 부하가 적을 때 실행되는 편향을 줄인다.

LR은 과거 forecast-query와 동일:
Standard/Query={3e-5,1e-4}, Head/Side={3e-4,1e-3}.
V 결과를 보고 재시도하거나 더 큰 grid를 추가하지 않는다.
사고로 중단된 fit도 attempt32 상한에 포함하며 완료 fit으로 표시하지 않는다.

## 선택, 평가, 비교 기준

각 dataset/seed/arm에서 2LR × 4시간 checkpoint 중 V primary 최소를 선택한다.
동률이면 이른 시간, 그 다음 작은 LR. step0도 합법적인 선택이다.
모든16 choices를 봉인한 뒤 E를 한 번 연다.
F0, 각 arm의 선택 예측, Query 자체 초기 예측을 같은 E origin에서 저장한다.
Query step0와 선택 모델은 같은 구현/precision으로 평가해
F0와의 BF16 rounding 차이를 학습 효과로 오해하지 않게 한다.
E 선택 모델 reload/replay, independent float64 metric replay, source/input hash를 남긴다.

Primary는 기존 raw-scale, train-std-scaled equal-channel 2-pinball이다.
MAE/MSE/coverage/width는 secondary다.
데이터셋별 두 seed 평균으로 Query와 모든 baseline을 비교한다.

별도 개발 지속 조건은 양쪽 dataset 모두:
1. Query seed-mean loss가 F0/Standard/Head/Side 각각의 seed-mean보다0.5% 이상 낮다.
2. 어느 seed도 해당 seed의 최선 baseline보다1% 이상 나쁘지 않다.
3. 모든 Query 선택 시간이0보다 크고, 같은 구현의 E step0보다 개선된다.
4. 정보/수치/시간/메모리 무결성이 모두 충족된다.

충족: CONTINUE_CANDIDATE_VALIDATION.
유효 실행이지만 미충족: STOP_CURRENT_QUERY.
구현/자원/타이밍 오류: INCONCLUSIVE_EXECUTION, 과학적 반증으로 합산하지 않는다.
이는 새로운 질문에 대한 자원 배분 규칙이며 통계적 유의성 기준이 아니다.
기존20% 메모리 감소 AND gate의 FAIL을 재분류하지 않는다.
새 규칙은 E 미개봉 상태에서 고정하며 E 확인 후 변경하지 않는다.

같은 dataset의 원점별 paired difference와 모든 seed별 raw 결과를 공개한다.
horizon/channel을 독립 표본으로 세지 않는다.
4origins씩 묶은4개 chronological block을1000회 bootstrap(seed51000)한 구간은
참고용으로만 보고한다. 블록4개와 optimization seed2개로 일반화/유의성을 강하게 주장하지 않는다.
품질이 유지되지만 속도만 빠른 경우 이 품질 개선 gate에는 통과시키지 않는다.
다른 목표를 사후 추가해 같은 run을 PASS로 바꾸지 않는다.

## 실행 전 구현해야 할 것과 완료 기준

이번 요청에서 완성한 것은 문서, machine-readable 설정, CPU 설계 감사다.
실험 완료나 즉시 실행 가능한 runner로 표시하지 않는다.
후속 구현에는 다음이 필요하다:
- 역사적 모델을 보존하는 Head/Side checkpoint wrapper와 gradient/update parity 검사.
- 실제 초 기반 scheduler, 전 구간 timing 장부, 사전 storage/순서 seal.
- 분리된 development/E loader, 32attempt 상한과 중단 상태 보존.
- 독립 prediction replay와 모든 baseline을 포함한 판정 보고서.

GPU는30초 외부 compute 없음/free≥4GiB/util<90%를 확인하고 시작한다.
step 경계마다 감시하며 외부 compute 등장/free<1GiB이면 대기한다.
startup 최대1시간, 전체 실행 최대2시간, 기존 RAM/GPU guard를 유지한다.
외부 프로세스 종료/시스템 driver 변경은 하지 않는다.
실행 코드와 CPU 테스트를 완료하여 실행 전 commit해야 한다.
기존 연구 자료와 결과는 수정하지 않는다.
