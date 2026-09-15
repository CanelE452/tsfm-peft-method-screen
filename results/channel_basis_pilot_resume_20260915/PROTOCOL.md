PEFT 우선순위 1·2 실행 계획 / CLI 전달본
작성: 2026-09-15
대상 저장소: CanelE452/tsfm-peft-method-screen
확인 기준 commit: 9051d03cebe23ae35d9689e7af576c63c4d5edd5
Time-PEFT 공개 코드 기준: ea4e7e1887bb35587bab7ea93e2af3685ac55852

이 문서는 두 작업의 구현·검사·조건부 학습·평가·보고까지 요청하는 실행 지시다.
계획서만 작성하고 종료하지 말고, 아래 전제 검사를 통과한 작업을 실제로 수행한다.
이 문서를 작성한 대화에서는 실제 저장소 수정이나 GPU 학습은 실행하지 않았다.
[확인]은 읽은 소스의 사실, [설계]는 이번에 정한 조건,
[미검증]은 효과·신규성이 입증되지 않은 가설이다.
새 경로·상태명은 새로 만들 이름이지, 기존 파일·함수가 존재한다는 주장이 아니다.

======================================================================
0. 이번에 실행할 두 작업과 하지 않을 일
======================================================================

1순위 Q: Query 수치 중단 정리 -> 공정한 자원 비교 완결 -> 조건부 최대 12 fits.
2순위 C: Time-PEFT의 채널별 변환을 공유하는 구조 비교 -> 최대 24 fits.

두 작업은 독립적인 연구 질문이다.
- Q에서 자원 이점이 없거나 Q 전용 수치 검사가 미해결이어도, 안전하고 독립적인 C는 진행한다.
- Q의 성능 PASS를 C의 시작 조건으로 삼지 않는다.
- CUDA/메모리 손상, 데이터 오염, 동시 작업 충돌 같은 공통 안전 문제가 있으면 둘 다 멈춘다.
- GPU 작업은 Q 다음 C 순서로 하나씩 실행한다. Q 측정 중 C를 함께 실행하지 않는다.
- Censor, FR, PatchPhase, Freshness, DualClock의 재튜닝은 이번 범위에서 제외한다.
- 결과가 좋다고 추가 seed, LR, 데이터, 길이, 다른 신규 후보를 자동으로 실행하지 않는다.

상한:
Q 수치 진단: 폐기용 optimizer updates 최대 192.
Q 자원 측정: 폐기용 optimizer updates 최대 512.
Q 본학습: 최대 12 fit attempts. 기존 120초 active-time 계약 유지.
C 모델·구조 점검: 폐기용 optimizer updates 최대 72.
C 본학습: 6 arms × 2원천 × 2seed = 최대 24 fit attempts.
C fit당 1,024 updates, 따라서 C 본학습 총 상한 24,576 updates.
본학습 합계 상한: 36 fit attempts. 위 폐기용 업데이트는 fits에 합치지 않는다.
실패 attempt도 각각의 상한에 포함하며, 이름만 바꿔 재시도하지 않는다.
Q GPU 단계 wall cap 4시간, C GPU 단계 wall cap 8시간.
이 시간들은 예산 상한이지 예상 완료시간이 아니다.

공통 환경·보호:
- 현재 branch/commit/dirty files/실행 프로세스/환경 버전을 기록한다.
- 이후 커밋에 같은 실험의 완료 결과가 있으면 먼저 읽고 중복 실행하지 않는다.
- 사용자 변경사항을 reset, clean, stash하거나 다른 프로세스를 종료하지 않는다.
- 이전 results/research의 결과와 판정은 hash manifest로 보존한다.
- 과거 9051d03의 INCONCLUSIVE_NUMERICS를 PASS로 덮어쓰지 않는다.
- 새 Q와 C는 별도 run ID, 결과 폴더, 계약, 체크포인트를 사용한다.
- 원격 push는 기존에 승인된 scoped commit/push 관행이 확인될 때만 한다.
  원자료·가중치·큰 캐시·자격증명은 올리지 않는다. 미업로드면 정직하게 표시한다.

새 출력 경로 제안:
  docs/PEFT_PRIORITY12_PROTOCOL_20260915.md
  results/query_budget_numeric_v2_20260915/
  .cache/query_budget_numeric_v2_20260915/
  results/channel_basis_pilot_20260915/
  .cache/channel_basis_pilot_20260915/
  results/priority12_20260915/REPORT.md
runner 파일명과 인자는 실제 구현 후 --help로 확인한다. 아래 경로를 기존 API로 착각하지 않는다.

======================================================================
1. 목적: 두 실험이 무엇에 답해야 하는가
======================================================================

최상위 목적:
'작고 명확한 조건에서, 강한 단순 대조군보다 추가 가치가 있는 PEFT 연구 방향을 고른다.'
소비처: 사용자와 지도교수가 후속 방법론 실험을 계속할지 결정하는 데 사용한다.
코드가 실행되었다는 사실과 방법이 좋아졌다는 사실, 논문 신규성은 따로 판단한다.

Q 질문:
'동일한 긴 과거 입력과 텐서 메모리·학습시간 예산에서 Query의 자원 이점이
Standard와 Side를 상대로 남으며, 그것이 예측 결과로 이어지는가?'

C 질문:
'채널마다 큰 출력 변환을 따로 만드는 대신 공통 변환 몇 개와 채널별 계수로
표현하면, 완전 공유보다 낫고 채널별 독립 변환보다 적은 파라미터로 예측할 수 있는가?'

[중요한 상류 제한]
- Q에는 실제 사용자가 요구한 소형 GPU 용량이 없다. 1/2/4/8 GiB 텐서 예산 실험이다.
- C의 공통 기저·채널별 계수는 알려진 파라미터 공유/분해 원리다. 그 자체를 새 발명이라 하지 않는다.
- C는 기존 채널을 학습하고 그 채널의 미래를 예측한다. 처음 보는 채널의 zero-shot 적응이 아니다.
- C는 가중치 공유를 바꾼다. 입력 채널 간 새 attention/정보 흐름을 추가하는 실험이 아니다.
- 두 작업 모두 기존 원천을 이용한 개발 비교다. 본문에서 미노출 독립 확증으로 표현하지 않는다.

행동의 이유와 그로부터 나온 검사 조건:
1) Q 수치 확인: 다른 학습을 같은 학습으로 오인하지 않기 + 단위에 따른 불필요 중단 방지.
   -> 그룹/손실/step 의미 검사와 부동소수점 차이 검사를 분리한다.
2) Q 자원 측정: 본학습 투자 가치 확인 + Standard의 단순 절약 전략이 충분한지 확인.
   -> Standard에 부분 재계산·origin 분할을 허용하고 고정 백본 비용도 포함한다.
3) C 구조 비교: 채널별 차이의 필요성 확인 + 공유로 파라미터 증가를 줄일 수 있는지 확인.
   -> 완전 공유, 동일 예산 공유, 고정 그룹, 학습 계수, 채널별 독립을 나란히 비교한다.
4) 공통 보고: 작은 효과와 미실행을 구별 + 향후 독립 평가를 보호.
   -> 모든 원점수/시도/개발 노출/초기값/불확실성을 함께 기록한다.

예상 결과 -> 목적 지지 -> 행동:
- Q 수치 의미가 같고 차이가 정밀도 수준에서 제한됨 -> 실행 무결성 지지 -> 자원 비교.
- Q 자원 이점 없음 -> 현재 효율 가설 불지지 -> Q 본학습 생략, C로 진행.
- Q 같은 예산에서 예측도 개선 -> 부분 지지 -> 후속 연구 검토. 논문 PASS는 아님.
- C 학습 계수가 같은 예산 공유/고정 그룹보다 나음 -> 채널별 공유 방식의 가치 지지.
- C 고정 그룹만으로 충분함 -> 학습 계수의 추가 목적 불지지 -> 새 계수 메커니즘 주장 제외.
- C 완전 공유만으로 충분함 -> 채널별 복잡성을 줄일 근거. 제안 모델 우위는 없음.
- C full-channel 참조가 단순 LoRA보다 나쁨 -> 그 참조를 이긴 것만으로 새 PEFT 우위라 하지 않음.
가지치기: '어떻게든 PASS'를 위해 데이터·기준을 바꾸는 목적은 버린다.

======================================================================
2. 먼저 읽을 근거와 선행연구 경계
======================================================================

[확인: Q 소스]
- results/query_budget_pilot_20260915/REPORT.md
- results/query_budget_pilot_20260915/PROTOCOL.md
- results/query_budget_pilot_20260915/contract.json
- results/query_budget_pilot_20260915/microbatch_stop_diagnosis.csv
- configs/query_budget_pilot_20260915.json
- scripts/run_query_budget_pilot.py
- src/tsfm_peft_screen/forecast_query/budget.py
- src/tsfm_peft_screen/forecast_query/equal_time.py
- src/tsfm_peft_screen/forecast_query/model.py
- src/tsfm_peft_screen/backbone.py
- src/tsfm_peft_screen/metrics.py
- tests/test_query_budget_pilot.py

[확인] 기존 Q는 A 폐기 업데이트 120회, B 0 fits로 끝났다.
같은 microbatch 크기의 재계산 비교는 통과했으나, microbatch1/2 비교 일부가
FP32 1e-5 기준을 넘었다. Standard에서는 gradient/update 상대차이도 넘었으므로
원출력만 scale로 나누는 수정으로 원인 규명 완료라고 하지 않는다.

[확인: 외부 원자료]
S1 PyTorch Numerical accuracy:
https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html
S2 FP32 matmul precision:
https://docs.pytorch.org/docs/2.14/generated/torch.set_float32_matmul_precision.html
S3 Activation checkpointing:
https://pytorch.org/blog/activation-checkpointing-techniques/
배치 연산과 나누어 계산한 연산은 부동소수점에서 bitwise 일치가 보장되지 않는다.
아래 수치 허용값은 논문의 보편 기준이 아니라 이번의 명시적인 운영 기준이다.
현재 설치 버전의 API를 확인한다. 웹 문서에 맞춘다는 이유로 기존 PyTorch를 업그레이드하지 않는다.

S4 Time-PEFT: Temporal and Multichannel Complexity-Based Fine-Tuning for
Time-Series Foundation Models(2026/ICML), 저자 공개 코드:
https://github.com/kaist-dmlab/TimePEFT/blob/ea4e7e1887bb35587bab7ea93e2af3685ac55852/run.py
https://github.com/kaist-dmlab/TimePEFT/blob/ea4e7e1887bb35587bab7ea93e2af3685ac55852/README.md
S5 MOMENT 작은 백본 모델 카드:
https://huggingface.co/AutonLab/MOMENT-1-small
S6 Channel-Aware Low-Rank Adaptation in Time Series Forecasting(2024/CIKM):
https://arxiv.org/html/2407.17246v1
공식 DOI: 10.1145/3627673.3679884
공식 코드: https://github.com/tongnie/C-LoRA

[확인] Time-PEFT 공개 run.py의 ChannelAdapter는 shared down_projection과
채널별 up_projections를 사용한다. 코드의 이 경로 자체는 채널 사이 값을 섞는
새 attention이 아니라 채널별 변환이다. TimePEFTPipeline은 LoRA, frequency adapter,
channel adapter, forecasting head를 학습한다.
[확인] 공개 train()은 best_loss 숫자는 관리하지만 best model state를 저장·복원하지
않고 마지막 model을 반환한다. 따라서 외부 train()을 그대로 호출해 'best checkpoint'라고
보고하지 않는다. 아래 공통 runner에서 모든 arm에 동일하게 best-state 저장/복원을 구현한다.
이 변경을 upstream discrepancy로 기록하고 원문 수치의 완전 재현이라고 부르지 않는다.

[확인] C-LoRA는 이미 공유 모델과 채널별 저랭크 성분을 결합한다.
이번 BASIS4를 '최초의 채널 공유 PEFT'라고 부르지 않는다.
C의 구현 전에 sources/RELATED_WORK.md에 세 항목을 분리해 적는다:
  기존 Time-PEFT에서 유지하는 것 / 이번 실험에서 바꾸는 것 / 이미 알려진 원리.
C-LoRA 및 시간계열의 공유·혼합 adapter와 수식이 동일하면 기존 방법의 통제 변형으로
정정한다. 신규성 미확정만으로 요청된 비교를 계획서 단계에서 묻어 두지는 않되,
결과가 좋아도 새 방법 또는 논문 준비 완료로 승격하지 않는다.
공식 C-LoRA 전체의 같은 백본 비교는 이번 24 fits에 포함하지 않는다. 이후 논문 주장에는 남은 비교다.

======================================================================
3. Q: 수치 중단을 제한적으로 정리하기
======================================================================

3.1 기록 보존과 허용되는 수정
기존 실패 폴더는 불변으로 남긴다. 새 run은 query_budget_numeric_v2_20260915.
수정 대상은 수치 검사의 단위·판정 및 실제 발견한 semantic bug에 한정한다.
모델 구조, 손실, rank, 학습 LR, 자료 구간, 자원 진행 기준은 바꾸지 않는다.
옛 결과를 이미 본 뒤 작성한 수치 정책 개정이라는 점을 명시한다.
아래 기준으로 새 train 검사 쌍을 검증하고, 통과까지 threshold를 반복 변경하지 않는다.

3.2 먼저 숫자 오차가 아닌 의미를 검사
- origin별 4개 채널을 함께 처리한다. 다른 origin 사이 group attention이 없어야 한다.
- 분할 손실은 각각 1/2 가중 후 gradient 누적. clipping과 Adam step은 마지막 한 번.
- padding/missing target의 분모가 full effective batch와 같아야 한다.
- 같은 parameter/Adam/RNG 상태에서 두 경로를 출발시킨다.
- dropout0 등 기존 Chronos 경로를 유지한다. eval이 RNG를 소비하면 복원한다.
- frozen weights가 바뀌지 않는지 hash로 확인한다.
- CPU FP64의 작은 독립 toy loss에서 full/partition 목적식·gradient를 확인한다.
- pinball의 잔차 부호가 0 근처에서 뒤집힌 개수·기여를 보고한다.
  필요하면 detached 고정 부호 손실로 gradient를 비교해 kink 영향을 진단한다.
  이 대조 손실을 본학습 손실로 쓰지 않는다.

3.3 실제 모델 검사
데이터·모델은 기존 Q와 동일하다. V/E 접근 없이 Train만 사용한다.
resource seed39000, 6개 원천/arm 상태를 동일 규칙으로 2회 warmup하여 상태를 준비한다.
기존 저장 artifact로 원래 차이를 재계산하고, 새 인증 배치는 다음 두 origin쌍으로 고정한다:
  [17280,18944], [20544,22016]. 모두 기존 Train grid에 속해야 함을 assert한다.
두 source, 세 arm, 두 쌍을 FP32와 학습용 BF16에서 micro1/2로 비교한다.

FP32 진단에는 autocast를 끄고 설치 버전에 맞춰 highest/IEEE matmul을 사용한다.
TF32 및 reduced-precision flag의 실제 값을 기록한다. 신·구 API를 섞어 설정하지 않는다.
모든 arm에 같은 backend를 적용하고 그 설정을 새 자원 측정에도 일관되게 사용한다.
모델 전체 .double()이 구현의 float cast 때문에 불가능하면 이를 기록하고,
독립 FP64 작은 연산 대조를 사용한다. full-model FP64 실행 자체를 새 관문으로 만들지 않는다.

비교량:
  z: native normalized output
  raw_scaled: raw output / 해당 채널의 고정 train std
  loss, unclipped/clipped gradient, actual update, Adam 1/2차 moment, RNG.
상대 L2 = ||a-b||2 / max(||a||2,1e-12), 모든 집계는 CPU float64.
raw 단위 max-abs는 보고하되, 단위와 무관한 하나의 raw 절대값으로 차단하지 않는다.

[설계: 이번 인증 한계]
같은 micro 크기의 activation-checkpoint 비교:
  기존 fp32 1e-5, bf16 1e-4 relative-L2 원칙 유지.
  출력의 절대차이를 볼 때 raw 대신 z/raw_scaled를 사용한다.
  정확히 보존되어야 하는 ID, mask, RNG, frozen-state hash는 정확 일치 검사.

micro1 대2 단일 업데이트:
  FP32:
    z/raw_scaled relative-L2 <= 1e-5, raw_scaled max-abs <= 1e-4.
    loss 차이 <= 1e-7 + 1e-5*abs(reference loss).
    전체 unclipped/clipped gradient 및 actual update relative-L2 <= 1e-4.
  BF16:
    z/raw_scaled relative-L2 <= 1e-2, raw_scaled max-abs <= 2e-2.
    loss 차이 <= 1e-6 + 1e-2*abs(reference loss).
    전체 gradient 및 actual update relative-L2 <= 1e-2.
  reference norm<1e-12이면 relative 대신 max-abs<=1e-6 사용.
  tensor별 차이도 전부 저장하고 큰 tensor 몇 개의 평균으로 이상값을 숨기지 않는다.
  Adam moment 차이는 별도 보고한다. moment 차이 하나로 무의미하게 차단하는 대신
  actual update와 아래 짧은 연속경로 검사를 함께 판단한다.

짧은 연속경로:
  각 6개 상태에서 같은 5개 effective batches를 micro1/2로 각각 실행한다.
  FP32/BF16 별도로 수행하고 모두 폐기한다. total delta=theta5-theta0를 비교한다.
  FP32 total-delta relative-L2 <= 1e-3, 끝 출력 raw_scaled max-abs <= 2e-4.
  BF16 total-delta relative-L2 <= 5e-2, 끝 출력 raw_scaled max-abs <= 5e-2.
  이 한계는 1,024회 학습의 동등성이나 예측 성능을 보증하지 않는다.

예산 산술:
  warmup12 + 단일인증48 + 5-step 두경로/두precision120 =180 updates.
  나머지12회는 명확히 기록한 진단용 여유다. 총192를 넘기지 않는다.
  파라미터 업데이트 없는 FP64·loss-only 대조는 별도 횟수로 기록한다.

판정:
- 의미 검사와 위 인증을 모두 만족: NUMERICS_BOUNDED. 원인을 완전히 규명했다고 하지 않는다.
- 의미 오류 발견: 제한된 수정 후 남은 예산 안에서 재검사. 코드 변경 전후를 보존.
- 한계 초과가 남음: INCONCLUSIVE_NUMERICS_V2. Q의 자원 선택/학습은 중단, C는 독립 진행.
- 실패한 Standard 옵션만 제외해 Query의 승리를 만들지 않는다.
- same-shape replay에는 동일 precision/backend에서 예측 복원을 확인한다.
  장치·shape가 바뀐 replay를 bitwise로 요구하지 않는다.

======================================================================
4. Q: 기존 자원 비교와 조건부 12 fits 완결
======================================================================

Q-수치가 통과하면 아래 전체를 수행한다.
기존 계약을 읽되 단독 파일만 전달받아도 실행 조건을 알 수 있도록 핵심을 재기록한다.

모델:
Chronos-2 revision 29ec3766d36d6f73f0696f85560a422f50e8498c.
Standard/Side/Query 각 trainable 1,179,648. 기존 출력층 고정.
context4096, horizon48, ETTm2/Electricity 기존 첫4채널.
2 origins × 4channels, BF16 autocast+FP32 parameter/Adam.
AdamW betas(.9,.999), eps1e-8, weight_decay0, global clip1.
LR: Standard/Query1e-4, Side1e-3. seeds39000/39001.

자료:
Train origins range(16384,22529,32), V range(23040,23713,96), E range(24576,26017,96).
scale: 기존 [16384,22576)의 std. 실제 ID/hash와 기존 데이터 계약 일치 확인.
과거 [o-4096,o), 미래 [o,o+48).
기존 E를 이미 본 개발 자료임을 명시. 원자료 재읽기는 기계적 staging과 구분한다.

옵션:
Standard CP blocks0/3/6/9/12; Side/Query0/12.
각 옵션 micro origins1/2 -> 18옵션/원천, 총36옵션.
실제 block index는 기존 budget.py의 block_indices를 유지한다.
모든 옵션의 frozen-pass/cache, transfer, loss, backward, clip, Adam을 시간에 포함한다.
상태 복구·검산 export·V 평가·저장은 active clock 밖, 총 wall에는 별도 포함한다.

측정:
기존과 같은 3 timing blocks, 각 block마다 3 complete updates.
update origin쌍 [17408,18432], [19456,20480], [17408,18432].
각 block 전 같은 warmed model/Adam/RNG로 복구한다.
block1 옵션 선택, block2/3 재확인. block2/3을 보고 옵션을 다시 선택하지 않는다.
순서 seed59000/59001/59002, CUDA synchronize, 기존 GPU guards 유지.
새 backend를 사용했다면 과거 timing을 이어 붙이지 말고 일괄 재측정한다.
cold Adam allocation과 warmed state peak를 모두 포함한다.
총 폐기 optimizer updates512 한도 안에서 warmup·CP 인증·측정을 수행한다.
이전 120회는 역사적 장부로 따로 기록한다.

예산:
1/2/4/8 GiB tensor allocated. 실제 GPU 전체 VRAM과 같다고 표현하지 않는다.
peak allocated/reserved, NVML, free memory와 model/load/eval peak도 따로 저장한다.
옵션 가능 조건: 검증한 peak <= .95*B.
B*: 두 원천에서 세 arm 모두 적어도 한 유효 옵션이 있는 가장 작은 공통 B.
각 원천/arm의 block1 중앙 step시간 최소 옵션을 선택한다.
최소시간2% 이내면 낮은 peak -> 적은 CP -> 큰 micro 순서로 동률 처리한다.
Standard의 제약 없는 최속 옵션이 B*에서 적어도 한 원천에서 불가능해야 한다.
선택된 Query가 Standard보다 block2와3 각각5% 이상 빠른 원천이 적어도 하나 있어야 한다.
모두 충족하면 RESOURCE_SIGNAL. 아니면 STOP_RESOURCE_SCREEN으로 Q 종료.
전 원천 승리 요구 없음. Query가 Side보다 빠를 필요도 없음.
어떤 예산에서 유리한지를 본 뒤 새로운 예산을 끼워 넣지 않는다.

Q 본학습:
2원천×2seed×3arm=12 attempts. A의 warmed weights를 본학습에 넘기지 않는다.
각 fit active120초, 체크포인트0/30/60/120초. 경계를 넘긴 첫 complete step에서 저장.
overshoot1초 이하; 안전 update 상한8192. 더 긴 step이면 본학습 전 timing 한계로 보고.
원천/seed별 8192개 origin쌍 stream 봉인, 각 arm은 같은 stream의 prefix를 사용한다.
빠른 모델은 더 많은 업데이트를 할 수 있다. 동일 updates 비교라고 하지 않는다.
총 학습시간과 전처리·로드·V·저장·대기를 분리한다.
V 최소 scaled_2pinball 선택, 동률은 빠른 시점. step0 허용.
12개 선택을 봉인·복원검사한 뒤 E scoring. 일부 fit 누락이면 주 E 비교는 열지 않는다.
Q의 개별 실패를 자원/수치/예측 실패로 구분한다.

Q 지표:
primary=기존 equal-channel train-std-scaled raw 2-pinball. 낮을수록 좋음.
G=100*(L_baseline-L_query)/L_baseline. loss ratio와 혼동하지 않는다.
Standard/Side/F0와 각각 비교하고 원천별 두seed raw loss를 먼저 평균한다.
다른 원천의 raw loss를 합쳐 평균하지 않는다. 필요시 원천별 상대gain을 macro 평균.
median MAE, coverage/width, updates, peak, active/wall, 초기 arm loss도 함께 보고.
기존 4블록 paired bootstrap1000회는 기술적 불확실성으로만 사용한다.

Q 기대 결과:
자원 이점과 예측 이점을 따로 판정한다. F0 승리를 모든 셀의 입장권으로 쓰지 않는다.
아주 작은 개선·예측 손해를 '동등'이라 부르지 않고 연속 수치로 보고한다.
Q를 종료하면 결과와 관계없이 다음 C의 독립 준비로 이동한다.

======================================================================
5. C: 채널 공유 구조 — 정식 실험 정의
======================================================================

[미검증 가설]
채널별 큰 변환의 일부를 공통 기저로 공유하면, 완전 공유의 제약을 완화하면서
채널별 독립 변환의 파라미터 증가를 줄일 수 있다.
이를 '채널 간 미래정보 교환', '새 채널 일반화', '최초의 tensor factorization'으로 주장하지 않는다.

C는 MOMENT-small과 Time-PEFT 공개 경로를 사용한다.
Q의 Chronos-2, 고정 head, 확률 loss를 가져와 같은 실험이라고 하지 않는다.
MOMENT 사용 이유: Time-PEFT의 실제 채널 어댑터를 기준으로 변경점 하나를 비교하기 위함.
이는 자원 제한을 위한 작은 백본·표준 원천의 개발 실험이며 원문38% 개선 재현을 약속하지 않는다.

5.1 환경과 upstream 일치
- 공개 TimePEFT 코드를 위 commit에 pin하고 license/attribution을 보존한다.
- 프로젝트 기존 PyTorch를 옛 requirements로 덮어쓰지 않는다.
- momentfm/peft 등이 필요하면 격리된 별도 venv에 설치한다.
  설치 버전/호환성/패키지 lock을 남기고 Q 환경은 변경하지 않는다.
- AutonLab/MOMENT-1-small을 읽고 실제 model revision SHA를 resolve/pin한 후 사용한다.
  숨은 가중치 경로나 revision을 추측하지 않는다.
- 기존 모델 캐시가 없으면 공식 공개 모델의 다운로드는 허용한다. 새 백본 사전학습은 금지.
- 공개 코드의 FrequencyAdapter/ChannelAdapter/TimePEFTPipeline/build_model을 읽고 재사용한다.
- 공식 train()의 best-state 미복원을 그대로 사용하지 말고 공통 best checkpoint runner를 구현한다.
- loss와 V 집계는 배치 평균의 단순 평균이 아니라 유효 원소 수를 합쳐 계산한다.
- torch.fft.rfft/irfft는 FP32로 수행한다. Fourier 경로의 half 정밀도 문제를 arm마다 다르게 처리하지 않는다.
- 사전학습 forecasting head가 없는 구성이라면 step0를 의미 있는 F0 forecast라고 부르지 않는다.
  초기 예측은 초기값 진단이고, 실용 기준은 학습한 LORA_HEAD다.

5.2 공통 모델 설정
- MOMENT-small 실제 hidden size D=512 확인. 다르면 자동으로 숫자를 바꿔 실행하지 않는다.
- context512, horizon96, 각 원천의 선택된 C=64개 채널.
- 패치 길이와 개수는 모델에서 읽어 확인한다. native MOMENT 경로를 그대로 사용.
- encoder LoRA rank8, alpha32, target q/k/v. 적용 위치는 공개 코드/실제 모듈로 확인.
- 원래 encoder/embedder/normalizer의 학습 권한은 공개 경로 기준으로 고정.
- forecasting head는 모든 arm에서 학습한다. head 구조와 초기 state는 같아야 한다.
- Time-PEFT 계열의 frequency top-k=3, down bottleneck r=256, ReLU, dropout.1, LayerNorm.
- 같은 head/LoRA/frequency 파라미터의 초기값은 seed별 master snapshot으로 공유한다.
- 모든 arm의 공통 요소와 arm별 요소를 나눠 trainable count를 다시 계산한다.

5.3 비교할 6개 arm (이번에 만드는 작업명)

LORA_HEAD:
  pretrained hidden -> forecasting head. encoder LoRA+head만 학습.
  frequency/channel adapter 없음. 표준 PEFT 참고선; BASIS4와 동일 파라미터 수는 아니다.

SPECIFIC:
  Time-PEFT 구조: common frequency/down_projection -> 채널별 독립 up_projection -> LayerNorm -> head.
  각 채널 up은 D×r weight와 D bias. r=256.
  공개 ChannelAdapter와 같은 forward 구조다. 아래 공통 초기화는 원문과 다를 수 있어 기록한다.

SHARED:
  SPECIFIC에서 모든 채널이 하나의 up_projection을 공유. r=256.
  작은 완전 공유 대조군. BASIS4보다 파라미터 수가 적다.

SHARED_WIDE:
  모든 채널은 같은 up을 공유하되 down/up bottleneck을 넓혀 BASIS4와 예산을 맞춘다.
  D512,C64,K4일 때 width513. 생성 코드가 hidden_dim_ratio 정수만 받으면
  새 명시적 width wrapper를 만들고 기존 constructor가513을 받는다고 가정하지 않는다.

GROUP4:
  down은 공유. 4개 up을 만들어 채널을 고정된 4그룹에 나눈다.
  64개 channel ID를 SHA256(고정seed,ID) 정렬 후 순위 modulo4로 배정한다.
  그룹은 성능·상관계수·평가 정답으로 고르지 않는다. 각 그룹16채널.
  통계적 군집화가 아니라 '정적 파라미터 분할로도 충분한가'를 보는 값싼 대조다.

BASIS4:
  4개 up basis와 채널별 4개 계수를 학습한다. r256,K4.
  u_c=ReLU(W_down concat(h_c,f_c)+b_down), dropout은 기존과 같음.
  W_c=sum(k=0..3) a[c,k]*B[k], b_c=sum(k=0..3) a[c,k]*b[k].
  output_c=LayerNorm(W_c u_c+b_c), 뒤의 head는 동일.
  a는 정적인 학습 파라미터: input에 따라 변하는 router가 아니며 signed 실수 계수다.
  softmax/별도 encoder/추가 regularizer/잔차 경로를 임의로 추가하지 않는다.
  모든 계수·basis·bias가 학습 파라미터에 포함된다.

[핵심 비교]
BASIS4 vs SHARED_WIDE: 동일한 예산에서도 채널별 계수가 의미가 있는가?
BASIS4 vs GROUP4: 4개로 나누는 것 자체 말고 학습된 조합이 필요한가?
BASIS4 vs SPECIFIC: 훨씬 적은 파라미터로 얼마나 예측을 유지하는가?
BASIS4 vs SHARED: 작은 완전 공유만으로 충분하지 않은가?
BASIS4 vs LORA_HEAD: 새 어댑터를 사용하는 실제 이유가 있는가?

5.4 파라미터 수 — down, up, bias, LayerNorm 포함
공통 LoRA/frequency/head는 아래 수에 포함하지 않는다. 최종 표에서는 반드시 더한다.
P_specific(r)=(2D+1)r + C*D*(r+1) + 2D.
P_shared(s)=(3D+1)s + 3D.
P_group(r)=(2D+1)r + K*D*(r+1) + 2D.
P_basis(r)=P_group(r)+C*K.

D512,C64,r256,K4일 때:
  SPECIFIC      8,684,800
  SHARED          395,008
  SHARED_WIDE     790,017  (width513)
  GROUP4          789,760
  BASIS4          790,016
SHARED_WIDE와 BASIS4의 channel block 차이는1개, GROUP4와는256개다.
소스 공식과 actual named_parameters 양쪽으로 확인한다.
동일 예산 비교는 위 세 arm이다. SPECIFIC/SHARED/LORA_HEAD까지 같은 예산이라 쓰지 않는다.
숫자를 맞추기 위한 사용되지 않는 dummy parameter를 만들지 않는다.
채널 block의 절약률을 전체 모델/전체 trainable 절약률로 둔갑시키지 않는다.
C16/32/64/128에서 이론·실제 constructor count도 CPU로 기록하되 추가 학습은 하지 않는다.
단일 C64 학습만으로 채널 수에 따른 정확도 스케일링을 입증했다고 쓰지 않는다.

5.5 초기화: 같은 출발 함수를 가능한 범위에서 유지
seed별 공통 down/frequency/head/LoRA/LayerNorm과 reference up U0,b0를 생성한다.
- SPECIFIC의 모든 up은 U0,b0로 값을 복사하되 서로 다른 Parameter로 생성한다.
- SHARED는 U0,b0. GROUP4의4개 up도 U0,b0로 복사한다.
- BASIS4는 B0=U0,b0. 나머지3개 basis는 독립 fan-in 초기화.
  모든 채널 a[c]=[1,0,0,0]로 초기화한다.
  따라서 step0 output은 공통 U0와 같지만, unused basis 방향에 대한 계수 gradient는 가능하다.
- SHARED_WIDE는 앞256 hidden행과 up열에 공통 값을 복사하고 추가 hidden행은 독립 초기화,
  추가 up열은0으로 둔다. 처음 출력은 같지만 extra parameter가 영구히 dead하지 않아야 한다.
- SPECIFIC~BASIS4의 초기 출력 일치는 model.eval(), FP32, 같은 입력에서 검사한다.
  LORA_HEAD와의 초기 동일성은 요구하지 않는다. 경로가 다르기 때문이다.
- dropout이 있는 학습 모드에서 다른 폭 모델들의 bitwise identity를 요구하지 않는다.
- unused basis matrix나 wide extra down의 첫 step gradient0는 정상일 수 있다.
  2~몇회 점검 후 경로가 활성화되는지 확인하고 모든 tensor가 매step nonzero라는 검사는 금지.

5.6 CPU/실모델 사전 검사
CPU FP64 작은 tensor로 다음을 확인한다:
  BASIS4를 effective W_c로 만든 결과와 basis별 합 결과 일치.
  K1 또는 모든 a 동일이면 shared 모델로 환원.
  고정 one-hot a이면 GROUP 구조로 환원.
  channel tensor와 ID/계수를 함께 순열하면 출력도 같은 순열로 바뀜.
  채널 ID가 아니라 현재 batch slot으로 계수를 고르는 오류가 없음.
  init 함수 일치, finite gradient, active path update, 실제 count 일치.
위 함수 일치의 CPU 기준은 atol/rtol1e-10. 신경망 성능 판정 기준이 아니다.

실모델:
공식 SPECIFIC과 wrapper에 동일 state를 주고 같은 입력 FP32 eval 출력/목적식 일치 확인.
이번 SPECIFIC 초기화가 공식 기본 initializer와 다르다는 점도 별도 기록.
두 source×6arms×2점검 updates=24회 기본 점검. 추가 검사를 합쳐72회 이내.
실제 model/loader/head shape, loss target align, frozen weights, finite raw loss를 검사.
예측을 읽는 동안 정답/정규화 정보 누출이 없는지 확인.
학습 그래프가 동작한다는 확인만 하면 된다. 여기에서 후보가 이겨야 본학습하는 규칙은 금지.

======================================================================
6. C: 데이터, 학습, 선택, 평가
======================================================================

6.1 두 원천
Electricity와 Traffic의 원본 다채널 배열을 사용한다.
전자의 full raw는 기존 Electricity manifest에서 실제 경로/hash를 읽어 찾는다.
후자는 다음 읽은 receipt를 진입점으로 사용한다:
  research/overnight_20260913/data_receipt.json
[확인] 이 receipt의 Traffic 원자료는 data/raw/overnight_20260913/traffic.txt.gz,
17544행, 시간 간격3600초로 기록되어 있다. 기존 작업이4채널만 썼더라도
원자료 전체 열 수를 직접 검사한 후64개를 선택한다. 전처리4채널 배열을64채널로 복제하지 않는다.
Electricity/Traffic은 기존 연구에서 본 원천이다. 새채널이 있다고 완전 독립 corpus라 하지 않는다.

선택/시간 분할은 예측 성능을 보지 않고 아래 식으로 고정한다:
  원본 전체 길이 N, train_end=floor(.6*N), val_end=floor(.8*N).
  train rows [0,train_end), V target rows [train_end,val_end), E [val_end,N).
  채널 적격성은 Train에서만 finite 전수 및 std>1e-6.
  원본 열 순서 기준 적격 채널 처음64개. 부족하면 BLOCKED_CHANNEL_SUPPORT.
  실제 ID, 원본 열 index, source hash, split index를 저장한다.
  V/E의 미래 값을 보고 채널을 교체하지 않는다.

Origin grids (0-based):
  Train range(512,train_end-96+1,24).
  V range(train_end,val_end-96+1,96).
  E range(val_end,N-96+1,96).
  x=[o-512,o), y=[o,o+96).
  V/E 각각 최소16개 origins가 있어야 한다. 부족하면 입력 조건 미충족으로 기록.
  V/E context가 바로 전 기간의 관측을 사용하는 것은 허용한다.
  학습 target이 V/E target로 넘어가서는 안 된다.

정규화:
Train 전체에서 채널별 mean/std(float64)를 fitting하고 모든 arm에 공유.
모델 input/output은 이 train-standardized 공간을 사용한다.
MOMENT의 내부 normalizer는 공식 경로대로 유지한다.
평가는 저장한 scaler 기준으로 한 번만 계산한다. 이미 표준화된 오차를 std로 다시 나누지 않는다.
V/E 결측이 있다면 input은 train mean(표준화0)으로 대체하고 별도 입력 mask 가능 여부를 확인,
target은 mask하여 평가한다. 미래 통계·보간으로 채우지 않는다.
mask를 타당하게 처리할 수 없으면 임의 완성하지 말고 해당 source를 BLOCKED_INPUT으로 남긴다.

6.2 같은 학습 기회
6arms×2sources×seeds40000/40001 =24 attempts.
fit마다 1,024 effective optimizer updates, batch2 origins×64channels.
source/seed별 표본 stream1,024쌍을 같은 train origin grid에서 복원추출하여 봉인한다.
각 arm은 같은 표본 순서와 노출 횟수. 이 비교를 동일 wall-time 비교라고 부르지 않는다.

모든 arm의 기본 정밀도는 FP32로 고정해 Q의 microbatch/BF16 문제를 여기로 옮기지 않는다.
모델·자원 준비에서 FP32 batch2가 모든 arm에 맞지 않으면,
훈련 점수를 보기 전에 모든 arm에 공통 origin microbatch1+gradient accumulation2를 선택한다.
각 micro loss1/2, clip/Adam은 마지막1회. group 관계를 바꾸지 않는다.
후보만 BF16/작은 배치로 바꾸지 않는다. 이 선택은 preflight 자원으로만 결정한다.
필요한 공통 encoder checkpoint는 모든 arm에 같은 방식으로 적용하고 초기에 봉인한다.
공통 배치/저장 전략으로도 물리 VRAM이 부족하면 BLOCKED_RESOURCE_C.
C=64를32로 줄이거나 작은데이터로 몰래 바꾸는 자동 fallback은 없다.

Optimizer: AdamW, LR1e-3, betas(.9,.999), eps1e-8, weight_decay.01, global clip1.
이는 공개 코드 LR와 AdamW 기본 weight decay를 참고한 단일 공통 recipe다.
최선 LR를 탐색한 비교는 아니며, 다른 크기 모델들의 최적화를 완전히 끝냈다고 주장하지 않는다.
LR은 update640 이후5e-4. 임의 plateau scheduler나 arm별 별도 스케줄은 금지.
모든 arm의 적용 규칙과 실제 optimizer parameter 목록을 저장한다.

V checkpoints: 0,128,256,512,1024.
매번 train/eval 모드와 RNG 복원을 정확히 한다.
검증 표준화 MSE 최소, 정확 동률은 이른 step.
선택 모델을 디스크에서 다시 불러 V prediction/score를 재검사한다.
원본 run.py의 마지막 state 반환 동작을 쓰지 않는다.
1,024에서 선택되고 말기 V가 내려가는 경우 BUDGET_LIMITED 표시.
이 경우 '수렴 후에도 안 됨'이라고 하지 않고 추가 자동 학습도 하지 않는다.

fit 순서는 seed60020 permutation으로24개 사전 봉인한다.
한 arm만 먼저 E를 보고 다른 arm 설계를 바꾸지 않는다.
24 fits 완료/모든 선택 봉인 후 E scoring한다.
일부 fit이 실패하면 예정된 나머지는 안전할 때 수행하되, 불완전한 source/seed 블록의
E를 주 비교에 넣지 않는다. 완성된 모든6arm×2seed의 source 블록만 별도 봉인 후 평가할 수 있다.
한 원천만 완성되면 one-source pilot로 표시한다. 실패 fit을 다른seed로 채우지 않는다.

참조 성능:
SPECIFIC이 LORA_HEAD보다 좋아져야만 나머지를 실행하는 규칙은 두지 않는다.
모든 결과를 보고 '이번 축소 setting에서 Time-PEFT 구조의 이득을 관측했는가'를 별도로 판단한다.
관측하지 못했다면 원문 전체를 반증하거나 파이프라인이 반드시 버그라고 단정하지 않는다.
원문의 혼돈계/심전도 조건과 작은 표준 원천 pilot은 다른 실험이다.

6.3 평가와 결과 판정
주지표: equal-channel train-standardized MSE. 낮을수록 좋다.
보조: standardized MAE, 원래 단위 채널별 오차, head/adapter/LoRA별 params,
총 trainable params, 모델 총params, 실제 allocated/reserved peak, fit wall/step latency.
학습하지 않는 가중치를 포함한 총 모델 크기와 trainable count를 혼동하지 않는다.

개선율 G[b]=100*(MSE_b-MSE_basis)/MSE_b.
원천별 두seed raw MSE 평균을 먼저 계산한 gain과 seed별gain을 같이 보고한다.
Q의 pinball과 C의 MSE를 하나의 평균 성능으로 합치지 않는다.

두 가지 목표를 따로 판정:
(A) 경량화 관찰:
  BASIS4 총 trainable params가 SPECIFIC의50% 이하이고,
  그 원천 두seed 평균 MSE가 SPECIFIC 대비1% 이내 열화이면 '제한된 경량화 신호'.
  이는1% 허용의 점추정 기준이며 통계적 동등성 증명이 아니다.
(B) 채널별 학습 계수의 추가 가치:
  적어도 한 원천에서 BASIS4가 SHARED_WIDE와 GROUP4 각각보다 평균0.5% 이상 낮은 MSE,
  해당 두seed 모두 같은 비교의 gain>0이면 '계수 활용 신호'.
  SHARED/LORA_HEAD 대비도 반드시 같이 보고하여 더 단순한 방법이 최선인지 확인한다.
  가장 간단한 LORA_HEAD가 더 정확하고 더 작으면 배포용 PEFT의 추가 가치가 있다고 쓰지 않는다.
수치0.5/1/50%는 이번 후속투자 기준이며 보편 논문 PASS 조건이 아니다.
모든 원천이 동시에 이겨야 한다는 규칙은 없다. 한 원천의 신호를 범용 효과로 확대하지 않는다.
A만 충족하면 단순 압축 관찰, B만 충족하면 계수의 효과 관찰로 구분한다.
새 방법 가능성은 A/B 및 가까운 선행 대비 차이를 함께 검토해야 하며 이번에 확정하지 않는다.

메커니즘 기술통계(새 fits 없음):
- 계수의 채널별 분산, effective W_c 간 차이, basis 사용량과 실제 update 기록.
- 평가 후 선택된 BASIS4에 채널 계수를 고정 순열로 교환한 추가 inference1회.
- 계수 교환 결과는 학습 중 비교나 인과 증명이 아닌 사후 진단이다.
- 배치 slot과channel ID를 같이 바꾼 permutation-equivariance 테스트와 혼동하지 않는다.
- 계수 차이가 거의 없고 SHARED와 비슷하면 '학습된 채널별 특화'를 주장하지 않는다.

불확실성:
source별 E origin을 시간순4개 묶음으로 paired block bootstrap1000회(seed60100).
끝의 불완전 블록은 버리지 말고 실제 길이를 유지해 결과에 모두 포함한다.
같은 draw를 모든 arm/seed/channel에 적용, 최종 draw 길이는 원래 origin수로 절단한다.
seed/채널을 독립 데이터셋으로 세지 않는다. 자료 재사용의 연구자 선택 편향까지 보정한 CI가 아니다.

======================================================================
7. 실행 운영: 분석만 하다가 끝나지 않게 하되, 오류를 성능으로 바꾸지 말 것
======================================================================

순서:
  1) 저장소/환경/두 source/공개 코드 확인, 공통 manifest.
  2) Q numeric_v2 프로토콜·tests·해시 봉인 -> 제한 진단.
  3) Q 통과 시 자원 측정 -> 조건부 Q12fits -> 평가·보고.
     미통과면 그 이유를 보고하고 Q를 종료.
  4) C isolated 환경/자료/구조·파라미터·upstream parity 검사.
  5) C 프로토콜·표본 stream·모델 revision·source hash 봉인.
  6) C24fits -> 선택 봉인 -> 평가·독립 검증.
  7) 통합 한국어 보고서와 기존 결과 색인에 새 항목 추가.

한 작업에 문제가 있으면 다른 작업과 공유되는 문제인지 판단해 독립 작업을 진행한다.
코드를 구현하느라 길어졌다는 이유만으로 '실행 명령은 다음에' 하고 종료하지 않는다.
그러나 실제 필수 자료·장치·수치 무결성이 막힌 경우에는 완료한 범위와 남은 것을 보고한다.

GPU:
- 기존 프로젝트 lock, worker/child PID 구분, single worker.
- startup30초 외부 compute 없음, free>=4GiB.
- timed Q update 중 외부 compute는 timing 오염으로 기록. 임의 좋은 반복만 남기지 않는다.
- 다른 GPU 작업은 종료하지 않음. graphics PID를 training PID로 오인하지 않음.
- free<1GiB 또는 외부 작업 출현 시 안전 경계에서 일시정지.
- 대기 누적600초/track이면 BLOCKED_GPU_BUSY. 무한 pgrep 대기 금지.
- C에 Q의 synthetic tensor budget을 강요하지 않음. 실제 안전 자원 한도를 기록.
- 학습 중 OOM -> 해당 fit 실행 실패, 임의batch축소·캐시삭제·재시작 금지.
- 공통 환경이 손상되면 두 track 모두 중단. Q의 개별 numeric 문제만으로 C를 막지 않음.
- dependency 설치는 C의 격리 환경에서만. 드라이버/시스템 CUDA 교체 금지.

평가 보안:
E 자료를 만든 staging과 E를 보고 모델을 선택하는 행위는 구분해서 기록.
모든 비교 arm의 selection seal 전 E 점수로 선택·중단·변형하지 않는다.
기존 개발 E를 재사용한 사실은 숨기지 않는다.
파일 해시 검사는 승인된 새 폴더·새 index 변경만 제외하고 기존 결과를 보호한다.

======================================================================
8. 산출물과 최종 사용자 보고
======================================================================

각 track에 다음을 남긴다:
- PROTOCOL.md: 질문, source-derived facts, 새 설계, 예상 결과와 해석 한계.
- contract/source/data/model/environment manifest와 해시.
- attempts.csv: 계획/실행/오류/실제fits/폐기업데이트/학습업데이트 분리.
- numeric_checks, parameter_inventory, leakage_checks, phase exit code.
- 자원 표: 모든 옵션/반복/예산, 초기화/최적화상태/피크범위 구분.
- trajectories.csv, checkpoints/selection_seal, prediction manifest와 원점수.
- independent_verification: 별도 scalar 구현과 vectorized metric 차이<=1e-10(float64 집계).
  같은 배열의 지표 재계산과 새 GPU 재추론을 다른 검증으로 기록.
- REPORT.md: 한국어. JSON은 내부 기록으로 가능하나 사용자 보고를 JSON 하나로 대신하지 않음.

통합 보고서는 아래 질문 순서에 답한다:
1. Q/C 각각 실제 무엇을 실행했나? 미실행을 성능0/실패로 쓰지 않음.
2. Q 수치 중단은 무슨 근거로 해소/미해소됐나? 옛 기준 변경을 숨기지 않음.
3. Standard의 강한 메모리 대조를 허용해도 Query 이점이 남았나?
4. C에서 무엇을 공유했고 파라미터 수는 정확히 얼마나 달랐나?
5. 같은 예산 공유/고정 그룹보다 BASIS4가 더 좋은가?
6. SPECIFIC과 LORA_HEAD 대비 실제 가치가 있는가?
7. 다음 연구를 계속할 근거/중단할 근거는 무엇인가? 추가 자동 학습은 없음.

공통 금지:
- 검사를 통과시키려고 허용오차를 계속 키우기.
- tiny gain, 구현 정상화, 학습 횟수를 논문 기여로 포장하기.
- 표준 baseline의 설정을 약하게 만들거나 실패한 baseline만 제외하기.
- 코드 수식과 다른 '시계열 특화' 설명을 붙이기.
- 공개 코드의 학회 전체 성능을 재현했다고 말하기.
- raw score 없이 F0=1 표만 보여주기.
- 계열별 std, 원천별 unit이 다른 숫자를 그대로 더해 극적인 결론 만들기.
- Q/C를 수행한 뒤 새 후보를 재귀적으로 만들고 학습하기.

최종 실행 결과는 예를 들어 다음 형식으로 답한다:
'Q: 수치검사 ..., 자원측정 ..., 본학습 n/12, 결과 ... .
 C: 구조검사 ..., 본학습 m/24, 같은 예산 비교 ..., 압축 관찰 ... .
 두 결과는 개발 증거이고 신규성/독립 확증은 ... .
 보고서 경로 ..., commit/push 상태 ... .'
이 예시의 빈칸을 실제 결과 없이 채우지 않는다.

끝. 이 문서는 Q와 C의 제한된 실행 승인을 해석한 지시이며,
한도가 끝나면 결과를 보고하고 새로운 실험으로 넘어가지 않는다.
