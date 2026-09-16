시계열 조건 PEFT 표본 설계 교정 및 6개 영향 트랙 재평가 실행 계약
버전: condition_sampling_repair_v1_20260917
작성 기준: CanelE452/tsfm-peft-method-screen commit b80e4a4af34c34827a24319258f334d9a171cf86
대상 저장소: CanelE452/tsfm-peft-method-screen

======================================================================
0. 이번 작업의 목적
======================================================================

이번 작업은 새 PEFT 후보를 발명하거나 기존 결과를 PASS로 만들기 위한 작업이 아니다.

직전 condition_studies_v1_20260916에서 공통 원점 선택 방식이
"하루 안의 phase를 먼저 맞추고 각 phase 안에서 linspace로 원점을 선택"하도록 되어 있어,
64개 원점을 사용했어도 실제로는 몇 개 index-day에 원점이 몰리는 문제가 확인됐다.

대표적인 실제 기록:
- N01 E: 64 origins / 5 index-days / 3 seven-day blocks
- N02 E: 64 origins / 6 index-days / 3 seven-day blocks
- N03 E: 64 origins / 23.75 hours span / 2 partial days
- R04 E: 64 origins / 5 index-days / 4 seven-day blocks
- N07 E: 64 origins / 5 index-days / 3 seven-day blocks
- R08 E: 64 origins / 6 index-days / 3 seven-day blocks
- R09 E: 64 origins / 64 index-days / 32 seven-day blocks

따라서 이번 작업의 최상위 질문은 하나다.

"이전 6개 학습 트랙의 결론이, 날짜/기간 다양성을 먼저 확보한 표본 설계에서도 유지되는가?"

이번에 고치는 것은 표본 선정 하나다.
다음은 바꾸지 않는다.
- 문제 정의
- 데이터 원천
- split 경계
- 모델/backbone
- LoRA 구성
- 각 방법의 수식
- loss
- LR 후보
- seed
- 총 업데이트 수
- checkpoint 후보
- 주지표
- 핵심 직접 대조
- 기존 선행 해석

새로운 구조, loss, dataset, seed, LR, 후속 후보를 자동 추가하지 않는다.

======================================================================
1. 최신 상태 감사와 불변 기준
======================================================================

시작 즉시 다음을 확인한다.

1) git HEAD와 dirty state.
2) 기준 commit b80e4a4 이후 변경사항.
3) condition_studies_v1_20260916의 다음 산출물 존재 및 hash:
   - MASTER_REPORT.md
   - FINAL_DECISION.md
   - origin_dispersion.csv
   - 각 track REPORT.md
   - 각 track origins.json/csv
   - fit/selections/predictions/verification 관련 receipt
4) 로컬 ignored cache의 기존 checkpoint/prediction 존재 여부.
5) Chronos-2 pinned revision과 rank8 LoRA 계약.
6) 원자료 hash:
   - electricity
   - traffic
   - ETTm1

이전 결과를 덮어쓰지 않는다.

새 루트:
- experiments/condition_sampling_repair_v1_20260917/
- results/condition_sampling_repair_v1_20260917/
- .cache/condition_sampling_repair_v1_20260917/
- docs/CONDITION_SAMPLING_REPAIR_20260917.md
- scripts/run_condition_sampling_repair.py

기존 condition_studies_v1_20260916의 코드/결과는 read-only reference로 취급한다.

======================================================================
2. 재실행 대상과 제외 대상
======================================================================

[재실행 대상 — 공통 원점 집중의 영향을 받은 6개]
S01 = N01 ASYNC
S02 = N02 ARCHIVE
S03 = N03 CLOCK
S04 = R04 REVISION
S05 = N07 SPECTRAL
S06 = R08 LEAD

각 트랙의 방법군은 기존과 완전히 동일하게 유지한다.

S01/N01:
A0 NATIVE
A1 RIDGE
A2 RESID
A3 AGE_RESID
핵심: A3 vs A2 / A1

S02/N02:
B0 SHORT
B1 LONG
B2 RETRIEVE
B3 DELTA_ADAPT
CPU blend 포함
핵심: B3 vs B2 / B1 / BLEND

S03/N03:
C0 GRID
C1 HOLD
C2 KERNEL
C3 LEARN_KERNEL
핵심: C3 vs C2 / C0

S04/R04:
D0 PLAIN
D1 STABLE
D2 CORR
D3 INNOV
CPU smoothing 포함
핵심: accuracy 1% 보호 하 D3 vs D2/D1/BLEND

S05/N07:
G0 BASE
G1 ENERGY
G2 PRED
G3 SHUFFLE
spectral ridge 포함
핵심: G2 vs G1/G3/G0

S06/R08:
H0 BASE
H1 STATIC
H2 LINEAR_H
H3 LEAD_H
frozen/ridge/blend 포함
핵심: H3 vs H2/H1/BLEND, FROZEN은 별도 참고

[재실행하지 않는 3개]

R05 VINTAGE:
- 기존 forecast-path 결과의 저장 예측 재분석.
- 이번 문제의 공통 origin selector로 학습한 track이 아니다.
- 새 neural fit 없이 75 test dates를 사용하는 별도 분석.
- 기존 결과를 reference로만 보존.

N06 JOINT:
- 기존 확률예측을 이용한 CPU dependence/coupling 비교.
- 이번 공통 neural origin selector와 무관.
- 새 neural fit 없음.
- 기존 결과를 reference로만 보존.

R09 MIXED:
- 기존 E 64 origins가 64 distinct index-days / 32 seven-day blocks에 분산되어 있음.
- 직전 공통 원점 집중 문제에 해당하지 않음.
- 이번 repair에서는 재학습하지 않는다.
- 기존 결과를 reference로만 보존.

이 세 트랙의 결과를 이번 repair의 새 증거라고 다시 세지 않는다.

======================================================================
3. 가장 중요한 변경 — DAY-FIRST ORIGIN SELECTION
======================================================================

기존 문제:
phase를 먼저 나눈 후 각 phase에서 첫/중간/끝 위치를 뽑아
서로 다른 phase가 같은 몇 날짜를 반복 선택했다.

이번에는 순서를 반대로 한다.

"기간을 먼저 넓게 분산 -> 서로 다른 날짜를 선택 -> 각 날짜 안에서 phase를 분산"

3.1 기본 단위

hourly source:
day_id = floor(origin_index / 24)
phase = origin_index mod 24

ETTm1 15-minute:
day_id = floor(origin_index / 96)
phase = origin_index mod 96

날짜 열이 없는 Electricity/Traffic에서 day_id/phase를 실제 현지 달력 시각이라고 부르지 않는다.
"index-day", "index-phase"라고 적는다.

3.2 role별 목표 원점 수

TRAIN: 64
V_SELECT: 32
V_CAL: 32
E_DISCOVERY: 64

원칙:
한 selected day에서 origin 하나만 사용한다.

따라서 목표 distinct day 수:
TRAIN 64
V_SELECT 32
V_CAL 32
E_DISCOVERY 64

예외적으로 해당 role의 합법적 distinct days가 목표보다 적으면
두 origins/day로 자동 완화하지 않는다.
그 track/role은 BLOCKED_DIVERSITY로 기록한다.

표본 수를 채우기 위해 기간 경계, 데이터셋, 채널, 조건을 바꾸지 않는다.

3.3 합법적 candidate origin을 먼저 만든다

각 track의 기존 context/horizon/special contract를 그대로 사용한다.

일반:
- context는 origin 이전만 사용
- target horizon은 자기 split role 경계를 넘지 않음
- natural missing이 있는 context/label은 기존 규칙대로 제외
- first 4 eligible numeric channels 유지
- TRAIN/V/E 경계는 기존 60/70/80/100% 유지

N01:
기존 C336/H48 조건과 mask 생성 규칙 유지.

N02:
C336 / LONG C1344 / H48 유지.
retrieval legal bank 조건 유지.
query target/self-overlap exclusion 유지.
bank 후보 >=32 조건 유지.

N03:
ETTm1 C336 slots / H48 slots 유지.
dense/delta4 train, delta2/irregular E 조건 유지.

R04:
paired origin o, o+24와 두 horizon이 role 경계를 모두 만족해야 함.

N07:
C336/H48 유지.

R08:
C336/H48 유지.

3.4 날짜를 먼저 고르는 알고리즘

각 role에서 합법적 candidate origins를 day_id로 group한다.
candidate_day는 적어도 origin 1개를 가진 day다.

D = sorted(candidate_day_ids)

require len(D) >= required_count

required_count:
TRAIN 64
V_SELECT 32
V_CAL 32
E_DISCOVERY 64

선택할 day index:
idx_k = round(k * (len(D)-1) / (required_count-1))
for k=0..required_count-1

반드시:
- idx가 strictly unique
- selected day가 strictly unique
- 첫 selected day와 마지막 selected day가 candidate day range의 양 끝을 포함
- selected day span / candidate day span >= 0.95

round 결과 중복이 생기면
성능과 무관한 deterministic nearest-unused index로 보정한다.
이 보정 규칙을 코드에 명시하고 unit test한다.

랜덤 seed로 좋은 날짜를 찾지 않는다.

3.5 선택된 날짜 안에서 phase 분산

role의 origin count = n.
period P:
hourly P=24
ETTm1 P=96

target phase roster:
phase_k = floor(k * P / n), k=0..n-1

n>P이면 같은 phase가 자연스럽게 반복될 수 있다.
n<P이면 서로 다른 phase가 가능한 한 넓게 분산된다.

selected day k에서:
- 그 day의 합법 candidate origin 중
- circular phase distance가 target phase_k에 최소인 origin 선택
- tie: 작은 phase -> 작은 origin index

circular distance:
min(abs(p-q), P-abs(p-q))

성능값, target값, model prediction으로 선택하지 않는다.

3.6 phase diversity 사전 검사

hourly TRAIN/E 64:
- distinct phase >= 20
- phase별 count max-min <= 2

hourly V 32:
- distinct phase >= 16

ETTm1 TRAIN/E 64:
- distinct phase >= 56

ETTm1 V 32:
- distinct phase >= 28

기준을 못 만족하면 BLOCKED_DIVERSITY.
결과를 보기 전에 차단한다.

3.7 time-block diversity 사전 검사

각 role에서 보고:
- distinct days
- first/last day
- selected day span
- candidate day span
- span ratio
- distinct 7-day blocks
- distinct phases
- phase histogram
- origin-to-origin gap distribution
- target timestamp unique count
- total target timestamp count
- target uniqueness ratio
- context timestamp unique count
- overlap multiplicity histogram

최소 7-day block 기준:
TRAIN/E 64 origins: >= 8
V_SELECT/V_CAL 32 origins: >= 4

못 만족하면 BLOCKED_DIVERSITY.

3.8 N03에 대한 추가 강제 조건

이전 N03:
TRAIN span 23.5h
V_SELECT span 11.5h
E span 23.75h

이번에는 위 상태를 절대 허용하지 않는다.

N03:
- TRAIN 64 distinct index-days
- E 64 distinct index-days
- V_SELECT 32 distinct index-days
- V_CAL 32 distinct index-days
- E span >= 80% of legal E day range
- TRAIN span >= 80% of legal TRAIN day range

이 검사 통과 전 GPU 0 updates.

======================================================================
4. OLD vs REPAIRED 원점 비교 감사
======================================================================

학습 전에 다음 파일을 만든다.

results/condition_sampling_repair_v1_20260917/ORIGIN_REPAIR_AUDIT.md
results/.../old_vs_new_origin_dispersion.csv
results/.../new_origins/<track>.json
results/.../new_origins/<track>.csv

트랙별로 표:
- old TRAIN origins
- new TRAIN origins
- old/new distinct days
- old/new seven-day blocks
- old/new phase count
- old/new origin span
- old/new target uniqueness ratio

특히 N03 old/new를 별도 표시.

새 origin을 만든 후 hash seal.
그 뒤 방법/학습 설정을 바꾸지 않는다.

======================================================================
5. 모델/방법/학습 설정 — 이전 계약 그대로
======================================================================

이번 repair의 과학적 변경은 origin selection뿐이다.

공통:
- amazon/chronos-2
- 이전 pinned revision 그대로
- FP32
- TF32 off
- backbone dropout 0
- original backbone/head frozen
- rank8 LoRA 동일
- 동일 method-specific auxiliary parameters
- AdamW(beta=.9/.999, eps=1e-8, weight_decay=0)
- clip norm 1
- LR candidates {1e-4, 3e-5}
- no scheduler
- TRAIN64 x 8 epochs = 512 updates/fit
- checkpoints {0,256,512}
- LR-selection seed 73100
- repeat seeds 73101,73102
- selection seed E는 주결과에 포함하지 않음
- exact tie: smaller LR, earlier checkpoint
- 기존 primary metric 그대로
- 기존 CPU controls 그대로

각 track의 method 수:
6 tracks x 4 methods = 24 method groups

fit 수:
선택 seed: 24 methods x 2 LR = 48 fits
repeat: 24 methods x 2 seeds = 48 fits
합계 = 96 fits

updates:
96 x 512 = 49,152 main optimizer updates

smoke:
24 method groups x 2 updates = 48 discarded smoke updates

전체 optimizer hard cap:
49,200 updates

남은 예산을 새 후보/추가 LR/추가 seed에 사용하지 않는다.

======================================================================
6. 선택과 평가
======================================================================

6.1 설정 선택

이전과 동일하게 V_SELECT만 사용.

각 method:
seed73100에서 LR 2개 비교.
각 LR의 checkpoints 0/256/512 평가.
기존 track-specific selection rule 유지.

R04:
기존 accuracy <= selected D0 V accuracy * 1.01 조건 아래 revision RMS 선택 규칙 유지.

나머지:
V primary accuracy 최저.
tie: smaller LR -> earlier checkpoint.

고정 후 repeat seed73101/73102 실행.

6.2 evaluation seal

repeat seed의 selections를 모두 저장.
CPU controls를 V에서만 고정.
그 뒤 evaluation seal 작성.
그 뒤 E label을 scorer가 읽는다.

평가 결과를 보고:
- LR 변경 금지
- checkpoint 변경 금지
- origin 변경 금지
- condition 변경 금지
- seed 추가 금지

6.3 primary contrasts

이전과 동일.

S01/N01:
A3 vs A2
A3 vs A1
A3 vs A0

S02/N02:
B3 vs B2
B3 vs B1
B3 vs BLEND

S03/N03:
C3 vs C2
C3 vs C0
C3 vs C1

S04/R04:
D3 vs D2
D3 vs D1
D3 vs BLEND
accuracy/revision 분리

S05/N07:
G2 vs G1
G2 vs G3
G2 vs G0

S06/R08:
H3 vs H2
H3 vs H1
H3 vs BLEND
FROZEN 별도 참고

======================================================================
7. 이번 repair에 추가하는 zero-training cross-evaluation
======================================================================

목적:
결론 변화가
(a) 평가 원점 분포 때문인지
(b) 새 분산 TRAIN/V로 다시 적응한 영향인지
부분적으로 분해한다.

기존 selected checkpoints가 로컬 cache에서 hash-valid하게 존재하는 경우에만 수행.
없으면 BLOCKED_OLD_CHECKPOINT로 기록하고 본 repair는 계속한다.
기존 모델을 재학습하여 old checkpoint를 복원하지 않는다.

각 track/method/repeat seed에 대해 가능하면:

A. OLD_MODEL_OLD_E
- 기존 published score 참조

B. OLD_MODEL_NEW_E
- 기존 selected weight
- 새 repaired E origins
- 새 E의 동일 method-specific preprocessing
- 추가 optimizer 0

C. NEW_MODEL_NEW_E
- 이번 repair의 주 결과

D. NEW_MODEL_OLD_E
- 이번 새 selected weight
- 기존 old E origins
- 추가 optimizer 0
- optional but 가능한 경우 수행

해석:
OLD_MODEL_NEW_E vs OLD_MODEL_OLD_E:
평가 분포 변경 민감성

NEW_MODEL_NEW_E vs OLD_MODEL_NEW_E:
분산 TRAIN/V 재학습 및 selection 변경 효과

NEW_MODEL_OLD_E vs OLD_MODEL_OLD_E:
새 TRAIN/V가 옛 좁은 E에서도 무엇을 바꿨는지 참고

이 decomposition은 인과효과 완전 분해라고 부르지 않는다.
TRAIN/V/E가 함께 변한 복합 효과임을 명시.

======================================================================
8. 불확실성과 표본 의존성
======================================================================

paired block bootstrap은 유지하지만,
이번에는 원점이 넓게 분산되었음을 먼저 확인한다.

hourly:
7-day index blocks

ETTm1:
actual 7-day = 672 15-min slots

2,000 replicates, fixed seed 유지.

추가 보고:
- number of distinct blocks actually represented
- overlap between horizons
- effective unique target timestamps
- seed별 effect sign
- old vs repaired effect sign

CI는:
- 관측한 원천
- 선택한 4 channels
- 2 repeat seeds
에 조건부다.

새 도메인 일반화나 6-track multiple-selection을 해결한다고 쓰지 않는다.

======================================================================
9. 기존 결론과 repair 결론을 비교하는 판정
======================================================================

각 track에 다음 중 하나를 부여.

ROBUST_NEGATIVE
조건:
- repaired primary proposed-vs-closest-baseline mean gain < 0
- repeat 2 seeds 모두 negative
- paired CI upper < 0
- sampling repair 후에도 방향이 기존 negative와 일치

NEGATIVE_UNCERTAIN
- mean negative
- CI includes 0 또는 seed 방향 불일치

REOPEN_CANDIDATE
조건:
- repaired primary closest-baseline gain > 0
- repeat 2 seeds 모두 positive
- paired CI lower > 0
- 단순 baseline보다 실제 추가 가치
- 기존 result와 달라도 그 차이를 sampling sensitivity로 기록
주의:
이 태그도 NEW_METHOD/PAPER_PASS가 아니다.
정식 선행 비교와 독립 source가 다음 단계.

POSITIVE_UNCERTAIN
- mean positive
- CI includes 0 또는 seed 방향 불일치

SAMPLING_SENSITIVE
- old primary effect와 repaired effect의 sign이 반전
또는
- 효과 크기가 크게 바뀌어 이전 결론의 안정성이 낮음
이 태그는 위 evidence tag와 병행 가능.

SIMPLE_METHOD_SUFFICIENT
- learned proposal보다 fixed/simple control이 더 좋고,
- 그 단순 control의 이득이 반복적으로 유지됨.

NO_SELECTED_ADAPTATION
- 반복 selected model이 INIT/FROZEN으로 돌아가고
- fixed512도 proposal의 필요성을 지지하지 않음.

서로 다른 목적의 R04를 accuracy track과 한 숫자로 합치지 않는다.

======================================================================
10. R05 / N06 / R09의 처리
======================================================================

이 세 track은 재학습 금지.

최종 report에 "UNAFFECTED_REFERENCE" 섹션으로만 포함.

R05:
- 기존 monotone policy result
- reused E / known control / two-model cost 명시

N06:
- 기존 AR1 sum-CRPS 개선
- simple known coupling
- new PEFT evidence 아님

R09:
- 기존 64 distinct E index-days
- NULL penalty negative vs MIXED
- MIXED/FINE_ONLY 정보량 차이 유지
- 이번 origin repair 대상 아님

이 세 결과를 새 repair와 합산해 우승 track을 만들지 않는다.

======================================================================
11. correctness 검사
======================================================================

모든 기존 condition-specific 검사를 유지하고 아래를 추가.

공통:
1) role split 경계.
2) E label poison -> input unchanged.
3) TRAIN stats only.
4) LoRA/extra finite gradients.
5) frozen backbone/head/buffer unchanged.
6) checkpoint restore.
7) native/preprocess parity.
8) metric scalar replay rtol/atol 1e-10.
9) no target-based origin selection.
10) no result-dependent origin replacement.

origin repair 검사:
11) selected days exactly unique.
12) selected day count == origin count.
13) selected day span ratio >= .95.
14) minimum week-block diversity.
15) phase diversity threshold.
16) no same-day duplicate origin.
17) origin list deterministic across two independent calls.
18) change all y/E labels -> selected origins bitwise unchanged.
19) shuffle candidate value columns while keeping missingness metadata -> day selection unchanged.
20) original candidate ordering을 reverse해도 selection 결과가 같도록 sort를 명시.
21) N03 old pathological span을 재현하는 regression test와
    repaired selector가 같은 pathology를 만들지 않는 test.

각 track special checks:
N01:
masked suffix poison invariance, donor causal access.

N02:
query-target exclusion, retrieval continuation already observed,
bank TRAIN-only.

N03:
all observed timestamps < origin,
same physical 12h target grid,
kernel observed overwrite 0.

R04:
early/late exact overlapping future timestamps,
late info -> early prediction leak 0.

N07:
weights TRAIN-only, FFT symmetry, Parseval.

R08:
observed donor timestamp < origin,
predicted donor uses model prediction only,
no circular corrected donor.

======================================================================
12. 자원/재개
======================================================================

one GPU / one worker.

기존 guard 사용.
- startup free >=4GiB
- update boundary free >=1GiB
- disk free >=10GiB
- cache <=100GiB
- controller hard safety <=24h
- waiting <=1800s

main updates cap 49,152
smoke cap 48
total optimizer cap 49,200

모든 update:
scalar journal first.
complete resume state each epoch.
checkpoints only 0/256/512.
pending update receipt.
ambiguous update automatic replay 금지.

완료 예측을 약속하는 시간이 아니라 safety bound다.

======================================================================
13. 반드시 만들어야 할 산출물
======================================================================

root:
MASTER_PROTOCOL.md
REPOSITORY_AUDIT.md
ORIGIN_REPAIR_AUDIT.md
old_vs_new_origin_dispersion.csv
MASTER_MANIFEST.json
MASTER_SEAL.json
QUEUE_STATUS.json
BUDGET_LEDGER.csv
MASTER_REPORT.md
FINAL_DECISION.md

track별:
TOPIC_ONEPAGE.md
PROTOCOL.json
old_origins.csv
new_origins.csv
origin_diversity.json
data_receipt.json
permissions.json
fit_manifest.csv
optimizer_log.csv
LR_selection.json
selections.json
predictions_manifest.json
cross_evaluation_manifest.json
raw_scores.csv
contrasts.csv
old_vs_repaired_contrasts.csv
uncertainty.csv
resources.csv
verification.json
REPORT.md
LIMITATIONS.md
STATUS.json

필수 그림:
1) old vs repaired origin distribution
   x=time/index-day, y=phase
   old/new separate figure or clear panels
2) old vs repaired primary gain
   track별 old/repaired repeat seed + mean
3) repaired condition tradeoff
   기존 track 목적에 맞는 figure

N03:
별도로 old 23.75h E vs repaired multi-day E의 origin scatter 필수.

======================================================================
14. 최종 REPORT에서 반드시 답할 질문
======================================================================

1) 기존 표본 선정에서 정확히 무엇이 잘못됐는가?
2) 새 selector는 날짜 다양성과 phase 다양성을 동시에 확보했는가?
3) 각 track의 old 결론이 repaired sampling에서 유지됐는가?
4) 어떤 track은 ROBUST_NEGATIVE인가?
5) 어떤 track은 SAMPLING_SENSITIVE인가?
6) 어떤 track은 REOPEN_CANDIDATE 또는 POSITIVE_UNCERTAIN인가?
7) 결과 변화가 old-model/new-E에서 이미 나타났는가?
8) retraining까지 해야 달라졌는가?
9) simple baseline은 repaired sampling에서도 충분한가?
10) R05/N06/R09를 왜 다시 돌리지 않았는가?
11) 현재 새 PEFT 구성요소로 채택할 것이 있는가?
12) 있다면 다음 필수 단계가 정식 선행 비교인지, 독립 source인지, 둘 다인지?
13) 없다면 "이번 repair에서도 새 PEFT 구성요소 미확보"라고 정확히 쓴다.

======================================================================
15. 최종 결정 규칙
======================================================================

FINAL_DECISION은 최대 2개만 다음 단계 후보로 남길 수 있다.
0개도 정상.

다음 단계 후보의 최소 조건:
- repaired sampling의 closest baseline 대비 positive mean
- repeat 2 seeds same positive direction
- CI lower >0
- 단순 baseline으로 설명되지 않는 추가 가치
- implementation/correctness complete

이 조건은 논문 acceptance 기준이 아니라
"독립 source와 정식 선행 비교에 추가 투자할 후보" 기준이다.

후보를 남겨도 자동으로 후속 학습하지 않는다.

반대로:
- old negative + repaired robust negative -> 현재 작은 방법 종료
- old/repaired sign flip -> sampling-sensitive, 이전 결론 강도 하향
- simple fixed method best -> simple method 보존, 새 module 자동 생성 금지
- CI includes 0 -> uncertain; equivalence라고 쓰지 않음

======================================================================
16. 이번 작업에서 금지할 것
======================================================================

- 7번째 학습 track 추가
- R05/N06/R09 재학습
- 새 dataset 추가
- seed 73103 등 추가
- LR grid 확장
- 1024 updates 등 학습량 확장
- 방법 v2/v3 생성
- 결과를 보고 phase/date 선정 기준 수정
- E에서 잘 나온 날짜/채널만 선택
- old negative result 삭제/덮어쓰기
- repaired 결과가 좋다는 이유로 독립 test라고 부르기
- 6개 중 하나가 좋다고 9개 탐색의 multiple selection 문제를 무시하기

======================================================================
17. CLI가 실제로 수행할 순서
======================================================================

Phase A — audit
1) repo/latest/dirty/cache audit
2) old result hashes 확인
3) old origin pathology 재계산
4) affected/unaffected track 분류

Phase B — sample repair only
5) 6개 track candidate origins 생성
6) DAY-FIRST selector 실행
7) diversity assertions
8) old/new origin audit 작성
9) 모든 new origin hash seal

Phase C — implementation parity
10) old method definitions와 새 runner가 동일한지 검사
11) smoke 48 updates
12) 특수 정보권한 검사

Phase D — 96 fits
13) 24 method groups × 2 LR on seed73100 = 48 fits
14) V_SELECT로 LR 고정
15) repeat seeds73101/73102 = 48 fits
16) selection seal

Phase E — evaluation
17) CPU controls V에서 고정
18) E prediction manifest 먼저 작성
19) E labels open
20) repaired metrics/scalar verification
21) old checkpoint가 있으면 cross-evaluation

Phase F — interpretation
22) old vs repaired effect table
23) sampling sensitivity classification
24) per-track REPORT
25) MASTER_REPORT
26) FINAL_DECISION
27) scoped commit/push

======================================================================
18. CLI에 그대로 전달할 실행 문장
======================================================================

이 문서 하나를 condition sampling repair의 유일한 실행 계약으로 사용해.

새 후보를 만들지 말고,
직전 condition_studies_v1_20260916에서 날짜 다양성이 부족했던
N01/N02/N03/R04/N07/R08 여섯 트랙만 같은 방법 정의로 다시 평가해.

가장 중요한 수정은 origin selection이다.
기존처럼 phase별로 먼저 뽑지 말고,
각 split에서 합법적인 서로 다른 날짜를 기간 전체에 먼저 균등하게 선택한 뒤
각 날짜 안에서 phase를 분산해.
TRAIN64/E64는 반드시64 distinct days,
V_SELECT32/V_CAL32는 반드시32 distinct days를 요구해.
충족할 수 없으면 숫자를 채우려고 완화하지 말고 BLOCKED_DIVERSITY로 기록해.

R05/N06는 기존 저장 예측 CPU 분석이고,
R09는 기존부터 E64가64 distinct index-days에 분산돼 있으므로 재학습하지 마.

데이터 원천, split, 채널, 방법 수식, loss, LR grid, seed,
512 updates, checkpoint, metric, 핵심 직접 대조는 바꾸지 마.
표본 선정 하나만 과학적 변경으로 허용해.

GPU를 쓰기 전에 old/new origin 분산표를 작성하고
day/span/week/phase/target-overlap 검사를 통과시켜.
특히 N03가 다시 하루 안에 몰리면 실행하지 마.

총 새 본학습은 최대96 fits / 49,152 updates,
smoke48updates로 고정해.
불리한 결과 때문에 추가 seed/LR/방법/dataset을 만들지 마.

가능하면 기존 selected checkpoint를 새 E origins에 zero-training cross-evaluate해서
평가 표본 변화와 retraining 효과를 부분적으로 분리해.
old checkpoint cache가 없으면 이를 재학습하지 말고 BLOCKED_OLD_CHECKPOINT로 기록하고
본 repair는 계속해.

모든 선택을 봉인한 뒤 E를 채점해.
최종 보고서에는 old effect와 repaired effect를 나란히 두고
ROBUST_NEGATIVE / NEGATIVE_UNCERTAIN / POSITIVE_UNCERTAIN /
REOPEN_CANDIDATE / SAMPLING_SENSITIVE / SIMPLE_METHOD_SUFFICIENT를 구분해.

실행 완료와 논문 성공을 섞지 마.
repaired sampling에서도 새 구성요소의 근거가 없으면 그렇게 종료해.
근거가 있으면 최대2개만 다음 투자 후보로 남기되,
정식 선행 비교와 독립 source가 아직 남았음을 명시하고
후속 학습을 자동 시작하지 마.

최종적으로 한국어 MASTER_REPORT.md와 FINAL_DECISION.md를 작성하고
검산 후 scoped commit/push해.
