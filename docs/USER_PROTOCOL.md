[최상위 지시]
새로운 Time-Series Foundation Model(TSFM) PEFT 방법 후보 7개를 빠르지만 공정하게 스크리닝하는 독립 연구 저장소를 처음부터 구축하고, 각 후보에 대해 construct gate → 최소 구현 → 소규모 GPU pilot → GO/STOP 판정을 수행해라.

이번 작업의 목표는 분석을 계속 늘리는 것이 아니다. 각 후보의 실제 proposed method를 구현하고 strongest simple baseline과 직접 비교한다. 기존 `mltimeseries` 저장소는 READ-ONLY reference로만 사용하고, 기존 study/results/runs를 새 결과처럼 복사하지 않는다.

===============================================================================
0. 연구 목표와 금지사항
===============================================================================

최상위 목적:
- 시계열의 구조적 특성 때문에 일반 PEFT가 부족할 수 있는 지점을 7개 후보로 나눈다.
- 후보마다 ① 문제 자체가 실제로 존재하는지, ② proposed PEFT가 실제 future forecasting primary metric을 개선하는지, ③ 단순 feature/augmentation/regularization으로 설명되지 않는지, ④ 직접 겹치는 기존 방법이 없는지, ⑤ 계산비용 대비 다음 단계 가치가 있는지 확인한다.
- 7개를 모두 성공시키려 하지 말고, 상위 1~2개만 다음 단계로 보낸다.

절대 금지:
- E 결과를 본 뒤 LR/λ/grid/split/metric/threshold 수정
- 한 seed만 좋은 결과를 평균으로 숨기기
- proposed arm만 더 많은 실제 과거 정보나 미래정보 사용
- diagnostic metric만 좋아졌는데 방법 성공 판정
- direct-collision 선행이 있는데 이름만 바꿔 계속 실행
- 7개 후보를 처음부터 대규모 benchmark로 확장
- selector/freezing/controller를 이번 screen에 추가
- 기존 `mltimeseries`를 수정하거나 새 결과를 거기에 저장

===============================================================================
1. 새 저장소
===============================================================================

기존 `mltimeseries`의 parent directory 아래 sibling directory로 새 저장소를 만든다.

local:
    tsfm-peft-method-screen

GitHub:
    CanelE452/tsfm-peft-method-screen

가능하면 private로 생성.

시작 전에:
    gh auth status

동일 이름 repo가 이미 존재하면 덮어쓰지 말고 STOP.

새 repo에서:
    git init
    git branch -M main
    gh repo create CanelE452/tsfm-peft-method-screen --private --source . --remote origin

별도 branch 생성 금지.
PR 불필요.
force push 금지.
모든 결과는 main에 직접 commit/push.

===============================================================================
2. 기존 repo에서 참고할 최소 계약
===============================================================================

기존 `mltimeseries`는 runtime dependency가 아니다. 필요한 계약만 읽어서 새 repo에 독립 구현한다.

A. Chronos-2
- amazon/chronos-2
- exact revision 및 local model hash 재확인
- context length 336
- forecast horizon 48
- probabilistic 21 quantiles

B. Standard LoRA baseline
- attention q/k/v/o
- 12 transformer blocks
- 각 block의 2 attention layers
- 총 96 projection modules
- rank=8
- alpha=16
- dropout=0
- native output head frozen
- pretrained base frozen

C. metric
- train-only target scale
- scaled probabilistic 2-pinball (primary)
- median MAE
- qmean MSE
- interval80 coverage
- interval80 width

D. reproducibility
- deterministic seed
- checkpoint replay
- saved predictions
- source/input/model hashes
- independent metric replay

금지:
    from mltimeseries ...
    sys.path.append(...mltimeseries...)

===============================================================================
3. 새 repo 구조
===============================================================================

tsfm-peft-method-screen/
├─ README.md
├─ .gitignore
├─ pyproject.toml
├─ requirements-lock.txt
├─ configs/
│  ├─ chronos2.yaml
│  ├─ common_screen.yaml
│  └─ candidate_01.yaml ~ candidate_07.yaml
├─ src/tsfm_peft_screen/
│  ├─ backbone.py
│  ├─ lora.py
│  ├─ data.py
│  ├─ metrics.py
│  ├─ selection.py
│  ├─ reproducibility.py
│  ├─ candidates/
│  │  ├─ freshness.py
│  │  ├─ dualclock.py
│  │  ├─ patchphase.py
│  │  ├─ fr_lora.py
│  │  ├─ maturity.py
│  │  ├─ jointpath.py
│  │  └─ censor.py
│  └─ runners/
│     ├─ common_fit.py
│     ├─ common_eval.py
│     └─ streaming_eval.py
├─ scripts/
│  ├─ audit_environment.py
│  ├─ prepare_common_data.py
│  ├─ run_candidate_01.py ~ run_candidate_07.py
│  ├─ screen_all.py
│  └─ verify_all.py
├─ tests/
├─ docs/
│  ├─ MASTER_SCREEN_PROTOCOL.md
│  ├─ NOVELTY_BOUNDARY.md
│  └─ CANDIDATE_01.md ~ CANDIDATE_07.md
└─ results/
   ├─ candidate_01/ ... candidate_07/
   └─ screening_summary/

raw data, HF model files, large checkpoints, prediction caches는 git에 넣지 않는다.

===============================================================================
4. 공통 실험 계약
===============================================================================

기본 split:
    Train -> V -> E

Train: optimization only
V: LR/λ/checkpoint selection only
E: V 선택 봉인 후 한 번만 open

Round 1은 development screen이며 final paper evidence라고 부르지 않는다.

공통 model:
    Chronos-2

context:
    336

horizon:
    48

primary:
    scaled probabilistic 2-pinball (lower better)

secondary:
    median MAE, qmean MSE, interval80 coverage, interval80 width

Round1 seed:
    30000

Round2 seed:
    30000, 30001

공통 LR 후보:
    3e-5
    1e-4

checkpoint:
    0,4,8,15,30,60,120,180,240,360

max steps:
    360

optimizer 기본:
    AdamW, weight_decay=0, gradient_clip=1.0

모든 adaptation arm에서 동일:
- train examples
- effective batch
- optimizer update 수
- checkpoint opportunity
- evaluation origins
- metric
- random seed 및 sampling stream

===============================================================================
5. 공통 무결성 게이트
===============================================================================

GPU 본학습 전에 반드시 PASS.

A. F0 identity
LoRA 초기화 직후 prediction(step0 LoRA)와 F0가 일치. max abs error 기록, 가능하면 <=1e-6.

B. Trainable audit
Standard LoRA에서 LoRA tensor만 trainable. native output head/base pretrained weights frozen.

C. Metric replay
worker score와 저장 prediction의 독립 재계산 차이 <=1e-10.

D. No future leakage
각 forecast origin context에 origin 이후 timestamp가 포함되지 않음을 assert.

E. Checkpoint replay
선택 checkpoint reload 후 prediction 재현.

F. Selection seal
V winner 선택 후 selection.json + hash 저장. 그 후 E open.

실패 시 해당 candidate STOP.

===============================================================================
6. 실행 단계
===============================================================================

ROUND 0
- 문제/construct/data gate
- CPU 또는 F0 inference 우선
- 문제 자체가 약하면 GPU fit 0으로 종료

ROUND 1
- 후보당 1 dataset × 1 seed
- 약 4~8 fits/streams
- actual proposed PEFT vs strongest simple baseline

ROUND 2
- Round1 상위 1~2개만
- second dataset × 2 seeds
- Round1 결과를 본 뒤 LR/λ retune 금지

이번 CLI에서는 Round0 + Round1까지만 자동 실행하고 STOP.
Round2는 사용자 검토 후 별도 승인.

===============================================================================
7. Candidate 01 — Observation Freshness / Availability Conditioned PEFT
===============================================================================

임시명: Freshness-Gated LoRA

질문:
관측 block-missing/update-rate shift가 있는 다변량 시계열에서 observation state가 LoRA correction 자체를 조절하면, 동일 정보를 단순 feature로 주는 것보다 unseen observation process에서 robust한가?

[Round0]
Jena clean evaluation origins 고정.
context에만 다음 corruption 생성:
- 6h contiguous block missing
- 12h block missing
- channel refresh every 2h
- channel refresh every 4h
- stale forward-fill

target untouched.
F0 degradation = 100*(L_corrupt-L_clean)/L_clean.

problem gate:
- 평균 degradation >=2%
- 최소 2개 corruption에서 >=1%
둘 다 아니면 NO_PROBLEM, GPU fit 0.

[Proposed]
각 channel/time에
    r_ct = [observed_mask, normalized_time_since_last_observation, recent_availability_rate]

low-rank correction:
    delta_h = B( g_psi(r) ⊙ A h )

trainable: A,B,small gate only.

[Baselines]
1. STANDARD_LORA
2. FEATURE_LORA: 같은 r을 단순 feature/adapter input으로 제공하되 LoRA gating 없음
3. FRESHNESS_GATED_LORA

모두 동일 corrupted train examples 사용.

[Round1]
Dataset: Jena
Seed: 30000
3 methods × 2 LR = 6 fits

Train corruption:
- 6h block
- 2h refresh

E unseen:
- 12h block
- 24h block
- 4h refresh
- 8h refresh
+ clean E

PASS:
- Proposed > FEATURE_LORA by >=1.0% (F0-normalized effect)
- Proposed > STANDARD_LORA
- clean degradation vs best baseline <=0.5%
- 특정 corruption에서 catastrophic worsening >3% 없음

===============================================================================
8. Candidate 02 — Dual Calendar-Time / Event-Time PEFT
===============================================================================

임시명: DualClock-PEFT

질문:
zero-heavy intermittent demand에서 regular calendar representation만 적응하는 것보다 같은 context 내부 event-time representation을 작은 adapter로 결합하면 추가 가치가 있는가?

[Data]
Round1: M5.
train-only 기준으로 intermittent series 256개 deterministic selection.
조건 예:
- positive occurrence 최소 개수
- zero fraction threshold
- item id/hash ordering
정확한 item list를 E 전 manifest로 봉인.

context 밖 이벤트 추가 사용 금지.

[Event representation]
동일 336-step context에서:
- occurrence positions
- inter-event gaps
- positive magnitudes
- time since last event

small event encoder hidden<=64.

fusion 예:
    h' = h + B sigma(A [h;e])

[Baselines]
1. STANDARD_LORA
2. EVENT_SUMMARY_LORA: last gap, mean gap, last magnitude, positive rate 등 summary만 제공
3. DUALCLOCK_ADAPTER: full event sequence encoder + low-rank fusion

모두 같은 output head와 target loss.
Hurdle head 사용 금지.

[Round1]
M5 subset, seed30000
3 methods × 2 LR = 6 fits

Primary:
scaled probabilistic loss 또는 구현이 명확한 WQL 하나로 고정.

Secondary:
RMSSE, occurrence Brier, positive-demand MAE.

PASS:
- DualClock > Standard >=1%
- DualClock > EventSummary >=1%
- improvement가 극소수 series에만 의존하지 않음
- zero-heavy subgroup에서도 방향 유지

===============================================================================
9. Candidate 03 — Patch Boundary / Patch Phase Robust PEFT
===============================================================================

임시명: PatchPhase-PEFT

질문:
동일한 실제 관측 정보가 다른 patch boundary에 배치될 때 pretrained prediction이 민감하다면 작은 phase-conditioned adapter가 그 민감도를 줄이는가?

[Round0 construct test]
`patch_phase(x, phase)` 구현 후 반드시:
    unpatch(patch_phase(x,phase)) == x

또한 동일해야 함:
- raw values
- timestamps
- masks
- forecast origin
- target horizon
- context information

하나라도 다르면 INVALID_CONSTRUCT, STOP.

[F0 sensitivity]
phase={0,4,8,12}, 동일 128 origins.
측정:
- phase별 loss range
- target-scale-normalized prediction discrepancy

problem gate:
- loss range >=0.5% 또는
- normalized discrepancy >=0.25%
둘 다 아니면 NO_PROBLEM.

[Methods]
1. STANDARD_LORA
2. PHASE_AUGMENTED_ADAPTER: phase variants를 보여주되 ordinary adapter
3. PHASE_CONDITIONED_ADAPTER: phase가 low-rank correction을 조절

모든 방법 동일 phase augmentation.

[Round1]
ETTm2, seed30000
train phase={0,8}
E unseen={4,12}
3 methods ×2 LR = 6 fits

PASS:
- Proposed unseen-phase loss >=0.5% improvement vs PHASE_AUGMENTED
- phase forecast variance >=30% 감소
- canonical phase degradation <=0.5%

===============================================================================
10. Candidate 04 — Forecast Revision Regularized LoRA
===============================================================================

임시명: FR-LoRA

질문:
raw forecast 자체를 안정화하는 것보다 F0 대비 adaptation correction의 revision을 안정화하면 future forecasting accuracy가 더 좋아지는가?

[Paired origins]
origin o, o+24h
horizon48h
shared future=[o+24,o+48)
동일 absolute timestamp 24h overlap만 revision term에 사용.

correction:
    d_phi(o,u)=f_phi(o,u)-f0(o,u)
train-scale normalized.

Proposed:
    L_FR = mean SmoothL1(d_phi(o,u)-d_phi(o+24,u))

Task:
    Ltask=0.5*(Lforecast(o)+Lforecast(o+24))

Total:
    L=Ltask+lambda*L_FR

[Baselines]
1. STANDARD_LORA: Ltask only
2. RAW_STABILITY_LORA: f_phi(o,u)-f_phi(o+24,u) regularization
3. F0_ANCHOR_LORA: f_phi-f0 magnitude regularization
4. FR_LORA

모든 arm 동일 paired batches.

[Unit tests]
- lambda=0 FR == Standard: loss/gradient/first update 동일
- constant nonzero correction이면 L_FR=0, L_anchor>0
- absolute timestamp alignment exact

[Round1]
Jena, seed30000
4 methods ×2 LR = 8 fits
lambda는 task/reg loss scale 확인 후 E 전에 단일 값으로 고정.

PASS:
- FR > Standard >=1%
- FR > RawStability >=1%
- FR > Anchor
- primary forecast loss도 개선
- revision diagnostic만 좋아지면 FAIL

===============================================================================
11. Candidate 05 — Delayed-Label / Maturity-Aware PEFT
===============================================================================

임시명: Maturity-PEFT / Protected-Horizon LoRA

질문:
multi-horizon forecast에서 일부 horizon 정답만 먼저 도착했을 때, 도착한 정답으로 적응하면서 아직 정답 없는 horizon prediction을 보호하면 immediate adaptation보다 online forecasting이 좋아지는가?

[Streaming contract]
context336, horizon48, origin stride24h.
각 origin:
1. prediction issue + save
2. 당시 available label만 update
3. 다음 origin

평가에는 issue 당시 prediction만 사용.
사후 수정 forecast 사용 금지.

[Methods]
1. F0
2. IMMEDIATE_LORA
3. WAIT_FULL
4. TAFAS-like baseline (가능한 범위에서 최소 재현)
5. MATURITY_PEFT

[Proposed]
update 직전 model f_pre.

matured horizon:
    supervised loss

unmatured horizon:
    lambda * D(f_phi(unmatured), stopgrad(f_pre(unmatured)))

label 도착 시 preservation 대상에서 supervised 대상로 이동.

[Round1]
Jena
30 streaming origins
seed30000
5 methods = 5 streaming runs
LR/λ one recipe씩만 사전 고정.

Primary:
prequential scaled 2-pinball of issued forecasts.

Secondary:
- unrevealed-horizon drift
- worst 5-origin mean loss
- adaptation wall-clock

PASS:
- Proposed > best online baseline >=1%
- no leakage
- compute overhead <=20%
- 개선이 마지막 몇 origin 하나에만 의존하지 않음

===============================================================================
12. Candidate 06 — Joint Future Path PEFT
===============================================================================

임시명: Conditional Path Adapter

질문:
pretrained TSFM marginal forecast는 그대로 유지하면서 hidden-state-conditioned small low-rank adapter로 horizon/channel joint dependence를 학습하면 correlated future paths가 더 정확해지는가?

[Setup]
Round1: Electricity.
metadata ordering으로 first 4 valid channels를 E 전에 고정.
Chronos marginal quantiles는 모든 방법에서 완전히 동일하게 frozen.

[Methods]
1. INDEPENDENT
2. STATIC_GAUSSIAN_COPULA: train residual rank correlation
3. UNCONDITIONAL_LOW_RANK_COPULA
4. CONDITIONAL_PATH_ADAPTER

예:
    Sigma_t=U_t U_t^T + diag(d)
conditional은 h_t -> small U_t.

[Round1]
learned methods unconditional/conditional ×2 init seeds = 4 fits.
Independent/static은 neural fit 없음 또는 최소 fitting.

Primary:
- Energy Score
- Variogram Score

Secondary:
- aggregate total demand interval coverage
- simultaneous exceedance Brier
- aggregate-sum CRPS 가능하면

Invariant:
모든 방법의 marginal quantile prediction exact identical.

PASS:
- conditional > static 및 unconditional >=2% on at least one primary path score
- 다른 primary path score worsening <=0.5%
- aggregate/event calibration 개선 또는 neutral
- marginal prediction unchanged

===============================================================================
13. Candidate 07 — Censor / Stockout Aware PEFT
===============================================================================

임시명: Censor-Preserve LoRA

질문:
stockout 때문에 observed sales가 latent demand lower bound만 제공할 때 censor-aware loss + pretrained forecast preservation을 이용한 PEFT가 naive/censor-only LoRA보다 latent demand recovery를 개선하는가?

[평가 원칙]
Round1은 controlled re-censoring.
완전히 관측된 true demand y에서:
    s=min(y,c)
synthetic censoring 생성.
진짜 y로 평가 가능.

c는 train distribution만으로 고정.
E를 본 뒤 변경 금지.

[Methods]
1. NAIVE_LORA: observed s를 true y로 취급
2. DROP_CENSORED_LORA
3. CENSORED_LOSS_LORA
4. CENSOR_PRESERVE_LORA

[Proposed]
uncensored:
    standard probabilistic loss

censored sale s:
    y>=s
21-quantile 기반 monotone interpolated CDF F_hat.

censor term:
    -log(1-F_hat_phi(s))

preserve term:
    uncensored/train-safe positions에서 D(f_phi,f0)

Total:
    L_uncensored + lambda_c*L_censor + lambda_p*L_preserve

quantile monotonicity/interpolation unit tests 필수.

[Round1]
M5 controlled re-censoring
seed30000
4 methods ×2 LR = 8 fits
lambda_c/lambda_p는 train-loss scale 기준으로 single pair 사전 고정.

Primary:
- true latent y scaled pinball
- censored-position pinball

Secondary:
- RMSSE
- censored underprediction bias
- uncensored-position loss

PASS:
- Proposed > CensoredLoss >=1%
- Proposed > Naive
- censored underprediction bias 개선
- uncensored degradation <=0.5%

===============================================================================
14. Round1 계산 상한
===============================================================================

모든 Round0가 통과한 최악의 경우:

Candidate01: 6 fits
Candidate02: 6 fits
Candidate03: 6 fits
Candidate04: 8 fits
Candidate05: 5 streaming runs
Candidate06: 4 fits
Candidate07: 8 fits

총 38 standard fits + 5 streaming runs.

이 이상 자동 실행 금지.
Round0 fail 후보는 GPU fit 0 또는 최소 smoke만 수행.

===============================================================================
15. 짧은 novelty collision audit
===============================================================================

각 candidate 구현 전 최대 20~30분 direct collision check만 수행.
장기 systematic review 금지.

Candidate01:
- missing modality PEFT
- irregular observation PEFT
- time-since-last-observation conditioned LoRA
- asynchronous covariate TSFM

Candidate02:
- intermittent demand event-time adapter
- renewal TSFM PEFT
- event sequence + pretrained TSFM

Candidate03:
- patch shift equivariant TSFM
- patch phase adapter
- patch boundary PEFT

Candidate04:
- forecast stability regularization
- forecast revision consistency
- pretrained correction consistency

Candidate05:
- partial-label TTA forecasting
- delayed-label online PEFT
- horizon maturity adaptation

Candidate06:
- TSFM copula
- correlated sample paths TSFM
- conditional low-rank copula adapter

Candidate07:
- censored demand LoRA
- stockout PEFT
- censor-aware TSFM adaptation

`docs/NOVELTY_BOUNDARY.md`에 각 candidate마다:
- closest paper
- overlapping mechanism
- non-overlap
- direct collision YES/NO
- confidence HIGH/MEDIUM/LOW

Direct collision이면 GPU pilot 금지.

===============================================================================
16. Candidate별 결과 형식
===============================================================================

각 `results/candidate_XX/`에:
- status.json
- contract.json
- integrity.json
- metrics.csv
- selections.csv
- trajectories.csv
- resource_usage.json
- RESULT.md
- figures/primary.png
- figures/diagnostics.png

RESULT.md 필수:
[문제]
[방법]
[강한 단순 baseline]
[데이터]
[학습 파라미터]
[무결성]
[raw 결과]
[relative 결과]
[성공/실패 판정]
[말할 수 없는 것]
[Round2 추천 여부]

사실/해석 태그:
[확인]
[추정]
[판정]
[미검증]

===============================================================================
17. 전체 랭킹
===============================================================================

`results/screening_summary/` 생성:
- all_candidates.csv
- ranking.md
- compute_summary.json
- integrity_summary.json
- final_verdict.json

표 컬럼:
- candidate
- problem_gate
- strongest_baseline
- proposed_primary
- proposed_gain
- method_specificity
- robustness
- compute_cost
- novelty_risk
- major_failure_mode
- verdict

판정 값은 정확히:
PASS
WEAK
FAIL
NO_PROBLEM
INVALID_CONSTRUCT
NOVELTY_COLLISION
IMPLEMENTATION_BLOCKED

Round2 추천은 최대 2개.

선정 기준:
A. Forecast benefit
B. Method specificity
C. Robustness
D. Novelty/cost

단순 점수 합으로 자동 winner를 만들지 말 것.

===============================================================================
18. Git / push
===============================================================================

모든 작업 main.
추천 commit:
1. Initialize seven-candidate TSFM PEFT screen
2. Add common Chronos-2 and LoRA reproduction baseline
3. Add candidate construct gates and unit tests
4. Implement candidate PEFT methods
5. Add Round 1 screening results

각 commit 전:
    git status --short
    git diff --check
    pytest -q

통과 후:
    git add ...
    git commit -m "..."
    git push origin main

대형 model/checkpoint/raw prediction add 금지.
force push 금지.

===============================================================================
19. 실행 안전성
===============================================================================

single GPU serial.
각 GPU job:
- timeout
- GPU memory guard
- RAM/commit guard
- NaN/Inf abort
- nonzero exit artifact 보존

한 후보 실패 때문에 다른 후보 결과 삭제 금지.

===============================================================================
20. 최종 보고 후 STOP
===============================================================================

Round0 + Round1 종료 후 Round2 자동 실행 금지.

최종 CLI 출력:

[REPOSITORY]
- local path
- GitHub URL
- branch=main
- final commit SHA
- git status --short

[COMMON INTEGRITY]
- Chronos-2 checkpoint/revision/hash
- F0 reproduction
- Standard LoRA step0 identity
- trainable count
- metric replay

[CANDIDATE 01~07]
각각:
- problem gate
- direct novelty collision 여부
- fit/stream count
- strongest baseline
- proposed raw primary metric
- proposed gain
- method-specific diagnostic
- compute cost
- verdict

[RANKING]
1~7위

[ROUND2 RECOMMENDATION]
- 최대 2개
- 추천 이유
- 다음 dataset/seed
- Round2 미실행 명시

작업 종료.

가장 중요한 원칙:

    문제 존재?
        ↓
      YES
        ↓
    실제 proposed PEFT 구현
        ↓
    strongest simple baseline과 직접 비교
        ↓
    안 좋음 -> 버림
    좋음 -> 다음 dataset/seed로 확장

이번 프로젝트의 목표는 7개를 모두 살리는 것이 아니라,
"논문 방법으로 확장할 이유가 있는 후보를 적은 비용으로 한두 개 찾는 것"이다.