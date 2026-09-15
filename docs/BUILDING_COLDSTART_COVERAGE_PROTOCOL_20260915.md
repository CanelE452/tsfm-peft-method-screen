건물 cold-start / 짧은 이력 PEFT 파일럿 — CLI 실행 지시문
작성일: 2026-09-15 KST
대상 저장소: CanelE452/tsfm-peft-method-screen
확인 기준 commit: 9c9cdc29d4a1e598dbbcda6ae6bda588158bc046
제안 run ID: building_coldstart_coverage_peft_v1_20260915

======================================================================
0. 이번 작업의 위치와 절대 범위
======================================================================

[확인] 현재 저장소의 최신 완료 상태는 Query R1이 BF16 수치 검사에서 종료됐고,
채널 공유 R2는 24/24 fits를 완료했으나 LH(LoRA+HEAD)가 최강 단순 대조군이었다.
이번 작업은 Query·채널 공유·Censor·FR의 추가 튜닝이 아니다.

[확인] 최신 선행에는 다음이 이미 존재한다.
- Time-PEFT (ICML 2026): temporal/multichannel complexity를 기준으로 frequency/channel adapter를 설계.
- Probabilistic Forecasting for Building Energy Systems using TSFMs (Energy and Buildings, 2025):
  건물 에너지에서 Chronos full FT/LoRA의 제한 데이터 적응을 분석.
- BuildingsBench (NeurIPS Datasets & Benchmarks 2023):
  실제 건물의 zero-shot/transfer learning 평가 환경 제공.
- A cross-building few-shot energy load forecasting framework via transferable temporal learning and residual correction
  (Energy and Buildings, 2026):
  unseen building의 few-shot cross-building transfer + hour/season residual correction을 이미 다룸.
- In-Context Fine-Tuning for Time-Series Foundation Models (ICML 2025):
  관련 시계열 예시를 이용하는 few-shot 적응을 이미 다룸.

따라서 다음을 새 기여라고 전제하지 않는다.
- "새 건물 few-shot forecasting" 자체
- "cross-building transfer" 자체
- "시간대/계절 residual correction" 자체
- "관련 시계열을 같이 넣는다" 자체
- "LoRA를 건물 데이터에 쓴다" 자체

[설계] 이번 1차 파일럿은 cross-building source transfer를 의도적으로 제외한다.
이유는 두 가지다.
1) 위의 2026 CFTR 선행과 문제·메커니즘을 불필요하게 겹치지 않기 위해서.
2) "짧은 타깃 이력에서 관측하지 못한 운영 상태까지 표준 LoRA가 같은 방식으로 바꾸는가"라는
   더 작은 질문을 먼저 분리하기 위해서.

[설계] 이번 작업의 목표는 완성 논문 방법을 자동으로 발명하는 것이 아니다.
먼저 문제 현상이 존재하는지 확인하고, 존재할 때만 매우 작은 coverage-aware PEFT 규칙을 시험한다.

이번 작업에서 자동으로 해도 되는 최대 범위:
- 공식/공개 BuildingsBench real-building evaluation data 준비
- repo/환경/데이터 audit
- LoRA recipe 선택용 최대 8 fits
- 문제 현상 screen용 최대 16 LoRA fits
- coverage-aware scaling은 저장된 F0/LoRA 예측으로 계산하므로 추가 GPU fit 0
- 문제+방법 신호가 모두 통과할 때 held-out building 최대 24 LoRA fits
- 총 실제 LoRA fits 상한 48
- smoke/수치 검사 updates는 별도 장부. forecasting fit으로 세지 않는다.
- held-out 평가가 끝난 뒤 새 adapter, source transfer, 추가 dataset, 추가 seed, 새 LR를 자동으로 만들지 않는다.

기존 results/research 파일을 덮어쓰지 않는다.
현재 repo에 같은 run이 이미 존재하면 중복 실행하지 말고 상태를 읽는다.

======================================================================
1. Phase 0 — 목적 트리
======================================================================

[설계] 최상위 목표:
"새 건물/새 계량기의 타깃 이력이 매우 짧을 때,
미래 운영 상태가 타깃 이력에서 충분히 관측되지 않았다는 사실을 이용해
표준 LoRA의 불필요한 target-specific correction을 줄일 수 있는지 검증한다."

소비처:
사용자와 지도교수가 다음을 결정한다.
- 이 문제를 PEFT 방법론 논문 후보로 계속할지
- 단순 LoRA/출력 보정으로 충분하므로 닫을지

1수 — 목표 고정
- 예측 문제: hourly building/load series의 next-24-hour forecasting.
- target history만 짧게 제한한다.
- 숨긴 과거/미래는 target normalization, target adaptation, target model selection에 사용하지 않는다.
- 이번 파일럿은 개발 증거다. 논문 최종 확증이 아니다.

2수 — 왜 가지치기
행동 A: 짧은 이력에서 coverage 현상을 먼저 확인.
  목적 후보1: 표준 LoRA의 이득이 단순히 history length가 아니라 관측한 운영 상태에 따라 달라지는지 확인.
  목적 후보2: coverage-aware 방법을 만들 근거가 실제로 있는지 확인.
  이것이 없으면 새 방법 전체를 중단한다.

행동 B: 새 학습 구조 전에 prediction-level shrinkage를 시험.
  목적 후보1: coverage 정보만으로 LoRA correction을 줄이는 단순 규칙이 충분한지 확인.
  목적 후보2: trainable adapter를 추가하기 전에 가장 싼 대조군을 세운다.
  이것이 실패하면 새 coverage adapter를 자동으로 만들지 않는다.

행동 C: held-out building에서 고정 규칙 확인.
  목적 후보1: dev building에 맞춘 규칙인지 확인.
  목적 후보2: 주장 범위를 "짧은 이력 cold-start"로 제한할 근거 확보.

3수 — 제대로 달성하려면
- building ID 단위 split. 같은 건물의 다른 시점을 독립 building으로 세지 않는다.
- target cold-start history 외의 target 과거를 normalization/feature/selection에 사용하지 않는다.
- 미래의 day-of-week는 예측 시점에 알려진 calendar 정보이므로 사용 가능하되,
  실제 미래 load/temperature/occupancy를 새 정보로 넣지 않는다.
- 모든 baseline과 후보에 같은 target history와 같은 forecast origin을 준다.
- LoRA recipe는 별도 recipe-building에서 결정한 뒤 method-dev/held-out에 고정한다.
- coverage scaling의 hyperparameter도 method-dev에서 고정한 뒤 held-out에서 바꾸지 않는다.
- 결과를 본 뒤 history length, forecast weekday, 건물 selection을 변경하지 않는다.
- CFTR처럼 source-building transfer/residual correction을 이번 방법의 이득과 섞지 않는다.

4수 — 역주행 검증

예상 결과 A:
short-history에서 미래 day-type이 target adaptation data에 없을 때 LoRA gain이 더 약함.
목적 지지: 지지
최상위 도달: 부분
독자 첫 질문: "그게 그냥 Saturday가 어려워서 그런 것 아닌가?"
-> 같은 building의 longer-history에서 day-type coverage가 생겼을 때 gap이 줄어드는 interaction을 같이 본다.

예상 결과 B:
coverage-aware scaling이 uniform shrink/fallback/affine보다 좋음.
목적 지지: 지지
최상위 도달: 닿음
독자 첫 질문: "그냥 F0로 돌아가면 되는 것 아닌가?"
-> binary fallback과 반드시 비교한다.

예상 결과 C:
coverage 현상은 있지만 simple fallback/uniform shrink가 후보만큼 좋음.
목적 지지: 문제는 존재하지만 새 방법 지지 불가
최상위 도달: 새 PEFT 방법으로는 안 닿음
행동: 새 adapter를 만들지 않고 종료.

예상 결과 D:
coverage 현상 자체가 없음.
목적 지지: 불가
최상위 도달: 안 닿음
행동: 이 cold-start coverage 주제를 종료. 숫자를 맞추기 위해 조건을 바꾸지 않는다.

======================================================================
2. Phase 2 — 무결성 태그와 선행 경계
======================================================================

실행기 결과 문서의 모든 사실 주장에는 [확인] 또는 [추정]을 붙인다.
- 파일·수치·실행 로그·논문 원문으로 확인한 것 = [확인]
- 결과 원인, 일반화, 메커니즘 해석 = [추정]
- 새 protocol의 선택 = [설계]

[확인해야 할 선행 최소 목록]
실행 전에 최신 웹/논문 원문을 다시 확인하고 literature_boundary.md에 적는다.
최소:
1. Time-PEFT: Temporal and Multichannel Complexity-Based Fine-Tuning for Time-Series Foundation Models
   ICML 2026.
2. Probabilistic Forecasting for Building Energy Systems using Time-Series Foundation Models
   Energy and Buildings, 2025, 116446.
3. BuildingsBench: A Large-Scale Dataset of 900K Buildings and Benchmark for Short-Term Load Forecasting
   NeurIPS Datasets & Benchmarks 2023.
4. In-Context Fine-Tuning for Time-Series Foundation Models
   ICML 2025.
5. A cross-building few-shot energy load forecasting framework via transferable temporal learning and residual correction
   Energy and Buildings 2026, 117530.
6. 필요하면 building cold-start / few-shot / personalized TSFM adapter의 2025-2026 주요 학회·저널을 추가.

[설계] novelty audit 질문:
- forecast-state coverage count를 이용해 target-specific PEFT correction magnitude를 조절하는 직접 선행이 있는가?
- building few-shot에서 weekday/weekend coverage를 adapter strength에 직접 연결한 방법이 있는가?
- 동일한 q = q0 + alpha(c)*(q_lora-q0) 형태가 이미 핵심 방법으로 제안됐는가?

직접 충돌이 있으면 GPU를 무조건 막지는 말고,
"NOVELTY_COLLISION" 또는 "KNOWN_RULE"로 기록한 뒤 사용자에게 과학적 차이를 설명한다.
그러나 exact-equivalent한 방법이면 새 방법 발견으로 주장하지 않는다.

[확인] cross-building few-shot + residual correction은 이미 2026 Energy and Buildings에 있으므로
이번 파일럿의 main claim으로 쓰지 않는다.

======================================================================
3. 저장소 audit와 새 경로
======================================================================

[확인 기준] 시작 시 기준 commit:
9c9cdc29d4a1e598dbbcda6ae6bda588158bc046

먼저:
- current branch/commit
- dirty files
- 기준 commit 이후 신규 commit
- 실행 중 동일 프로젝트 worker
- 기존 GPU lock
- 최근 RESULTS_INDEX
를 읽는다.

기준 commit 이후 사용자 작업이 있으면 먼저 diff를 검토한다.
reset/stash/clean 금지.
원시 데이터·cache·checkpoint 삭제 금지.

기존 Query/채널/Censor/FR 결과를 수정하거나 재판정하지 않는다.

새 경로 제안:
docs/BUILDING_COLDSTART_COVERAGE_PROTOCOL_20260915.md
configs/building_coldstart_coverage_v1_20260915.json
experiments/building_coldstart_coverage_v1_20260915/
scripts/run_building_coldstart_coverage_v1.py
scripts/finalize_building_coldstart_coverage_v1.py
results/building_coldstart_coverage_v1_20260915/
.cache/building_coldstart_coverage_v1_20260915/

위 경로는 새로 만들 이름이다. 기존 파일이라고 가정하지 않는다.

======================================================================
4. 데이터 — BuildingsBench real-building subset
======================================================================

[확인] BuildingsBench는 실제 building energy benchmark를 제공하고
zero-shot / transfer-learning task와 building-level loader를 제공한다.
BDG-2는 real building smart-meter data가 포함된 evaluation source다.

[설계] 우선 데이터 선택:
- BuildingsBench 공식 evaluation archive의 실제 건물 데이터.
- 1차 후보는 BDG-2의 hourly load/electricity 계열.
- 정확한 dataset/site/building schema는 공식 loader를 실제로 읽고 확정한다.
- 아직 보지 않은 내부 column/key 이름을 만들어 쓰지 않는다.

로컬에 official BuildingsBench evaluation data가 있으면 재사용하고 hash를 기록한다.
없으면:
- 공식 BuildingsBench source/documentation에서 evaluation archive 위치를 확인한다.
- 공개 evaluation data만 받는다. 900K pretraining 110GB 전체를 받지 않는다.
- 다운로드 URL, version/commit, 파일 hash, 크기를 manifest에 남긴다.
- 네트워크/권한으로 불가능하면 BLOCKED_DATA.
- 임의로 기존 Electricity/Traffic으로 바꾸지 않는다.

[설계] target series eligibility:
공식 schema를 확인한 뒤 아래 기계적 조건만 사용한다.
- real building/load series
- hourly timestamp가 검증 가능
- 적어도 120 consecutive days의 finite target load 구간 확보
- 해당 구간의 train-only std > 1e-6
- 성능·future target 통계로 building을 고르지 않는다.

결측:
- 각 cold-start episode의 exposed history와 forecast target 24h가 모두 finite한 episode를 우선 사용.
- 충분한 building이 있으면 imputation 없이 finite episode만 사용.
- 충분하지 않으면 protocol을 바꾸지 말고 BLOCKED_DATA 또는 INSUFFICIENT_ELIGIBLE_BUILDINGS.
- future target을 impute해서 정답을 만들지 않는다.

[설계] 필요한 eligible building 수:
최소 14개.
- recipe buildings: 4
- method-development buildings: 4
- held-out buildings: 6

building ID를 UTF-8 문자열로 canonicalize한 뒤 sha256 정렬.
성능을 보기 전에 4/4/6으로 고정한다.
같은 물리적 building의 복수 meter가 있으면 동일 building group으로 묶어 split한다.
그룹 규칙은 official metadata/schema를 보고 확정하고 기록한다.

14개 미만이면 building 수를 임의로 줄여 PASS를 만들지 않는다.
INSUFFICIENT_ELIGIBLE_BUILDINGS로 종료하고 후보 dataset 대안을 제안만 한다.

======================================================================
5. Cold-start episode 정의 — 실제 가용 정보 제한
======================================================================

[설계] 예측:
- target: next 24 hourly load values
- forecast origin: dataset calendar에서 00:00로 확인되는 시점
- inference context: 바로 직전 24h
- target adaptation history H ∈ {3 days, 14 days}
- H보다 이전의 target load는 해당 episode의 model input, normalization, adaptation, selection에 사용 금지.

[설계] 운영 상태의 1차 정의:
weekday/weekend.
- official timestamp/day-of-week 정보만 사용.
- weekday={Mon,Tue,Wed,Thu,Fri}
- weekend={Sat,Sun}
- holiday/occupancy를 추측해 추가하지 않는다.
- timezone가 문서화되지 않았으면 제공된 timestamp calendar를 그대로 사용하고 한계를 기록한다.

각 building에서 deterministic하게 두 종류의 forecast episode를 만든다.
1) COVERED candidate:
   forecast day = Wednesday
2) UNCOVERED candidate:
   forecast day = Saturday

가능하면 같은 calendar week 안의 Wednesday/Saturday pair를 사용한다.
두 forecast origin 모두:
- 앞 14d history finite
- 다음24h target finite
- 같은 season/month 범위
를 만족해야 한다.

선택 규칙:
- eligible 120d block 안에서 위 조건을 만족하는 첫 deterministic pair를 사용한다.
- 성능을 보고 다른 날짜로 바꾸지 않는다.
- exact dates를 manifest에 기록한다.

왜 Wednesday/Saturday:
[설계] H=3d에서 daily adaptation target window를 만들면
- Wednesday forecast 전 3d에서 weekday target 예시는 존재할 수 있음.
- Saturday forecast 전 3d에는 weekend target 예시가 없도록 구성 가능.
H=14d에서는 두 day-type이 모두 관측된다.
이 interaction으로 단순 Saturday difficulty와 "관측 범위"를 일부 분리한다.

실제 episode 생성 뒤 반드시 assertion:
- H=3d Wednesday: coverage_count(weekday) > 0
- H=3d Saturday: coverage_count(weekend) == 0
- H=14d Wednesday/Saturday: 각 forecast day-type coverage_count > 0
하나라도 성립하지 않으면 해당 building을 성능과 무관한 다음 hash building으로 교체한다.
교체 이유와 ID를 기록한다.

======================================================================
6. Fine-tuning sample 계약
======================================================================

[설계] 모든 target adaptation window:
- context 24h
- horizon 24h
- daily origin at 00:00
- context와 target이 exposed H-day interval 안에 완전히 들어가야 함.
- stride = 24h
- overlapping future targets 금지.

따라서 대략:
- H=3d -> 2개의 daily supervised windows
- H=14d -> 13개의 daily supervised windows
정확한 count는 timestamp assertion 후 기록한다.

coverage_count c:
forecast할 day-type과 같은 day-type을 target으로 갖는 adaptation window 개수.
이 값은 target history에서만 계산한다.
미래 target load를 보지 않는다.

모든 방법에서:
- target normalization/statistics가 필요하면 exposed H-day history만 사용.
- hidden earlier target history를 scale/mean/std 계산에 사용 금지.
- pretrained model 내부의 고정 normalization은 그대로 둘 수 있으나,
  target-specific statistic을 추가하면 위 규칙을 지킨다.

======================================================================
7. 모델 — Chronos-2 고정
======================================================================

현재 저장소의 검증된 Chronos-2 로더와 revision을 먼저 읽는다.
기존 revision이 계속 유효하면 그대로 사용한다.
다른 revision으로 자동 교체하지 않는다.

[설계] main backbone:
amazon/chronos-2, 현재 repo가 pin한 revision.
현재 repo의 rank-8 Standard LoRA 부착 위치를 재사용하되
horizon/context 변경으로 API contract가 맞는지 smoke에서 검증한다.

[확인해야 함]
- output horizon 24가 native API에서 정상 지원되는지
- target-only univariate input에서 LoRA path가 정상 동작하는지
- F0와 LoRA step0 identity
- frozen base/head 정책
- trainable parameter count
- quantile ordering / finite output
- 실제 update 발생
- selected checkpoint reload parity

출력 head를 임의로 풀지 않는다.
이번 질문은 "standard target LoRA correction을 coverage로 조절할 이유"이므로
HEAD 정책을 중간에 바꾸면 다른 질문이 된다.

======================================================================
8. LoRA recipe 선택 — 별도 4개 building에서만
======================================================================

method-dev/held-out의 future를 보고 LR/step을 고르지 않는다.

recipe buildings 4개에서만 global recipe를 선택한다.

[설계] recipe cell:
- H=14d
- Wednesday episode만
- exposed 14d 내부에서만 chronological inner split
  예: 앞쪽 fit windows / 마지막 가능한 daily windows를 validation으로 사용.
  exact count는 각 building의 available windows를 보고 성능 전에 고정한다.
- 미래 episode target 24h는 recipe 선택에 사용하지 않는다.

LR grid:
{3e-5, 1e-4}

checkpoint updates:
{15, 30, 60, 120}
step0도 기록하되 global recipe는 실제 adaptation 여부와 함께 보고한다.

4 buildings × 2 LR = 최대 8 fits.
한 fit 안에서 네 checkpoint를 저장한다.

global recipe:
4 recipe buildings의 inner-V primary를 building-macro 평균한 값이 최소인 (LR, step).
동률:
1) 작은 step
2) 작은 LR.

이 recipe를 method-dev와 held-out 전체에 그대로 고정한다.
H=3d에 불리하더라도 결과를 보고 새 recipe를 만들지 않는다.

======================================================================
9. 평가 지표
======================================================================

[설계] primary:
episode-level scaled_RMSE

scaled_RMSE =
RMSE(next24 median forecast, actual next24)
/
max(std(exposed target history), 1e-6)

- std는 해당 episode가 실제로 볼 수 있는 H-day target history에서 계산.
- evaluation future로 denominator를 만들지 않는다.
- 낮을수록 좋음.

secondary:
- raw RMSE
- raw MAE
- probabilistic scaled 2-pinball (Chronos quantiles 사용)
- BuildingsBench official NRMSE가 해당 loader/task에 제공되면 별도 표로 보고.
  official metric을 primary와 섞지 않는다.

building마다 load scale이 다르므로 raw MSE/RMSE를 건물 간 그대로 평균해
"한 건물당 같은 의미"라고 주장하지 않는다.
primary는 building episode를 같은 가중치로 macro average한다.

LoRA gain vs F0:
G = 100 * (Err_F0 - Err_LoRA) / Err_F0
양수 = LoRA가 좋음.

후보 gain도 같은 방식으로 "기준 이름"을 명시한다.

======================================================================
10. Stage A — 문제 현상 screen
======================================================================

method-development buildings 4개만 사용.

각 building:
2 forecast types {Wednesday, Saturday}
× H {3d,14d}
= 4 cells.

각 cell에서:
- F0 prediction
- Standard LoRA fit/prediction
- simple affine output correction (CPU 가능)
를 계산한다.

Standard LoRA:
고정된 global recipe 사용.
4 buildings ×4 cells = 최대16 fits.

Affine baseline:
F0가 exposed adaptation windows에서 낸 median/quantile prediction에
positive scale a>0 + bias b를 맞춘다.
- exposed history target만 사용.
- quantile 전체에 같은 positive scale+bias를 적용해 order 보존.
- 2 parameters.
- fitting objective와 solver를 성능 보기 전에 고정.
- fit 실패/ill-conditioned면 failure를 기록하고 임의 방식으로 교체하지 않는다.

[확인할 핵심 수치]
각 building b에 대해:
D3_b  = G(H3, Wednesday) - G(H3, Saturday)
D14_b = G(H14, Wednesday) - G(H14, Saturday)
I_b   = D3_b - D14_b

해석:
- D3>0: short history에서 covered weekday LoRA gain이 uncovered weekend보다 큼.
- I>0: longer history에서 coverage가 생기면서 그 gap이 줄었다는 방향.

[설계] 문제 현상 continuation gate:
다음을 모두 만족하면 COVERAGE_PROBLEM_SIGNAL.
1) I_b > 0 인 building이 4개 중 최소3개.
2) mean(I_b) >= 0.5 percentage points.
3) H3 Saturday에서 Standard LoRA가 F0보다 항상 나빠야 한다는 조건은 요구하지 않는다.
   핵심은 coverage interaction이다.
4) affine baseline 하나만으로 H3 Saturday의 LoRA 손해/격차가 사실상 사라지는지 같이 보고한다.

0.5pp는 후속 GPU 투자를 제한하기 위한 운영 기준이지
통계적 유의성/논문 합격선이 아니다.

미충족:
NO_COVERAGE_PROBLEM_SIGNAL.
Stage B/C를 실행하지 않는다.
Saturday/Wednesday 대신 다른 요일을 사후 탐색하지 않는다.
history length를 새로 추가하지 않는다.

======================================================================
11. Stage B — 새 학습 모델보다 먼저, 가장 싼 coverage 규칙
======================================================================

Stage A 통과 시에만 수행.
추가 GPU fit 0.
저장된 F0와 Standard LoRA quantile prediction을 사용한다.

이 단계의 목적:
"coverage 정보를 쓰면 좋은가?"와
"새 trainable adapter가 필요한가?"를 분리.

모든 quantile q에 대해 convex interpolation:
q_alpha = q_F0 + alpha * (q_LoRA - q_F0)

quantile order는 두 monotone quantile vector의 convex combination이면 보존되는지 검사한다.

비교 규칙:

B0. STANDARD
alpha=1

B1. UNIFORM_SHRINK
alpha ∈ {0,0.25,0.5,0.75,1}
method-dev 전체에서 하나의 alpha를 선택.
forecast state와 관계없이 동일.

B2. BINARY_FALLBACK
coverage_count c == 0 -> alpha=0
c > 0              -> alpha=1

B3. COVERAGE_SCALE — [미검증 후보 / 작업명]
alpha(c;tau) = c / (c + tau)
tau ∈ {0.5,1,2,4}
method-dev 전체에서 하나의 tau를 선택.
target building별/episode별 tau 튜닝 금지.

B4. AFFINE
Stage A에서 fit한 target-specific affine baseline.

alpha/tau 선택:
method-dev 4 buildings의 primary macro가 최소인 값.
같은 dev data에서 후보를 고르는 단계이므로 이것을 확증이라고 부르지 않는다.

[설계] Stage B continuation gate:
고정된 최선 COVERAGE_SCALE이 method-dev에서 다음을 모두 만족할 때 METHOD_SIGNAL.
1) H=3d 전체 macro primary가 STANDARD보다 >=0.3% 낮음.
2) H=3d macro primary가
   min(고정 UNIFORM_SHRINK, BINARY_FALLBACK, AFFINE)보다 >=0.3% 낮음.
3) H3 Saturday cell에서 building 4개 중 최소3개가
   best simple baseline 대비 양의 improvement.
4) H=14d macro primary가 STANDARD보다 0.2% 이상 나빠지지 않음.

0.3/0.2%는 투자 판단용 내부 기준.
논문 합격선이 아니다.

특히:
- COVERAGE_SCALE ≈ BINARY_FALLBACK이면 "새 scaling의 추가 가치 없음".
- UNIFORM_SHRINK가 같거나 더 좋으면 coverage-specific mechanism 필요성 없음.
- AFFINE이 같거나 더 좋으면 내부 PEFT correction을 건드릴 이유가 약함.

미충족:
METHOD_SIGNAL_FAIL.
held-out building을 열지 않는다.
새 gate/router/state-specific LoRA를 자동으로 만들지 않는다.

======================================================================
12. Stage C — held-out building 6개
======================================================================

Stage A와 B를 모두 통과한 경우에만 수행.

held-out building IDs, episode dates, recipe, alpha*, tau*,
affine fitting rule, metrics를 전부 seal한 뒤 실행.

각 held-out building:
2 forecast types {Wednesday,Saturday}
× H {3d,14d}
=4 cells.

Standard LoRA fit:
6 buildings ×4 cells = 최대24 fits.
global recipe 그대로.
새 LR/step 선택 없음.
target future로 early stopping 없음.

F0 / Standard / UniformShrink(alpha*) / BinaryFallback /
CoverageScale(tau*) / Affine를 모두 계산.

[설계] held-out continuation signal:
다음을 "논문 PASS"가 아니라 HELDOUT_METHOD_SIGNAL로 정의.
1) CoverageScale macro primary <
   Standard macro primary.
2) CoverageScale macro primary <
   best fixed simple baseline
   {UniformShrink, BinaryFallback, Affine}.
3) best simple baseline 대비 building-level 평균 gain이 양수인 building >=4/6.
4) macro improvement가 best simple baseline 대비 >=0.3%.
5) H14에서 Standard 대비 macro degradation <=0.5%.

모든 숫자와 실패 building을 그대로 보고한다.
6개 building을 독립 논문 확증의 충분한 sample이라고 주장하지 않는다.

불확실성:
building ID를 unit으로 2000회 paired bootstrap.
같은 building의 Wednesday/Saturday/H3/H14 episode를 독립 sample처럼 따로 bootstrap하지 않는다.
CI는 개발 파일럿의 기술적 불확실성으로만 보고한다.

[중요]
held-out 결과를 보고 tau/alpha/LR/step/history/day-type을 변경하면
그 building들은 더 이상 held-out이 아니다.
변경 금지.

======================================================================
13. Stage C 이후 해석
======================================================================

HELDOUT_METHOD_SIGNAL이면:
- "coverage-aware correction scaling"의 개발 근거가 생긴 것.
- 아직 논문 신규성 확정 아님.
- cross-building source transfer를 아직 사용하지 않았으므로
  CFTR와 같은 문제를 해결했다고 주장하지 않는다.
- 다음 단계는 사용자 review 후
  (a) novelty deep audit,
  (b) 더 많은 building/source,
  (c) 가까운 PEFT baseline,
  (d) 필요할 경우 training-time coverage-aware adapter
  중 무엇을 할지 결정.
자동 실행 금지.

signal이 없으면:
- 이 후보 종료.
- state 종류를 7 weekday/holiday/season으로 늘려 사후 탐색하지 않는다.
- tau grid를 계속 늘리지 않는다.
- source-building transfer를 붙여 결과를 살리지 않는다.
- "building cold-start 전체가 실패"라고 일반화하지 않고
  이번 weekday/weekend coverage hypothesis가 지지되지 않았다고 쓴다.

======================================================================
14. 데이터 누출·불공정 비교 금지
======================================================================

금지:
- 숨긴 target 과거로 mean/std/normalization 계산.
- hidden future target으로 recipe/checkpoint/alpha/tau 선택.
- 후보만 future calendar 외의 추가 정보를 사용.
- candidate만 다른 building history를 사용.
- same building의 다른 meter를 source/held-out 양쪽에 나눠 넣기.
- 성능을 보고 building/episode 교체.
- Saturday가 안 좋으면 다른 요일로 바꾸기.
- H=3/14가 안 좋으면 5/7/21일을 추가해서 제일 좋은 길이 선택.
- held-out 결과를 본 뒤 LR/step/tau 변경.
- CFTR나 In-Context FT의 결과를 현재 후보의 예상 성능으로 그대로 인용.

허용:
- 예측 시점에 이미 확정된 calendar day-of-week.
- official metadata로 기계적 building eligibility 확인.
- pretrained foundation model의 기존 지식.
- recipe/dev/held-out building separation.

======================================================================
15. GPU / 실행 안전
======================================================================

현재 repo의 GPU lock/guard를 재사용하되 실제 코드를 읽고 사용한다.
존재하지 않는 함수/CLI option을 추측해서 호출하지 않는다.

- 시작 전 외부 compute 확인.
- 다른 프로젝트 PID 종료 금지.
- 자원 대기는 기존 안전 규칙 안에서 제한.
- nonfinite loss/gradient/output 즉시 해당 fit EXECUTION_ERROR.
- OOM을 보고 batch/context/history를 자동 축소하지 않는다.
- 실패 fit을 새 seed로 대체하지 않는다.
- smoke state를 main fit에 넘기지 않는다.

각 fit:
- actual optimizer updates
- wall time
- active train time
- peak allocated/reserved
- trainable count
- frozen hash
를 남긴다.

이번 연구의 중심은 accuracy/coverage 문제다.
Query에서 했던 microbatch/activation-checkpointing 자원 토너먼트를 새로 열지 않는다.

======================================================================
16. 구현 전 smoke / correctness
======================================================================

CPU tests:
1) episode history에 H 이전 target load가 들어오지 않는지.
2) forecast next24 target이 adaptation sample에 들어오지 않는지.
3) daily adaptation windows가 exposed H 안에 완전히 포함되는지.
4) H3 Wednesday/Saturday coverage assertion.
5) building-level split disjoint.
6) same physical building meter grouping disjoint.
7) coverage_count가 target history만으로 계산되는지.
8) alpha interpolation quantile monotonicity.
9) scaled_RMSE scalar implementation과 vector implementation 일치.
10) candidate alpha=0 -> F0 exact, alpha=1 -> LoRA exact.

실제 model smoke:
- recipe/dev 각각 최소1 building에서
  F0 forward
  LoRA identity
  one optimizer update
  frozen weights unchanged
  nonzero intended update
  H24 forecast shape
  checkpoint save/reload
를 검증.
smoke update는 fit으로 세지 않는다.

======================================================================
17. 산출물
======================================================================

필수:
- PROTOCOL.md
- literature_boundary.md
- repo_audit.json
- data_manifest.json
- eligibility.csv
- building_split.json
- episode_manifest.csv
- recipe_selection.csv / recipe_seal.json
- fit_attempts.csv
- train_trajectories.csv
- predictions manifest + hashes
- stageA_metrics.csv
- stageA_interaction.csv
- stageB_rules.csv
- stageB_selection.json
- heldout_seal.json (Stage C 진입 시)
- heldout_metrics.csv (Stage C 실행 시)
- independent_verification.json
- resource_usage.csv
- REPORT.md (한국어)

REPORT 흐름:
1. 연구 질문 한 문장.
2. 왜 Query/BASIS/Censor가 아니라 cold-start coverage를 시험했는가.
3. 선행연구 때문에 무엇을 novelty로 주장하지 않는가.
4. 데이터와 "진짜 짧은 타깃 이력" 계약.
5. Stage A: coverage problem이 실제로 있었나.
6. Stage B: uniform shrink/fallback/affine보다 coverage scaling이 추가 가치가 있었나.
7. Stage C를 실행했나. 했다면 held-out 결과.
8. 실제 fits/updates/자원.
9. 확인된 사실과 추정 원인 분리.
10. STOP / HELDOUT_METHOD_SIGNAL / EXECUTION_INCONCLUSIVE 판정.
11. 자동 후속 없음.

그림:
Stage A까지만 실행되면
- building별 D3/D14/I
- H×forecast-state별 F0/LoRA gain
최소2개.

Stage B 통과 시
- alpha/tau rule comparison
추가1개.

Stage C 실행 시
- held-out building paired gains
- history length별 method comparison
추가2개.

빈 Stage C 결과를 그래프로 만들지 않는다.

======================================================================
18. 실행 인터페이스
======================================================================

새 runner에 최소:
prepare
recipe
screen
rules
heldout
verify
status
를 구현한다.

권장 흐름:
1) prepare
2) recipe
3) screen
4) Stage A gate 확인
5) 통과 시 rules
6) Stage B gate 확인
7) 통과 시 heldout
8) verify
9) report
10) scoped commit/push

계획서만 작성하고 멈추지 않는다.
조건을 통과하면 승인된 상한 안에서 다음 단계를 실행한다.
조건 미충족이면 해당 STOP 결과와 근거를 저장하고 종료한다.

기존 repo의 verified scoped commit/push workflow가 현재도 존재하고 사용자 변경과 충돌하지 않을 때만
새 작업 파일을 commit/push한다.
원시 data/model weights/cache/credential은 push하지 않는다.

======================================================================
19. Phase 3 — 스토리 흐름 검증
======================================================================

최종 REPORT가 아래 질문에 순서대로 답하는지 검사한다.

1) 무슨 문제야?
   새 building의 target history가 며칠뿐이면,
   미래 운영 상태를 target에서 아직 보지 못했을 수 있다.

2) 왜 중요해?
   일반 LoRA가 짧은 history의 몇 개 패턴을 전체 미래 상태에 같은 방식으로 적용할 수 있다.
   단, 이 문제가 실제로 존재하는지는 Stage A에서 확인한다.

3) 왜 이 방법/대조군이야?
   coverage-aware scaling은 LoRA correction을 증거량 c에 따라 줄이는 가장 작은 개입.
   uniform shrink, binary fallback, affine가 더 간단한 대조.

4) 설정 근거는?
   3d vs14d, Wednesday vsSaturday, next24는 결과를 보기 전에 protocol에서 고정.
   cold-start 실사용 문제와 weekly operating pattern을 분리하기 위한 개발 설정.

5) 결과가 주장을 지지해?
   Stage A 현상 → Stage B simple-baseline 추가 가치 → Stage C held-out 순서로만 말한다.

6) 한계·다음은?
   building 수 적음, weekday/weekend 단순화, 관련 source building 미사용,
   기존 development pilot, foundation pretraining overlap 불명확,
   novelty 미확정.
   다음 실험은 사용자 review 전 자동 실행하지 않는다.

독자가 "그냥 F0로 돌아가면?", "그냥 uniform shrink면?",
"그냥 affine correction이면?", "CFTR랑 뭐가 달라?"라고 물었을 때
REPORT 안에서 직접 답이 있어야 한다.

======================================================================
20. CLI에게 보내는 최종 실행 문장
======================================================================

이 지시문을 받은 CLI는:

- 최신 repo delta를 먼저 감사하고,
- official BuildingsBench real-building data를 확인/준비하고,
- Phase A 문제 존재를 먼저 검증하고,
- A가 통과할 때만 저장 예측으로 Phase B coverage rule을 비교하고,
- B까지 통과할 때만 6개 held-out building을 열어 Stage C를 수행하라.

중간 결과를 보고 조건·데이터·threshold를 바꾸지 마라.
새 adapter 구조를 자동으로 발명하지 마라.
cross-building transfer나 residual correction을 추가하지 마라.
결과는 JSON만 주지 말고 한국어 REPORT.md로 설명하라.

======================================================================
21. 참고 출처 — 실행 전에 최신 상태 재확인
======================================================================

저장소:
https://github.com/CanelE452/tsfm-peft-method-screen

BuildingsBench:
https://github.com/NatLabRockies/BuildingsBench

Building Data Genome 2:
https://github.com/buds-lab/building-data-genome-project-2

Time-PEFT:
https://github.com/kaist-dmlab/TimePEFT

논문:
- Time-PEFT: Temporal and Multichannel Complexity-Based Fine-Tuning for Time-Series Foundation Models
  ICML 2026.
- BuildingsBench: A Large-Scale Dataset of 900K Buildings and Benchmark for Short-Term Load Forecasting
  NeurIPS 2023.
- In-Context Fine-Tuning for Time-Series Foundation Models
  ICML 2025.
- Probabilistic Forecasting for Building Energy Systems using Time-Series Foundation Models
  Energy and Buildings 2025, 116446.
- A cross-building few-shot energy load forecasting framework via transferable temporal learning and residual correction
  Energy and Buildings 2026, 117530.

이 파일의 3d/14d, Wednesday/Saturday, 4/4/6 buildings,
0.5pp problem gate, 0.3%/0.2% method gate, tau/alpha grid, 최대48 fits는
이번 개발 파일럿을 빠르게 죽이거나 살리기 위한 [설계] 값이다.
문헌의 보편적 PASS 기준이나 통계적 유의성 기준이 아니다.
