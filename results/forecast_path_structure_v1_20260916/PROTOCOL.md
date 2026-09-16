예보 입력의 시간적 연결을 보존하는 LoRA 학습 — 단일 CLI 실험 계약
작성: 2026-09-16 KST
대상 저장소: CanelE452/tsfm-peft-method-screen
확인 기준: a5cbff35572d01cfb5d5204f71ae571eab61efc1
새 run ID: forecast_path_structure_v1_20260916

0. 먼저 읽을 결론과 이번 작업의 경계

[설계] 연구 범위는 “시간에 따라 갱신되는 기상예보를 입력으로 받아 다음 날 전력 부하를 예측하는 적응” 하나다.
MOMENT의 일반 다변량 attention, PRIOR/SIDE, 건물 cold-start, Censor, Query, source-bank로 돌아가지 않는다.
이 파일 하나를 실행 계약으로 삼는다. 이전의 이름이 비슷한 지시문을 합쳐 실행하지 않는다.

[확인 S1-S4] 기존 한 타깃의 작은 개발 비교에서는 과거 예보 버전을 섞어 학습한 LoRA가 최신 예보만 학습한 LoRA보다 좋았다.
버전을 순환/역순/무작위 순서로 제시한 세 방식은 고정120 updates에서 비슷했다.
그것은 epoch 간 제시 순서를 바꾼 비교이지, 한 24시간 입력 안의 예보 버전 연결을 바꾼 비교가 아니다.
한 타깃, TRAIN8/V4/D16 원점, 재사용 기간이라는 한계가 있다. 새 PEFT 원리는 확보되지 않았다.

[미검증 가설] 예보의 각 시간 값이 조금씩 흔들리는 것뿐 아니라,
24시간에 걸쳐 함께 변하는 경로 구조를 학습하는 것이 실제 예보 입력에 대한 적응에 중요할 수 있다.
반례는 “시간별 값의 분포만 맞춰도 충분하다”, “단순 채널 dropout/출력 보정이면 충분하다”다.

이번에 할 일:
- 기존 타깃 하나와 성능으로 고르지 않은 추가 타깃 둘에서 직접 비교한다.
- 동일 시점의 동일 예보값 노출량을 맞춘 PATH 대 POINT가 핵심이다.
- 최신 예보 LoRA와 exogenous dropout을 강한 단순 대조군으로 둔다.
- 학습/선택/보정 이후의 달력 구간에서 한 번 평가한다.
- 지표가 애매하다고 조건을 바꾸지 않는다. 실행 가능한 네 군은 모두 끝낸다.

하지 않을 일:
- attention, confidence gate, 증류, FR/consistency loss, 새로운 LoRA 행렬 구조를 자동 추가하지 않는다.
- “먼저 LoRA가 F0에 져야 한다” 또는 “교사가 먼저 N% 좋아야 한다”는 진입 gate를 만들지 않는다.
- PATH가 좋아졌다고 새로운 PEFT 아키텍처라고 부르지 않는다. PATH 자체는 알려진 vintage augmentation 계열이다.
- POINT는 정보 통제용 합성 경로다. 물리적으로 발행된 예보 또는 실용 배포 후보라고 주장하지 않는다.

최종 소비처: 사용자가 이 조건에서 시간적으로 구조화된 입력을 다루는 PEFT 연구를 더 할지,
기존 단순 학습법으로 충분하므로 새 모듈 설계를 중단할지 결정한다.
이 실험은 문제 가설과 알려진 학습 방식의 직접 비교다. 논문 채택/새 방법 성공을 보장하지 않는다.

1. 목적 트리와 예상 결과

A. 가용시각과 데이터 분리 검사
왜: 사용할 수 없는 미래 예보가 입력에 섞이는 일을 막고, 모사된 정보 계약을 실제 배포 증거와 구분한다.
조건: 예보 available_at <= forecast origin, 부하 입력 timestamp < origin, 정답 별도 보관.
예상: 모사된 as-of 입력을 재현 가능하게 구성. 이것이 틀리면 방법 성능을 해석하지 못한다.

B. PATH와 POINT의 정보량·예산 통제
왜: 과거 버전이라는 추가 정보의 효과와, 그 정보를 시간적으로 묶어 주는 효과를 분리한다.
조건: 4-epoch 블록마다 각 origin/예측시간/기상변수에서 사용한 네 값의 multiset이 정확히 같아야 한다.
예상: PATH가 POINT보다 좋으면, 단순 개별시점 노출만으로 효과를 설명하기 어렵다.
한계: 시간적으로 섞은 경로의 비현실성·연속성·국소 smoothness 등이 함께 바뀐다. 특정 인과원인 하나의 증명은 아니다.

C. LATEST/DROP 및 동일 출력 보정과의 비교
왜: 그냥 다양하게 학습하거나 편향을 고치는 것으로 충분한지 확인하고, 새 구조가 필요한 척하지 않는다.
조건: 같은 백본, 동일 target 정답 횟수, 같은 최적화 기회. 모든 군에 같은 사후 보정 권한.
예상: 단순 대조가 충분하면 그것을 결과로 남긴다. 더 복잡한 후보를 자동 발명하지 않는다.

D. 별도 달력 구간과 추가 타깃에서 고정 평가
왜: 예전 8개 TRAIN 원점에만 맞춘 신호인지, 다른 타깃/기간에서도 남는지 확인한다.
조건: 평가 전 모든 선택 봉인. 동일 데이터셋의 타깃 세 개를 독립 도메인 세 개라고 세지 않는다.
예상: 방향이 반복되면 이 문제의 후속 연구 근거. 작거나 일관되지 않으면 불확실/추가 가치 미확보.

가지치기:
실제 weather 정답으로 신뢰도를 학습하는 gate, 여러 날씨 경로의 teacher distillation,
새 랭크 배치 및 MOMENT 재실험은 이번 질문에 필요하지 않아 제외한다.
최종 독자 질문: “같은 정보를 시간별로 섞어도 되는가?”, “그냥 dropout/보정이면 충분하지 않은가?”,
“실제 발행 로그인가?”, “새 방법인가 아니면 좋은 기존 학습 규칙인가?”에 결과가 직접 답해야 한다.

2. 선행과 신규성 경계 — 실행 전 짧게 확인, 새 아이디어 토너먼트 금지

[확인 S5] Exogenous Dropout: A Simple, Strong Baseline for Corruption-Robust Time Series Forecasting with Covariates
(2026/arXiv 공개본, 이번 조사에서 정식 채택은 확인하지 못함)는 전체 외생채널 dropout을 강한 대조로 제안한다.
§3.5의 같은 채널 과거/미래 공유 mask와 inverted scaling을 사용한다. 해당 논문 전체 재현은 아니다.
[확인 S6] TFMAdapter: Lightweight Instance-Level Adaptation of Foundation Models for Forecasting with Covariates
(2025/CIKM)는 기존 예측과 공변량으로 출력 보정하는 가까운 대안이다.
이번의 21개 분위수 상수 보정은 TFMAdapter의 GP/pseudo-forecast 전체 재현이 아니다.
[확인 S7] UniCA(2026/ICLR)는 다양한 공변량을 결합하는 어댑터다. 공변량을 넣는 것 자체는 기여가 아니다.
[확인 S8] CoRA(확인한 판본은 2025 arXiv v1)는 동결 표현·조건 주입을 다룬다. 동적 gate의 신규성도 자동 인정하지 않는다.

literature_boundary.md에 다음만 적는다:
- vintage/path augmentation, whole-channel dropout, 시점별/블록 perturbation, covariate adaptation의 가장 가까운 선행.
- 정확히 읽은 절과 코드 여부. 미확인 게재상태는 확정하지 않는다.
- 본 비교의 차이: 같은 예보값의 시점별 노출을 맞춘 24시간 경로 연결 검사.
- 알려진 증강 규칙을 새 구조라고 부르지 않음; 좋은 결과도 신기술 확정은 아님.

예보 관련 값의 평균·분산을 진짜 기상 앙상블 확률로 부르지 않는다.
논문에 예보라는 단어가 나온다는 이유만으로 직접 동치라 판정하지 않는다.
이번 계약의 방법은 이미 완전히 정했다. 문헌을 읽다가 새 방법으로 바꿔 실행하지 않는다.
직접 동일 선행을 찾으면 KNOWN_REPLICATION으로 명시한 채 정해진 비교를 수행할 수 있다.

3. 기존 저장소 연결과 보존

시작 때 branch/commit/dirty files/실행 중 worker/관련 run을 기록한다.
기준 이후 동일 비교가 이미 있으면 내용을 읽고 정확히 같은 셀만 재사용한다. 자동 중복 실행 금지.
사용자 파일 reset/stash/clean/삭제 금지. 다른 프로젝트 프로세스 종료 금지.

읽은 기존 경로:
- docs/RESULTS_INDEX.md
- research/covariate_availability_audit_20260916/{REPORT.md,download_manifest.json,schema_summary.json}
- research/covariate_reliability_diagnostic_20260916/{REPORT.md,LITERATURE_BOUNDARY.md}
- experiments/covariate_vintage_reference_20260916/{run.py,common.py}
- experiments/covariate_lora_controls_20260916/{common.py,run.py}
- results/covariate_lora_controls_20260916/{REPORT.md,seal.json,scaling.json}
- results/covariate_order_control_20260916/{REPORT.md,INTERPRETATION.md}
- results/covariate_calibration_control_20260916/{REPORT.md,INTERPRETATION.md}
- src/tsfm_peft_screen/{backbone.py,lora.py}

확인한 실제 재사용 지점:
- vintage_reference/run.py: build_inputs(load,weather,o).
  available_at<=o 필터, timestamp별 최신순 순위, 과거 rank0/미래 rank0..3, 15분값 4개 평균.
- covariate_lora_controls/common.py: LoRA, make, payload, tensors, forward, predict, score, configure.
  파일 전역 OUT/FEATURES 및 고정 job schema를 그대로 다른 run에 돌려 쓰지 않는다.
- backbone.py: MODEL_ID/REVISION/QUANTILES/load_base.
- MODULES의 실제 이름 96개를 다시 읽어 사용한다. 새로운 내부 이름을 추측하지 않는다.

새로 만들 경로(현재 존재한다고 가정하지 않음):
experiments/forecast_path_structure_v1_20260916/
results/forecast_path_structure_v1_20260916/
.cache/forecast_path_structure_v1_20260916/
docs/FORECAST_PATH_STRUCTURE_20260916.md
scripts/run_forecast_path_structure.py

기존 runner를 import하면서 전역 OUT을 바꿔 옛 폴더에 쓰지 않는다.
순수 함수를 복사·분리하고 실제 import 경로를 기록한다. 옛 결과의 원점수/판정은 그대로 둔다.

4. 데이터 범위와 타깃 고정

[확인 S4,S9] 데이터: OpenSTEF/liander2024-energy-forecasting-benchmark.
고정 revision dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044.
사용 파일: 해당 타깃의 load_measurements 및 weather_forecasts_versioned.
weather_measurements, version 없는 weather_forecasts, EPEX, profiles는 사용하지 않는다.
카드의 모사 가용시각을 따르므로 결과 전체를 SIMULATED_ASOF라고 표시한다.
실제 예보 발행 cycle/ensemble-member/실제 공개 지연을 복원했다고 주장하지 않는다.

최대 타깃 3개:
T0 = 기존 공식 예제 mv_feeder / OS Gorredijk.
T1,T2 = 같은 mv_feeder 그룹에서 성능과 무관하게 고른 추가 두 타깃.
공식 liander2024_targets.yaml을 읽고 (group_name,name)을 NFC 문자열로 canonicalize해 SHA256 정렬한다.
T0 및 기존 예보 연구에서 사용된 다른 타깃이 있으면 그 이력을 기록하고 추가 타깃 후보에서 제외한다.
이름만 다른 동일 물리 target/중복 파일은 메타데이터·내용 hash로 확인해 중복 제외한다.
처음 최대8개 후보의 TRAIN 구간만 검사하여 다음 조건을 만족하는 첫2개를 고른다:
- 아래 TRAIN 기간에서 전체 context/4경로/target이 유한한 origin이 64개 이상.
- train 부하 표준편차 >1e-6, 기상 세 변수의 train 표준편차 >1e-6.
- 중복 timestamp 및 중복 (timestamp,available_at)이 없음.
선정에 V/CAL/TEST의 값·평균·상관·성능, metadata upper/lower_limit은 사용하지 않는다.
선정 탈락 이유와 순서를 보존한다. TEST가 나쁘다고 교체하지 않는다.

T1/T2를 확보하지 못하면 가능한 타깃의 비교는 수행하고 범위를 PARTIAL_TARGET_COVERAGE로 명시한다.
없는 타깃을 Electricity/Traffic/BDG2로 대체하지 않는다. 타깃 수를 채우려고 다른 그룹을 섞지 않는다.
같은 배전망의 세 타깃은 날씨·부하가 상관될 수 있다. 독립 원천 세 개가 아니다.
각 타깃 모델은 자기 TRAIN으로만 적응한다. 이번에 shared/source-pooled 모델을 만들지 않는다.

다운로드는 고정 revision의 metadata + 선택에 필요한 파일만, 총2GiB 상한.
기존 cache는 hash 확인 후 재사용. 원자료 전체를 git에 올리지 않는다.
라이선스는 데이터 카드와 원천 조건을 기록하되 새로운 법적 해석을 하지 않는다.

5. 시각과 분할: 한 타깃의 과거만으로 학습하고 뒤 기간에서 평가

UTC, 매일08:00가 후보 origin. context336시간, horizon24시간.
원자료 quarter-hour 부하는 각 시간의 연속4값을 평균한다. W의 시간 평균이며 에너지 합계라고 하지 않는다.
기상도 같은 유효시각 4값을 평균한다. 이미 interpolated 자료라는 카드의 한계를 남긴다.

역할                 후보 origin(2024년)               고정 개수
TRAIN                 03-01~06-29, 매일08:00             64
V_SELECT              07-01~07-30, 매일08:00             16
V_CALIBRATE           08-01~08-30, 매일08:00             16
BUFFER                09월                              학습정답·선택·보정·채점 미사용
TEST                  10-01~12-30, 매일08:00             최대91

9월을 별도 채점하지 않는다. TEST 시점의 336시간 context에 이미 관측된 9월 값이 포함되는 것은 허용한다.
각 split에서 모든 target horizon이 그 역할의 달력 종료 경계 안에 있어야 한다.
TRAIN/V_SELECT/V_CALIBRATE는 유효한 시간순 origin에서
floor(linspace(0,N-1,n))의 n개 인덱스를 택한다. N>=n이면 unique assertion.
N<n이면 그 타깃의 준비 실패를 명시한다. 기간/표본 수를 성능에 맞춰 늘리지 않는다.
TEST는 위 기간의 모든 유효 origin을 사용하며 성능 기반 subsampling을 하지 않는다.

유효성은 missing/availability/경계만으로 정한다. TEST 정답은 scorer가 선택 봉인 뒤 확인한다.
네 입력 case 중 하나라도 필요한 값이 없거나 target24에 결측이 있으면, 그 origin을 모든 군·case에서 함께 제외한다.
입력 availability 제외와 target 결측 제외를 별도 기록. TEST 구간의 값 보간 금지.
TEST 유효 origin이 타깃당45개 미만이면 그 타깃 결과는 LIMITED_EVALUATION으로 표시한다. 다른 타깃으로 교체하지 않는다.

최근 실행에서 위 TEST 기간의 모델 성능을 이미 봤는지 exposure_ledger에 확인한다.
- 안 봤다면 held-out time-period replication(학습은 같은 대상의 앞 기간)이라고 한정한다.
- 봤다면 DISCOVERY_REUSED_PERIODS로 낮추고 같은 계획을 수행한다. 새 깨끗한 기간을 임의로 찾아 바꾸지 않는다.
- 일반 Chronos-2의 사전학습 corpus 중복 부재는 입증하지 않았다고 항상 쓴다.
T0의 과거 개발 노출과 T1/T2의 새 적응 평가를 별도로 표시한다.

6. as-of 입력의 완전한 정의

forecast origin o마다 과거 부하 timestamp u∈[o-336h,o), available_at<=o인 값만 입력한다.
예측 정답 load timestamp v∈[o,o+24h)는 입력 함수에 전달하지 않는다.
labels는 TRAIN optimizer / V scorer / CAL fitter / TEST scorer의 권한에 따라 별도 파일·인자로 전달한다.

기상은 weather_forecasts_versioned에서 available_at<=o인 행만 남긴다.
각 valid timestamp별 available_at 내림차순의 순위를 k=0,1,2,3으로 정의한다.
과거 기상: 모든 군에서 각 과거 timestamp의 k=0 사용. 과거 측정 날씨 파일은 읽지 않는다.
미래 기상: k별로 같은 순위의 행들을 연결하고 hourly mean하여 Z[o,k,feature,h]를 만든다.
feature 순서: temperature_2m, wind_speed_10m, shortwave_radiation.
각 선택 행의 available_at을 남기고 실제 age=o-available_at 및 lead=valid_time-available_at 분포를 기록한다.

k는 예보 버전의 순위이지 정확히 k일 전이라는 보장이 없다.
같은 k의 24시간 경로도 단일 physical issue/run임을 보장하지 않는다.
PATH라는 이름은 “이 데이터의 같은 순위 선택 경로를 보존한다”는 뜻이다.
진짜 앙상블 확률, 관측 노이즈의 참분포, 기상 오차를 복원했다고 쓰지 않는다.

테스트 조건 S0/S1/S2/S3:
동일한 과거 부하·과거 기상에서 미래 경로만 k=0/1/2/3으로 바꾼다.
S0=허용된 최신 경로. 정답 기상을 주는 clean/oracle이라고 부르지 않는다.
S1..S3=이전 순위 경로를 일부러 제공하는 통제된 입력 품질/갱신 지연 시험.
최신 부하·과거 기상이 정상인 상태에서 미래 예보만 오래된 조건이며 전 시스템 통신 장애를 재현한 것은 아니다.
훈련용 POINT 합성 경로를 주 TEST 입력으로 사용하지 않는다.

정규화:
기상 external mean/std는 TRAIN에서 노출된 과거 기상 배열만 사용한다.
같은 (valid timestamp,available_at,feature) 반복 행은 external 통계에서 한 번만 센다.
타깃별 주 평가 scale sigma_y는 TRAIN 입력+TRAIN 정답의 unique hourly timestamp 부하에서 계산해 고정한다.
모델 부하 입력은 기존 native instance normalization을 유지한다. 모델 외부에서 부하를 다시 정규화하지 않는다.
외부 기상 정규화 후 dropout을 적용한다. NaN/실제 결측을 zero로 바꾸어 정상값으로 숨기지 않는다.
V/CAL/TEST로 통계를 갱신하지 않는다. native normalization은 각 합법적 context만 읽는다.
이 scale은 이전 origin별 context std 지표와 다르므로 옛 점수와 개선율을 직접 합치지 않는다.

7. 네 학습군 — 모델은 완전히 동일, 데이터 제시만 다름

공통 모델:
amazon/chronos-2 revision 29ec3766d36d6f73f0696f85560a422f50e8498c.
원래 backbone/native 확률 head frozen. 기존 96개 projection의 rank1 LoRA, alpha2.
기존 확인 학습 파라미터147,456개/192텐서이며 실제 numel과 목록을 다시 검증한다.
FP32, TF32 off, dropout0, native quantile loss. 입력4행(부하1+기상3), context336, future24.
전체32 output positions 중 부하24 positions만 supervised; 나머지 native mask/reduction 유지.
21 quantiles는 config에서 확인하고 median/0.1/0.9 위치는 확률값으로 찾는다.
raw quantile 출력을 보존하고 평가에서는 모두 같은 increasing rearrangement를 쓴다.
학습 label 값이 model prediction forward에 들어가지 않는 poison 검사를 유지한다.

A. LATEST
매 epoch 같은 origin의 Z[k=0]만 사용. 미래 예보 최신값으로 학습하는 표준 LoRA.

B. DROP
LATEST와 같되, 각 기상 채널마다 m~Bernoulli(0.7)를 뽑아
그 채널의 과거336h와 미래24h를 함께 m/0.7배 한다. 부하 입력·정답에는 적용하지 않는다.
p=.3, 추가 파라미터0, inference dropout 없음. S5의 학습 규칙을 적용한 대조다.

C. PATH
epoch e=0..7에서 k=e mod4인 기존 Z[k]의 미래24시간 전체를 제공한다.
각 4epoch 안에서 모든 origin이 네 버전을 각각 한 번씩 본다.
과거 기상은 늘 k=0. epoch 간의 순서를 고급 curriculum이라고 부르지 않는다.

D. POINT
동일 origin의 시간 h마다 다른 k를 선택해 만든 경로를 제공한다.
4epoch 블록 b=floor(e/4)와 origin마다 offsets[24]를 한 번 생성한다:
[0,1,2,3]을 각6개씩 넣고 독립 deterministic RNG로 섞는다.
k(e,h)=(e mod4+offsets[h]) mod4.
같은 h의 세 기상변수는 반드시 같은 k에서 가져와 동시점 기상변수 묶음을 유지한다.
한 4epoch 블록 전체에서 각 origin,h,feature가 본 네 값은 PATH와 정확히 같다.
과거 부하·과거 기상·정답·외부 정규화·origin 순서·모델 초기값·업데이트 수는 PATH와 같다.
달라지는 것은 24시간 안에서 여러 시점의 버전값을 함께 보여주는 방식이다.
POINT를 “독립 Gaussian noise” 또는 “시간별 참 주변분포 샘플”이라고 부르지 않는다.

RNG:
학습 origin 순서는 np.random.default_rng(seed+epoch)의 64개 순열; 네 군과 두 LR에 공통.
POINT offsets/dropout은 모델 RNG와 분리한 hash(target_id,origin,seed,block 또는step)로 만든다.
POINT/dropout을 생성해서 다른 군의 모델 RNG를 바꾸지 않는다.
모든 augmentation의 index/mask를 TRAIN 데이터만으로 prepare 때 저장·봉인한다.

구별해야 할 효과:
PATH 대 LATEST에는 추가 과거 예보 정보와 데이터 다양화가 함께 들어간다.
PATH 대 POINT에는 같은 시점별 값 노출에서 경로 연결 차이가 들어간다.
PATH 대 DROP은 알려진 단순 규제보다 유용한지의 실용 비교지만 같은 예보 정보량 비교는 아니다.
PATH+dropout 결합, 상태별gate 등은 이번에 시험하지 않았다. 결과로 그 가능성까지 단정하지 않는다.

8. 학습·선택 예산

타깃당 네 군×두 LR×두 seed:
LR={1e-4,3e-5}, seeds={61730,61731}.
타깃3×군4×LR2×seed2=최대48 trajectories.
모두64 TRAIN origins×8epoch=512 optimizer updates. 전체최대24,576 updates.
AdamW beta(.9,.999),eps1e-8,wd0,clip global norm1,constant LR, effective batch1 origin/group4.
512 전부 수행하고 step{0,256,512}에서만 V_SELECT 평가·선택용 checkpoint를 남긴다.
이 시점들은 complete 4epoch augmentation block의 끝이다. 미완결 prefix의 버전 노출 차이를 줄인다.
불리한 중간 성능 때문에 일부 군을 중단하지 않는다. nonfinite/자원/입력 오류는 별도 중단.
512에서도 개선중이면 BUDGET_LIMITED 표시. 자동 증량/LR 변경/seed 추가 금지.

V_SELECT는 실제 S0..S3 모두 평가한다. 네 case 평균의 primary가 선택 기준이다.
각 target/arm/LR/seed에서 0/256/512 중 최소 V primary(동률이면 이른step).
target/arm별로 두 seed의 최소 V primary 평균이 작은 LR 하나를 고정(정확 동률이면3e-5).
선택 LR에서 seed별 step을 반환한다.
두 seed를 LR 선택에 사용했으므로 새 independent tuning-free seed라고 부르지 않는다.
LATEST도 네 상태에서 선택한다. latest-only 기준으로 불리하게 고르지 않는다.
선택과 별도로 모든 군의 FIXED512도 동일정보/업데이트 대조로 사전 지정한다.

저장: INIT,256,512와 epoch당 최신 완전 resume state(Adam/RNG/step/stream 포함).
매 step 대형 checkpoint 전부 저장하지 않는다. scalar log는 매step, 상태는 안전하게 원자저장한다.
중단 후 state가 불완전하면 해당 fit을 INCOMPLETE로 보존한다. 점수가 나쁜 것을 실행 오류로 바꾸지 않는다.
같은 코드를 옛120step에 이어붙여 512step이라고 하지 않는다. 새 TRAIN이므로 옛 weights warm-start 금지.

9. 지표와 단순 출력 보정

Primary per target, origin, input-case:
L = mean_{21 tau,24 h}[2*max(tau*(y-q),(tau-1)*(y-q))] / sigma_y.
sigma_y는 해당 타깃 TRAIN에서 고정. 값이낮을수록좋음.
주 평가는 S0..S3를 동일 비중으로 평균한 스트레스 점수다.
이1/4 가중치는 실제 배포 빈도를 추정한 것이 아니다. 실제운영 평균이나 참 uncertainty distribution으로 부르지 않는다.
최신S0, 각S1/S2/S3, max_k(타깃별 평균손실)의 worst-scenario도 반드시 별도 보고한다.
per-origin max와 max-of-means는 다르며 뒤의 정의만 사용한다.

Secondary: raw mean2pinball, median RMSE/MAE, 80%구간 포함률/폭, per-lead-hour score.
raw 오차를 다른 부하 규모의 타깃끼리 그대로 평균하지 않는다.
각 타깃의 primary→날짜평균→seed평균→타깃동일가중 평균 순서.
모든 gain은100*(baseline_score-method_score)/baseline_score. 분모의 비교군과 case를 표시한다.

고정 단순 대조(추가 neural fit0):
- FROZEN_WEATHER: 같은 원래 Chronos-2와 같은 S0..S3 입력, LoRA 없는 상태.
- FROZEN_HISTORY: 부하context만 제공한 원래 모델. future weather 사용가치 참고선.
- LAST_DAY: 직전24h 부하 반복, 점예측 보조지표만 보고.
FROZEN_WEATHER는 S0..S3의 서로 다른 입력 예측이다. 모든군 공통 정답과 scale을 사용한다.
FROZEN_HISTORY의 1변수와 weather의4변수 forward 차이를 새 architecture효과라고 하지 않는다.

출력 보정(모든 선택 모델·FROZEN에 같은 권한, V_CALIBRATE만 사용):
각 quantile tau에 대해 delta_tau=empirical_tau_quantile(y-q_tau)을
V_CALIBRATE의 모든origin×S0..S3×hour를 같은 빈도로 모아 계산한다(np.quantile,method='linear').
새 예측 q_cal_tau=q_tau+delta_tau, 이후 같은 increasing rearrangement.
타깃/arm/seed마다21개 상수만 사용; case별/시간대별 새 보정기는 만들지 않는다.
선택된모델에만이 보정 적용, FIXED512는 raw 공통예산 진단으로만 보고.
같은 target이 case별 반복되는 것은 calibration risk의1/4평균을 구현한 것; 독립레이블4배로 세지 않는다.
CAL을 checkpoint/LR 선택에 사용하지 않는다. raw와calibrated를 모두 미리 지정해 TEST에 한 번 평가한다.
유리한 버전만 주 결과로 바꾸지 않는다. 기본 primary=raw, 동일보정 후 추가가치=필수감사 표.
보정으로PATH 이득이사라지면 “일정한 분위수 위치 보정으로설명될수있음”으로약화한다.
이 단순 보정은 완전한 calibration 보장이나 TFMAdapter의 재현이 아니다.

10. 봉인 평가와 해석

모든48경로(또는 준비단계에서기계적으로정한적은타깃의전체경로)를 완료한 뒤
선택 checkpoint/LR, V_CAL 보정값, 원점·case·제외규칙·metric 코드 hash를 봉인한다.
그 후 TEST 예측을 수행하고, 예측파일 해시를 저장한 뒤 scorer가 TEST부하정답을 읽는다.
같은 checkpoint의 중복 역할은 예측을한번수행하고참조한다.
TEST가 이미노출됐다면그사실을표시하되성능이좋은다른기간으로바꾸지않는다.

고정 대비:
1) PATH vs LATEST: 과거예보 학습의 이득이 새 조건에서 재현되는가?
2) PATH vs DROP: 단순 외생채널 dropout을 넘는가?
3) PATH vs POINT: 동일 시점별 예보값 노출에서 시간 연결이 중요한가? [핵심 대비]
4) 각군 vs FROZEN_WEATHER: 적응의 실제 이득/손해, 성공을위한입장gate는아님.
5) raw 대 common-calibrated 대비: 단순 분위수편향 보정만으로 차이가설명되는가?
6) 선택모델 대 FIXED512: prefix선택과학습효과구분. FIXED512를사후우승정책으로승격금지.

각 target/seed/case의 양수·음수효과를전부남긴다. 가장좋은target·seed만추리지않는다.
TEST 날짜의연속7일블록으로paired bootstrap2000회(seed61916).
동일 resample 날짜 인덱스를 군/seed/case/동시기 타깃에 공통 적용해 공통날씨상관을유지한다.
누락origin은미리기록한mask로처리하고 각타깃/seed의분모를재계산한다.
시간block CI는 관측한3타깃과2학습seed에조건부인 기술적 구간이다.
타깃모집단/seed모집단 전체의불확실성이나다중검정교정을대체하지않는다.
타깃별결과,월별결과,두추가타깃만의결과를함께제시; 종합p값만으로결론내리지않는다.

데이터 값으로 고르는 별도 신뢰도 subgroup/gate를 만들지 않는다.
버전간차이의크기,시간차분량,lag1상관 등은 TRAIN의개입강도를설명하는진단으로만계산하고
그지표로TEST 좋은날짜만선택하거나새threshold를만들지않는다.

11. 결과의 의미 — PASS를 억지로 만들지 않음

EXECUTION_COMPLETE는 학습/평가/검산 완료만 의미한다.
표의 연속적인 effect size·불확실성·최신/오래된입력 손익이 판정보다 먼저다.

- PATH가 LATEST보다좋고 POINT/DROP과차이가없음:
  알려진증강/일반규제로충분할가능성. 시간경로특화장치의추가근거미확보.
- PATH가 POINT보다좋고 DROP보다도좋으며같은보정후에도차이가남음:
  이정보계약에서경로연결을보존할가치의개발근거. 특정신경망구조가필요하다는증명은아님.
- raw이득이calibration후사라짐:
  단순출력위치보정이강한설명. 새PEFT구조주장을약화.
- 최신S0은좋지만이전S3이악화,또는그반대:
  TRADEOFF. 네case평균만으로둘다해결했다고부르지않음.
- 모델이INIT로선택됨:
  선택정책의무적응결과와FIXED512 실제학습결과를구분.
- 평균양성이나불확실성큼/타깃반전:
  POSITIVE_BUT_UNCERTAIN. 부재나성공으로강제하지않음.
- 기계적으로입력이같아져PATH=POINT가된경우:
  NO_DISTINCT_INTERVENTION. 완전히같은계산은중복학습하지않고참조하며일반불가능주장금지.

1% 등과 같은 임의성능문턱을중간진입조건으로두지않는다.
향후최소효과크기가필요하면1%기준의감도표만참고로보고현재원결과를재명명하지않는다.
최종 NEW_PEFT_METHOD_ESTABLISHED는 이 실행에서 부여하지 않는다.
PATH는 알려진 학습 규칙이고 POINT는 통제이므로, 좋은 결과는 신규성을 자동 완성하지 않는다.
최종 추천은 하나만: 이조건에서다음방법을정의할근거있음 / 단순대안으로충분 / 추가근거미확보.
다음방법에대한한단락제안은가능하나구현·학습자동실행금지. 다른주제로이동하지않는다.

12. 필수 correctness와 예산 안전

CPU:
- target별 TRAIN/V/CAL/TEST label interval disjoint, origin하루간격에서 H24 비중첩.
- 각입력선택행 available_at<=o, 모든load입력timestamp<o.
- 미래load정답값을poison해도입력/augmentation/모델예측불변.
- unavailable기상행을poison해도입력불변; 과거available기상값변경에는입력이반응해야함.
- PATH/POINT의 4epoch블록 per-origin/per-hour/per-feature multiset exact 동일.
- 한시간의세변수는같은k에서선택; offsets재사용범위와각k의6시간배분검산.
- 네경로가동일하면PATH와POINT의입력이동일; 서로다르면적어도지정합성예제에서는다름.
- DROP은기상만과거/미래같은mask,부하/target변경0; surviving scale1/.7.
- float64 scalar pinball/median metric과vectorized metric검산.
- config.quantiles와반환shape,calibration target권한,quantile rearrangement검산.

실제모델smoke:
최대3타깃×4군×2updates=24updates, 본학습과별도, 폐기.
공통초기parameterhash, step0nativepipeline일치(같은shape/정밀도),frozenweights/buffer불변,
유한loss/gradient/실제LoRA update,복원후같은입력의예측검사.
LoRA B0때 첫Agradient0일수있음을오류로세지않는다.
BF16/FP32·서로다른batch의완전동일성gate를가져오지않는다. 본실험은고정FP32한경로.
복원검사 normalized maxabs<=1e-5와 mismatch기록; 최초비트일치여부도남김.
저장예측의scalar계산은float64 rtol/atol1e-10. 실패하면오류원인기록,기준상향금지.

자원:
main48fits/24,576updates, smoke24, 추가폐기resource최대24, 총신규optimizer상한24,624.
native forward는학습·평가·검산용을분리계수. target별S0..S3는스트레스평가4회이지배포ensemble4회가아님.
한GPU한worker, 검증된프로젝트lock/guard사용,다른프로젝트PID종료금지.
시작free>=4GiB, 업데이트경계free>=1GiB, 누적안전대기600초, controller wall4시간상한.
이시간은안전상한이지완료시간약속이아님. checkpoint용량과여유디스크를미리확인;기존cache삭제금지.
전역환경/driver변경금지. 기존Chronos용환경을우선재사용하고버전기록.
nonfinite/OOM/I/O오류는실행판정불가. 한군만context/batch를줄여다른문제를만들지않는다.
정확한resume state로같은fit재개가능하나seed/recipe교체금지. 반영여부가모호한step자동재실행금지.
출력이 불리해도 remaining4군을 중단하지 않는다. 자원상한도달하면 부분결과와미실행범위를보고한다.

13. 산출물과 실행 순서

새 runner가 prepare -> tests/smoke -> train -> select -> calibrate/seal -> test -> verify -> report를 수행한다.
--help/status/resume은실제구현후확인한다. 없는API옵션이나helper를추측해서호출하지않는다.
계획서만만들고끝내지않는다. 필수입력/실행무결성이확보되면정한학습을실행한다.

필수파일:
PROTOCOL.md, TOPIC_ONEPAGE.md, literature_boundary.md,
repo_audit.json, data_receipt.json, target_manifest.csv, origin_manifest.csv, exposure_ledger.md,
selected_weather_rows manifest, train_scaling.json, augmentation_schedule.npz + hash,
fit_manifest.csv, fit_attempts.csv, train_curves.csv, checkpoint_manifest.json,
selection_seal.json, calibration_parameters.json, evaluation_seal.json,
scores_by_target_seed_case.csv, raw_and_calibrated.csv, contrasts.csv, uncertainty.csv,
resource_usage.csv, independent_verification.json, REPORT.md, FINAL_DECISION.md.
원자료/모델/큰예측배열은로컬ignoredcache. JSON만사용자에게주지않는다.
이전결과hash보존. 같은프로젝트에서기존승인된scoped commit/push가유효할때새파일과index만업로드.
기존승인/접근권한이없으면localartifact와미업로드상태를보고. 외부공개범위를임의확대하지않는다.

한국어 REPORT 흐름:
1) 무엇이문제인가: 미래기상입력이정답이아닌버전별예보.
2) 기존에어떤신호가있었고왜같은실험반복이아닌가: 추가타깃·후속기간·경로내정보통제.
3) 실제발행로그아닌모사as-of임과타깃/기간/정답/비용계약.
4) 실제완료fits/updates/미실행범위와불리한결과포함전체표.
5) 같은예보노출에서PATH와POINT가달랐는가,실제intervention이구별됐는가.
6) dropout과동일출력보정으로충분했는가.
7) 최신/오래된입력손익, 타깃별·월별불확실성.
8) 이것이새PEFT가아닌부분과,다음연구가정당화되는정확한범위.
종료후새후보자동생성/학습없음.

그림은 점수 계산 후 최소 세 개만: 타깃별 4case 오차, PATH 대 대조군 paired gain,
raw/동일보정 비교. 실제측정없는곡선/완료하지않은타깃을그리지않는다.
이 실험에서 기대한 결과를 얻지 못해도 기존조건의기록을남기고종료한다.

14. 출처 — 확인된 사실과 이번 설계의 구분

[S1] 기존 LoRA 대조:
https://github.com/CanelE452/tsfm-peft-method-screen/blob/a5cbff35572d01cfb5d5204f71ae571eab61efc1/results/covariate_lora_controls_20260916/REPORT.md
[S2] 순서 대조:
https://github.com/CanelE452/tsfm-peft-method-screen/blob/a5cbff35572d01cfb5d5204f71ae571eab61efc1/results/covariate_order_control_20260916/REPORT.md
[S3] 실제 입력·LoRA 구현:
https://github.com/CanelE452/tsfm-peft-method-screen/blob/a5cbff35572d01cfb5d5204f71ae571eab61efc1/experiments/covariate_vintage_reference_20260916/run.py
https://github.com/CanelE452/tsfm-peft-method-screen/blob/a5cbff35572d01cfb5d5204f71ae571eab61efc1/experiments/covariate_lora_controls_20260916/common.py
[S4] as-of 데이터 감사:
https://github.com/CanelE452/tsfm-peft-method-screen/blob/a5cbff35572d01cfb5d5204f71ae571eab61efc1/research/covariate_availability_audit_20260916/REPORT.md
[S5] Exogenous Dropout(2026/arXiv 공개본), §3.5:
https://arxiv.org/html/2607.05452v1
[S6] TFMAdapter(2025/CIKM), §3–4:
https://arxiv.org/html/2509.13906v1
[S7] UniCA(2026/ICLR):
https://proceedings.iclr.cc/paper_files/paper/2026/file/0b5eb45a22ff33956c043dd271f244ea-Paper-Conference.pdf
[S8] CoRA(확인한 판본2025/arXiv v1):
https://arxiv.org/html/2510.12681v1
[S9] 공식 데이터 카드, 실제사용은고정revision과맞춰재확인:
https://huggingface.co/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark

타깃3개, 기간, 64/16/16/91원점, 8epochs, 두LR·두seed, offsets 통제,
21상수보정, 동일case가중치 등은 이문서가정한 [설계]이지 논문표준이아니다.
작성자는 로컬원자료·실제GPU실험을실행하지않았다.
아래 부록은 작은합성배열로검산한참조코드이며, 실제파일·모델연결검사를대체하지않는다.

15. CLI에 보내는 실행 요청

이 파일 하나를 사용해 최신저장소중복여부를확인하고, as-of입력·PATH/POINT정보통제를검사한뒤
최대48경로의직접비교와봉인평가·독립검산까지실행해.
평균성능이나간접gate가안좋다고다른주제로옮기지말고, 네학습군의계획된비교를완료해.
미래날씨정답·비가용예보·평가정답으로학습/선택하지마.
POINT는통제입력이고PATH는기존학습규칙임을숨기지마.
신규성·실행완료·예측근거를각각보고하고, 논문PASS를강제로만들지마.
추가구조·다른데이터셋·후속학습을자동실행하지마.


부록 A. PATH/POINT 정보 통제와 지표의 작은 CPU 참조 코드
------------------------------------------------------
# Python 3 + NumPy. 아래 함수는 입력 구조/노출 통제만 검산한다.
import hashlib
import numpy as np


def stable_seed(*parts):
    s = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big")


def point_offsets(target_id, origin_iso, model_seed, block):
    seed = stable_seed("POINT_V1", target_id, origin_iso, model_seed, block)
    rng = np.random.default_rng(seed)
    return rng.permutation(np.tile(np.arange(4, dtype=np.int64), 6))


def select_path(paths, epoch, arm, offsets=None):
    # paths: (4 vintage ranks, 3 weather variables, 24 future hours)
    a = np.asarray(paths)
    if a.shape != (4, 3, 24) or not np.isfinite(a).all():
        raise ValueError("Expected finite paths with shape (4,3,24)")
    if arm in ("LATEST", "DROP"):
        return a[0].copy()
    if arm == "PATH":
        return a[epoch % 4].copy()
    if arm != "POINT":
        raise ValueError("Unknown arm")
    d = np.asarray(offsets, dtype=np.int64)
    if d.shape != (24,) or not np.array_equal(np.bincount(d, minlength=4), [6]*4):
        raise ValueError("POINT offsets must have 6 occurrences per vintage offset")
    k = (epoch % 4 + d) % 4
    return np.stack([a[k[h], :, h] for h in range(24)], axis=1)


def check_four_epoch_multiset(paths, offsets):
    pp = np.stack([select_path(paths, e, "PATH") for e in range(4)])
    qq = np.stack([select_path(paths, e, "POINT", offsets) for e in range(4)])
    assert np.array_equal(np.sort(pp, axis=0), np.sort(qq, axis=0))
    return bool(np.any(pp != qq))


def two_pinball(q, y, taus, sigma):
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    if q.shape != (len(taus), len(y)) or sigma <= 0:
        raise ValueError("Invalid score input")
    if not (np.isfinite(q).all() and np.isfinite(y).all()):
        raise ValueError("Nonfinite score input")
    e = y[None, :] - q
    return float((2 * np.maximum(taus[:,None]*e, (taus[:,None]-1)*e)).mean()/sigma)


def quantile_bias_fit(q, y, taus):
    # q: origins x 4 cases x quantiles x hours; y: origins x hours
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    if q.ndim != 4 or q.shape[1] != 4 or q.shape[2] != len(taus):
        raise ValueError("Invalid calibration shape")
    if y.shape != (q.shape[0], q.shape[-1]):
        raise ValueError("Invalid calibration labels")
    residual = y[:,None,None,:] - q
    return np.array([
        np.quantile(residual[:,:,j,:].reshape(-1), t, method="linear")
        for j,t in enumerate(taus)
    ])


def quantile_bias_apply(q, delta):
    q = np.asarray(q, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if q.shape[-2] != len(delta):
        raise ValueError("Quantile count mismatch")
    return np.sort(q + delta[:,None], axis=-2)

# 작성 단계에서는 offsets의결정성, 4epochmultiset동일성, 시간별3변수묶음,
# 동일한4경로의퇴화동작, pinball독립합산, 보정의monotonicity와분할날짜산술을검사했다.
# 실제 3타깃의유효성, 학습성능, rawmetadata발행로그는검증하지않았다.
