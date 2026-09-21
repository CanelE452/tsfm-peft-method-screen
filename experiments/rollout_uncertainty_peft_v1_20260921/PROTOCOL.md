# Rollout uncertainty PEFT v1 실행 규약

유일한 실행 계약은 `contract/MASTER_CLI.txt` (SHA-256 `8b4a50ed2ca3215b4a2b9ea45cada6e3e7d6febef17a72b55f440448e826bfac`)다. 이 문서는 구현 선택을 구체화하며 계약을 대체하지 않는다. HIER/MAG/FR 결과·학습가중치는 사용하지 않는다.

## 고정 설계

- Electricity TRAIN 적격 열의 SHA-256 순서 첫 32개, ETTh1 7개를 독립 단변량으로 예측한다. 공식 저자 raw source의 해시·upstream revision·실제 계열 및 원점은 DATA_AND_SPLIT_AUDIT.json에 기록한다.
- 원자료 순서 60/10/10/20% TRAIN/CALIBRATION/VALIDATION/TEST, 원점 row%24=0, context512와 future256 유한성만 확인한다. 모든 합법 원점을 사용한다. TRAIN population std로 loss·평가를 나누며 외부 데이터 표준화는 없다.
- Bolt-small FP32, TF32 off, dropout eval. 64시간씩 네 호출이며 context는 512/576/640/704로 증가한다. q/v LoRA r8 alpha16 dropout0, AdamW LR {1e-4,3e-4}, weight decay0, gradient clip1, batch8, 512 updates.
- R은 LoRA만, S/U는 공통 516→8→512 GELU adapter 8,744개. S의 fourth feature는 generated lead², U는 generated log1p(width/s0). 출력 projection 0 초기화, generated patch에만 적용하고 REG는 건드리지 않는다.
- 매 호출의 raw quantiles를 보존하고 정렬한 값으로 pinball loss·median/width feedback을 만든다. median과 width 모두 detach. 네 loss/4 backward 후 optimizer step 한 번. 미래 y는 loss만 사용한다.
- source2×arm3×(선택 seed92120 LR2 + 반복 seed92121/92122 각 고정LR1)=24 fits, 12,288 main updates. 별도 smoke 12 updates. 체크포인트0/128/256/512 중 전체 VALIDATION scaled pinball 최소. LR 동점 작은LR, checkpoint 동점 이른step. 선택 seed는 반복 평균에서 제외한다.
- 각 source/seed의 sampling packet 512×8×2를 먼저 저장하고 LR/arm 간 공유한다. 모델 RNG와 데이터 RNG를 분리한다.

## 강한 대조와 출력 보정

현재 설치 Chronos 2.3.2의 공식 `ChronosBoltPipeline.predict`를 F0_NATIVE/R_NATIVE에서 직접 호출한다. 첫 block 뒤9개 경로, 다음81개 출력을9quantiles로 줄이며 context를 유지한다. 공식 구현은 median-only가 아니다. 설치 source와 현재 upstream의 byte 일치 여부는 manifest에 기록한다.

F0_MEDIAN, F0_NATIVE, R/S/U, 선택 R_NATIVE, R_MC16, CHRONOS2_DIRECT에 같은 CAL affine35grid 기회를 준다. R_MC16은 첫64 analytic quantiles, 나머지는 고정16경로의 선형 empirical quantiles다. 매 horizon 독립 U로 정렬된9Q inverse CDF 보간, [.1,.9] 바깥 clamped tails. sample seed1701/1702, 원점→계열→block→particle→lead 배열 순서를 고정해 microbatch가 난수 draw를 바꾸지 않는다. 유한 경로 대조이며 joint distribution 보장으로 해석하지 않는다.

Chronos2 입력은 list의 각 항목을 [1,512] 단일 target group으로 포장하고 cross_learning=False, native256 및 .1–.9를 직접 요청한다. CUDA tensor를 공식 DataLoader에 넣으면 pin_memory 오류가 발생하므로 공식 API 입력은 CPU tensor다. 이 수정은 학습 이전 API 검사에서 확인했고 모델 연산/설정은 변경하지 않았다.

Affine q'=median+beta*sigma+alpha*(q−median)는 최종 출력에만 적용한다. alpha{.5,.75,1,1.25,1.5,2,3}, beta{−.5,−.25,0,.25,.5}; source/model/block별 CAL pinball 최소. 정확 동점은 (alpha−1)²+beta², alpha, |beta|, 최종 signed beta 순. 같은 기회를 모든 모델에 준다. 모든 neural 선택과 CAL 계수를 봉인한 뒤 TEST 예측을 저장한다. 모든 TEST 예측 저장 후 채점한다.

## 검산과 예산 보존

공급 CPU13tests는 실모델 검증을 대신하지 않는다. 실제 TRAIN 입력의 native64/256 parity, S/U count/init hash, 최초256예측 일치, metadata NumPy 참조 일치, 실제 raw future cells 치환 후 데이터 로더 경유 동일성, context/metadata 길이, 배치분할·계열순서, detached feedback, 유한 gradient, 2update 변화, 원 backbone weights/buffers 불변, perturb 후 checkpoint 복원을 검사한다. FP32 배치 연산 허용 오차는 max absolute prediction scale의2e−5이며 실제 최대차도 보고한다.

각 optimizer step 전에 intent, 완료 직후 optimizer/RNG/schedule index 포함 atomic checkpoint, 이후 committed ledger를 기록한다. 모호한 crash 또는 checkpoint/ledger mismatch 시 재학습하지 않는다. main 12,288/smoke12 caps 및 duplicate key를 강제한다. 초기 API pin-memory 및 smoke event 직렬화 문제는 implementation failure로 기록하며 과학적 음수와 섞지 않는다.

## 평가·비용·해석

TRAIN-scale mean twice-pinball이 주 지표이며 finite9Q 점수다. exact CRPS나 순수 calibration으로 부르지 않는다. series 동일 가중; PCE와 RMSE는 series 내부 집계 후 평균. 각64block, prefix64/128/192/256, full256, tail129–256을 저장한다. raw/ordered/affine을 구분하고 raw crossing은 정렬 전 adjacent8쌍 비율 및 any-crossing 비율을 기록한다.

7개 연속 TEST 원점의 비순환 moving-block paired bootstrap2000회, 필요 블록을 복원추출 후 총 원점수에서 자른다. series와 horizons를 함께 유지하고 seed별 및 두 반복 seed 평균의 CI를 분리한다. 이는 고정 seed·계열의 원점 불확실성으로, optimizer 전체/과거 탐색을 보정하지 않는다. 256h target overlap과 고정24h 시작 phase, 공개 개발 데이터/사전학습 노출 가능성, 비표준 ETT split을 명시한다.

RTX4070에서 batch1/8, 1warmup+3timed repetitions, median walltime 및 peak allocated VRAM을 측정한다. 모델 로딩을 별도 기록하고 input GPU전송·metadata·모델·sort·선택 affine·CPU반환을 포함한다. 각 source/model/seed에 동일 범위를 적용한다. conditional context 수4/28/49는 실제 시간비율이 아니다. Chronos2는 크기/사전학습이 다른 실용 대조다.

보고 범주는 MASTER의 설명에 따라 결과를 해석해 지정하며 자동 유의성/PASS gate를 새로 만들지 않는다. 신규성·논문 PASS를 선언하지 않고 부정적인 제한예산 결과를 rollout PEFT 전체의 반증으로 쓰지 않는다. 추가 실험0으로 종료한다.

## 실행 및 보존

`python experiments/rollout_uncertainty_peft_v1_20260921/provenance.py`

`python experiments/rollout_uncertainty_peft_v1_20260921/data.py`

`python -m pytest -q experiments/rollout_uncertainty_peft_v1_20260921`

`python experiments/rollout_uncertainty_peft_v1_20260921/smoke.py`

`python scripts/run_rollout_uncertainty_peft.py --stage all`

현재 검증된 Python3.11 CUDA venv를 읽기전용 재사용한다. 버전 lock/environment receipt를 결과 경로에 보존한다. 원자료·모델·체크포인트·전체예측은 기존 ignored .cache의 새 실험 하위에만 보존한다. GitHub에는 해시 manifest와 aggregate/검산을 남기므로 GitHub만으로 raw numerical replay 파일까지 제공된다고 주장하지 않는다. 저장소 AGENTS의 results index 갱신은 이번 사용자 지정 새 경로만 수정 지시가 우선하므로 기존 docs index는 수정하지 않는다.

선행: [ICLR 논문](https://arxiv.org/html/2510.16060v2), [저자 코드](https://github.com/Coaster41/Beyond-Accuracy-TSFM-Calibration/tree/b60bfd92ff836525773c71039a83ba2fe3d123bf), [Chronos 공식 소스](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py). 저자 코드는 base 및 고정 문맥 shift를 사용한다. 이번 small/문맥증가/데이터는 전체 논문 재현이 아니다.
