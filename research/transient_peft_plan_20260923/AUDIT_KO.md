# A0 감사 — 전환 과도응답 PEFT 후보

2026-09-23. 읽기 전용 감사. 학습 0, 모델 추론 0, 시뮬레이션 0, 설치 0, 기존 파일 변경 0, commit/push 0.

## 0. 한 줄 결론

데이터가 없다. 설비 운전·제어 기록도, 그 대안인 BOPTEST 실행 환경도 이 머신에 없다.
동시에 이 저장소에는 조건부·시간반응 adapter 계열 선행이 5건 있고 **전부 추가 가치 미확보**다.
두 사실이 겹치므로 이번 후보는 `DATA_BLOCKED` + `DUPLICATE_RISK` 로 보고하고 A1 이후를 요청하지 않는다.

## 1. 작업 루트 — 불일치를 먼저 보고한다

| 항목 | 실측 |
|---|---|
| 현재 작업 폴더 | `/home/minjae/Documents/github/forecast-revision-peft` |
| 그 Git 루트 / HEAD | 같은 경로 / `237d245 Add pilot v1 results` / dirty 0건 |
| 이 문서를 쓴 위치 | `tsfm-peft-method-screen` (HEAD `caaf270`, dirty 0건) |
| `mltimeseries` | **로컬에 없음** |

지시문이 근거로 든 [R1] `mltimeseries/experiments/peft_lora_only_diagnostic_v1/PURPOSE.md` 는
로컬에서 읽을 수 없다. GitHub 경로는 지시문에 적힌 대로이며 로컬 존재를 보장하지 않는다고
지시문 스스로 명시했고, 실제로 없었다.

현재 작업 폴더(`forecast-revision-peft`)는 FR-LoRA 파일럿 저장소로 이번 후보의 대상이 아니다.
TSFM PEFT 방법론 작업은 `tsfm-peft-method-screen` 에서 이뤄져 왔고 그곳에 `research/` 체계가
이미 있으므로(25개 폴더) 문서를 그곳에 만들었다. 이 판단이 틀렸다면 옮긴다.

## 2. 프로세스·GPU (읽기만, 종료·재시작 없음)

```
GPU        RTX 3080, 923 MiB / 10240 MiB 사용, util 8%
점유 프로세스 rustdesk 272 MiB 뿐. 학습·추론 프로세스 없음
```

## 3. 데이터 — DATA_BLOCKED

### 3.1 설비 운전·제어 기록: 없음

지시문 §3.2 가 요구한 의미(목표 온도, 과거 제어 명령, 실제 실행 상태, 명령 발행·유효 시각,
외기 조건, 표본 간격·권한)에 대응하는 필드를 가진 데이터를 찾지 못했다.

탐색 범위와 결과:

```
find ~/Documents ~/Downloads -maxdepth 4 -type d  ( *hvac* *boptest* *heat*pump* *building* *setpoint* *hydronic* )
  -> 히트펌프·설비 폴더 0건. building_* 은 전부 이 저장소의 과거 실험 폴더였다.
find ~/Documents -maxdepth 5 -type f ( *setpoint* *boptest* *heatpump* *zone*temp* )
  -> 0건
```

이 저장소가 가진 시계열 원자료는 `data/raw/` 의 electricity, ETTm2, jena, m5 다.
**명령 의미를 가진 필드가 없다.** 지시문 §3.2 마지막 줄대로, 온도·전력 곡선에서 임의의 상태
레이블을 만들어 제어 명령처럼 쓰지 않는다. 검색 결과가 없다는 것만으로 사용자에게 데이터가
없다고 단정하지는 않으며, 접근 권한이 있는 다른 위치는 사용자만 안다.

### 3.2 BOPTEST 대안: 현재 실행 불가

```
python -c "import boptest"   ModuleNotFoundError
pip list | grep boptest      없음
which docker                 없음          <- BOPTEST 표준 배포는 컨테이너다
```

공식 문서상 `bestest_hydronic_heat_pump` 사례가 단일 열 구역 + 공기-물 히트펌프 + 바닥난방을
포함한다는 것은 지시문 [S1] 에 적힌 대로이나, **로컬 버전·available inputs/measurements·단위를
대조할 대상이 없다.** 지시문이 금지한 설치·인스턴스 시작·advance 를 하지 않았으므로 여기서 멈춘다.

제어 가능한 물리량이 zone setpoint 인지, heat-pump modulation 인지, supply setpoint 인지의 구별도
로컬 메타데이터 없이는 확정할 수 없다. 문서상의 이름을 실제 신호명으로 승격하지 않는다.

### 3.3 판정

```
DATA_BLOCKED
```

실측 데이터 부재 + 시뮬레이션 환경 부재가 겹친다. `DATA_READY_SIM_PROPOSAL` 로 쓰려면
최소한 docker 와 BOPTEST 이미지가 있어야 하는데 둘 다 없다. 설치는 이번 단계 금지 항목이다.

## 4. 선행 중복 — DUPLICATE_RISK

이 저장소에서 조건부·gate·시간반응 adapter 계열이 이미 여러 번 시도됐고 판정이 모두 남아 있다.

| 선행 | 무엇을 했나 | 판정 |
|---|---|---|
| `temporal_response_peft_20260919` (TRP) | 시간 반응 보존 **손실** `\|rθ(Tx)−rθ(x)\|` | NO_METHOD_EVIDENCE_AT_FIXED_PROTOCOL |
| `learned_gate_comparison_20260919` | 학습형 gate 2종 통제 대조 | LIMITED_COMPONENT_EVIDENCE |
| `petsa_cell_comparison_20260919` | PETSA 계열 조건부 cell 대조 | ADDITIONAL_BASELINE_SUPPORT_NOT_ESTABLISHED |
| `delta_adapter_comparison_20260919` | DELTA adapter 예산 대조 | COMPLETE_CONTROLLED_PRIOR_COMPARISON (논문 PASS 아님) |
| `temporal_lora_init_v1_20260922` | 시간 의존 LoRA 초기화 | HOLD |
| `building_peft_topic_decision_20260916` | 짧은 이력 PEFT 주제 결정, 120 fits | NO_METHOD_TOPIC_THIS_RUN (후보 가치 0) |

### 4.1 이번 제안 P 와의 관계

P 의 핵심은 구조다.

```
m[j,t] = exp(-dt / tau[j,s_t]) * m[j,t-1] + phi_j(u_t - u_(t-1))
h'     = W0 h + B diag(1 + c(m)) A h
```

TRP 와는 **층위가 다르다** — TRP 는 목적함수를 바꾸고 구조는 일반 adapter 였으며, P 는 구조를
바꾸고 loss 는 native 를 쓴다(지시문 §6.3). 따라서 TRP 의 직접 재실행은 아니다.

그러나 S(상태 조건부 LoRA)와 G(GRU 조건부 LoRA)는 `learned_gate_comparison` 및
`petsa_cell_comparison` 과 겹칠 소지가 크다. 두 선행 모두 "조건부 변조가 추가 가치를 주지
못했다" 쪽 결론이다. P 는 그 위에 "조건을 감쇠 상태로 만든다"를 얹는 구조인데, 그 감쇠 상태를
**입력 특징으로 주는 것**(L1)과 **변조로 쓰는 것**(P)의 차이가 이 저장소에서 아직 분리된 적은 없다.

TRP 의 [선행 신규성 감사](../temporal_response_method_20260919/NOVELTY_AUDIT_KO.md)는 같은
저장소에서 이미 "구현 차이가 있다는 사실만으로 학술적 신규성이 생기지 않는다"는 기준을 세워
두었다. P 도 같은 기준을 받는다.

### 4.2 외부 선행

지시문 [S5] STAR 는 상태 조건부 저랭크 adapter 이고 temporal encoder 를 이미 쓴다.
P 와의 차이는 "조건이 정적 상태 ID 인가, 물리 시간으로 감쇠하는 동적 상태인가" 한 겹이다.
이 차이만으로 독립 신규성을 주장할 수 없다는 것은 지시문 §2 가 이미 [미검증] 으로 적어 두었고,
이번 감사에서 그 판단을 뒤집을 근거를 찾지 못했다. 원문 수식·삽입 위치 수준 대조는 하지 않았다
(논문 본문 미확인).

```
NOVELTY_UNRESOLVED  +  DUPLICATE_RISK(S/G 축)
```

## 5. Chronos-2 코드 확인 (소스만, 모델 로드 없음)

경로: `.venv/lib/python3.11/site-packages/chronos/chronos2/`

| 확인 대상 | 실측 |
|---|---|
| 출력층 | `model.py:265` `self.output_patch_embedding = ResidualBlock(...)`, 예측은 `:732` 에서 생성 |
| native loss | `model.py:518 _compute_loss`, `:551` quantile loss, `:563` `loss = quantile_loss * loss_mask` — 마스크 곱이 이미 존재 |
| 미래 공변량 | `preprocess.py:22` `future_covariates: (n_variates, prediction_length) float32`, `:63` 없으면 NaN 패딩 |
| NaN 검사 | `model.py:465` future_covariates 의 비마스크 위치 NaN 을 거부 |
| 그룹/시간 토큰 | `model.py:124–128` `group_ids` 로 group mask, `:165` group-time mask 결합 |
| 기본 LoRA 대상 | `pipeline.py:207–216` r=8, alpha=16, `self_attention.q/k/v/o` + **`output_patch_embedding.output_layer`** |

**주의 1.** 기본 LoRA 대상에 출력층이 포함된다. 지시문 §6.3 은 "native 출력층 동결"을 요구하므로
기본값을 그대로 쓰면 계약 위반이다. `lora_config` 를 명시해야 한다.

**주의 2.** 지시문 §6.1 은 `h' = W0 h + B diag(1+c(m)) A h` 를 attention projection 에 넣는 구상인데,
m 은 시간축 상태이고 이 백본은 patch·group·time 토큰 배치를 쓴다. m 과 토큰의 대응은
`model.py:144–165` 를 더 읽어야 확정되며 이번 감사 범위에서 설계를 확정하지 않았다.

v2 실험에서 확인된 사실(fit 이 모델을 새로 만들어 인스턴스 런타임 패치가 학습에 전달되지 않음,
sm80+ 에서 bf16/tf32 자동 적용, `**extra_trainer_kwargs` 로 seed·save_strategy 전달 가능)은
`results/service_axis_v2_20260922/` 에 실측으로 남아 있어 그대로 재사용할 수 있다.

## 6. 누출·재포장 위험

- **재포장 위험**: 이 저장소의 인접 선행 5건이 모두 "미확보"이므로, P 가 그중 하나의 이름만 바꾼
  것이 되지 않도록 S/G 를 반드시 같은 계약으로 **실제 학습**해야 한다(지시문 §5.2 와 같은 취지).
- **정보 권한 누출**: 제안 P 에만 명령 이력을 주고 L0 에 주지 않으면 정보 대조가 아니다.
  지시문 §2 가 이미 금지했고 Chronos-2 는 공변량을 zero-shot 으로 받으므로 기술적으로도 불필요하다.
- **시뮬레이터 내부값 누출**: BOPTEST 를 쓰게 되면 숨은 열 상태·열용량을 입력에 넣지 않는다.
  현재는 환경이 없어 해당 없음.
- **기존 수치 이식**: BMRA/Jena/M5 의 split·seed·성능을 이번 설비 과제 근거로 쓰지 않는다.

## 7. 실행 내역

```
모델 학습        0
모델 추론·채점    0
시뮬레이션 생성   0
패키지 설치       0
기존 파일 변경    0
git commit/push  0
새로 만든 것      이 폴더의 문서 3개뿐
```
