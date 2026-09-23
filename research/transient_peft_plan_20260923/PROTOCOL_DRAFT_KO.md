# 실행 계약 초안 — 전환 과도응답 PEFT

2026-09-23. **미확정 문서다.** A0 감사에서 확인된 칸만 채웠고 나머지는 빈 채로 둔다.
데이터가 확보되기 전에는 이 수치를 실행 계약으로 승격하지 않는다.

## 0. 상태

```
데이터 준비도   DATA_BLOCKED
신규성          NOVELTY_UNRESOLVED + DUPLICATE_RISK (S/G 축)
따라서          A1 이후를 요청하지 않는다. 아래는 데이터가 생겼을 때의 골격이다.
```

## 1. 확인된 칸

| 항목 | 값 | 근거 |
|---|---|---|
| 백본 | Chronos-2 (chronos-forecasting 2.3.2) | `.venv` 설치본, v2 실험에서 검증 |
| 추론 정밀도 | bf16, sm86 에서 tf32/bf16 자동 | `pipeline.py:281–282`, v2 실측 |
| native loss | quantile loss × loss_mask | `model.py:551, :563` |
| 출력층 | `output_patch_embedding` | `model.py:265, :732` |
| 미래 공변량 | `(n_variates, prediction_length) float32`, 없으면 NaN | `preprocess.py:22, :63` |
| LoRA 기본 대상 | q/k/v/o + **output_layer** | `pipeline.py:207–216` |
| **출력층 동결 방법** | `lora_config` 를 명시해 `output_patch_embedding.output_layer` 를 대상에서 뺀다 | 기본값 그대로 쓰면 §6.3 위반 |
| seed·체크포인트 배선 | `fit(**extra_trainer_kwargs)` 로 `seed`, `save_strategy="steps"`, `save_steps` 전달 | v2 실측 (`V0F`·smoke) |
| fit 의 모델 재생성 | fit 은 새 모델을 만들어 state_dict 를 복사한다 → 인스턴스 런타임 패치는 학습에 전달되지 않음. 클래스 수준 교체 + 프로세스 분리 필요 | v2 `V0F_NZ_WIRING.json` (loc 3.0 대 1.5) |
| GPU | RTX 3080 10GB, 현재 유휴 | 이번 감사 |

## 2. 빈 칸 — 데이터 없이 채울 수 없다

| 항목 | 상태 |
|---|---|
| 데이터 출처·권한·기간 | **미정** |
| 실제 컬럼명·단위·시각 의미 | **미정** — 추측 금지 |
| 제어 가능한 물리량(zone setpoint / HP modulation / supply setpoint) | **미정** |
| 명령 발행시각 대 유효시각 대 실행값의 구분 | **미정** |
| 표본 간격 (제안 15분) | **미봉인** — 신호 특성 확인 후 |
| context / horizon (제안 192점 / 24점) | **미봉인** — tau0 확인 후 |
| tau0 (과도응답 시간) | **미정** — 공학 문서 또는 TRAIN step-response/ARX 로 추정, 불일치 시 NOT_IDENTIFIED |
| 전환의 정의 | **미정** — 명령 의미·해상도에 근거해야 함 |
| TRAIN/V/DEV 기간 | **미정** |
| 유효 전환 수 (하한 TRAIN 80 / V 40 / DEV 80) | **미정** — 미달이면 DATA_INSUFFICIENT |
| m 과 patch·group·time 토큰의 대응 | **미정** — `model.py:144–165` 추가 확인 필요 |
| s_t (운전 상태) 정의 | **미정** — 연속 명령뿐이면 임의 구간화 금지 |

## 3. 데이터가 생겼을 때의 골격 (지시문 원안 유지)

arm: F0 / L0 / L1 / S / G / P / O 6개 신경망군 + SYS.
모든 arm 에 같은 원시 정보 권한. L1·S·G·P·O 는 같은 고정 event feature.

예산: 6군 × 2LR × 3seed = 36 fits (L0/L1 의 12 fits 는 A2 와 공유), 조건부 소자료 대조 9 fits.
총 45 fits / 46,080 updates 상한.

주 지표: TRANSIENT 원점의 6시간 MAE(°C). 날짜 동일가중.
불확실성: 날짜 블록 paired bootstrap 2,000회, 기본 block 48h.

진행 기준: P 가 L1 보다 평균 3% 이상 좋고 3 seed 모두 개선, 그리고 G/S 보다도 좋아야 함.

## 4. 이 저장소의 선행이 부과하는 추가 조건

인접 계열 5건이 전부 추가 가치 미확보이므로, 이번 계약에는 다음이 추가돼야 한다.

1. **S/G 를 실제로 학습한다.** 과거 `learned_gate_comparison`·`petsa_cell_comparison` 의 수치를
   이번 대조로 이식하지 않는다(데이터가 다르다).
2. **L1 대 P 의 분리를 주 질문으로 둔다.** 감쇠 상태를 입력 특징으로 주는 것과 변조로 쓰는 것의
   차이는 이 저장소에서 아직 분리된 적이 없다. 이것이 P 의 유일한 고유 질문이다.
3. **TRP 의 신규성 기준을 상속한다.** "구현 차이가 있다는 사실만으로 학술적 신규성이 생기지
   않는다"(`temporal_response_method_20260919/NOVELTY_AUDIT_KO.md`).
4. **STAR 원문 수식·삽입 위치 대조를 A1 착수 전에 끝낸다.** 이번 감사에서는 하지 않았다.

## 5. 판정 토큰 (지시문 §7.4 그대로)

```
INVALID / BLOCKED_DATA / DATA_INSUFFICIENT / NO_STRUCTURAL_SIGNAL /
NO_ADDED_METHOD_VALUE / PROMISING_SIM_ONLY / PROMISING_REAL_DEV /
INCONCLUSIVE / RESOURCE_STOP
```
