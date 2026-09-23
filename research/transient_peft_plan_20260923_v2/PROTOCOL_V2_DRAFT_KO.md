# 실행 계약 초안 v2 — 전환 과도응답 PEFT

2026-09-23. HEAD `db3879c`. **미확정 문서다.** 데이터가 없으므로 수치를 봉인하지 않는다.
이전 초안: `research/transient_peft_plan_20260923/PROTOCOL_DRAFT_KO.md` (수정·삭제하지 않았다).

## 1. 먼저 difficulty 를 검증한다 — G0~G3

본학습 전에 네 질문을 순서대로 답한다. 어느 하나라도 실패하면 다음으로 가지 않는다.

```
Q1  같은 현재 상태·온도라도 과거 전환 이력 때문에 미래 반응이 달라지는가
Q2  그 차이가 current-state feature 보다 history-aware 표현에서 더 잘 설명되는가
Q3  그 정보를 일반 LoRA 에 똑같이 줘도 residual difficulty 가 남는가
Q4  남는 경우에만, structured PEFT 가 그 residual 을 줄이는가
```

| gate | 내용 | 실패 시 |
|---|---|---|
| **G0 DATA SEMANTICS** | 제어 명령 의미, 발행/유효/적용 시각 정렬, target 단위, 미래 계획 권한이 모두 명확한가 | `BLOCKED_DATA` |
| **G1 TRANSIENT ID** | TRAIN-only 에서 step-response/ARX 또는 공학 정의로 tau0 추정. DEV/TEST 를 보고 고르지 않는다 | `NOT_IDENTIFIED` |
| **G2 HISTORY SIGNAL** | SYS0(현재 상태+context) / SYS1(+명령 이력·경과시간) / SYS2(정규화 ARX·switching) / SYS3(고정 다중 시간척도 상태) 를 TRAIN/V 에서 비교. TRANSIENT·SETTLED·OVERLAPPING 분리 | `NO_STRUCTURAL_SIGNAL` |
| **G3 STANDARD-ADAPTATION RESIDUAL** | 동일 정보 권한으로 F0 / L0 / FI 를 먼저 학습. FI 만으로 difficulty 가 사실상 해결되면 modulation PEFT 를 자동 실행하지 않는다 | 여기서 멈춤 |

## 2. 2×2 재설계 — 교란 제거

이전 초안의 L1 대 P 비교는 두 요인이 동시에 바뀌었다. 분리한다.

```
                    INPUT (covariate 로 제공)      MOD (LoRA 저랭크 업데이트를 변조)
FIXED  (tau 봉인)        FI                              FM
LEARNED (tau 학습)       LI                              LM   <- 제안 후보
```

| 대조 | 답하는 질문 |
|---|---|
| FM vs FI | 상태가 고정일 때 **주입 위치** 효과 |
| LM vs LI | 상태가 학습될 때 **주입 위치** 효과 |
| LI vs FI | 입력 경로에서 **상태 학습** 효과 |
| LM vs FM | 변조 경로에서 **상태 학습** 효과 |

FIXED 는 TRAIN-only 에서 봉인한 tau set 을 쓰고 학습하지 않는다.
LEARNED 는 같은 초기 tau set 에서 시작하고 tau·state generator 만 학습 가능하다.
네 arm 이 **같은 상태 정의**를 공유하므로 두 축이 분리된다.

## 3. arm 전체

| arm | 내용 |
|---|---|
| F0 | 동결 TSFM. raw / feature 입력 view 모두 보존 |
| L0 | 일반 attention LoRA (raw view 기준선) |
| FI | 고정 response state → 입력·공변량 |
| FM | 같은 고정 state → LoRA 변조 |
| LI | 학습 response state → 입력·공변량 |
| LM | 같은 학습 state → LoRA 변조 **(제안)** |
| G | 일반 causal GRU 이력 conditioner → LoRA 변조 |
| O | LM 과 같은 학습 state 로 **출력 residual 만** 보정 |
| SYS | ARX / switching / state-space 소형 baseline |

주 대조: `LM vs LI` · `LM vs FM` · `LM vs G` · `LM vs L0` · `LM vs O` · `LM vs SYS`.
**"LM 이 F0 보다 좋다" 만으로 성공 판정하지 않는다.**

## 4. 공통 정보 권한 (모든 arm 동일)

허용: t 까지 관측된 target·context, t 까지 발행·적용된 명령 이력, t 에 이미 확정된 미래 명령 계획,
t 당시 합법적으로 쓸 수 있던 외생 예보.
금지: 미래 truth, 사후 실제 명령, 시뮬레이터 hidden state.

## 5. 확인된 구현 제약 (CHRONOS_INSERTION_AUDIT.md 요약)

- 기본 `lora_config` 가 `output_patch_embedding.output_layer` 를 포함한다.
  **출력층 동결을 지키려면 config 를 명시해야 한다.**
- `fit()` 이 `Chronos2Model(config)` 로 새 모델을 만들고 `state_dict` 만 복사한다.
  custom module 은 **클래스 수준 교체**로만 학습 경로에 전달된다. 프로세스 분리 필수.
- m(t) 는 `input_patch_size` 간격으로 패치 집계한 뒤 group·quantile 축으로 확장해야 한다.
- 삽입 pseudocode 와 초기 동일성 보장 방법은 감사 문서 §7 에 있다.

## 6. split / horizon — 물리 시간으로 정의

step 수가 아니라 tau0 기준으로 적는다. A1 에서 tau0 를 얻은 뒤
`sampling_interval / tau0`, `context / tau0`, `horizon / tau0` 를 기록한다.
**15min / 48h / 6h 는 아직 계약값이 아니다.**

```
분할        시간순. window random split 금지
gap         같은 transition response 가 여러 split target 에 반복되지 않게 확보
group id    parent weather / initialization / schedule seed 를 보존.
            simulation seed 를 독립 건물이라고 부르지 않는다
panels      TRANSIENT (time_since_transition <= tau0)
            SETTLED   (>= 2*tau0)
            INTERMEDIATE
            OVERLAPPING_RESPONSE (tau0 안에 transition 2개 이상)
```

주 지표: TRANSIENT panel 의 물리 단위 MAE.
보조: 전체·SETTLED MAE, 1h/3h/full prefix, P90 절대오차, native probabilistic loss·coverage,
per-day/block 효과, 학습·추론 비용.

## 7. 예산

**기존 45 fits 를 상속하지 않는다.** arm 이 6개에서 9개로 늘었고 데이터 규모를 모른다.
실제 데이터가 생기고 §5 의 구현 구조가 확정된 뒤 다시 계산한다.

단계별로 끊는다. G3(F0 / L0 / FI) 를 먼저 돌리고 멈춘다. FI 만으로 해결되면 거기서 끝난다.

## 8. 빈 칸 — 데이터 없이 채울 수 없다

데이터 출처·권한·기간, 컬럼명·단위·시각 의미, 제어 가능한 물리량의 정체,
명령 발행/유효/실행값 구분, 표본 간격, tau0, 전환의 정의, TRAIN/V/DEV 기간,
유효 전환 수, s_t 정의, 패치 집계 규칙.
