# A0 정정 — 이전 결론에서 넓어진 부분

2026-09-23. HEAD `db3879c`. 이전 문서: `research/transient_peft_plan_20260923/`(삭제·수정하지 않았다).

## 정정 1 — BOPTEST 실행경로

```
이전   "docker 가 없으므로 BOPTEST 시뮬레이션 경로도 막혔다" -> DATA_BLOCKED
실제   public web-service 가 문서상 존재하고 설치 요구사항이 없다 (api.boptest.net, v0.9.0).
       로컬 경로만 보고 내린 결론이었다.
지금   AUTH_OR_POLICY_BLOCKED — 이 환경에서 DNS 해석 실패로 도달 불가. 서비스 생존은 미확인.
```

상세는 [BOPTEST_PUBLIC_AUDIT.md](BOPTEST_PUBLIC_AUDIT.md).
데이터 판정은 여전히 진행 불가이지만 **사유가 다르다.** "경로가 없다"가 아니라 "이 환경에서 못 닿는다"다.
해제 조건이 생겼다는 점이 실질적 차이다.

## 정정 2 — 선행 실험

```
이전   "인접 선행 5건이 전부 추가 가치 미확보"
실제   verdict 가 서로 다르다. learned_gate_comparison 은 사전 문턱을 충족했고(True),
       고정 MAG 가 학습형 gate 를 +1.72~+3.29% 이겼다. "논문 PASS 아님"과 "효과 0"은 다르다.
지금   실험별로 과제·데이터·정보 권한·개입 위치·비교군·verdict·겹침 정도를 분리했다.
```

상세는 [PRIOR_OVERLAP_MATRIX.md](PRIOR_OVERLAP_MATRIX.md).

## 정정 3 — STAR

```
이전   "STAR 는 상태 ID 에 따라 weight 를 고르는 정적 조건부 adapter"
실제   원문 확인 결과 정적 ID 가 아니다.
```

원문(arXiv 2510.16014v1) 확인 내용:

| 항목 | STAR |
|---|---|
| 상태 encoder 입력 | point-wise 상태 임베딩 `S_point ∈ R^(T×C_s×d)`, **전체 시퀀스 길이 T**. 백본 patch 설정에 맞춰 m 개 patch 로 분할 (§3.2.2, Eq 6) |
| 상태 표현 생성 | learnable **State Memory** `S ∈ R^(N×d)` + Memory Router 의 soft selection (Eq 1–3), load balancing (Eq 5), `S_point = W_mask ⊗ S` |
| 저랭크 변조 | `R_init, D = g₁(S_patch^t)` (Eq 12), `M = sigmoid((R_mask − Γ)/ε)` (Eq 14), **`ΔW = A R B ⊙ D`** (Eq 9), `R = M ⊙ R_init` |
| 삽입 위치 | **원문에 명시되지 않음.** "백본 전체 동결, STAR 만 학습"이라고만 적혀 있다 |
| 상태의 시간성 | 정적 범주 ID 가 아니다. 이산 상태 변수를 시간에 따라 point-wise 처리하고 **patch 별로** R, D 를 생성 |
| task | MTSAD(다변량 이상탐지). 예측이 아니다 |
| 미래 정보 | 사용하지 않음. 학습·추론 모두 patch 경계 안에서 causal |

### 그래서 P 와의 실제 차이

변조 **형태**는 같은 계열이다.

```
STAR   ΔW = A R B ⊙ D,        R·D 는 상태로부터 생성
P      h' = W0 h + B diag(1 + c(m)) A h,   diag 는 상태 m 으로부터 생성
```

다른 곳은 **조건을 만드는 방식**이다.

| | STAR | P |
|---|---|---|
| 상태 생성 | learnable State Memory + Memory Router(soft selection) | 물리 시간 상수 tau 로 지수 감쇠하는 명시적 동역학 `m[j,t] = exp(-dt/tau[j,s_t])·m[j,t-1] + phi_j(Δu_t)` |
| 입력 | 이산 상태 변수 | **제어 명령 변화량 Δu** |
| tau | 없음 | TRAIN 에서 추정한 tau0 주변에서 시작, 물리적으로 해석 가능 |
| task | 이상탐지 | 예측 |

**P 의 차별점은 "조건 생성이 물리 시간 상수를 가진 명시적 감쇠 동역학"이라는 한 겹이다.**
변조 형태 자체의 신규성은 주장할 수 없다. 이전 A0 가 STAR 를 과소평가해 P 의 차별점을 실제보다
넓게 잡았다.

### 함께 확인한 것

- **UniCA** (ICLR 2026): 공변량 적응의 직접 선행. 이번 확인 범위에서는 제목·초록 수준이며
  본문 수식 대조는 하지 않았다 [미확인].
- **Time-PEFT**: 주파수·채널 구조를 구현한 별도 방법. TRP 감사에서 이미
  "우리 연구를 그 코드의 재현이나 head-to-head 비교로 쓰지 않는다"고 정리돼 있다.
- **event-response / event-covariate 계열** [S6]: 이벤트 이후 응답을 입력 공변량으로 만드는 접근.
  §5 의 FI/LI(입력 경로) 가 정확히 이 계열에 해당하므로 **주 대조로 반드시 유지해야 한다.**
- **switching / state-space 계열** [S7]: 전환 동역학 자체의 선행. SYS baseline 이 이 자리다.
  이번 확인 범위에서 본문 대조는 하지 않았다 [미확인].

## 정정 4 — L1 대 P 의 교란

지시문 [D] 지적이 맞다. 이전 설계에서 L1 과 P 는 두 요인이 동시에 바뀌었다.

```
L1  fixed decay feature  ->  input 으로 제공
P   learnable decay state ->  LoRA modulation
     (1) fixed vs learnable      (2) input injection vs modulation   둘 다 변함
```

`PROTOCOL_V2_DRAFT_KO.md` §2 의 2×2 로 분리했다.

## 정정하지 않은 것

- 설비 운전·제어 기록이 이 머신에 없다는 것 — 재확인해도 동일하다.
- Chronos-2 기본 `lora_config` 가 `output_patch_embedding.output_layer` 를 포함한다는 것 —
  `pipeline.py:207–216` 에서 다시 확인했다. 출력층 동결을 지키려면 config 를 명시해야 한다.
