# 선행 실험 겹침 행렬 — 한 문장으로 묶지 않는다

2026-09-23. 이전 A0 는 인접 선행 5건을 "전부 추가 가치 미확보"로 묶었다. 그것은 부정확했다.
verdict·데이터·개입 위치가 각각 다르므로 분리해 적는다.

## 1. 행렬

| 선행 | 과제 | 데이터 | 입력 정보 권한 | 개입 위치 | 비교군 | 실제 verdict | 양성 signal | 이번 transient 문제와 |
|---|---|---|---|---|---|---|---|---|
| `learned_gate_comparison_20260919` | 예측(전이) | electricity_transfer, NESO 2026 7–8월 | 명령 없음. 수치 이력만 | 잔차에 붙는 gate (TOKEN_GATE / TOKEN_GATE_ENTROPY) | 고정 MAG, B0, PLAIN, C3 | **LIMITED_COMPONENT_EVIDENCE** | **있음 — 단 방향이 반대.** 고정 MAG 가 학습형 gate 를 이겼다 (+3.29 / +2.78 / +2.13 / +1.72%, 네 구간 하한 > 0, 사전 문턱 충족 True) | **부분 겹침** |
| `petsa_cell_comparison_20260919` | 예측 | 동일 계열 | 수치 이력만 | 입력·출력 correction cell (PETSA GCM rank16) | MAG, B0 | ADDITIONAL_BASELINE_SUPPORT_NOT_ESTABLISHED | 미확보. PETSA cell 19,010 params vs MAG 8,712 로 예산도 다름 | **부분 겹침** (O 축) |
| `peft_gonogo12_v1_20260922` | 예측 | 동일 계열 | 수치 이력 / 관측집합 | H1 축소 gate, H2 관측집합 bias | local LoRA, shared, native F0, q/v LoRA | **둘 다 NO_GO_CURRENT_RECIPE** | H1 +1.314% (local LoRA 대비) 이나 공유 모델과 사실상 동일 | **부분 겹침** (조건부 gate 축) |
| `temporal_response_peft_20260919` (TRP) | 예측 | 동일 계열 | 수치 이력 | **목적함수** `\|rθ(Tx)−rθ(x)\|` (구조는 일반 adapter) | PLAIN, B0, ANCHOR, SHUFFLE, IDEAL, MAG/C3 | NO_METHOD_EVIDENCE_AT_FIXED_PROTOCOL | IDEAL 의 전력 SHIFT8 좁은 이득만 | **별개** (loss 축 대 구조 축) |
| `temporal_lora_init_v1_20260922` | 예측 | Electricity, ETTh1 | 수치 이력 | LoRA **초기화** | 일반 LoRA, SHUFFLE | HOLD | Electricity 0.168% / ETTh1 0.112% (SHUFFLE 대비 0.064 / 0.007%) | **별개** (초기화 축) |
| `building_peft_topic_decision_20260916` | 예측 | 건물 계열 | 수치 이력 | 주제 선정 screen, 120 fits | 세 방법 | NO_METHOD_TOPIC_THIS_RUN | 세 방법 모두 ZERO 선택 | **별개** (주제 선정) |

## 2. 이전 A0 가 틀린 지점

```
이전 A0    "조건부·시간반응 adapter 계열 선행 5건, 전부 추가 가치 미확보"
실제       verdict 가 다섯 가지로 서로 다르고, learned_gate 는 사전 문턱을 충족했다(True).
           "논문 PASS 아님" 과 "효과 0" 은 같지 않다.
```

## 3. 이번 후보에 실제로 부과되는 제약

**(a) 가장 무거운 것 — 학습형 조건화가 고정 방식에 진 직접 사례가 있다.**
`learned_gate_comparison` 에서 제안이 고정 MAG 였고 학습형 gate 두 종이 대조군이었으며, 고정 쪽이
이겼다. 이번 제안 LM(learned state → modulation)은 그 반대편에 서 있다. 다만 **데이터에 제어 명령이
없었다.** 전환 이력이라는 정보 자체가 없는 자료에서 학습형 조건화가 진 것을, 명령 이력이 있는 설비
자료의 근거로 이식하지 않는다. 그래서 "부분 겹침"이다.

**(b) O(출력 보정) 축은 이미 한 번 졌다.** `petsa_cell_comparison` 의 출력 correction cell 이
ADDITIONAL_BASELINE_SUPPORT_NOT_ESTABLISHED 다. O 를 필수 대조로 유지하되, 이 선행 때문에
O 가 이기면 그것은 새 정보가 아니라 재확인이다.

**(c) 조건부 gate 축도 한 번 졌다.** `peft_gonogo12_v1` H1 이 NO_GO_CURRENT_RECIPE 다.

**(d) TRP 와 초기화 계열은 별개다.** loss 축·초기화 축이라 이번 구조 축과 직접 겹치지 않는다.
다만 TRP 의 신규성 기준("구현 차이가 있다는 사실만으로 학술적 신규성이 생기지 않는다")은 상속한다.

## 4. 금지 사항 확인

- 위 실험들의 electricity/ETT/NESO 성능을 이번 설비 후보의 positive/negative evidence 로 계산하지 않는다.
- "과거 conditional adapter 가 별로였으니 이번도 별로" 라고 쓰지 않는다. (a) 가 그 유혹의 자리다.
- 반대로 "learned_gate 에서 고정이 이겼으니 FIXED 가 맞다" 고 미리 정하지도 않는다. 그것이 §5 의 2×2 가
  실제로 답할 질문이다.
