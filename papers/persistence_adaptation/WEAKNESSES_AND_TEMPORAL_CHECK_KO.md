# 남은 약점과 실제 보완 결과 — 2026-09-18

논문은 구체적인 PEFT 구현과 그 추가 효과를 다룰 수 있다. 다만 현재 가장 약한 것은 **C3의 지속성 규칙이 단순 RECENCY보다 필요하다는 근거**다. 모든 데이터에서 이길 필요는 없지만, 새 구성요소의 필요성은 별도로 설득해야 한다.

이번에는 이를 명확히 하고 반복 평가의 약점을 줄이기 위해 고정 모델의 새 기간 평가와 기존 저장 예측의 입력 조건별 분해를 실제로 수행했다. 규약 커밋은 `0863f50`이며 새 학습은 0회다. 기존 결과·기준·checkpoint를 바꾸지 않았다.

| 약점 | 이번에 완료한 보완 | 현재 판단 |
|---|---|---|
| 단순 RECENCY 대비 새 규칙의 필요성 | 입력 mask가 같은/다른 경우를 분리, 모든 상태·seed의 고정 가중치×gate 교차 분해 | 전력16계열의 다른 mask 집단에서도 상대 이득은0.0854%; gate 항은 음수. 규칙 자체의 필요성은 아직 약함 |
| 반복 사용한 개발 평가 | NESO 2026 H1의128일·26주를 봉인하고 기존 가중치로60개 예측 평가 | SHIFT8에서 C3/B0 +3.5802%, C3/C2 +1.9843%, 세 seed 모두 양수. 같은 제공자의 새로운 기간이며 독립 원천은 아님 |
| 단순 규칙과의 시간적 재현 | 새 기간에도 RECENCY·MEAN·ROTATE16와 직접 비교 | C3/RECENCY +0.0612%,95% CI [−0.0784,+0.1893]%. SHIFT8의256개 mask가 모두 같아 추론 규칙 차이는 식별 불가 |
| 원자료/오류 조건의 손해 | REFERENCE·fault6종·SHIFT4/8/POINT·shape9종 모두 보고 | 새 기간 REFERENCE는 B0 대비0.4871% 악화, F0 대비7.5176% 악화. FAULT는 B0 대비0.4515% 악화 |
| 신규성과 강한 선행 비교 | 기존 문헌·구현 대조의 범위 유지, 해결되지 않은 비교를 명시 | 공식 선행의 동일 정보/예산 직접 재현은 이번에 추가하지 않았으며 미완료 |
| 실제 사건에 대한 유효성 | 합성 변형과 실제 관측 자료의 경계 명시 | 실제 센서 오류/체제 변화 label은 여전히 없으며 해결했다고 주장할 수 없음 |

주요 세 대조의 Bonferroni 보정 구간도 B0/C2에서는 양수였지만 RECENCY는0을 포함했다. 구간은 고정된 세 seed에 조건부인 시간 block bootstrap이며 무한한 seed·데이터 모집단을 대표하지 않는다. 사후 진단의 유리한 하위집단을 새로운 주효과로 승격하지 않았다.

새 기간 결과는 **잘 적응된 예측기 위 추가 PEFT가 큰 지속 변화에서 좁은 이득을 낸다**는 주장을 보강한다. 반면 **C3의 지속성 규칙이 RECENCY보다 독자적으로 우수하다**는 주장은 아직 보강되지 않았다. 추가 학습을 많이 하거나 비교군을 약하게 만드는 것으로 이 차이를 메우지 않는다.

현 원고는 해당 구현의 조건부 성능·원자료 보존의 절충과 mask 식별 한계를 함께 설명해야 한다. 기존 ETTm1/ETTm2의 불리한 결과도 유지한다. 현재 근거로 구체적인 PEFT 연구를 작성할 수 있지만, 새 PEFT 방법의 독창성과 필수성이 충분하여 채택될 것이라는 뜻은 아니다. 투고처를 정한 다음 정식 선행 비교의 필요성을 결정할 수 있으며 이번에 자동 후속 학습은 실행하지 않았다.

- [전체 한국어 보고서](../../results/c3_identifiability_temporal_20260918/REPORT.md)
- [최종 판단](../../results/c3_identifiability_temporal_20260918/FINAL_DECISION.md)
- [규약](../../results/c3_identifiability_temporal_20260918/PROTOCOL.md)
- [새 기간 원점수](../../results/c3_identifiability_temporal_20260918/RAW_SCORES.csv), [seed별 효과](../../results/c3_identifiability_temporal_20260918/SEED_EFFECTS.csv)
- [새 기간 비교 그림](../../results/c3_identifiability_temporal_20260918/figures/F10_temporal_effects.png)
- [조건부 기전 그림](../../results/c3_identifiability_temporal_20260918/figures/F11_conditional_mechanism.png)
- [모든 변화 형태 비교 그림](../../results/c3_identifiability_temporal_20260918/figures/F12_temporal_shapes.png)
- [독립 검산과 GPU 안전 기록](../../results/c3_identifiability_temporal_20260918/VERIFICATION.json)

기존10쪽 검토 PDF는 이번60개 평가 전 자료를 담은 이전 버전이다. 이번 보완 메모와 연결된 보고서·그림을 함께 사용해야 한다. 기존 결과 문서는 재작성하지 않고 버전 이력을 보존했다.
