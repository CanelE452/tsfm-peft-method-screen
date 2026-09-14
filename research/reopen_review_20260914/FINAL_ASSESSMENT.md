# 최종 판단

허용된 재검토는 모두 완료했다. 과거 FAIL/STOP은 삭제하거나 PASS로 고치지 않았다. 새 forecasting 학습은 PatchPhase v2의 12 fits/8640 updates뿐이다. smoke6 updates, Query의 폐기되는 자원 측정134 optimizer steps는 별도 집계했다. FR/Censor 신규 fits와 optimizer update는 각각0이다.

## 우선 투자 후보 — 두 개

1. **Forecast-Query PEFT의 효율성 축.** 두 원천의 Query OFF/ON 모두 Standard partial CP 및 Side에 대해 품질·메모리·시간의 3축에서 지배되지 않았다. Query OFF는 약948.1MiB/126.7–130.7ms, ON은 약683.9MiB/195.2–195.4ms다. Standard 전체 CP는 약732.8MiB/168.3–169.6ms로 더 좋은 품질과 빠른 시간을 갖지만 Query ON보다 메모리는 크다. 반대로 Side는 작고 빠르지만 기존 평균 품질은 Query보다 나쁘다. 판정은 QUERY_RESOURCE_FRONTIER이고 현재 상태는 SYSTEMS_TRADEOFF_ONLY다. 독립 환경 재현·실제 메모리 제한 시 효용·선행 대비 차별성이 다음 판단에 필요하다.
2. **Censor tail-loss의 구현·설계 교정 축.** 두 저장 상태에서 검열 train 위치7957회 중217회/222회가 saturated zero-gradient tail이었다(2.7272%/2.7900%). 이들이 censor 손실 항의18.2471%/18.5491%를 차지했다. CENSOR_IMPLEMENTATION_LIMIT_MATERIAL로 기록한다. 이 비율은 전체 학습 손실 기여율이 아니며, 과거 실패 전체를 설명한다는 뜻도 아니다. 정확한 tail 모델과 단순 censored-loss 대조부터 검증할 가치가 있지만, corrected loss나 Censor v2는 이번에 구현·학습하지 않았다. 원래 ceiling에 의해 만든 합성 검열 문제와 실제 판매 검열 문제의 차이도 남는다.

두 후보 모두 논문 성공 판정은 아니다. Query는 실제 자원 절충의 근거가 있고, Censor는 수정할 구체적인 구현 실패가 있다는 서로 다른 이유로 선정했다.

## 우선순위를 낮춘 후보

- **PatchPhase v2 — PATCH_V2_MIXED.** Conditioned 대 augmented 정확도 변화가 seed30000 −1.382073%, seed30001 +0.000876%다. Phase variance 감소는 +0.433635%/+0.060328%, canonical 성능은 두 seed 모두 개선됐다. 작은 양의 robustness 효과는 보존하지만 accuracy의 일관된 추가 가치는 확인하지 못했다. Standard보다 ordinary adapter도 두 seed에서 소폭 나빴다. 현재 final-hidden 분기의 추가 투자 우선순위를 낮추며 v3를 만들지 않는다. mixed 결과를 모든 phase-conditioning 아이디어의 반증으로 확대하지 않는다.
- **FR — FR_REOPEN_POSSIBLE.** 네 상태 모두에서 F0보다 나은 적응과 비퇴화 correction revision을 관찰했다. 과거 step0 진단만으로 FR 원리를 폐기할 근거는 약하지만, 여기서는 FR penalty를 적용한 학습 성능을 측정하지 않았다. 이번 우선순위 두 개에는 넣지 않았다.
- **Anchoring — ANCHOR_CONDITIONAL.** Macro+0.494126%, sparse32−0.013348%, dense233+1.001601%, interaction−1.014949pp. Beijing 양수, ETTm2 0, Electricity 음수다. 조건부 기준선으로 보존하며 신규 방법 기여로 세지 않는다.

## 재개하지 않은 항목

- Freshness v1/v2: 수정 후에도 simple feature 기준선 대비 추가 가치 부족.
- DualClock 추가 연장: 360→1440의 공정한 세 방법 비교가 이미 끝났음.
- Maturity: 복구된 구현에서도 quality/cost 불리.
- Block-shape: 실제 update는 있었지만 acceptance가 pooled/simple control을 넘지 못함.
- Calibration-weighted anchor: uniform/shuffled보다 추가 가치 부족; uniform anchoring과 구분.
- Drift: 현재 지시에서 재개 제외, 새 근거 없이 재학습하지 않음.
- Adaptive freeze/probe/overlap: 실제 비용을 포함한 강한 단순 대조에 불리.
- Global/local activation compression: 가까운 simple compression 대비 memory/gradient 절충이 불리.
- Conditional Path: 성능 반증이 아니라 기여 차별화·novelty collision 문제.
- Context distillation: teacher headroom이 없어 학생 학습 진입을 하지 않은 것; 학생 실패로 세지 않음.

## 검증과 한계

PatchPhase366개 캐시 독립 재계산 오차 최대1.67e-16, Anchor30개 캐시8.33e-17, 추가978개 독립 검산,98개 테스트 통과. 기존 결과1091개 해시 유지. Query는 모든18개 설정의 수치 동등성 검사를 통과했다. Censor per-position gradient tensor 자체는 저장하지 않았으며 실제 원래 train sampling count와 batch별 수치 집계를 독립 검산했다. 원천/seed/선택/재사용 데이터 범위를 유지했다.

[전체 상태](STATUS.md) · [후보별 표](METHOD_REOPEN_MATRIX.csv) · [자원 그림](../../results/reopen_query_resource_20260914/quality_memory_time.png) · [독립 검산](verification.json)

이 보고서를 끝으로 STOP. 추가 후보·v3/v4·자동 후속 실행은 하지 않는다.
