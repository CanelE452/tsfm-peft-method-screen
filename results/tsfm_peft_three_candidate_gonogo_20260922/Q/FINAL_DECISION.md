# Q: GO_STANDARD_ONLY

이번 후보는 ‘실제 4bit 저장 제약에서 forecast sensitivity rank 배치가 좋은 초기화와 단순 mixed precision보다 미래 예측을 개선하는가’를 Electricity 개발 화면에서 시험했다. V로 선택한 직접 기준선은 seed92201 `Q_IO16`, seed92202 `Q_IO16`이며 후보의 TEST 개선율은 각각 -0.367%, -0.127%, 평균 점수 기준 -0.247%이다. 비용 조건은 48.08% BF16 bytes이며 median nMAE 변화는 기준선보다 +0.253%이다(양수는 손해). Q_IO16은 BF16 base의 53.79% 저장량으로 Q_FP 대비 primary 점수 손해가 약 0.05%여서 이번 압축 목적을 충족했다. Q_FORECAST는 48.08%로 더 작지만 추가 예측 개선은 없었다. QERA 계열은 balanced-factor 로컬 변형이라는 구현 범위를 적용한다. 판정은 **GO_STANDARD_ONLY**이며, 현재 forecast rank 배치 후보에 대한 후속 투자 근거는 확보되지 않았다. 신규성·논문 PASS는 판정하지 않았다.

- Mean gain: -0.247012%
- Seed gains: -0.366580%, -0.127359%
- 95% block CI: [-0.421501%, -0.173252%]
- Mechanical category: NO_GO_CURRENT
- Recommendation: NO_AUTOMATIC_FOLLOWUP

[전체 보고서](REPORT_KO.md) · [판정 근거](../EFFECTS.json)
