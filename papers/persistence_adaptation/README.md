# 지속 변화와 추가 PEFT — 논문 준비 근거 묶음

사용자 요청에 따라 필요한 직접 비교와 시각화를 수행한 자료다. 원고만 작성한 작업이 아니며, 투고·게재 완료를 뜻하지 않는다.

- **최신 영향 진단:** [규칙과 학습된 가중치의 역할 분해](../../results/c3_magnitude_diagnostic_20260918/INTERPRETATION_KO.md) — 추가학습0회·교차평가36개. C3 규칙의 추론 효과와 MAG 학습 가중치 효과가 반대 방향이며, 한 seed의 영향이 컸다. 앞선 평균 비교를 “C3 규칙 자체가 무용하다”로 해석하지 않는다.
- **최신 학습 대조:** [세 설명 대조의 결과와 논문 주장 수정](CONTROLS_AND_CLAIMS_KO.md) — 30 fits·54개 평가 완료. C3의 전력 이득은 남지만 MAG_ONLY를 넘지 못해 연속성 고유 가치가 강화되지 않았다. 아래 원고/PDF는 이 결과 반영 전 버전이다.
- **이전 보완:** [남은 약점·새 기간 평가·조건부 기전 분석](WEAKNESSES_AND_TEMPORAL_CHECK_KO.md) — 60개 새 평가, 학습0회. 아래 PDF는 이 보완 이전 버전이므로 함께 읽는다.
- [한국어 검토 PDF](EVIDENCE_BRIEF_KO.pdf) · [편집용 DOCX](EVIDENCE_BRIEF_KO.docx)
- [보강 실험 결과와 논문에 쓸 수 있는 주장](EVIDENCE_REPORT.md)
- [그림9종과 표 사용 안내](FIGURE_GUIDE.md)
- [선행 대조와 신규성의 경계](RELATED_WORK_AND_SCOPE.md)
- [방법·실험 설정의 정확한 명세](METHODS_AND_PROTOCOL.md)
- [재현 절차와 자료 제공 범위](REPRODUCIBILITY.md)
- [주장–근거 대응표](CLAIM_EVIDENCE.md)
- [남은 제출 단계](SUBMISSION_READINESS.md)

이전 근거 묶음은 새 학습 없이 기존 가중치의 기전 대조와 기준선 평가를 수행했다. 최신 보완에서는 C3/B0를 그대로 두고 설명 대조 세 가지의 30개 학습 경로를 추가했다. 전체 수치는 `tables/`,편집 가능한 벡터 그림은 `figures/`에 있다. 검산 결과는 [이번 검증 기록](../../results/paper_readiness_20260918/VERIFICATION.json)을 참조한다.
