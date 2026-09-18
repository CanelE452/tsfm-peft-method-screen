# 지속 변화와 추가 PEFT — 논문 준비 근거 묶음

사용자 요청에 따라 필요한 직접 비교와 시각화를 수행한 자료다. 원고만 작성한 작업이 아니며, 투고·게재 완료를 뜻하지 않는다.

- **논문 성립 가능성 검토(2026-09-19):** [기여·선행·남은 작업](../../research/paper_viability_review_20260919/REVIEW_KO.md) — PEFT 분석 원고의 근거와 새 방법의 우위를 구분했다. 아래 본문과 두 보충은 아직 하나의 최신 제출 원고로 통합되지 않았으며, 신규성과 독립 확인 범위가 남은 핵심 약점이다.
- **최신 내부 기전·논문 보충:** [PDF·DOCX·한국어 해석](internal_mechanism_v1_20260918/README.md) — 추가학습0회, 초기 gradient·160 checkpoint·192 E view 검산 완료. 같은 patch 표현에서도 B0의 역전파 신호와 초기 특징의 결합이 달랐고, C3/MAG 모두 학습 B0와의 조합 의존성을 보였다. 초기 미분 경로와 최종 예측 차이의 완전한 인과 설명은 구분한다.
- **최신 통제 실험·논문 보충:** [PDF·DOCX·한국어 해석](training_factorial_v1_20260918/README.md) — 신규24fits·기존8재사용, 192평가·독립검산 완료. 고정1,024 updates에서 B0×초기값 결합, V 선택에서는 B0 주효과가 가장 큰 점추정이었다. 아래 원고의 미분리 학습 요인 한계를 두 수준에서 보완했으며 범용 방법 우위는 주장하지 않는다.
- **이전 한국어 원고(요인 분리 이전):** [PDF](manuscript_v2_20260918/MANUSCRIPT_KO.pdf) · [DOCX](manuscript_v2_20260918/MANUSCRIPT_KO.docx) · [원고·논리·검산](manuscript_v2_20260918/README.md) — 설명 대조와 교차 진단, 초기 가설과 실제 결과의 불일치 및 아래 영향 감사를 반영했다. 투고 형식 적용 전 원고다.
- **최대 영향의 식별 범위:** [학습 실행·원점·계열 감사](../../research/c3_influence_audit_20260918/REPORT.md) — 추가 학습·추론0회, 60조건 검산. 평균 차이는 seed별 실행에 민감하지만 B0·초기화·순서가 함께 바뀌므로 최대 인과 요인은 미식별이다.
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
