# 학습 요인 분리 — 한국어 논문 보충자료

[읽기용 PDF](PAPER_ADDENDUM_KO.pdf) · [편집용 DOCX](PAPER_ADDENDUM_KO.docx) · [Markdown](PAPER_ADDENDUM_KO.md) · [문서 검산](AUDIT.json)

신규24 fits와 기존8 fits의2×2×2 요인 비교 결과다. 고정1,024 updates의 전력 전이 SHIFT8에서는 B0×초기값 상호작용이 가장 큰 점추정이고, V 선택 절차에서는 B0 주효과가 가장 컸다. 이는 선택한 두 수준과 해당 평가에 한정된 실증이며 신규 방법의 범용 우위가 아니다.

원점수·비용·코드는 [실험 보고서](../../../results/c3_training_factorial_20260918/REPORT.md), [최종 판단](../../../results/c3_training_factorial_20260918/FINAL_DECISION.md), [독립 검산](../../../results/c3_training_factorial_20260918/INDEPENDENT_AUDIT.json)을 참조한다. 그림2종의 PNG/PDF/SVG가 포함돼 있다. 기존 원고와81553·음성 결과는 보존했다.

저장소 루트에서 `.venv/bin/python -m experiments.c3_training_factorial_20260918.publish_paper`로 문서를 재생성한다. 원점수 생성에는 별도로 로컬 모델·TRAIN·예측 캐시가 필요하다.
