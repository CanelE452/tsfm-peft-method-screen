# C3 내부 기전 — 한국어 논문 보충

- [읽기용 PDF](PAPER_ADDENDUM_KO.pdf) · [편집용 DOCX](PAPER_ADDENDUM_KO.docx) · [원문 Markdown](PAPER_ADDENDUM_KO.md)
- [실험 보고서](../../../results/c3_internal_mechanism_20260918/REPORT.md) · [전체 조건](../../../results/c3_internal_mechanism_20260918/ALL_CONDITIONS.md) · [최종 판단](../../../results/c3_internal_mechanism_20260918/FINAL_DECISION.md)
- [독립 검산](../../../results/c3_internal_mechanism_20260918/INDEPENDENT_AUDIT.json) · [문서 검산](AUDIT.json)

새 학습 없이 기존32경로·160 checkpoint의 gradient와 B0 교환192개 E view를 분석했다. 초기 미분 경로를 확인했으나 최종 성능 차이의 완전한 인과 매개 분석은 아니다. 두 source의 재사용 개발자료이며 방법 신규성·범용 우위·논문 PASS를 주장하지 않는다. 기존 원고와 통제 실험 보충을 보존한 추가 결과다. SVG/PDF/PNG 그림3종은 figures에 있다. raw 데이터·가중치·예측·gradient 원벡터는 ignored 로컬 cache에 있고 hash manifest만 공개한다.
