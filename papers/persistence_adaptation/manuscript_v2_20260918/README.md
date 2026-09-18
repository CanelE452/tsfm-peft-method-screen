# 한국어 원고 — 최신 설명 대조와 영향 진단 반영

**중심 주장은 ‘추가 PEFT의 최종 순위와 추론 규칙의 효과를 구분해야 한다’는 조건부 실증이다.** 최초의 지속성 가설이 그대로 지지되지 않았다는 사실과, 이후의 교차 분석이 사후 진단이라는 순서를 명시했다. 설명을 바꾸는 것만으로 새 방법 성공이나 신규성이 확보됐다고 쓰지 않는다.

- [읽기용 PDF](MANUSCRIPT_KO.pdf) · [편집용 DOCX](MANUSCRIPT_KO.docx) · [원고 Markdown](MANUSCRIPT_KO.md)
- [논문의 논리와 예상 질문 답변](ARGUMENT_MAP_KO.md)
- [최신 주장–근거 대응표](CLAIM_EVIDENCE_KO.md)
- [근거 파일·그림 해시](EVIDENCE_MANIFEST.json) · [원고 검산](AUDIT.json)

본문은 초록, 서론, 관련 연구, 방법, 설계·연구 이력, 결과, 논의, 한계, 결론과 부록을 포함한다. 그림4종과 표5개를 포함하며 각 그림은 PNG/PDF/SVG로 제공한다. 기존 보고서·PDF·원고는 보존했다. 새 모델 학습·추론·bootstrap은0회다.

저자·소속·투고처를 꾸며 넣지 않았으며 특정 학회 양식이나 익명화가 끝난 제출본은 아니다. 이후 투고 양식에 맞춰 참고문헌 형식·분량·데이터 제공 문구를 확정해야 한다. 한국어 원고와 동일한 수치·근거를 유지해 영문으로 옮길 수 있다.

재생성은 저장소 루트에서 `.venv/bin/python papers/persistence_adaptation/manuscript_v2_20260918/build.py`로 표·Markdown·HTML·DOCX를 만든 뒤 `render.sh`로 PDF를 만든다. 원본 CSV와 Pandoc/Chrome이 필요하다. `audit.py`는 수치·해시·그림·문서 내용을 검사하며 학습하지 않는다.
