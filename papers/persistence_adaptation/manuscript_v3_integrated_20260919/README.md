# 한국어 통합 원고 v3 — 최종 성능과 관측 gate 효과의 불일치

기준 commit `7098e40`, 2026-09-19. 기존 원고 v2에 두 후속 연구와 선행 검토를 본문 구조로 통합했다. 보고서를 뒤에 단순 연결한 자료가 아니다.

- [읽기용 PDF](MANUSCRIPT_KO.pdf) · [편집용 DOCX](MANUSCRIPT_KO.docx) · [Markdown 원문](MANUSCRIPT_KO.md) · [HTML](MANUSCRIPT_KO.html)
- [주장–근거 대응표](CLAIM_EVIDENCE_KO.md) · [통합 변경 내역과 남은 작업](INTEGRATION_NOTES_KO.md)
- [검산 기록](AUDIT.json) · [원본 출처 manifest](EVIDENCE_MANIFEST.json)

중심 질문은 최종 방법 순위와 gate의 고정 가중치 효과가 왜 다른가이다. 기존 C3/MAG 결과 뒤에 학습 factorial, 초기 gradient 및 B0 교환을 연결했다. 본문은 좁은 통제 실증의 근거를 보고하며 신규성 확보·범용 새 방법 우위·논문 채택을 선언하지 않는다. 기존 원고·보고서·음성 결과는 보존했고 통합 과정의 학습·추론·새 bootstrap은0회다.

표9개, 그림8개(PNG/PDF/SVG), 참고문헌12개를 포함한다. 원자료·checkpoint·전체 예측은 기존과 같이 로컬 cache이며 GitHub만으로 전체 추론이 재현된다고 주장하지 않는다.

## 재생성

저장소 root에서 기존 공개 근거가 보존된 상태로 실행한다. pandas/numpy와 Pandoc(`/home/minjae/anaconda3/bin/pandoc`), Chrome, Noto Sans CJK KR, poppler가 필요하다. 환경별 binary 경로는 `build.py`에서 명시적으로 조정한다.

```bash
.venv/bin/python papers/persistence_adaptation/manuscript_v3_integrated_20260919/build.py
.venv/bin/python papers/persistence_adaptation/manuscript_v3_integrated_20260919/audit.py
```

이 두 명령은 모델을 호출하지 않는다. MD는 `MANUSCRIPT_TEMPLATE_KO.md`와 기존 CSV에서 생성되므로 본문 수정은 template에서 한다. 기존 v2의 표1–5는 그대로 보존하며 새 표6–8은 직접 CSV에서 생성한다. 표9의 과거 계산 장부는 부모 실행 기록과 검산한다. `AUDIT.json`의 완료는 원고 통합·자료 일치의 완료이며 독립 확인 실험이나 제출 준비 전체의 완료가 아니다.
