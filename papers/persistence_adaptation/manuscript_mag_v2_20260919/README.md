# MAG 한국어 방법론 통합 원고 v2

[PDF](MANUSCRIPT_KO.pdf) · [편집용 DOCX](MANUSCRIPT_KO.docx) · [Markdown](MANUSCRIPT_KO.md).

기존 v1을 보존하고 PETSA 공개 cell의 고정8회 비교와 추가 선행 검토를 통합했다. 실제 판정과 점수는 원고의 PETSA 표 및 원 실험 REPORT를 따른다. 결과에 맞춰 MAG를 변경하거나 추가 실험을 시작하지 않았다.

방법 수식과 제한된 경험적 이득을 설명하는 내부 방법론 원고이며, 충분한 신규성·투고 준비 완료·논문 PASS는 선언하지 않는다. PETSA는 공식 온라인 전체 방법이 아닌 offline 부품 이식이다. 이번 PETSA 평가의 모든 E는 이미 노출한 개발 자료다. AIRA/온라인 Kalman 선행과 구분되는 구체적 설계를 제시하되 넓은 gating/outlier-aware PEFT의 최초성은 주장하지 않는다.

저장소 root에서 `.venv/bin/python papers/persistence_adaptation/manuscript_mag_v2_20260919/build.py`와 `audit.py`로 생성·검산한다. Pandoc/Chrome/Noto CJK/poppler가 필요하다. 새 학습·모델 추론·bootstrap 없이 공개 표와 그림을 통합한다. 전체 실험 replay에는 공개되지 않은 로컬 raw/checkpoint/prediction cache가 별도로 필요하다. EVIDENCE_MANIFEST.json과 AUDIT.json에 출처 및 수치·렌더 검사를 기록한다.

현재 수치: PETSA cell 대비 SHIFT8 전력 전이 +0.969% (보정구간[0.372,1.507]), NESO +0.589% ([-0.583,1.669]). NESO 구간이0을 포함해 결합 기준은 미충족이다. [주장–근거](PAPER_CLAIM_EVIDENCE.md)와 [최종 판단](FINAL_DECISION.md)에 양성 근거·부정조건·미완료 범위를 구분했다. 11쪽, 그림4종, 원점수2240행을 포함한다.
