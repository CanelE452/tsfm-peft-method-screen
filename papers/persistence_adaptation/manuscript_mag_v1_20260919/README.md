# MAG 한국어 방법론 통합 원고 v1

[PDF](MANUSCRIPT_KO.pdf) · [편집용 DOCX](MANUSCRIPT_KO.docx) · [원문 Markdown](MANUSCRIPT_KO.md) · [HTML](MANUSCRIPT_KO.html).

2026-09-19까지 완료된 MAG/PLAIN/학습형 gate와 기존 δ 비교를 바탕으로 작성한 내부 검토본이다. 기존 분석 중심 원고를 덮어쓰지 않았다. 방법 수식·알고리즘·관련 연구·실험·부정 결과·자원·노출 한계를 하나의 원고에 통합했다. 저자/소속/투고처는 미정이며, 이 파일 생성은 방법론 신규성 확보나 투고 준비 완료를 뜻하지 않는다.

핵심 주장은 학습된 B0와 원래 관측을 유지하면서 고정 진폭 gate로 추가 잔차를 제한하는 설계가 특정 큰 합성 지속 변화에서 추가 이득을 보였다는 것이다. 최초 gating, 지속성 식별, 실제 사건 해결, 공식 전체 방법 대비 우위, 독립 원천 일반화는 주장하지 않는다. PETSA 부품 대조는 준비·승인 대기 중이며 성능을 만들어 넣지 않았다.

현재 남은 과학적 약점은 가까운 정식 선행과의 비교, 독립 검증, 충분한 신규성이다. 준비된 8회 PETSA cell 비교도 offline 통제 이식과 기존 개발 E 재사용에 한정되므로 그 하나가 모든 약점을 없애지는 않는다. 기존 승인 한도가 끝나 추가 학습은 실행하지 않았다.

재생: 저장소 root에서 `.venv/bin/python papers/persistence_adaptation/manuscript_mag_v1_20260919/build.py`, 이어서 같은 경로의 `audit.py`를 실행한다. pandas/numpy, Pandoc, Chrome, Noto Sans CJK KR, poppler가 필요하다. build.py는 도구 경로를 명시한다. 표·그림의 출처는 EVIDENCE_MANIFEST.json과 AUDIT.json으로 추적한다. 기존 공개 점수/그림으로 원고는 렌더링할 수 있지만 전체 학습·예측 재현에는 별도 로컬 캐시가 필요하다.

본 원고는 PETSA 승인 대기 시점의 스냅샷이다. 상태가 바뀌면 본문·manifest를 명시적으로 갱신해야 하며, build.py는 pending 상태를 조용히 완료로 바꾸지 않는다.
