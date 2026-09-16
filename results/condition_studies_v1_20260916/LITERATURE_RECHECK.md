# 실행 중 선행 범위 재확인

실험 조건과 봉인된 LITERATURE_BOUNDARY 문서는 변경하지 않았다. 다음은 보고를 위한 서지·범위 추가 확인이다.

- N02: [PMLR의 RAFT 공식 초록·서지](https://proceedings.mlr.press/v267/han25d.html)를 다시 읽었다. 학습 과거의 유사 구간과 그 구간 뒤 관측값을 사용하는 원리는 이미 알려져 있다. 이번 8계수 보정은 RAFT 정식 모델 재현이나 그 논문 대비 우위 시험이 아니다.
- N06: [저자 기관의 correlated sample paths 공개 페이지](https://www.amazon.science/publications/efficiently-generating-correlated-sample-paths-from-multi-step-time-series-foundation-models)를 다시 읽었다. 기존 다중시점 모델의 주변분포에서 copula로 결합 경로를 만드는 접근이며 게재 표기는 NeurIPS 2025 TSFM Workshop이다. 메인 학회 논문이나 이번 CPU 결합의 신규성을 주장하지 않는다.
- R04: [출판사 페이지](https://www.sciencedirect.com/science/article/pii/S0169207025000615)의 검색 수록 본문에서 정확도/수정 안정성의 복합 손실 및 동적 가중 설명을 확인했다. 직접 페이지 열기는 HTTP403으로 실패했고 최종 발행 연도는 출판사 화면에서 확인하지 못했다. 따라서 봉인 기록의 2024 공개/2025 v2 표기를 유지한다. 이번 고정 TRAIN 혁신량 가중과 LoRA 비교는 정식 N-BEATS-S/TARW 재현이 아니다.

추가 학습·후보·데이터·학습률 변경은 없다. 선행 효과와 이번 수치를 결합한 성능 주장은 하지 않는다.
