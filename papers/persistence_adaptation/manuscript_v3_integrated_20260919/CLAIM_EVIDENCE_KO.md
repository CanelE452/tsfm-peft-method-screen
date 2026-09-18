# 통합 원고 주장–근거 대응표

| 주장 | 본문 | 직접 근거 | 표현 한계 |
| --- | --- | --- | --- |
| C3는 일부 전력 변화에서 C2 위 이득이 있지만 단순 MAG 위 고유 우위는 부족 | 5.1–5.2, 표1–2 | T1_historical, T2_controls_all_intervals | ETT·원자료·오류의 손해를 함께 보고; 범용 우위 아님 |
| 최종 순위와 고정 가중치 gate 효과가 다른 사례가 있다 | 5.3, 표3 | T3_decomposition 및 원본 DECOMPOSITION.csv | 기존3seed selected 평균, 사후 함수 개입; E 최적 조합을 새 방법으로 선택하지 않음 |
| 이전 seed 묶음에는 여러 학습 요인이 들어 있다 | 5.4 | T4_seed_decomposition 및 기존 영향 감사 | 85.4%는 signed 평균 기여이고 초기화의 인과 기여율 아님 |
| 조건부 방법 격차는 B0·INIT·ORDER 결합과 선택에 민감 | 4.4, 5.7, 표6 | T6_factorial; 원본8셀 대조에서 contrast 재계산 | 두 수준, 셀 내 반복 없음, 최대 절대 점추정; 보편적인 최대 원인 아님 |
| 같은 시작 예측 아래서도 초기 특징·B0 신호가 gradient를 바꿀 경로가 존재 | 4.5, 5.8, 표7 | chain-rule140건; INITIAL_GRADIENT_DECOMPOSITION, T7_gradient | 알려진 국소 미분; 전체 최종 성능 매개나 새 이론 아님 |
| 학습 B0와 어댑터의 조합 의존성이 C3/MAG에 공통 | 5.9, 표8 | T8_pairing; RAW_SCORES의 균형2×2 집계 | fixed1024·두 B0, pulse/step 예외 보존; C3만의 지속성 효과 아님 |
| gradient·가중치 변화·함수 개입은 서로 다른 추정 대상 | 4.5, 6.1–6.3 | 세 분석의 설계·선택·분모를 분리 | 서로 다른 퍼센트를 합쳐 설명률로 만들지 않음 |
| 최근 결과의 일반 원리는 선행과 겹친다 | 2, 6.2, 참고문헌9–12 | Hayou2024, LoRA-GA, Lin2025, ReLoRA2026 preprint | 우리 설정과 정확히 동일한 실험이라는 주장도 하지 않음 |
| 공개 근거 검산과 독립 평가·외부 재현은 구분 | 부록B–C | EVIDENCE_MANIFEST, AUDIT 및 부모 기록 | 재사용 개발 E, 로컬 cache, 학습 장부와 원고 생성0회를 분리 |

새 표는 `tables/`, 모든 그림의 원본 대응은 EVIDENCE_MANIFEST.json에 있다. 이전 원고와 후속 보고서의 원점수는 변경하지 않았다. 이 표는 주장을 검토할 수 있게 연결하는 자료이며 논문 성공 판정표가 아니다.
