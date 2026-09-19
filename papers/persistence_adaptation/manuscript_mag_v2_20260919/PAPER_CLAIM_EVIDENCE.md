# 방법론 주장–근거–한계

| 주장 | 현재 근거 | 검증 | 확대하지 않을 범위 |
| --- | --- | --- | --- |
| 구체적 PEFT 방법 명세 | B0 동결, 원래 입력 보존, MAD 진폭 gate, 8,712개 추가 잔차 | 방법 수식·공식 구현·실제 초기 동일성/복원 검사 | 국소 residual bound를 최종 오차 보장으로 확대하지 않음 |
| 일반 잔차 대비 추가 효과 | 전력 전이 SHIFT8 +3.794%; NESO 후반 +1.941% | 동일 B0/학습 기회; seed·날짜 점수 보존 | ETTm1·원자료·오류·긴 변화 손해 존재 |
| 학습형 gate 대비 추가 효과 | 네 주 비교의 family4 구간 하한>0 및 모든 seed 양수 | 16fits 완료·독립검산 | 표현/초기 gate/파라미터 수가 달라 trainability만의 효과 아님 |
| PETSA 공개 cell 대비 추가 효과 | ADDITIONAL_BASELINE_SUPPORT_NOT_ESTABLISHED | 8fits/8192+4updates; family2 및 모든 보호조건 확인 | offline 이식·모든 E 재사용; 공식 온라인 전체 PETSA 우위 아님 |
| 기전 해석 | 고정 규칙 효과와 학습된 가중치 효과는 반대일 수 있음 | 과거 C3–MAG 교차 분해 및 B0 교환 | 지속성 식별·최대 원인·완전한 인과 매개 주장 안 함 |
| 신규성 | 관측 robust gate를 추가 내부 잔차에 적용한 구체적 조합 | GateRA/PETSA/δ/AIRA/온라인 Kalman과 범위 구분 | gating/outlier-aware PEFT 최초성·충분한 학술 신규성 미확정 |
| 자료 일반성 | 동일 Electricity의 추가 계열 및 NESO 시간 구간 | 모든 source/노출 이력·ETT 손해 보존 | 독립 source·실제 센서 사건 해결·범용 우위 미입증 |
| 자원 | 추가 MAG8712, Gate9225, PETSA19010 parameters | receipt의 시간·peak memory 공개 | B0 학습 비용 존재; timer가 달라 공정한 속도비 미확정 |

이번 원고 생성은 신규 학습0회다. 추가 PETSA 실험의8회와 기존 gate16회는 서로 별도 완료 예산이며 전체 연구 누적 비용을24회로 표시하지 않는다.
