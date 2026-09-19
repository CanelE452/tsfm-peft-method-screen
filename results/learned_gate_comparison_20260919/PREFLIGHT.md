# 실행 전 점검 완료

사용자 `자동시작해`에 따라 기존 고정 MAG와 두 학습형 gate 대조를 실행한다. CPU 및 새 NESO55일 데이터 검사 완료. 실제 Chronos 4설정×2updates=8smoke 완료: 초기B0/adapter off 동일, gate update, 동결 가중치·buffer 보존, 복원 동일. 본학습은 최대16fits/16384updates이며 smoke를 재실행하지 않는다. 성능 결과는 아직 없다.

실행: `.venv/bin/python -m experiments.learned_gate_comparison_20260919.execute`. 실제 학습→선택 봉인→전체192예측 저장→점수→독립 검산→한국어 보고·그림→scoped commit/push를 한 프로세스에 연결했다. 오류나 GPU 문제는 원인과 완료 범위를 보존하고 자동 재학습/새 후보는 시작하지 않는다. 실행 중에는 status.json 및 로컬 `.cache/learned_gate_comparison_20260919/run.log`를 확인한다.

새기간55일/8주블록, 기존CSV 재사용, 독립source 아님. 정답 구간의 이전H1겹침0, context겹침허용. 공식GateRA전체 재현 아님. 고정계약은 experiments의 PROTOCOL.md.
