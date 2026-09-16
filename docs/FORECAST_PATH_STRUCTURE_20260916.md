# 예보 경로 연결 직접 비교 — 완료

단일 계약 [PROTOCOL](../results/forecast_path_structure_v1_20260916/PROTOCOL.md)에 따라48/48fits·24,576updates, smoke24updates, 봉인 평가와151,200개 scalar 검산을 완료했다. 기존fits재사용0, 추가resourceoptimizer0, 자동 후속 학습0이다. GPU controller61.77분, 세 타깃 각각75 TEST원점을 공통 채점했다.

PATH의 DROP 대비 raw이득1.826%는 관측됐지만, 같은 시점별 예보값 노출 POINT 대비 이득0.033%는 불확실했다. 보정 뒤 POINT 대비0.668%도 전체 구간이0을 포함하고 추가 타깃 둘만으로는−0.008%였다. T0/T1 양성과 T2 반전을 모두 보존한다. 최종 추천은 **추가 근거 미확보** 하나이며 새로운 PEFT 방법 성립을 주장하지 않는다.

- [한국어 REPORT](../results/forecast_path_structure_v1_20260916/REPORT.md)
- [FINAL_DECISION](../results/forecast_path_structure_v1_20260916/FINAL_DECISION.md)
- [독립 지표·선택·보정 검산](../results/forecast_path_structure_v1_20260916/independent_verification.json)
- [48개 완전 재개 상태·630개 집계 검산](../results/forecast_path_structure_v1_20260916/completion_audit.json)
- [감사와 실행 후 유지보수 이력](../results/forecast_path_structure_v1_20260916/AUDIT_CORRECTIONS.md)

```bash
scripts/with_cuda.sh .venv/bin/python scripts/run_forecast_path_structure.py status
.venv/bin/python scripts/audit_forecast_path_structure.py
.venv/bin/python scripts/check_forecast_path_publication.py
```

실제 학습·채점 소스는622d79c의implementation_seal hash에 해당한다. 완료 후 미사용 resume256/512 체크포인트 경계를 보완했으며 CPU 모의 중단 회귀검사만 수행했다. 새 학습은 없고 `run`은 기존 시도의 중복 실행을 거부한다. source hash가 달라지면 기존 선택·채점의 자동 재실행은 거부한다.

원자료·가중치·원시예측·augmentation은 ignored 로컬 cache에 있다. GitHub만으로 모든 수치 재현 파일이 포함됐다고 주장하지 않는다. SIMULATED_ASOF이며 T0의 이전 개발 노출과 이번 후속 시간 구간을 구분한다. 세 타깃이 세 독립 도메인이나 사전학습 비중복을 의미하지 않는다.
