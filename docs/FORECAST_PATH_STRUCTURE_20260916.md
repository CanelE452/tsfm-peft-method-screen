# 예보 경로 연결 직접 비교 — 실행 중

단일 계약은 [PROTOCOL](../results/forecast_path_structure_v1_20260916/PROTOCOL.md)이다. 기존 a5cbff3 결과는 보존했다. 새 TRAIN/분할/타깃/seed/POINT 통제로 기존 fits 재사용0이며, 모사된 as-of 입력 및 시점별 동일 값 노출 CPU 검사와 세 타깃×네 군의 실제 smoke24updates를 통과했다.

2026-09-16 실행 시작. 최대48fits(각512updates)의 직접 비교가 진행 중이며 이 문서 시점에는 TEST 성능을 채점하지 않았다. 결과에 따른 다른 군 중단·튜닝·추가 후보는 없다. 최종 완료 여부는 결과폴더 status/REPORT를 확인한다.

```bash
scripts/with_cuda.sh .venv/bin/python scripts/run_forecast_path_structure.py status
```

`run`은 준비/검사 후 학습→선택→보정/봉인→TEST예측→채점→독립검산→보고서를 수행한다. `resume`은 완료 epoch의 원자 resume state와 update log가 정확히 맞는 경우만 허용하고, 모호한 반영 step은 재실행하지 않는다. 원자료·가중치·예측은 로컬 ignoredcache에 있다.
