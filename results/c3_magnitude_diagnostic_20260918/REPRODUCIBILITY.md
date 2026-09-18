# 재현 범위

기준 저장소 커밋은 c07d4af, 최초 진단 봉인 커밋은99493c5다. PROTOCOL.md에 고정한 0-update 범위로 수행했다. 코드 추출/집계 오류의 원본과 수정 이유는 IMPLEMENTATION_CORRECTION.md와 두 과거 봉인에 보존했다. 수정은 모델·과학적 조건을 변경하지 않는다.

현재 로컬 캐시가 있는 저장소 루트에서 다음 명령은 저장 예측의 CPU 분석과 검산·보고서 생성만 수행한다.

```bash
.venv/bin/python -m experiments.c3_magnitude_diagnostic_20260918.analyze
.venv/bin/python -m experiments.c3_magnitude_diagnostic_20260918.audit
.venv/bin/python -m experiments.c3_magnitude_diagnostic_20260918.report
```

runner는 실제 GPU 교차 예측을 생성한 실행기다. 학습은 없으며 완료 예측은 해시를 검증해 재사용한다. 결과를 읽기 위해 다시 실행할 필요가 없다. 원래 A/D 예측36개와 교차 B/C 예측36개, 선택된12가중치의 경로·해시는 REUSED_PREDICTIONS.json, PREDICTIONS.json, MODELS.json 및 SEAL.json에 있다.

INPUT_FEATURES.csv.gz의 channel은 패널 내 0부터 시작하는 위치다. CHANNEL_CONTRIBUTIONS.csv는 부모 DATA_MANIFEST의 실제 selected_columns ID를 사용한다. 각 행의 origin/draw/조건을 보존했고 같은 날짜를 독립 표본으로 늘리지 않았다.

DECOMPOSITION.csv의 양수 total/gate/weights는 MAG 방향의 nMAE 감소다. 분해의 합을 표시할 때는 원래 C3 오차를 공통 분모로 사용했다. 각 항의 구간도 같은 고정 분모로 표시한 설명용 구간이다. 기존 보고서에서 C3/MAG 비율을 쓴 수치와 분모가 다름을 유의한다. seed 기여율은 signed 항의 합에 대한 값이며 보편적 변수 중요도가 아니다.

새 예측을 채점하기 전에 ALL_PREDICTIONS_SAVED.json을 남겼다. scalar metric은72개 새/재사용 view의 모든 조건·draw에서 첫/마지막 원점·채널을 검사한5,472항목×2metric이며, 모든 원점을 scalar로 재생했다는 뜻은 아니다. 전체 원점은 벡터 점수로 집계했고, 분해·층화·계열 가중합을 별도 검산했다.

원자료·모델·전체 예측은 ignored 로컬 캐시에 있으며 GitHub에는 코드·해시·원점별 오차·검산·보고서·그림을 공개한다. GitHub만으로 전체 모델 재생이 가능하다고 주장하지 않는다. 기존 파일은 덮어쓰지 않았다. 원고는 이번 해석 메모와 함께 읽어야 하며, 과거 원고 자체의 전면 개정은 이번 영향 진단 범위가 아니다.
