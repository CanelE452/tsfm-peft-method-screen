# 방법론 주장 보완: 가까운 선행의 직접 대조

현재 작업은 C3/MAG를 새로 튜닝하거나 이름만 바꾸는 작업이 아니다. 고정된 현재 구조의 방법론 주장을 검토하기 위해, frozen forecaster에 작은 입력·출력 보정을 붙이는 가까운 선행과 직접 비교한다. 결과가 좋아도 기존 지속성 고유 가치/독립 확인/신규성의 미확보 상태가 자동 해소되지는 않는다.

- [실행 계약](../../experiments/delta_adapter_comparison_20260919/PROTOCOL.md): 최대16fits·16,384main+8smoke updates, 동일 TRAIN/V, 기존 비교 가중치·예측 재사용.
- [공식 원문](https://arxiv.org/html/2601.20280v1), [공식 고정 코드](https://github.com/Anoise/Adapter/blob/0add06ea7b4d2e0a84c364a8be72eef2676a92f2/Adapter-X%2BY/experiments/exp_online_xy_add.py): 원본 class output/input-gradient parity 검사 완료. hidden512와 parameter 근접 hidden7을 모두 비교한다.
- 공개 cell의 Chronos 연결은 별도 변경이다. 512→64 output 차원, raw 단위에 대한 관측 평균/TRAIN sigma 변환, 분위수별 공유 output cell, 기존2pinball objective를 명시했다. 논문의 전체 benchmark/MSE·Y-only·분포 calibrator 재현으로 쓰지 않는다.
- `runner.py`는 안전검사 → smoke → 모든 학습 → LR/모델 선택 봉인 → 전체 E예측 저장 → 채점으로 이어진다. `completion_hook.py`는 실제 runner PID와 creation time을 검증해 종료를 기다린 뒤 CPU 검산·한국어 보고서·그림을 작성한다. 학습을 재시작하거나 새 실험을 만들지 않는다.
- 아직 실행 중이면 결과 파일이 없을 수 있다. 실제 단계는 [진행 상태와 후속 연결 snapshot](../../results/delta_adapter_comparison_20260919/RUN_PROGRESS_SNAPSHOT.json)에 기록한다. GitHub의 진행 snapshot은 실시간 상태가 아니며 새 push 시점의 값이다.

완료 뒤의 보고서: `results/delta_adapter_comparison_20260919/REPORT.md`, `FINAL_DECISION.md`, `PUBLICATION_AUDIT.json`. 완료 전에는 이 파일들이 존재한다고 주장하지 않는다. 모델·예측 배열은 ignored 로컬 cache, 결과·해시는 공개한다.
