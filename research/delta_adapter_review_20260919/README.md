# 방법론 주장 보완: 가까운 선행의 직접 대조

**실행·평가·검산 완료.** 신규16fits,16,384main+8smoke,144prediction views를 완료했다. 전력16계열 SHIFT8에서 C3는 δ 두구성보다7.986~8.353% 좋지만 MAG보다0.413% 나빴다. 선행 대비 좁은 양성을 확인했으며 C3 지속성 규칙의 필요성과 신규성은 별도로 남았다. 모든 조건의 우위나 논문 PASS를 주장하지 않는다.

현재 작업은 C3/MAG를 새로 튜닝하거나 이름만 바꾸는 작업이 아니다. 고정된 현재 구조의 방법론 주장을 검토하기 위해, frozen forecaster에 작은 입력·출력 보정을 붙이는 가까운 선행과 직접 비교한다. 결과가 좋아도 기존 지속성 고유 가치/독립 확인/신규성의 미확보 상태가 자동 해소되지는 않는다.

- [실행 계약](../../experiments/delta_adapter_comparison_20260919/PROTOCOL.md): 최대16fits·16,384main+8smoke updates, 동일 TRAIN/V, 기존 비교 가중치·예측 재사용.
- [공식 원문](https://arxiv.org/html/2601.20280v1), [공식 고정 코드](https://github.com/Anoise/Adapter/blob/0add06ea7b4d2e0a84c364a8be72eef2676a92f2/Adapter-X%2BY/experiments/exp_online_xy_add.py): 원본 class output/input-gradient parity 검사 완료. hidden512와 parameter 근접 hidden7을 모두 비교한다.
- 공개 cell의 Chronos 연결은 별도 변경이다. 512→64 output 차원, raw 단위에 대한 관측 평균/TRAIN sigma 변환, 분위수별 공유 output cell, 기존2pinball objective를 명시했다. 논문의 전체 benchmark/MSE·Y-only·분포 calibrator 재현으로 쓰지 않는다.
- `runner.py`는 안전검사 → smoke → 모든 학습 → LR/모델 선택 봉인 → 전체 E예측 저장 → 채점으로 이어진다. `completion_hook.py`는 실제 runner PID와 creation time을 검증해 종료를 기다린 뒤 CPU 검산·한국어 보고서·그림을 작성한다. 학습을 재시작하거나 새 실험을 만들지 않는다.
- [완료 snapshot](../../results/delta_adapter_comparison_20260919/RUN_PROGRESS_SNAPSHOT.json)은 runner와 hook의 종료를 실제 process identity로 확인한 기록이다. 후속 학습은 실행하지 않았다.

[한국어 보고서](../../results/delta_adapter_comparison_20260919/REPORT.md), [최종 결정](../../results/delta_adapter_comparison_20260919/FINAL_DECISION.md), [기본 검산](../../results/delta_adapter_comparison_20260919/PUBLICATION_AUDIT.json), [bootstrap·seed·비용 검토](../../results/delta_adapter_comparison_20260919/COMPARISON_REVIEW_KO.md), [재검산 절차](REPRODUCTION_KO.md)를 제공한다. 모델·예측 배열은 ignored 로컬 cache, 결과·해시는 공개한다.

보고서 코드는 실행 후 변하는 hook 상태·진행 snapshot·로그를 완료 결과 해시에서 제외해 해시가 곧바로 무효화되는 것을 방지했다. 별도 CPU 비교 감사는 기본 감사에 의존하므로 순환 해시를 만들지 않는다. 봉인된 모델·학습·선택·평가 코드의 변경은 없다.
