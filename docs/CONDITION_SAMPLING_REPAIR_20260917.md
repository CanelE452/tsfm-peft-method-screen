# 조건 실험 날짜 다양성 교정 실행 기록

단일 계약: Downloads/condition_sampling_repair_cli_20260917.txt. 기준 b80e4a4. 이전 실험은 read-only reference로 보존한다.

새 실행기: `scripts/run_condition_sampling_repair.py`의 prepare-all / run-all / status / resume-all / verify-all / report. GPU 명령은 `scripts/with_cuda.sh .venv/bin/python scripts/run_condition_sampling_repair.py run-all`이다. 기존 상태가 있으면 run-all 중복 실행을 거부한다. epoch64마다 optimizer/RNG/parameters를 저장하며 불확실한 update는 자동 재생하지 않는다.

GPU 사전 감사 결과 N02/N03만 모든 날짜·phase 조건을 통과했다. N01/R04/N07/R08은 TRAIN 날짜64개와 모든 phase24개를 얻었으나, 계약의 정확한 boundary circular tie 규칙 때문에 phase 개수1~4개(차이3)로 한도2를 넘었다. BLOCKED_DIVERSITY는 성능 실패가 아니다. 날짜·기간·phase를 바꾸지 않고 해당 트랙의 smoke/본학습/교차 평가를 생략한다.

전체 한도96fits/49,152main+48smoke 중 이번 적격 범위는32fits/16,384main+16smoke다. 남는 예산은 사용하지 않는다. 모든 적격 트랙의 smoke → LR 선택 경로 → 반복 경로 → CPU 선택 → 전역 평가 봉인 → 평가·교차 평가 순서를 지킨다. R05/N06/R09는 실행하지 않는다.

[원점 사전 감사](../results/condition_sampling_repair_v1_20260917/ORIGIN_REPAIR_AUDIT.md), [전체 보고서](../results/condition_sampling_repair_v1_20260917/MASTER_REPORT.md), [실행 상태](../results/condition_sampling_repair_v1_20260917/QUEUE_STATUS.json), [최종 결정](../results/condition_sampling_repair_v1_20260917/FINAL_DECISION.md).

교차 평가는 기존 가중치/새 E와 새 가중치/기존 E를 모두 수행하되 추가 optimizer0회다. 미래 정답을 선택에 쓰지 않으며 기존 weight가 없다면 복원 재학습을 하지 않는다. 큰 cache는 로컬에 남고 GitHub에는 보고서·원점수·manifest·검산 기록을 게시한다.

## 최종 완료

적격32fits/16,384 본업데이트·smoke16, 양방향 교차 평가를 완료했다. 30,416 nativeforward 장부와 기존2,952파일 hash 보존을 검산했다. N02/N03는 POSITIVE_UNCERTAIN + SAMPLING_SENSITIVE; 나머지4트랙은 BLOCKED_DIVERSITY다. 재개 후보0개이며 추가 학습은 없다.
