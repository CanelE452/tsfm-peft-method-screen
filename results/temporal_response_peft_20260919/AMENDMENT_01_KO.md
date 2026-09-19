# 평가 예측 재사용 식별 보완 — 학습 정의 불변

E 평가·채점 전에 코드 감사에서 발견한 오류다. 어댑터 checkpoint hash만 같으면 기반 B0도 같다고 볼 수 없는데, 최초 evaluator에는 B0 seed 조건이 빠져 있었다. 기존 ETTm1 MAG_ONLY selected step0에서 seed81552의 adapter와 같은 hash를 갖는 B0 seed81551 예측이 먼저 매칭되는 실제 사례를 확인했다.

source·기반 B0 seed·방법·checkpoint hash·panel·kind를 함께 확인하도록 수정했다. 같은 어댑터를 다른 B0에 얹은 예측은 이제 거절한다. 이전 factorial 파일에서는 이 조건을 이미 확인하고 있었으며, 수정 대상은 이번 새 evaluator다. 기존 실험 결과를 바꾸지 않는다.

원래 봉인은 SEAL_INITIAL.json으로 보존하고 SEAL_AMENDMENT_01.json에 바뀐 파일 hash를 기록했다. 방법 수식, loss, LR, lambda, data, seed, checkpoint, metric, 판정 기준은 그대로다. 당시 새 E 예측0개·채점0개였으며, 완료5fits와 진행 경로32updates를 보존하여 총5,152updates 다음부터 재개한다. 추가 smoke나 main update 반복은 없다.

CPU 검사10개 통과. 실제 과거 manifest에서도 B0 seed81551의 충돌 항목을 거절하고81552의 올바른 항목만 남기는 것을 확인했다. 재개 검사: RESUME_AUDIT_01.json.

PLANNED_PAUSE_01.json의 USER_STOP_AT_EPOCH_BOUNDARY는 공통 실행기의 일반 signal 예외 이름이다. 이번 중단은 사용자의 중단 요청이나 성능 실패가 아니라 assistant가 평가 정확성 수정을 위해 요청한 계획된 일시 정지다.
