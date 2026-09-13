# 전체 STOP 이후의 1회 후속 연구

사용자 승인: "하고 나서 만약에 다 fail이면 안되는 이유를 분석해서 다시 주제 정해서 해줘"

현재 실행의 소스·프로토콜·결과는 수정하지 않고 별도 감시기를 추가했다.
감시 중에는 모델을 호출하지 않는다. 부모 큐가 완료되고 독립 검증이 성공한 뒤,
anchor/drift/distill 세 후보 모두 PILOT_STOP 또는 STOP_NO_TEACHER_HEADROOM일 때만
기존 대화의 정확한 session ID를 재개한다. 새 세션이나 하위 에이전트를 만들지 않는다.
현재 대화의 실행 중인 turn이 끝나고 60초간 변경이 없을 때까지 기다린다.

하나라도 PILOT_PASS이면 새 연구를 자동 시작하지 않는다.
오류·미완료·검증 실패는 과학적 FAIL로 간주하지 않으며 감시 상태에 원인을 남긴다.
취소·시간 초과된 원래 큐는 재시작하지 않는다.
기존 프로세스 종료와 GPU lock 해제를 확인하고 finalize_overnight.py --verify-only를 다시 통과해야 한다.

후속 지시는 continuation.md에 고정한다. 실제 결과로 원인을 분석한 후 주제를 최대 2개 선정하고,
구현·검증·실행까지 하도록 한다. 최대 64 fit attempts, 후속 GPU 큐 최대 8시간이다.
가설 자체가 construct/novelty/headroom 검사에서 반증되면 그 진단을 기록하며 억지 학습하지 않는다.
기존 FAIL의 기준을 낮추거나 기존 결과를 덮어쓰지 않는다.
학습 데이터와 평가 구간의 새로운 설계를 별도로 봉인한다.
현재 구현·실행에서 과거 E는 이미 노출된 development라는 사실을 유지한다.

재개 명령은 현재 로그인과 사용자 설정·실행 규칙을 그대로 사용한다.
workspace-write 및 on-request 승인 제어를 사용하며 bypass 옵션은 없다.
CLI 실행 형식은 [OpenAI 공식 non-interactive 문서](https://learn.chatgpt.com/docs/non-interactive-mode)의
특정 session ID resume 기능과 로컬 codex exec resume --help로 확인했다.
실제 Codex 재개는 조건 충족 전에는 실행하지 않으므로 사후 인증·도구 실행 결과는 미리 보장하지 않는다.
재개 오류는 CONTINUATION_ERROR에 기록하고 자동 재시도하지 않는다.
CLI가 0으로 끝났더라도 후속 과학 실험이 끝났다고 판정하지 않는다.

감시기는 최대 12시간 기다리며 한 번만 재개한다. 재개 turn의 프로세스 상한은 2시간이다.
분리된 후속 학습 큐는 자체 8시간 상한을 지켜야 한다.
같은 조건으로 또 다음 자동 연구를 예약하는 재귀 동작은 금지한다.

상태 확인:

    .venv/bin/python automation/overnight_followup/watch.py status

후속 자동 재개만 취소(원래 실험 큐는 계속됨):

    .venv/bin/python automation/overnight_followup/watch.py cancel

로컬 상태:
- .cache/overnight_followup/status.json
- .cache/overnight_followup/trigger.json — 실제 조건 충족 후에만 생성
- .cache/overnight_followup/continuation.jsonl — 재개가 실행된 경우 로그
- .cache/overnight_followup/final_message.md — 후속 turn의 최종 응답 사본

후속 연구 요약 저장 위치는 automation/overnight_followup/FOLLOWUP_RESULT.md로 지시했다.
이 파일은 아직 연구가 실행되지 않았다면 존재하지 않는다.
개인 session ID와 경로는 ignored .cache의 config에만 저장한다.
