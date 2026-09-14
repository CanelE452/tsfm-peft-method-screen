# GPU 실행 예외 — 사용자 후속 지시

사용자는 RustDesk compute 점유(272MiB)를 알리고 예외 여부를 물은 뒤 “학습우선으로 해줘”라고 지시했다. 이 세션에서는 이를 기존 원격접속 프로세스와 함께 학습을 진행하라는 승인으로 적용한다.

허용 범위는 정확한 실행 경로 `/usr/share/rustdesk/rustdesk`이며 GPU 메모리 512MiB 이하일 때만 예외로 한다. 다른 외부 compute와 메모리 부족은 기존 대기 조건을 유지한다. 다른 프로세스를 종료하거나 드라이버를 변경하지 않는다. `resources_preflight.json`과 `resources_run.json`, GPU telemetry가 실제 적용을 기록한다.
