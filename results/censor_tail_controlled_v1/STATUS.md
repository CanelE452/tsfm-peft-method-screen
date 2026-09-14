# Censor tail controlled v1 — 준비 완료, 실모델 실행 전

사용자 지시문에 따른 최대 6-fit 실행기와 꼬리 손실을 구현했다. 데이터·상한·스케일·샘플링·모델 해시 계약을 검사하고 CPU 검증을 통과했다. 실제 모델 smoke와 본학습은 아직 실행하지 않았다.

GPU 조회에서 RustDesk(`/usr/share/rustdesk/rustdesk`, PID394545)가 compute 유형으로 272MiB를 사용 중이다. 이번 지시문은 외부 compute가 없는 상태를 요구한다. 해당 프로세스만 예외로 허용할지 사용자에게 확인 중이며, 다른 프로세스를 종료하지 않는다.

- 실제 본학습: 0/6 fits, 0/2160 updates.
- 실제 모델 smoke: 0/6 updates.
- CPU 검사: [준비 검증 기록](prepare_cpu_validation.json).
- 데이터·소스 계약: [manifest](manifest.json), [고정 설정](config.json).
- 원 지시문: [PROTOCOL.md](PROTOCOL.md).
- 초기 prepare 이후 실모델 실행 전 코드 점검 보완은 manifest의 preflight_amendment에 원 해시와 함께 보존했다. 학습 설정·판정 기준은 변경하지 않았다.

일반 GPU 조건에서는 `preflight`, 이어서 `run`, `verify`, `report`를 순서대로 실행한다. RustDesk 예외가 명시적으로 승인된 경우에만 `preflight`와 `run`에 `--allow-rustdesk`를 사용할 수 있다. 원래 6-fit/6-smoke 상한과 외부 작업 보호를 유지한다. 이미 시작된 시도는 재시도하지 않는다.
