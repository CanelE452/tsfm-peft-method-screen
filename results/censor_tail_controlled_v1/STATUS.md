# Censor tail controlled v1 — 실제 학습 실행 중

본학습 전 CPU 검사와 실제 모델 smoke 6 updates가 통과했다. 세 arm의 초기 LoRA/F0 예측 일치, 고정 가중치 보존, V checkpoint reload 및 동일 task gradient를 확인했다. 본학습은 최대 6 fits / 2,160 updates이며 자동 추가 학습은 없다.

- 최신 실제 진행 수: [status.json](status.json). 이 문서는 단계 설명이며 live count는 실행기가 기록한다.
- 사전검사: [preflight.json](preflight.json), [CPU 검사](prepare_cpu_validation.json).
- 고정 lambda: [lambda.json](lambda.json). 첫 16개 F0 train batches만 사용했으며 V/E로 조절하지 않는다.
- GPU 예외 근거: [사용자 학습 우선 지시](GPU_AUTHORIZATION.md).
- [실행 프로토콜](PROTOCOL.md), [manifest](manifest.json), [고정 설정](config.json).

6개 V 선택을 모두 봉인하고 disk reload를 확인한 뒤 reused E_dev를 평가한다. 보고와 독립 검증까지 실행 명령을 연결했다. 완료 숫자와 성능은 종료 후 결과 문서에서 확정한다.
