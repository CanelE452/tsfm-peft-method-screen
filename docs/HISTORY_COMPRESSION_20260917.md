# 긴 이력 압축 PEFT 개발 파일럿

사용자 승인: 2026-09-17 “그렇게 해서 실험해줘”. 별도 후보 자동 탐색을 재개하지 않고, 직전 대화에서 제안한 예측 정보를 보존하는 과거 토큰 압축을 한정 비교한다.

- [봉인 프로토콜](../results/history_compression_v1_20260917/PROTOCOL.md)
- [정확한 수식·예산·판정](../results/history_compression_v1_20260917/PROTOCOL.json)
- [기존 SHORT/LONG 8 fits 재사용 감사](../results/history_compression_v1_20260917/REUSE_RECEIPT.json)
- [실제 입력 독립 검산](../results/history_compression_v1_20260917/independent_input_audit.json)
- [실제 GPU 학습·복원 검사](../results/history_compression_v1_20260917/smoke.json)

이번은 Traffic 4채널의 기존 개발 표본을 재사용한다. TRAIN/E 각 64 distinct index-days, V 32일, native Chronos-2 FP32 rank8·원점 순서·512updates·두 LR·선택 seed 하나/반복 seed 둘을 유지한다. SHORT/LONG 기준선은 동일 비교의 기존 학습을 검증 후 참조한다.

새 비교군은 STATS_SHORT, POOL, POOL_KD, LEARN, LEARN_KD다. 총 20개 신규 경로/10,240 본업데이트, 군별 2회씩 10개 폐기 smoke 업데이트로 제한한다. 교사 모델은 기존 LONG의 V 선택 상태로 고정하며, 새 교사 학습은 없다. 같은 TRAIN 정답으로 학습된 교사의 추가 정보·계산 비용을 명시한다. 모든 LR/checkpoint 선택을 봉인한 뒤 E를 채점한다.

원점·모델 조건을 결과에 맞춰 변경하지 않는다. 성능 gate로 다른 비교군을 건너뛰지 않는다. 단순 풀링/증류를 새 알고리즘으로 선언하지 않으며, 독립 source나 선행 재현을 수행한 것처럼 보고하지 않는다.

실행:

```bash
.venv/bin/python scripts/run_history_compression.py prepare
.venv/bin/python -u scripts/run_history_compression.py run
.venv/bin/python scripts/run_history_compression.py status
.venv/bin/python scripts/run_history_compression.py verify
.venv/bin/python scripts/run_history_compression.py report
```

`run`은 완료 경로와 해시가 검증된 예측을 재사용한다. 중단이 epoch 경계 밖이면 모호한 업데이트의 무검증 재실행을 거절한다. GPU는 RustDesk만 기존 사용자 승인 예외로 허용하며, 외부 학습·여유 메모리 부족 시 경계에서 대기한다. 학습 다음 평가·자원측정·검산·보고서가 동일 controller에 연결된다. 로그는 로컬 `.cache/history_compression_v1_20260917/controller.log`다.
