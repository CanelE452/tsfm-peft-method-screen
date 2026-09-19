# 시간 반응 보존 PEFT — 실행 전 검증 완료

상태: **사전 검사 완료·본학습 진행 단계**. 결과 보고서 또는 논문 성공 판정이 아니다.

CPU 부품 검사5개 통과, 두 source × 네 신규 군 ×2updates =16smoke updates 완료. 실모델 초기 B0 동일성, 기존 PLAIN 초기 어댑터 동일성, 고정 가중치·버퍼 불변, 두 어댑터 행렬의 실제 업데이트, 저장·복원 정확성, adapter off B0 동일성을 확인했다. smoke 최대 할당 VRAM은 0.504GiB였다. 외부 compute는 기존 허용 RustDesk만 있었다.

기존 PLAIN4개 경로의 20개 checkpoint와 TRAIN/V 데이터 hash를 검증했다. 175개 코드·입력·가중치 파일을 봉인했다. 신규16fits/16,384 main updates 및 전체 평가가 남아 있다. 상한에는 이미 완료된 smoke16updates를 별도로 포함한다.

분리 백그라운드 프로세스 PID1331278이 실제로 유지되고 본학습 중임을 같은 실행 권한의 psutil·GPU 목록·update ledger로 확인했다. 처음 제한 환경의 PID 조회가 이를 보지 못해 재실행을 시도했지만 GPU lock이 학습 전 차단했고 중복 optimizer update는 없었다. 이후 생존 확인은 원 실행과 같은 권한에서 수행한다. status.json과 UPDATE_LEDGER.jsonl은 로컬 실행 중 갱신되며 이 문서는 봉인·smoke 단계의 고정 기록이다.

실행: `.venv/bin/python -m experiments.temporal_response_peft_20260919.runner`. 완료 receipt는 재사용하고, 미완료 epoch나 해결되지 않은 optimizer intent는 임의로 반복하지 않는다. 원본 cache가 필요하다.

[고정 방법·비교·판단](../../experiments/temporal_response_peft_20260919/PROTOCOL.md), [CPU](CPU_TESTS.json), [실모델 smoke](SMOKE.json), [재사용](REUSE_AUDIT.json), [봉인](SEAL.json), [Time-PEFT 실제 코드 확인](PRIOR_CODE_AUDIT.json). 본학습 뒤 REPORT.md·FINAL_DECISION.md와 모든 원점수·자원 기록을 별도로 작성한다.
