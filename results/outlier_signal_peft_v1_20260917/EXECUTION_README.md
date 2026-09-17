# 고정 비교 실행

사용자 직접 구현 승인 이후 CPU10검사·공개 모델98 series·smoke24 updates를 완료했다. run-all은 여섯 군×두 원천의 선택24 fits, LR봉인, 반복24 fits, 선택봉인, 모든 E예측 저장, E채점, 독립 검산, 한국어 보고서로 이어진다. 성능 부진을 종료 gate로 쓰지 않는다.

```bash
.venv/bin/python scripts/run_outlier_signal_peft.py status
.venv/bin/python scripts/run_outlier_signal_peft.py run-all
```

동일 worker가 실행 중일 때 run-all을 추가 실행하지 않는다. GPU lock 및 RustDesk-only 예외를 적용한다. 실행 기록은 `.cache/outlier_signal_peft_v1_20260917/run.log`, update 장부는 `optimizer.jsonl`이다. epoch resume와 journal이 불일치하면 조용히 추가 학습하지 않는다. 이미 완료된 fit와 제한 재현/smoke를 재사용한다.

보고서/상태 직렬화 오류 및 재개 checkpoint 일정 오류를 고쳤다. 각각 reporting_fix_receipt.json/resume_schedule_fix_receipt.json에 이전 source hash와 변경 이유가 있다. 원래 과학적 LR/seed/업데이트/선택 기준은 유지했다. 416-step에서 추가 계산된 V 진단은 원문 기록을 보존하고 선택 후보에서 제외했다. 완료 fit의 실제 선택은1024로 변하지 않았다. 모든 optimizer update를 보존해 재실행하지 않았다.

48 fits/49,152 main updates +24 smoke, 신규 cache50GiB, GPU실행 hard24h 상한이다. 체크포인트 선택은0/256/512/768/1024만 허용한다. 원자료·가중치·예측 cache는 로컬에 보관한다. 실험 중 코드 hash를 바꾸면 다음 fit 진입에서 차단되며, 변경이 필요한 경우 epoch 저장 경계에서 정지하고 원인을 별도 기록해야 한다.
