# 실행·재개·감사

유일한 실행 계약: PROTOCOL.md. 이 문서는 실행 경로를 설명하며 과학적 설정을 바꾸지 않는다.

```bash
.venv/bin/python scripts/run_outlier_signal_followup.py run-all
```

순서: immutable v1 audit → optimizer0 diagnosis → matched TRAIN 준비 → CPU/actual-model 검사 → 24 smoke updates → source/data/settings seal → 24 LR-selection fits → 24 repeat fits → 전체 E prediction 저장 → scoring → 독립 검산. 단일 프로세스가 한 GPU lock을 잡는다. 허용된 RustDesk 외 compute를 감지하면 update boundary에서 기다리거나 자원 상태로 중단한다.

실행 상태: status.json, fits/*/receipt.json, optimizer.jsonl, GPU journal. V checkpoint마다 상태를 갱신하므로 status의 update 수는 실시간 journal보다 조금 늦을 수 있다.

종료 후 학습 없는 보고서 생성:

```bash
.venv/bin/python -m experiments.outlier_signal_followup_v2_20260917.render_analysis
.venv/bin/python -m experiments.outlier_signal_followup_v2_20260917.report
```

report는 검산 결과와 명시적인 scientific_decision.json을 요구한다. 과학적 판정은 임의의 1% gate로 자동 생성하지 않는다.

재개 원칙:

- 완료 receipt가 있는 fit은 반복하지 않는다.
- epoch32-update boundary의 resume.pt에 optimizer/RNG/학습 파라미터를 보존한다.
- optimizer step 전 intent, 실제 step 후 fsync journal을 남긴다.
- journal count와 resume.step이 다르거나 unresolved intent가 있으면 자동 replay하지 않는다.
- SIGTERM/SIGINT는 현재 epoch를 완료하고 exact state를 저장한 뒤 중단한다.
- 강제 종료 후 불명확한 업데이트를 지우거나 예산을 새로 시작하지 않는다.
- smoke 도중 중단도 임의 재실행하지 않는다.
- source/data seal이 바뀌면 학습을 차단한다. reporting-only 모듈은 selection/model을 변경하지 않는다.
- 이미 저장된 E prediction은 hash 확인 후 재사용한다.

원자료·model snapshot·checkpoint·predictions·진단의 전체 example cache는 .cache 아래 local ignored 파일이다. GitHub에는 실행 코드, manifest/hash, journal, 점수, 검산, 보고서를 공개한다. GitHub 파일만으로 local cache 전체를 복원할 수 있다고 주장하지 않는다.

이 runner는 다른 과거 트랙이나 새 후보를 호출하지 않는다. 48 fits 완료 후 자동 후속 학습은 없다.
