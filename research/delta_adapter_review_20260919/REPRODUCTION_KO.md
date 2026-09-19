# δ-Adapter 통제 비교의 재검산과 재현 범위

이 문서는 이번 고정 비교의 증거를 확인하는 절차다. 새 후보나 추가 학습을 승인하는 지시문이 아니다. 모델·원자료·예측 배열은 ignored 로컬 cache이므로 GitHub checkout만으로 모든 수치가 바로 재현된다고 주장하지 않는다.

## 공개된 것과 로컬 입력

- 코드와 계약: `experiments/delta_adapter_comparison_20260919/`. `SEAL.json`이 코드·기존 자료·B0 가중치·E 배열의 SHA256을 고정한다.
- 공식 선행: `PRIOR_CODE_RECEIPTS.json`에 원본 commit·URL·SHA256이 있다. 원본 XY class와의 CPU 검사는 공개 코드 `prepare.cpu_check()`에 있다. 공식 전체 benchmark 재현은 아니다.
- 학습 이력: `fits/*/receipt.json`, `validation.json`, `intent.json`, `UPDATE_LEDGER.jsonl`, `SMOKE_LEDGER.jsonl`. 완료 시 새16fits·16,384main+8smoke이며 총80개 선택 후보 checkpoint가 존재해야 한다.
- 모델 선택: `LR_SELECTION.json`, `MODEL_SELECTION.json`, `EVALUATION_SEAL.json`. seed81550의 두LR 비교 후 반복 seed81551/81552를 사용한다. checkpoint0도 사전에 허용된 선택지다.
- 예측: 기존96view는 `REUSED_PREDICTIONS.json`, 전체144view는 `PREDICTIONS.json`. 배열은 각각의 `path`에 있는 로컬 `.npy`다. 동일 checkpoint의 selected/fixed view는 같은 파일을 가리킬 수 있다. view 수를 독립 학습 수로 세지 않는다.
- 점수: `ORIGIN_SCORES.csv.gz`, `RAW_SCORES.csv`, `CHANNEL_SCORES.csv`, `EFFECTS.csv`. raw에는 원단위 MAE와 TRAIN scale로 정규화한 nMAE, normalized2pinball이 있다. FAULT는6개 POINT/BURST 상태 평균이다.
- 자원: `gpu_delta.jsonl`, `gpu_budget.json`, `RESOURCES.csv`. 보고서는 PyTorch peak allocated와 전체 GPU 사용량을 구분한다. `MATCHED_REPEAT_RESOURCES.csv`는 해시가 연결된 과거 대조군 receipt도 포함한다.

## 완료된 결과를 학습 없이 다시 검사하기

저장소 루트와 기존 `.venv`에서 실행한다. 아래 코드는 optimizer update나 새 모델 추론을 하지 않는다. 완료 표시가 없는 실행에 적용하면 중단한다.

```bash
OPENBLAS_NUM_THREADS=4 .venv/bin/python research/delta_adapter_review_20260919/finalize.py
OPENBLAS_NUM_THREADS=4 .venv/bin/python research/delta_adapter_review_20260919/audit_comparison.py
```

첫 명령은 로컬 checkpoint·prediction·봉인 파일을 읽고 실제 학습 장부, 선택, 복원 기록, 저장된 origin 점수와 effect를 검증한다. scalar 점수 검사는 원 runner의 `score.py`에서 실제 저장 예측과 평가 정답으로 이루어지며 `VERIFICATION.json`에 범위가 남는다. 두 번째 명령은 주 단위 합계 방식으로 bootstrap 구간을 별도 계산하고 기존 checkpoint hash에 맞는 자원 receipt를 연결한다. 시간 구간을 재표집한 신뢰구간은 두 seed와 현재 source에 조건부다.

점수 자체를 prediction과 label로 다시 계산하려면 로컬 배열이 모두 존재해야 하며, 봉인이 맞는 완료된 결과에서 `experiments.delta_adapter_comparison_20260919.score.score()`를 호출할 수 있다. 이는 CPU 재채점이고 새 학습이 아니다. 기존 결과 파일이 다시 기록되므로 원본 파일의 해시를 먼저 보관한다. 전체 재채점 후에는 두 검산 명령을 다시 실행해야 한다.

## 실행·중단·재개

원 runner는 안전 확인→smoke→학습→선택 봉인→전체예측 저장→채점을 순서대로 실행한다. PID 기반 completion hook은 **같은 runner가 실제 종료**된 뒤 CPU 보고서만 작성한다. hook에는 학습 재시작 기능이 없다.

학습 재개는 동일 PID가 이미 실행 중인지 먼저 확인해야 한다. 타임아웃이나 잠깐 갱신되지 않은 파일만으로 재실행하지 않는다. epoch 경계의 optimizer/RNG/model resume 및 update 장부가 일치할 때만 기존 runner의 재개 로직을 사용할 수 있다. 부분 epoch·미확정 optimizer intent는 예산을 다시 쓰지 않도록 자동 재생을 거부한다. 현재 계약은 다른 후보·LR·seed·새 데이터 실행을 허용하지 않는다.

완료된 실행을 확인하는 데 runner를 다시 호출할 필요는 없다. GitHub에는 수치·시각화·검산의 검토 경로를 제공하고, 로컬 입력이 없는 환경에서는 미확인 부분을 명시한다.
