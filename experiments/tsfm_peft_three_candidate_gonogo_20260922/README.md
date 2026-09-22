# Three-candidate TSFM PEFT screen

단일 최상위 계약은 [MASTER_PLAN.txt](contract/MASTER_PLAN.txt)이다. 사용자는 별도로 실행을 승인했다. [고정 구현 규칙](PROTOCOL.md), [판정/검산 범위](DECISION_DETAILS.md), [원 실행 지시](contract/USER_EXECUTION_REQUEST.txt)를 함께 읽는다.

결과: [한국어 통합 보고서](../../results/tsfm_peft_three_candidate_gonogo_20260922/TRIAGE_SUMMARY_KO.md).

이 경로는 완료된 고정 예산 화면이다. 기존 MAG/HIER/rollout/FR 결과와 학습 checkpoint를 초기화에 사용하지 않았다. source seal과 update ledger가 있는 완료 실행에 학습 명령을 다시 호출해 덮어쓰지 않는다. 후속 실행에는 별도 승인·경로가 필요하다.

실제 실행 순서는 bootstrap/data audit → Q preparation/smoke/seal/train → T bridge packet/smoke/seal/teacher-cache/student → F smoke/seal/train → packed reload verification → fresh-process resource measurement → V selection seal → all TEST predictions seal → CPU scoring → plots/report/verification였다.

주요 모듈:

- `model.py`, `quantization.py`: native64·LoRA·packed NF4·QERA-diag·고정 rank allocation.
- `data.py`, `bridge_data.py`: 공식 Electricity 감사·time splits·student-safe BRIDGE packet.
- `run_q.py`, `run_t.py`, `run_f.py`: 서로 독립인 fixed-budget runner.
- `engine.py`, `common.py`: true-target 학습/검증, save/restore, 업데이트 호출 전 영구 장부 예약.
- `resource.py`, `verify_artifacts.py`: 별도 프로세스 비용과 실제 packed artifact 재현성.
- `evaluate.py`, `scoring.py`: V 선택, TEST 예측 전체 저장, 독립 metric/gain replay 및 paired bootstrap.
- `plots.py`, `report.py`, `verify_run.py`: 그림·한국어 보고서·hash manifest.

기존 Python3.11 환경은 읽기 전용 재사용했고, 이번 bitsandbytes wheel만 `.cache/<scope>/python_packages` overlay에 설치했다. 실제 버전은 결과의 requirements-lock.txt/ENVIRONMENT.json/UPSTREAM.json을 따른다. raw 데이터·HF weights·checkpoint·prediction cache는 Git에 포함하지 않는다. GitHub만으로 기존 수치의 직접 replay가 완결되지 않으며 로컬 cache 또는 독립적인 재실행이 필요하다.

`forensics/F_smoke_original.py`는 실패 source를 보존한 기록이다. 실행용 runner가 아니다. 실패한 smoke1 update도 전체24회 상한에 포함하며 F main 전에 수정했다.
