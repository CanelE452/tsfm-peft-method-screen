# 완료된 새 건물 전이 실험

[한국어 REPORT](../../results/building_transfer_subspace_v1_20260915/REPORT.md) 및 [해석](../../results/building_transfer_subspace_v1_20260915/INTERPRETATION.md).

57 fits / 8,616 updates, STOP_NO_TRANSFER_SIGNAL. 추가36 fits 및 과거 heldout은 미실행. 중복 학습·자동 재시도 금지. RustDesk만 사용자 승인 compute 예외.

최초 실행 명령은 저장소에서 `scripts/with_cuda.sh .venv/bin/python experiments/building_transfer_subspace_v1_20260915/run.py prepare`, 이어서 같은 경로 `run`이었다. 이미 종료된 ID에서 다시 실행하면 봉인/원장 검사로 거절된다. 이 README는 재학습 명령이 아니다.

종료 후 **미실행 coefficient 분기**의 최강 비교군 목록에 LOCAL_FIXED120을 추가했다. 실제 실행 소스는 [snapshot](../../results/building_transfer_subspace_v1_20260915/executed_source/run.py), 정확한 한 줄 수정은 [patch record](../../results/building_transfer_subspace_v1_20260915/post_run_patch.json)에 있다. 현재 source와 실행 시 source가 다른 이유를 숨기지 않는다. 기존 seal은 변경하지 않았으며 finalizer는 snapshot의 원래 해시와 현재 파일의 한 줄 차이를 모두 확인한다. Snapshot은 provenance용으로 경로를 옮겨 보존한 파일이므로 그 위치에서 실행하지 않는다.

로컬 ignored cache가 있을 때의 검산:

```bash
scripts/with_cuda.sh .venv/bin/python -m unittest discover -s experiments/building_transfer_subspace_v1_20260915 -p 'test_*.py' -v
scripts/with_cuda.sh .venv/bin/python scripts/finalize_building_transfer_subspace_v1.py
.venv/bin/python scripts/analyze_building_transfer_subspace_v1.py
```

Finalizer는 CPU metric·selection·원본 결과 해시 확인과 GPU fresh-model forward 2건을 수행하며 optimizer update는 없다. GPU가 다른 학습에 사용 중이면 안전 감시로 대기/중단한다. 분석 스크립트는 scores.csv만 읽으며 추가 학습이나 판정 변경을 하지 않는다. 외부 raw/model/cache가 없는 GitHub checkout만으로 numerical replay가 완결되지는 않는다.
