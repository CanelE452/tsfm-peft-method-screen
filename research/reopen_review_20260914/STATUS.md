# Bounded PEFT reopen review

허용된 재검토를 완료했다. 과거 FAIL/STOP은 해당 고정 실험의 판정으로 보존하며 방법 전체의 반증으로 바꾸지 않았다.

- Anchor: ANCHOR_CONDITIONAL; macro +0.494126%, sparse/dense interaction -1.014949pp. 기존48 fits 재실행0.
- PatchPhase v2: PATCH_V2_MIXED; 12 fits/8640 updates 완료, smoke6 updates 별도.
- Query: QUERY_RESOURCE_FRONTIER; 신규 forecasting fits0, 부분 checkpointing의 함수·gradient·Adam 동등성 후 기존 quality와 결합.
- FR: FR_REOPEN_POSSIBLE; 신규 fits0. 실제 미래 타깃이 있는 사후 개발 쌍에서 진입 조건만 검사.
- Censor: 신규 fits0. 선택된 두 상태에서 원래360 train batches를 재생한 tail-gradient 비율 보고.

구체적인 후보 투자 우선순위와 tail materiality의 수치 해석은 FINAL_ASSESSMENT.md에 기록한다. 연구 가치·방법론 성공·진입 조건을 구분한다. 추가 연구는 자동 생성하거나 실행하지 않는다.

[Anchor](ANCHOR_REVIEW.md) · [PatchPhase](PATCHPHASE_V2_RESULT.md) · [Query](QUERY_RESOURCE_FRONTIER.md) · [FR](FR_ENTRY_DIAGNOSTIC.md) · [Censor](CENSOR_TAIL_DIAGNOSTIC.md) · [방법별 표](METHOD_REOPEN_MATRIX.csv)

실행 명령(이미 완료한 run은 재실행 금지):
```bash
scripts/with_cuda.sh .venv/bin/python scripts/run_patchphase_support_v2.py smoke
scripts/with_cuda.sh .venv/bin/python scripts/run_patchphase_support_v2.py run
scripts/with_cuda.sh .venv/bin/python scripts/run_reopen_query_resources.py
scripts/with_cuda.sh .venv/bin/python scripts/run_reopen_fr_censor.py fr
scripts/with_cuda.sh .venv/bin/python scripts/run_reopen_fr_censor.py censor
scripts/with_cuda.sh env CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests automation/research_followup/test_observe.py automation/overnight_followup/test_watch.py
```
