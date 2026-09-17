# 실행/재개

` .venv/bin/python scripts/run_additive_b0_adapter.py run-all `

입력/기존결과 hash감사 → CPU/model checks → 12 smoke → seal → 12 selection fits →12repeatfits →new E prediction전체저장 + C0기존예측참조 →score →independent verify 순서다.

one worker/GPU lock. RustDesk만예외. SIGTERM/SIGINT는epoch32updates경계에서optimizer/RNG state를저장한뒤정지한다. Journal/update intent/epoch resume가불일치하면자동replay하지않는다. 이미완료된fit과예측은hash확인후재사용한다. C0학습을반복하지않는다.

C1은기존B0 LoRA만계속학습하며, C2/C3는B0가중치를모두동결한다. 시작모델이같고추가학습순서/입력/정답/선택규칙을공유한다. 초기B0동등성이학습후성능보장을뜻하지않는다.

E는이전실험에노출된개발기간이다. 저장된C0도같은paired source/seed checkpoint이며예전전체모델평균을대신쓰지않는다. weights/raw/prediction cache는local ignored, manifest/검산/점수/보고서를GitHub에공개한다.
