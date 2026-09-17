# 단일 계약과 사전 구현 명세

유일한 실행 계약은 [MASTER_CLI.txt](MASTER_CLI.txt)다. 다운로드 원본 이름은 additive_persistence_paper_cli_20260917.txt이며 해당 문서의 버전과 기준 commit, 58 fits/59,428 전체 updates가 사용자 요청과 일치한다. 첨부 reference_checks.py/ZIP은 발견되지 않아 사용자의 Python 직접 작성 승인에 따라 CPU 참조 검사 20개와 실제 runner를 로컬 작성했다. 이를 첨부 코드 복원이라고 주장하지 않는다.

기존 C3 수식·threshold3·window8·patch16·rank8·cap·원래 raw 입력을 유지한다. 새 구조는 없다. MEAN/ROTATE16/RECENCY와 C2는 같은 초기 adapter, 같은 source/seed B0, 같은 학습 순서를 사용한다.

학습은 기존 source의 세 번째 seed B0/core8 fits → 원래 source 기전 대조30 fits → ETTm2 B0/core20 fits 순서다. 총58×1024=59,392 main updates, 폐기용 smoke18그룹×2=36, 전체59,428 cap. 초기 foundation wiring을 확인하는 ETTm2 smoke와 이후 실제 학습된 B0에서의 main fit 초기 동일성 검사를 구분한다. smoke weights는 main에 사용하지 않는다.

원래 source의 LR/표본/변형 cache는 그대로 재사용한다. ETTm2 source key는 ettm2, 선택85550/반복85551~85553, 날짜 phase seed85700/85701/85702다. generator는 기존 v2 코드의 입력 source/path만 ETTm2로 변경한다. 모든 source의 학습 shuffle은 기존 rng(84100,source,seed,epoch)다. FP32/TF32off/dropout0, AdamW(.9,.999), eps1e-8, wd0, clip1, batch32, 1024updates, V 다섯 조건 동가중, checkpoint0/256/512/768/1024를 고정한다. OOM에만 micro32/16/8/4 누적을 사용한다.

전력 canonical ID는 원본의 0-based 열 번호 문자열이다. NUL을 포함한 sha256('additive-persistence-v1\0'+id)로 정렬한다. 미사용을 입증한 계열은0개이며 contract의 fallback에 따라 현재 B0 미학습 적격16개를 탐색적으로 평가한다. 계열 ID와 노출은 SERIES_PANEL_MANIFEST.json에 고정한다. target TRAIN sigma 접근은 허용하되 target-data-free라고 부르지 않는다. ETTm2도 과거 성능 노출 자료다.

평가는 selected 모델54개를 source별로 적용하며 전력 transfer에는 전력 모델21개를 그대로 쓴다. selected/fixed1024를 분리한다. fixed1024는 추가 적응의 마지막 상태이며 출발 C0는 동일 V-selected B0를 유지한다. 동일 가중치는 추론도 재사용한다. C2_SHRINK/C0_BIAS는 원래 네 계열 V로 source/seed별로 선택하고 transfer에는 그대로 사용한다. E 전체 예측을 먼저 저장한 후 한 번 채점한다.

형태 패널은 index순 등간격64 원점을 사용해 지정된8형태를 +/- 동가중으로 평가한다. PULSE의 동일 과거/다른 미래를 검산하기 위해 이미 정의된 SHIFT8_D32의 +/- 균형 짝도 저장한다(새 변화 형태나 학습 후보 아님). 기존 SHIFT8의 두 random draw는 부호가 항상 반반이지 않으므로, 이에 더해 기존 SHIFT8 입력·예측을 정확히 재사용한 PULSE_LEGACY_MATCH 표를 별도로 둔다. 이 표는 동일 원점/채널/draw의 과거·부호를 완전히 보존한다. 양 부호 균형 검사와 기존 draw 일치 검사를 섞지 않는다.

2000회 paired bootstrap은 86700+1000×panelID+10×conditionID+seed_scopeID. panelID는 electricity0/ettm1 1/ettm2 2/transfer3, 조건은 코드 statistics.CONDITIONS에 고정한다. 날짜7일 block을 모든 방법·계열·seed에 같이 적용한다. transfer에는 계열 재표집을 추가한 구간도 보고한다. 기전3대비는 nominal95%와 Bonferroni3 구간을 같이 쓴다. 서로 다른 목적의 종합 우승 점수나 임의 합격선은 없다.

본 실행 전에 MASTER_SEAL.json에 계약·자료/노출·가중치·구현 hash를 기록한다. main update마다 intent/journal, epoch마다 정확한 resume state를 저장하며 모호한 update를 재실행하지 않는다. 과거 결과·사용자 변경은 보존한다. 보고서와 주장 근거표 작성 및 검산 후 scoped commit/push하며 자동 추가 연구는 없다.
