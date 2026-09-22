# Residual feedback PEFT: 실행 계약 적용 기록

사용자가 승인한 [00_MASTER_CLI.txt](contract/00_MASTER_CLI.txt)가 유일한 실행 계약이다. 01_A/02_B는 참조 사본이며 설정을 합치지 않는다. A_AMORTIZED와 B_PARTIAL은 독립 학습·선택·평가하며 새 경로만 사용한다. 기존 결과·환경 패키지·Git 설정은 수정하지 않는다.

ZIP SHA256: cd5bc5265a08fee7127abedcc5d4b3679a2aad535b4d53babf1ef2ba4b44fc30. MASTER 및 참조 부품 bytes는 원본 그대로 보존한다. Git text 변환을 끄고 source seal을 학습 전에 만든다.

## 계약을 구체화한 구현

- 원본 column ID와 내부 48열 index는 SERIES_SPLIT의 대응으로 명시한다. packet/발행 장부의 series는 내부 index다. 정확한 ID 선택과 시간·packet hash는 data audit가 먼저 고정한다.
- G0 BASE_TRAIN은 합법한 매시간 origin, META_TRAIN 공통 packet은 8시간 grid에서 seed별 고정 표본이다. STATIC과 A/B 생성기 모두 같은512×4query를 사용한다.
- DEV checkpoint는 각 방법·seed의0/128/512 중 같은 primary 최소, 동점 이른 checkpoint. A_LOCAL/COEFF의 k는 두seed 평균DEV로 하나씩 고정하며 TEST에서도0/1/4/8 전체 trajectory를 저장한다. 선택된k만 primary다.
- baseline winner는 해당 track의 TIME을 제외한 모든 합법 비교법의 두seed 평균 DEV 최소다. 직접 SET 및 NOERROR/FULLGEN 비교는 별도로 항상 남긴다. A 효율 baseline은 LOCAL/COEFF 중 DEV 최소다.
- 생성기 입력은 prefix accessor만 받아 구성한다. Offline donor query 정답과 scorer 정답은 별도 접근한다. B_FULLGEN은 미완성 task residual/mask만0으로 하고 실제 age/visible_fraction metadata는 유지한다.
- B 고정 G0 reference는 각 logical issue의 과거context만으로 재현하고 불변cache/장부에 저장한다. DEV/TEST stream의 tstart−64부터 8시간 grid를 순서대로 채운 후 각 현재시각이 참조할 수 있는 issue만 사용한다. 과거 온라인 가중치가 있었다는 주장을 하지 않는다.
- q/v의 functional delta는 기존 LoRA forward를 대체한다. c를 Parameter로 복사하지 않으며 graph를 유지한다. COEFF 대조의 직접 최적화 c만 독립 Parameter다.
- GRU eval backward의 cuDNN 제한으로 cuDNN backend를 비활성화한다. dropout0/eval/FP32/TF32off는 유지하며 native parity를 다시 검사했다. 성능을 보고 바꾼 설정이 아닌 main 이전 구현 수정이다. 실패 전 smoke2회도 포함하여 검사누적14updates, 상한32 이내다.
- A는 episode마다 bank/c/optimizer reset, B optimizer 대조는 같은 series stream48ticks 안에서만 상태 유지한다. 생성기는 평가 중 동결한다. 발행 npz는 존재하면 덮어쓰기를 거절하며 장부는 flush/fsync한다.
- COSA는 고정 blob의 SimpleOutputAdapter class와 ADAPTER_TYPES를 그대로 추출했다. CC BY-NC-SA4.0 원문과 출처를 vendor에 보존한다. quantile9채널/고정clock/pinball 이식이며 원논문 전체 재현이 아니다.
- 현장 optimizer 비교를 포함한 모든 update는 intent/commit으로 센다. 정상 실행 최대 offline8192+A6144+B12288=26624, smoke32 상한. 실적은 장부로 다시 계산한다.
- read-only forward benchmark는 고정 첫 TEST context에서 각 snapshot5회, seed2는 방법순서를 반대로 한다. G0 support4 forecast 계산을 별도 측정해 각 생성기의 cold 비용에 더한다. Cache hit와 cold 비용을 구분한다. 추가 optimizer benchmark는 없다.
- 학습 source/config는 main 전 봉인한다. 별도 analysis 경로는 저장예측 채점·검산·보고서만 담당하고 추가 모델 학습을 하지 않는다. 자동 후속 실험은 없다.

## 실행

검증된 기존 Python3.11 가상환경에서 data audit → reference pytest → preflight.py → runner.py 순서다. 완료된 경로의 optimizer를 자동 재생하지 않는다. 하나의 숨겨진 executor에 로그/PID/독점 lock을 둔다. 정상 종료 후 분석·scoped commit/push와 remote SHA를 확인한다.
