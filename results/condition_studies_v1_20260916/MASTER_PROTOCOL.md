시계열 조건 중심 PEFT — 새·이전 후보 9개 순차 실행 계약
버전: condition_studies_v1_20260916 / 단일 최종본
작성일: 2026-09-16 KST
대상: CanelE452/tsfm-peft-method-screen
읽고 확인한 기준 commit: 485b15b3990236d0372fc074f1c80c7f1df057e2

======================================================================
0. 이 문서의 범위와 결정권
======================================================================

[설계] 이 파일 하나가 이번 배치의 실행 계약이다. 앞선 서로 다른 TXT들을 합치지 않는다.
최근 두 답변의 후보 목록을 아래 9개 ID로 통합했다. ID는 결과에 따라 바꾸지 않는다.

N01 ASYNC    일부 채널만 최근 관측이 지연됨 (Freshness의 새 조건)
N02 ARCHIVE  최근 입력창 밖의 관측 완료된 과거 사례 활용 (새 후보)
N03 CLOCK    관측 간격/밀도 변화와 불규칙한 시간 정보 (두 제안을 통합)
R04 REVISION 같은 미래의 예측 수정 안정성 (이전 FR의 새 직접 비교)
R05 VINTAGE  최신 예보 정확도와 오래된 예보 강건성 (기존 결과 재분석)
N06 JOINT    시점별 분포를 유지하며 하루 전체 확률 위험을 예측 (새 목표)
N07 SPECTRAL 작은 에너지이지만 예측 가능한 변동의 적응 (앞선 새 후보)
R08 LEAD     다른 채널의 선행 신호가 이미 관측된 예측 구간 활용 (이전 관찰)
R09 MIXED    상세/집계 관측이 섞인 학습에서 세부 패턴 보존 (이전 합계 감독)

실행 순서: N01 -> N02 -> R05 -> N06 -> N03 -> R04 -> N07 -> R08 -> R09.
R05/N06는 큰 새 신경망 학습보다 저장 예측의 저비용 직접 비교를 먼저 한다.
후보 하나의 성능 부진/로컬 구현 차단은 다음 후보를 막지 않는다.
공통 데이터 오염, GPU 안전 문제, 전역 예산 소진만 전체 실행을 멈출 수 있다.

[설계] 이번에 승인된 것은 '9개 완성 논문 방법'이 아니라 9개 제한된 증거 비교다.
각 실험에 구체적인 변경과 단순 대조를 아래에서 고정했다.
CLI가 결과를 보고 10번째 후보, v2/v3, 다른 loss/데이터/seed를 발명하지 않는다.
새 설계는 [미검증]이며 정확한 신규성을 확인하지 않은 것은 KNOWN_COMPONENTS로 둔다.
좋은 숫자만으로 NEW_METHOD/논문 PASS를 부여하지 않는다.

[설계] 대규모 공식 논문 전체 재현은 이번 예산에 포함하지 않는다.
정의한 단순 대조가 가까운 주요 논문의 전체 재현은 아니므로,
그 대조를 이겨도 해당 논문을 이겼다고 쓰지 않는다.
실제 후속 기여를 주장하려면 직접 선행 구현 비교가 남았다는 칸을 반드시 둔다.

용어:
- C는 모델에 주는 과거 입력 길이, H는 미래 예측 길이다. 실제 시간 단위도 함께 적는다.
- origin은 예측을 발행하는 시점이다. donor는 해당 타깃 예측에 참고하는 다른 채널이다.
- fit/학습 경로는 하나의 방법·설정·seed로 학습을 시작해 지정 횟수까지 진행한 기록이다.
- seed는 초기화와 표본 순서를 고정하는 난수값이고, 새로운 데이터 원천을 뜻하지 않는다.
- TRAIN은 학습, V_SELECT는 설정 선택, V_CAL은 허용한 보정 추정, E는 평가 역할이다.
- frozen은 가중치를 갱신하지 않는다는 뜻이다. 보조 입력과 forward 계산이 없는 것은 아니다.
- alias는 다른 이름의 군이 같은 계산/목적함수로 축소돼 한 결과를 참조하는 경우다.
- 나머지 세부 지표와 수식은 각 실험 절에서 정의한다.

재개 제외:
- 현재 PRIOR/SIDE 변형, BASIS 압축, Censor 꼬리 보정 재튜닝,
  Query의 BF16 배치 동등성, 기여 기반 freeze/probe, 기존 주말 coverage 규칙.
- 이는 분야 전체의 불가능 판정이 아니라 현재 배치와 연결되지 않는 기존 방법의 재실행 제외다.
- R04/R08/R09는 옛 결과를 덮어쓰는 재시도가 아니라 질문/정보/목적을 명시한 새 비교다.

실행 권한:
- 계획 명세, 순수 수학 검사, 모델 연결 검사 후 지정된 비교를 실제 실행한다.
- 계획서만 만들고 종료하지 않는다.
- 성능 gate로 직접 비교 진입을 막지 않는다. F0보다 LoRA가 먼저 나빠야 할 필요가 없다.
- 데이터/논리/수치가 유효하지 않으면 결과를 만들지 않고 BLOCKED_*로 기록한 뒤 다음으로 간다.
- 파일 삭제, 사용자 작업 reset/stash, 다른 프로세스 종료, 전역 환경/드라이버 변경 금지.
- 기존 승인된 scoped commit/push만 사용한다. 허가 범위를 넓히거나 private 자료를 새로 공개하지 않는다.

======================================================================
1. 최상위 목적, 목적 트리, 예상 결과
======================================================================

최상위 목표:
"예측 시점에 정의할 수 있는 시계열 조건에서, 이미 관측 가능한 정보를 더 잘 이용하는
작은 적응 변경의 추가 가치를 직접 비교하고, 후속 연구에 투자할 문제를 최대 2개 남긴다."
소비처: 사용자와 지도교수의 다음 연구 투자 판단.

목표는 네 가지로 구분한다.
- 정확도: N01/N02/N03/N07/R08/R09.
- 정확도를 크게 해치지 않는 수정 안정성: R04.
- 최신 성능을 제한적으로 보존한 지연 강건성: R05.
- 시점별 분포와 별개인 집계/연속 사건의 확률 예측: N06.
원점수를 다른 목표 사이에서 합쳐 하나의 우승 순위를 만들지 않는다.

행동별 이유(복수 목적):
A. 실제 정보 계약 확인:
   (i) 미래/숨긴 정답 누출 방지; (ii) 개선에 이용할 정보가 어디에 남는지 명시.
B. 직접 단순 대조:
   (i) 새 변경의 필요성 확인; (ii) 알려진 싼 방법으로 충분할 때 그것을 채택.
C. 반복 seed와 뒤 구간 평가:
   (i) 우연한 초기화와 기간 특성을 구분; (ii) 효과의 불확실성과 범위를 공개.
D. 모든 후보의 독립 실행:
   (i) 한 간접 가설 실패가 다른 문제를 닫지 않음; (ii) 후속 선택에 근거를 남김.
E. 저비용 R05/N06:
   (i) 이미 얻은 예측에서 새 목적의 가능성을 확인; (ii) 필요성이 약한 새 신경망 제작을 줄임.

제대로 달성하기 위한 조건:
- 문제/조건/방법/데이터를 각각 한 줄로 구분.
- TRAIN에서만 생성한 통계·검색 은행·시차·관측 마스크를 사용.
- 비교군별 정보 권한, 파라미터, 학습량, 입력 길이, 모델 호출 비용을 병렬 표로 제시.
- 후보가 단순 대조와 수학적으로 동일해지는 경우 먼저 탐지.
- 주 평가 대상과 부가 지표를 결과 전에 고정.
- 평가 정답으로 조건/타깃/설정을 소급 선택하지 않음.
- 한 후보의 결과를 보고 뒤 후보 명세를 바꾸지 않음: 모든 명세를 먼저 봉인.

역주행 판단:
예상 결과 1: 후보가 같은 정보의 단순 대조를 넘고, 겨냥한 조건에서 개선.
  목적 지지=지지 / 최상위 도달=부분 / 독자 질문="정식 선행보다도 필요한가?"
  행동=제한된 개발 근거로 남김. 신규성과 독립 확증을 별도로 표시.
예상 결과 2: 단순 복원/검색/보정만으로 충분.
  목적 지지=새 방법에는 불가, 실용 문제 해결에는 지지 / 도달=일부.
  행동=단순 방법 보존. 특별한 후보를 살리려는 후속 변형 금지.
예상 결과 3: 추가 정보가 도움이 안 되거나 효과가 작고 불안정.
  목적 지지=불확실 / 도달=미확정.
  행동=효과·부호·구간을 남김. 분야 전체 반증 또는 자동 성공으로 바꾸지 않음.
예상 결과 4: 후보와 대조가 같은 계산 또는 관측 조건이 주장을 시험하지 못함.
  목적 지지=불가 / 도달=안 닿음.
  행동=수학적으로 같은 군은 alias, 잘못된 실험은 BLOCKED_CONSTRUCT.

======================================================================
2. 실제 확인한 저장소 연결 / 아직 확인하지 않은 것
======================================================================

[확인] 기준 commit에는 forecast_path_structure_v1_20260916의 48경로 완료 기록이 있다.
PATH/POINT의 일반적 우위와 새 PEFT는 확보되지 않았고, 일부 타깃의 손익은 남아 있다.
[확인] 아래 파일 내용/경로를 읽었다.
- src/tsfm_peft_screen/backbone.py
- src/tsfm_peft_screen/lora.py
- src/tsfm_peft_screen/data.py
- docs/CANDIDATE_04.md
- research/overnight_20260913/data_receipt.json
- experiments/forecast_path_structure_v1_20260916/{common.py,data.py,runtime.py,evaluate.py}
- results/forecast_path_structure_v1_20260916/{REPORT.md,FINAL_DECISION.md}
- CanelE452/mltimeseries의 README.md (과거 방법/합계 감독 기록 연결)

[확인] 기본 Chronos-2 로더:
MODEL_ID amazon/chronos-2
REVISION 29ec3766d36d6f73f0696f85560a422f50e8498c
기본 quantile 21개, 중앙 슬롯 tau=.5, output patch16, 원래 head frozen.
기존 rank8 attach는 96개 q/k/v/o projection, 학습 파라미터 1,179,648개다.
전체 모델/라이브러리 버전, snapshot 해시, MODULES의 실제 이름은 시작 때 재확인한다.

[확인] 원자료 경로의 근거:
- data/raw/electricity.txt.gz: data.py에서 사용.
- data/raw/overnight_20260913/traffic.txt.gz: data_receipt에서 확인.
- data/raw/overnight_20260913/ETTm1.csv: 같은 receipt에서 확인, 15분 단위.
- data/raw/overnight_20260913/ETTh1.csv: 같은 receipt에서 확인, 1시간 단위.
원시 배열의 현재 존재/무결성은 이 문서 작성 단계에서 직접 검사하지 않았다.
로컬에 없으면 해당 receipt의 공식 URL과 hash로만 복구한다. 임의 데이터 대체 금지.

[확인] 기존 forecast() helper는 context336/horizon48을 가정한다.
N02 LONG1344와 R09 H24에서 그 함수를 억지로 호출하거나 잘라 맞추지 않는다.
실제 native model forward/preprocess를 읽고 새 run-local 범용 wrapper를 구현한다.
미확인 내부 변수/키를 추측하지 않는다. 아래 수학 기호는 새 API 명세이지 기존 키가 아니다.

[설계] 새 루트:
experiments/condition_studies_v1_20260916/
results/condition_studies_v1_20260916/
.cache/condition_studies_v1_20260916/
docs/CONDITION_STUDIES_20260916.md
scripts/run_condition_studies.py

기존 runner 전역 OUT을 monkey-patch하여 옛 결과 폴더에 쓰지 않는다.
mltimeseries/covariate-trust-pilot는 읽기만 한다. 이번 변경은 대상 저장소 새 루트에 한정.

======================================================================
3. 공통 데이터 계약 — 새 실험이지 이전 점수의 재현이 아님
======================================================================

N01/N02/N03/R04/N07/R08/R09는 단일 원천의 제한된 첫 비교다.
각 후보의 원천은 아래 절에서 고정했다. 결과를 보고 원천을 바꾸지 않는다.
전체 9개 후보를 비교한 결과는 탐색 증거이며 최종 논문 확증이 아니다.

일반 분할(N03은 15분 raw grid, 나머지는 원래 hourly grid):
유효한 원시 시간축 길이 N을 메타데이터/행 수로 확인.
b1=floor(.60N), b2=floor(.70N), b3=floor(.80N).
TRAIN=[0,b1), V_SELECT=[b1,b2), V_CAL=[b2,b3), E_DISCOVERY=[b3,N).
모든 horizon, R04의 두 번째 horizon까지 자기 역할 경계 안에 포함.
C보다 앞의 합법적 과거는 문맥으로 가능. 현재/미래 타깃으로 train/selection하지 않음.
원자료를 실제로 읽은 뒤 분할 경계와 허용 채널을 고정한다. 배열을 읽었다는 것과
그 뒤 구간의 정답을 학습·모델 선택에 사용했다는 것은 구분하여 기록한다.
N02 archive의 warmup은 별도 조건을 따른다.

변수: 원본 순서에서 TRAIN finite, TRAIN std>1e-6인 첫 4개 숫자 신호.
날짜 열/ID는 제외한다. 판독할 실제 열 이름은 원자료를 읽어 manifest에 기록.
자연 결측이 있으면 공통 complete-input/label 원점만 쓴다.
N01/N03/R09의 인위적 관측 가림은 이 자연 결측 처리와 구분.
평가 결과/상관으로 좋은 채널을 선택하지 않는다.

원점 수: TRAIN64, V_SELECT32, V_CAL32, E_DISCOVERY64.
필요한 문맥/label 경계를 만족하는 후보 시점에서 값과 무관하게 고정한다.
- hourly: source index mod24의 위상을 균형 있게 배치.
- 15min: mod96 중 64개 위상을 균등하게 정해 네 역할에 같은 roster 사용.
- 원본 날짜가 없는 Electricity/Traffic의 index 위상을 실제 현지 시각이라 부르지 않는다.
- R09는 집계 완료 경계에 맞는 mod24=0만 네 역할 모두 사용. 불일치가 없다.
- 후보별 각 위상 수와 날짜 분산을 audit. 고정 stride로 TRAIN/V/E 위상이 어긋나지 않게 한다.
- 동일 role의 예측 구간이 겹칠 수 있다. 독립 표본64*H개라고 세지 않는다.
- 최소 개수를 못 확보하면 BLOCKED_DATA. 숫자를 채우려고 기간이나 값을 바꾸지 않는다.

권장 결정적 선택 구현:
각 위상별 유효 origin을 시간순 정렬, quota를 같은 순서로 배분,
각 위상 내부 floor(linspace(0,n-1,quota))를 선택한다.
부족 위상의 잔여 quota는 사전 고정된 다른 위상 순서로 재배분하되 네 역할 분포를 공개.
어떤 role에도 그 위상을 확보할 수 없다면 공통 위상집합부터 다시 기계적으로 고정한다.
모든 결과 계산 전에 origins.json을 저장하고 hash를 봉인한다.

정규화:
원칙적으로 TRAIN 원시 값에서 채널별 mu,sigma를 고정한다.
R09는 원시 숨긴 상세값이 아니라 허용된 상세값/집계 복원값으로만 계산한다.
모델 입력을 외부에서 표준화할지 여부는 전 군 동일하게 고정한다.
기본은 raw target + native instance normalization; 외부 mu,sigma는 보조 계산과 지표에만 사용.
어댑터에서 사용하는 z=(x-mu)/sigma는 TRAIN 통계만 사용.

노출:
이 원천과 기간 상당수는 이전 프로젝트에서 이미 사용됐다.
exposure_ledger에 기존 결과/스키마/값 열람을 적고 원천 비중복을 주장하지 않는다.
Chronos 사전학습 비중복도 미확인이다.
9개 비교의 E는 이번 명세 안에서는 봉인 평가지만 연구프로그램 전체에서는 개발 평가다.

======================================================================
4. 공통 모델·학습 계약 / 예산
======================================================================

[설계] 7개 학습 실험은 '점예측'을 기본 문제로 삼는다.
원래 Chronos의 tau=.5 출력 슬롯을 점예측 yhat으로 읽고, 아래 MSE로 학습한다.
학습 후 이 슬롯을 정확한 조건부 중앙값이라고 주장하지 않는다.
다른 quantile은 지도하지 않으므로 학습된 모델의 확률 calibration 개선을 주장하지 않는다.
N06는 이 점예측 학습 모델을 쓰지 않고 기존 확률예측을 사용한다.

L_task=mean_{supervised target channels,h} [ (yhat-y)/sigma_train ]^2.
원래 head/backbone은 frozen, rank8 LoRA만 학습 + 해당 절의 작은 명시 모듈.
입력 보조행 수와 padding 길이가 늘어도 L_task 분모는 실제 supervised target 위치뿐이다.
따라서 4행/12행 입력의 native loss dilution을 도입하지 않는다.
[설계 차이] 기존 native quantile-loss 연구의 재현이 아니다. 점예측 목적에 맞춘 새 계약이다.
모든 방법의 기본 loss는 각 실험 안에서 동일하다. 과거 native-loss 점수와 직접 합치지 않는다.
이 제한된 두 학습률만으로 가능한 모든 LoRA recipe보다 좋다고 주장하지 않는다.

학습 가능한 입력 변환의 gradient 계약:
- N01의 A2/A3, N02의 B3, N03의 C3는 실제 torch 연산으로 native forward에 연결한다.
- 고정 원자료와 검색/마스크 준비는 NumPy를 써도 되지만, 학습 계수를 적용한 뒤
  .detach(), .numpy(), Python scalar 변환 또는 새 torch.tensor(...) 재생성으로 gradient를 끊지 않는다.
- 기존 from_list_of_dicts는 고정 입력 템플릿/행 매핑을 만드는 데 사용할 수 있다.
  실제 로컬 구현을 읽고, learnable 변환은 gradient가 유지되는 context/future_covariates 텐서에 적용한다.
- 입력 행별 실제 target/covariate/group 역할을 기록한다. 필요한 torch-native 전처리는
  고정 계수일 때 공식 입력과 같은 출력인지 검사한다. 학습 중 pipeline.predict의 no-grad 경로를 쓰지 않는다.
- 보조행/가림 이후에도 loss는 지정한 진짜 타깃만 지도한다. 보조 검색 경로나 복원값을 새 정답으로 세지 않는다.
- smoke 두 업데이트 중 적어도 하나는 해당 보조 모듈이 작동하는 비퇴화 조건을 쓴다.
  예: N01은 실제 suffix가 가려진 창, N03은 누락이 있는 창. clean 창만 검사하고 모듈이 정상이라고 하지 않는다.
- 일부 경로가 원리상 0 gradient인 상황(LoRA B 초기0에서 A, 완전관측에서 kernel)은 허용한다.
  반면 영향을 주어야 하는 추가 계수가 모든 유효 조건에서 0 gradient이면 해당 군을 차단한다.

공통 조건:
- Chronos-2 동일 pinned revision, FP32, TF32 off, backbone dropout0.
- task별 모든 군의 초기 LoRA A/B와 원점 순서 동일.
- AdamW(beta .9/.999, eps1e-8, weight_decay0), global clip norm1.
- LR 후보 {1e-4,3e-5}, 일정 LR, 자동 scheduler 없음.
- TRAIN64 원점 x8epochs=512 optimizer updates/경로.
- checkpoints {0,256,512}. 점수와 별개로 정해진 512회까지 모든 군 수행.
- LR 선택용 seed 73100; 반복 seed 73101/73102.
  선택 seed에서 방법당 두 LR을 비교해 V 최저 checkpoint 기준으로 LR 하나 고정.
  반복 seed는 각각 고정 LR로 새로 학습하고 V에서 0/256/512 중 선택한다.
  선택 seed의 E는 주 결과에 넣지 않는다. 반복2seed 모두 공개한다.
- 정확 동률이면 작은 LR, 이른 checkpoint.
- R04의 선택 제약은 해당 절의 정확도-안정성 목적을 따른다.
- 최종512점은 동일 업데이트 진단. 주 선택모델을 사후 FIXED512로 바꾸지 않는다.

파라미터 공정성:
공통 LoRA 예산은 같지만 작은 보조 파라미터가 붙는 군은 전체 수가 다르다.
전체 trainable/frozen/보조 통계/CPU 회귀 계수를 모두 세고 '완전히 equal-budget'이라 부르지 않는다.
더 많은 용량으로만 설명되는 결과는 후속 용량 대조가 남았다고 표시한다.
기존 Time-PEFT 전체나 공식 retrieval/irregular baseline의 재현을 이번 코드와 동일시하지 않는다.

fit 상한(방법당 선택2경로+반복2경로=4):
N01 4방법 ->16 fits
N02 4방법 ->16 fits
N03 4방법 ->16 fits
R04 4방법 ->16 fits
N07 4방법 ->16 fits
R08 4방법 ->16 fits
R09 5방법 ->20 fits
합계 116 fits x 512 = 59,392 본학습 updates.
이는 116개 아이디어가 아니라 7개 학습 주제의 29개 방법군에서 선택·반복하는 경로 수다.
R05/N06의 neural fit=0. CPU 보정/통계 학습은 별도 장부.
각 고유 모델군 2-update smoke: 29군 x2=58 updates.
추가 허용 smoke/수식 연결 점검까지 합쳐 폐기 optimizer 총96 상한.
총 신규 optimizer 최대59,488. 결과가 안 좋아 남는 예산을 다른 후보에 사용하지 않는다.
같은 군/가중치가 수학적으로 동일하면 alias로 중복 fit을 줄이고 이유를 기록.

자원:
한 GPU 한 worker. 기존 검증된 lock/guard를 읽어 재사용.
start free4GiB, update 경계 free1GiB를 설계상 안전 기준으로 사용.
외부 미승인 compute와 겹치면 해당 update 비용을 오염으로 표시하고 기존 guard대로 일시정지.
다른 프로세스 종료 금지. 누적 대기상한1800초, 전체 controller 안전상한24시간.
이 시간은 완료 예측/약속이 아니다. 도달하면 PAUSED_BUDGET와 남은 queue를 보존.
원자료 다운로드2GiB 상한, 새 checkpoint/cache100GiB 상한, 디스크 free 최소10GiB.
모델 snapshot 없는 경우 임의 다른 backbone으로 대체하지 않고 해당 track BLOCKED_MODEL.
전체 native forward20만회 상한. train/eval/smoke/재생을 따로 계수.
매 update 대형 snapshot 저장 금지: scalar journal 매step, 완전 재개 상태 epoch별,
선택용 checkpoint0/256/512만 보존. 파일은 원자적으로 저장.

======================================================================
5. N01 ASYNC — 타깃의 최신값은 늦고 다른 채널은 최신일 때
======================================================================

데이터: Electricity, TRAIN에서 정한4채널, C336h,H48h.
문제: 타깃의 미관측 suffix를 아직 관측된 donor 채널 정보로 보완할 수 있는가?
선행 단서: t-PatchGNN(ICML2024)의 비동기 채널 관계. 이번 간단한 ridge/imputer는 그 논문 재현이 아님.
기존 Freshness와 차이: 모든 채널을 함께 지연시키지 않고 한 타깃만 지연.

관측 조건:
TRAIN 원점마다 타깃 c=(origin_index mod4), 지연 d={0,6,24}를 epoch별 순환.
타깃의 [o-d,o)값만 감춘다. 다른3채널은 o까지 관측.
V/E에서는 모든 타깃을 순회하고 d=0/6/24를 공통 평가.
추가 E 진단: d=12(학습 안 한 길이), 모든 채널 d=24(도움될 최신 donor 부재).
주지표는 부분 지연 d=6/24를 동일 가중한 타깃 normalized RMSE.
정상d0 손해와 d12/all-delayed는 별도 표. 실제 지연 빈도라고 부르지 않는다.
loss는 해당 원점의 지정 타깃만 사용. 모든 군은 동일한 true training target을 받음.

허용 정보: 가려지지 않은 값, 관측 mask M, 각 위치의 마지막 관측에서 경과한 시간 A.
가린 target context 값으로 auxiliary reconstruction supervision을 새로 주지 않는다.
TRAIN 완전관측 과거로 imputer 회귀식을 fitting하는 권한은 모든 군에 동일하게 허용.
따라서 ridge fitting에는 기본 예측 학습과 다른 within-TRAIN 회귀 쌍이 추가된다.
그 회귀 쌍 수를 별도로 세며 A1 대 A0를 순수한 구조 효과라고 부르지 않는다.
핵심 A3/A2 비교는 같은 회귀식·같은 회귀 학습 정보를 공유한다.
보조 mask/age는 모든 군에 past-covariate 행으로 제공. native target NaN과 구분.
NaN을 숫자0으로 바꾸면서 실제 관측값이라고 속이지 않는다.

고정 단순 회귀 imputer:
z값 TRAIN에서 채널c를 다른3채널의 동시점 값+상수로 ridge 예측.
ridge alpha={.1,1,10}, TRAIN 내부3개 forward-chaining fold로 선택. E/V를 사용하지 않는다.
마지막 관측 유지값 l_c(t), ridge r_c(t)를 계산.
donor도 없는 진단 상태는 donor LOCF와 mask 사용; 전체 회귀량을 관측값으로 취급하지 않음.

4개 학습군:
A0 NATIVE: target suffix NaN을 native mask로 처리 + 공통 M/A + LoRA.
A1 RIDGE: 결측 위치만 r_c(t)로 채움 + M/A + LoRA.
A2 RESID: r_c(t)+theta_c^T phi(t)로 채움 + M/A + LoRA.
   phi=[1,l_c,r_c, donor_z*M(3), donor_M(3)] (총9), theta 초기0.
A3 AGE_RESID: A2와 같되 + eta_c * a_c(t)*(r_c(t)-l_c(t)).
   a=min(A/24,2), eta초기0. 관측 위치는 항상 원래 값 그대로.
   이 추가항은 [미검증] 작은 입력 보정이며 '새 age-aware 기법'이라고 전제하지 않는다.
A2/A3 보정은 missing positions만 적용, 실제 원입력 mask는 유지해 model이 fill을 구분.
계산값은 표준화 공간에서 만들고 raw 단위로 복원해 native 입력에 넣는다.

직접 대비: A3 vs A2 (age 상호작용), A3 vs A1 (학습 가능한 보정의 필요성), A3 vs A0.
A3 추가 계수4개도 파라미터 표에 포함. A2보다 큰 용량일 수 있음을 숨기지 않음.
기대: 부분 지연의 예측 개선, all-delayed에서는 장점 약화 가능.
반례: A1이나 native grouping만으로 충분하면 별도 imputer PEFT 필요성 약함.
상관이 강한 채널만 E에서 골라 결과를 살리지 않음.

필수검사: 숨긴 suffix poison 입력불변, donor 변화에는 입력반응, 미래donor 사용0,
관측값 불변, 네 target의 노출균형, A2/A3 초기=A1, eta=0 축소 동작.
한 wrapper 초기결과가 native F0와 다른 것은 imputation 차이일 수 있으므로
초기 identity는 '같은 전처리의 LoRA0'와 비교한다.

======================================================================
6. N02 ARCHIVE — 최근 창 밖의 관측 완료된 과거 사례
======================================================================

데이터: Traffic 4채널, 기본 C336h/H48h.
문제: 이미 관측 완료된 예전의 유사한 패턴과 그 이후 값이 추가 예측정보를 주는가?
선행: Retrieval Augmented Time Series Forecasting(ICML2025, RAFT).
검색 자체/검색 후값 사용은 알려진 방법. 아래 간단한 보정도 신규성 미확인.

과거 은행:
채널별 r>=336인 TRAIN 원점에서 stride24로 (과거336, 이후48)을 저장.
현재 o에서 검색 가능한 r은 r+48 <= min(TRAIN_end, o-336-24).
현재 문맥과 은행의 전체 정답 사이에도24h 간격을 둔다.
TRAIN query에도 동일 규칙. query의 target 또는 query와 겹치는 continuation 검색 금지.
TRAIN query는 o>=2048부터 선택한다. 은행후보>=32인 원점만 기계적으로 유효 처리.
은행 최대2048개: 많으면 TRAIN 원점의 deterministic 균등 인덱스.
V/E 은행은 TRAIN에 고정. V/E의 새로운 과거로 은행을 갱신하지 않는다.

검색:
채널별 query의 마지막336과 은행past336을 각 창의 mean/std로 표준화.
Euclidean mean squared distance로 상위K=2. 동률 r가 이른 것을 선택.
정답 continuation을 유사도 계산에 넣지 않는다.
과거와 후속 전체를 query의 past mean/std로 단위 변환한 두 사례를 만든다.
후속값이 현재에서 이미 과거인 시각이라는 증명을 각 retrieval에 남긴다.

4개 학습군:
B0 SHORT: 원래 C336 그룹 + LoRA.
B1 LONG: 원래 C1344 그룹 + 동일 LoRA. H48 동일.
   더 긴 문맥은 강한 직접 대안. 입력 정보량/비용 차이를 명시.
B2 RETRIEVE: C336 원래 그룹 + 채널마다K2의 검색past를 보조 행으로,
   검색continuation을 그 보조행의 'known future feature'로 제공 + LoRA.
   이 feature는 실제 미래 측정값이 아니라 이미 관측된 과거 사례를 정렬한 합성feature다.
B3 DELTA_ADAPT: B2와 같고 검색 future만 다음처럼 조정:
   u_last = 검색past의 마지막 값(이미 query 단위로 변환)
   u_future' = u_last + exp(clip(a_c,-2,2))*(u_future-u_last) + b_c*sigma_query.
   a_c=b_c=0 초기. 추가8 scalar. 검색past 자체는 바꾸지 않는다.
   모든 검색행의 affine를 past/future에 같이 적용하면 native normalization이
   효과를 지울 수 있으므로 그런 구현으로 대체하지 않는다.
   clipping saturation율과 실제 a/b gradient를 기록.

추가 CPU 대조: 검색continuation 평균만 예측, B0 예측과 그 평균의 전역 convex blend.
blend alpha={0,.25,.5,.75,1}, V_SELECT에서만 선택. 비용에 검색시간 포함.
본체가 같은 조합은 prediction reuse. CPU fitting도 기록.

주지표: 뒤 구간 전체 normalized RMSE.
핵심 대비 B3 vs B2, B3 vs LONG, B3 vs 단순 blend.
B2가 SHORT만 이기는 결과는 추가정보 효과이지 새 PEFT 효과가 아님.
LONG으로 충분하면 문맥 제한을 의도적으로 줄여 retrieval을 살리지 않는다.

필수검사:
query target poison -> 검색순위/변환/입력 불변,
모든 r+H <= o-C-24, bank TRAIN 경계, 자기 자신/겹침 exclusion,
B3 초기=B2, 바꾸는 future delta가 native 전처리 후에도 실제로 달라짐,
보조후속은 label이 아니라 합법적 이미 관측된 feature임을 시간표로 증명.

======================================================================
7. N03 CLOCK — 같은 실제 미래를 다른 관측 밀도로 예측
======================================================================

데이터: ETTm1의15분 원자료,4채널. C336slots=84h, H48slots=12h.
ETTh1과 같은 물리적 시스템/다른 해상도일 수 있으므로 독립 도메인으로 세지 않는다.
문제: 같은84h 관측 구간을 듬성듬성 읽을 때 실제시간을 보존하는 단순처리로 충분한가?
선행: t-PatchGNN, FlowState(ICML2026). 아래 kernel은 두 논문의 재현이 아니다.
초기 실험은 기록된 규칙 자료의 관측을 가리는 모사다. 실제 irregular benchmark 성공 아님.

관측:
TRAIN epoch마다 dense(delta=1)와60분(delta=4) 교대.
각 채널의 sampling phase는 hash(origin,channel,epoch)로 정한 0..delta-1.
모든 군에 같은 관측값·시각·mask·age 제공.
V_SELECT는 delta1/4 같은 가중치. E 주평가는 미학습delta2(30분)와
간격{1,3}를 교대로 섞은 irregular pattern(평균2slot)의 같은 가중치.
irregular의 시작phase/순서는 고정hash. 마지막관측은 항상 origin미만.
E의 dense1/60분4는 부가표. 예측은 모든 경우 같은 미래12h의15분 grid.
현재 원점 사례에서 가린 값은 입력·보간·native context normalization·보조 복원 손실에서 읽지 않는다.
공통 TRAIN mu/sigma는 dense 학습 조건에서 관측을 허용한 TRAIN 자료만으로 고정한다.
이 실험은 TRAIN 전체가 불규칙 관측인 설정이 아니라, dense/60분 조건으로 학습한 뒤
새 관측 밀도에 평가하는 설정이다. 실제 완전 미관측 TRAIN 값을 알고 정규화했다고 혼동하지 않는다.

4개 학습군:
C0 GRID: canonical15분 grid의 누락을 NaN으로 유지 + M/A + LoRA.
C1 HOLD: forward-fill(이전 관측 유지), 첫 관측 전은 TRAIN mean + M/A + LoRA.
C2 KERNEL: 누락grid t를 이용 가능한 가장 가까운4개 관측으로 kernel 보간 + M/A + LoRA.
   w_i=exp(-|t-t_i|/tau0), tau0=4 raw slots. 관측 위치는 원래 값 그대로.
   양쪽 관측 모두 origin 이전이면 현재 forecast에서 이용 가능하다.
   실시간 각 과거 t에 이미 알려졌다는 주장이 아니라 원점o에 합법적인 smoothing이다.
C3 LEARN_KERNEL: C2와 같고 채널별 tau_c=0.5+7.5*sigmoid(theta_c).
   tau_c=4가 되도록 초기화. 추가4파라미터. 인접4개 선택은 고정, kernel만 학습.
   누락이 없는 창은 C2/C3가 입력상 동일하고 tau gradient0일 수 있다.

대조의 정보 권한은 동일. C0도 mask/age와 canonical 실제시간 위치를 받는다.
단순 재표본화 baseline에서 시간정보를 일부러 빼지 않는다.
primary: 동일 실제12h의 normalized RMSE (delta2/irregular 평균).
기대: 학습에서 본 관측밀도에만 맞춘 성능이 아니라 중간/불규칙 조건의 이득.
반례: C0/C1/C2로 충분 -> 학습 kernel 추가가치 없음.
이 결과만으로 다양한 관측률의 모든 TSFM이나 실제 비동기자료를 해결했다고 쓰지 않는다.

필수검사: 모든 방법 동일 observed tuple(value,timestamp), 정답 physicaltime 일치,
원점 이후 관측 접근0, kernel row weight 합1, observed overwrite0,
tau0일 때C3=C2, raw timestamp/단위 변환 검산.

======================================================================
8. R04 REVISION — 정확도를 유지하며 불필요한 예측 수정 줄이기
======================================================================

데이터: Traffic 4채널, C336h/H48h, paired origins o와o+24.
같은 미래를 예측한 구간은 early[24:48]와 late[0:24]. 다른 lead를 그대로 빼지 않는다.
TRAIN/V/E의 두 horizon이 각 role 경계 안에 있어야 한다.
두 입력은 서로 다른 forward 또는 서로 다른 group ID. 뒤 문맥이 앞 예측에 새어 들어가면 안 된다.

[확인] 옛 Candidate04도 paired origin과 F0 대비 correction revision을 다뤘다.
이번 차이: 고정 point 목적과 새로운 공통 분할, innovation 정보의 작은 직접 대조.
새 데이터 독립 재현이나 원 논문 전체 재현이라고 부르지 않는다.
선행: Using dynamic loss weighting to boost improvements in forecast stability.
예측 안정성 손실/동적 가중 자체는 알려진 접근이다.

기본 L_task = 두 origin의 채널별 normalized MSE 평균.
R_raw = mean[((late[0:24]-early[24:48])/sigma)^2].
Delta = 현재 예측 - 같은 입력의 고정 F0 점예측.
R_corr = mean[((Delta_late[0:24]-Delta_early[24:48])/sigma)^2].
F0 예측은 원점별 캐시 가능. 미래 정답 없는 no-grad 호출만 사용.

새로 관측된 정보:
v_c(o)=mean[((observed_y[o:o+24]-F0_early[0:24])/sigma_c)^2].
이는 late 원점에서는 이미 관측된 값이며 late미래정답이 아니다.
w_c(o)=1/(1+v_c(o)). TRAIN 원점 전부에서 구한 mean(w)로 나눠 평균1.
V/E에는 TRAIN mean을 고정 사용. v를 미래 정답/최종 forecast error로 정의하지 않는다.

단일 lambda:
TRAIN의 첫16pair에서 F0의 L_task와 R_raw를 기록.
lambda=clip(.05*median(L_task)/max(median(R_raw),1e-8),1e-4,1).
초기 R_corr=0을 분모로 쓰지 않는다. 모든 규제군에 같은lambda.
값/gradient 비율은 다를 수 있으므로 큰 clipping 빈도도 보고. 사후lambda재튜닝 금지.

4개 학습군:
D0 PLAIN: L_task.
D1 STABLE: L_task+lambda*R_raw (기존 안정성 손실의 제한된 대조).
D2 CORR: L_task+lambda*R_corr (옛 FR원리의 새 데이터 비교).
D3 INNOV: L_task+lambda*mean[w_c*((Delta_late-Delta_early)/sigma)^2].
D3는 '새 정보가 크게 들어왔으면 보정수정을 덜 억제'하는 [미검증] 가중 규칙.
coefficient가 작은 것만으로 좋아지지 않도록 w의 TRAIN 평균을1로 맞췄다.

CPU 단순 대조: 선택된D0 late 예측과 이미 발행한early 겹침 예측의 blend.
alpha={0,.25,.5,.75,1}, V에서 선택, 같은 실제 미래만 blend.
발행한 과거 예측 파일은 나중 모델로 덮어쓰지 않는다.

선택/목표:
D0은 V accuracy최저. 다른 군은 각 checkpoint의 V accuracy가
선택한D0 V normalizedRMSE의1.01배 이하인 후보 중 revision RMS최저를 선택.
조건을 만족하는 checkpoint가 없으면 그 군의 accuracy최저와 INFEASIBLE를 기록.
LR 후보 모두에게 같은 rule. 1%는 이번 안정성 목적의 설계허용치이며 논문합격선 아님.
주 결과: accuracy vs raw revision의 paired frontier.
E에서도 accuracy보호를 만족하는지 별도 확인. 수정량만 줄어든 것을 성공이라 하지 않음.
'불필요한' 수정의 진짜 레이블은 없으므로 accuracy제약하 revision이라는 조작적 정의.
큰 혁신 구간에서 반응이 느려지는지 TRAIN 기준v 상위quartile을 진단하되 E로 cutoff를 정하지 않음.

필수검사: exact target timestamps overlap, late정보->early전달0,
innovation값의 as-of 가용성, F0 disabledLoRA복원, lambda초기분모유한,
두forward 비용을 한forward효율이라고 세지 않음.

======================================================================
9. R05 VINTAGE — 저장 예측으로 최신/지연 손익 정책부터 비교
======================================================================

neural fits=0. 기존 forecast_path_structure_v1_20260916의 선택된 LATEST/PATH를 고정.
원래 LR/step을 새 목적에 유리하게 다시 선택하지 않는다.
현재 결과에서 'LATEST'는 이전 네 상태평균의 V 선택모델이었다는 한계도 표시.

타깃: 기존T0/T1/T2, seed61730/61731, S0..S3 원래 입력case.
V_SELECT7월로 아래 정책만 고정하고 E10~12월을 재분석.
E는 이미 점수가 노출됐다. 명칭은 REANALYSIS_REUSED_E; 새독립평가가 아니다.
큰 예측파일이 없으면 기존 해시가 맞는 checkpoint+입력으로 필요한 예측만 재생 가능.
추론을 재생했다고 LoRA fit으로 세지 않는다. 가중치를 새로 학습하지 않는다.

q_alpha(k)=(1-alpha_k)*q_LATEST(k)+alpha_k*q_PATH(k).
quantile vector의 convex mixture이며 확률분포 mixture의 quantile이라고 부르지 않는다.
원래 정렬된 quantile을 사용. 같은tau끼리 조합하고 교차 여부 검사.

5정책:
E0 LATEST: alpha=[0,0,0,0].
E1 PATH: alpha=[1,1,1,1].
E2 GLOBAL: alpha공통, {0,.25,.5,.75,1} 중 선택.
E3 BINARY: alpha=[0,1,1,1] 고정.
E4 MONOTONE: 같은5값에서 0<=a0<=a1<=a2<=a3<=1인70개 조합.
모든 점수는 V의 same origin/case/target 사용. 타깃별계수는 두 seed평균V로 하나.
E2/E4 선택: V S0 loss <= E0 S0 loss*1.01인 조합 중 평균S1..S3 loss 최소.
동률: 평균alpha작은 것, 사전lexicographic순서.
alpha0=0인E4는S0에서LATEST와정확히같다. 그조건에서LATEST보다개선한다고 기대/요구하지 않음.
E4의 직접상대는E2뿐아니라 E3 BINARY. 동일행동이면중복실험이아니라같음으로기록.

보정은 추가하지 않는다. 원래 raw와 원래 각모델보정결과가 모두 있으면 보정판은 민감도표만.
주목표: 최신S0 손해1% 이내에서 S1..S3 평균/최악손해 감소.
실제 갱신지연빈도나 rank별확률을 추정했다고 쓰지 않는다.
S0/S3 개별표, 전체타깃/추가타깃만, seed별표 공개.
두모델 추론비용 포함. alpha끝점은한모델, 중간이면두모델이며 단일PEFT 실행아님.
모사 available_at, rank != 실제예보나이/정확도/ensemble member 한계를 유지.

정책유용성은 확인 가능하나 학습방법신규성은 주장불가.
E4가E3보다특별히좋지않으면복잡한age PEFT를만들근거는약하다.
이것이기상예보학습전체가불가능하다는판정은아니다.

======================================================================
10. N06 JOINT — 주변분포를 동일하게 유지한 하루 전체 위험
======================================================================

neural fits=0. 기존 예보 실험의 FROZEN_WEATHER S0 raw quantiles를 고정.
point MSE로 학습한 이번다른track의모델을확률모델로사용하지않는다.
타깃T0/T1/T2; V_CALIBRATE8월로 dependence만 fitting; E는 재사용개발평가.
목표: 같은시점별분포로도 다른시간결합이 하루총량/연속위험 예측에 영향을 주는가?

선행: TACTiS-2(ICLR2024),
Efficiently Generating Correlated Sample Paths from Multi-step TSFMs
(NeurIPS2025 시계열 foundation model workshop; 메인학회 아님).
copula·잔차rank 재배열 자체는 기존방법이다. 이실험은 알려진 대조의 유용성 확인.

각origin H24, sampleN=256.
각lead에 공통 u_j=(j+.5)/256을 원래 quantile함수 선형보간으로 매핑한 sorted256값을 만든다.
u가 .01보다작거나 .99보다큰부분은같은공통끝값으로clamp.
이는공통finite-tail 가정이다. 극단꼬리추정성과/무한지지확률의정확성을주장하지않음.
모든coupling은 각lead의 이256값의 순열만 바꾼다.
따라서 경험적주변분포와각lead의ensemble CRPS는정확히같아야한다.
독립적marginal quantile을개선했다고결론내릴수없는설계다.

V_CAL에서 진짜값의quantile CDF 위치u를 구함:
동일q값은대응tau의중간값으로병합,단조보간, u clip [.01,.99].
z=Phi_inverse(u). calibration 16경로뿐이므로 covariance불확실성 큼.

4 coupling:
F0 INDEPENDENT: lead별독립permutation.
F1 AR1: V_CAL z의인접lead lag1 correlation rho를 pooled추정,
   clip rho[-.95,.95], R_ij=rho^|i-j|.
F2 SHRUNK: V_CAL z의24x24상관행렬 S, R=.5I+.5S; 대각1재정규화.
   symmetrize+고정eigen floor1e-6후재정규화. 더좋은shrinkage탐색금지.
F3 EMPIRICAL: V_CAL의24차원PIT rank경로16개를반복표집해256template만들고
   각lead template순위로공통256예측값을재배열(Schaake-style대조).
F1/F2도 Gaussian R 표본 256개의 각 lead 순위로 공통 예측값을 재배열.
V_CAL에서 상수인 lead는 상관 추정이 정의되지 않으므로 해당 off-diagonal을 0,
diagonal을 1로 두고 그 위치를 보고한다. 상수/결측을 임의 작은 noise로 숨기지 않는다.
고정 seed 73260과 동일한 tie-break를 기록한다. empirical template의 동점은
별도 고정 난수 순위로 해소하되, E 정답을 이용하지 않는다. 256 표본이라는 Monte Carlo
근사의 한계도 보고하고, seed를 바꿔 유리한 샘플 경로를 고르지 않는다.

primary: 하루24시간합계의sample CRPS/(24*sigma_TRAIN).
CRPS_N=mean|s_i-y_sum| - (1/(2N^2))*sum_ij|s_i-s_j|.
부가: 전체경로energy score, p=.5 variogram score, 연속6h평균 최대값의초과확률 Brier.
임계값은TRAIN의rolling6h평균90분위수로만정함. E에서좋은threshold선택금지.
에너지는W의시간평균 x1h의합이면Wh이며 원자료단위를다시확인.
원래각시간의pinball/CRPS동일성표를필수로붙임.
전체 energy pair합은 256x256, 큰 배열은 origin 단위로 계산해 메모리를 제한한다.
energy score = mean_i ||s_i-y||_2 - .5*mean_ij ||s_i-s_j||_2 (각 값은 TRAIN sigma로 표준화).
variogram score = mean_{h<k} (|y_h-y_k|^.5 - mean_i|s_ih-s_ik|^.5)^2,
모든 h<k에 같은 가중치. Brier=(예측 사건확률-실제 사건0/1)^2.
이것들은 보조 지표이며 sum-CRPS보다 유리하게 나왔다고 primary를 교체하지 않는다.

필수검사: 각lead sorted samples exact동일, correlation PSD, quantile순서,
V_CAL/E분리, 두시간공동/교대toy에서동일주변분포·다른합계위험을재현.
분포결합의효과와marginal misspecification/finite-tail한계를구분.
N06는새PEFT를이미설계한것이아니다. 기존간단copula로충분하면그것을채택할근거다.

======================================================================
11. N07 SPECTRAL — 작은 크기와 예측 가능성을 분리한 학습
======================================================================

데이터: Electricity4채널, C336/H48.
문제: 큰 패턴에 비해 에너지는 작아도 과거로 예측 가능한 주파수성분을 적응이 놓치는가?
선행: Fredformer(KDD2024), MSFT(NeurIPS2025).
주파수강조/다중스케일학습 자체가 새 기여가 아니며 아래 고정가중 loss도 [미검증]이다.

채널별 TRAIN64 원점에서 H48 future의 unitary full FFT Z_k를 계산.
E_k=mean|Z_k|^2 (신호는TRAIN mu/sigma로표준화).
과거48/직전48의FFT에서 각k의실수·허수와상수5개로 future Z_k를ridge예측.
TRAIN64를시간순4block으로나누고 앞1block->다음block,앞2->다음,앞3->다음의
forward-chaining 예측을 모아 skill_k=1-MSE_pred/MSE_train_mean 계산.
각 fold 정규화와 평균은 그fold 앞부분만. 겹친label/context에 대한48slot purge 적용.
유효한 fold가부족하면 해당skill을0으로놓고그이유기록. E/V로보완하지않음.
skill은[0,1]로clip. DC와Nyquist는재가중대상에서제외.

가중치:
w_BASE=1.
w_ENERGY: positive-frequency E_k^-1/2을 mean비로정규화후[.25,4]clip.
w_PRED: E_k가positive frequency중앙값이하인bin에서
   u_k=skill_k*min(4,median(E)/(E_k+eps)), 그외u_k=0; w=1+u.
w_SHUFFLE: w_PRED의positive-frequency순서를고정seed로섞음.
각가중치를negative frequency로mirror,DC/Nyquist 원값1,
마지막에전체H개의mean이1이되도록정규화. real-valued signal에대칭가중치.
eps=1e-8*max(mean(E),1). 값과공식은TRAIN에서봉인.

4개 학습군:
G0 BASE / G1 ENERGY / G2 PRED / G3 SHUFFLE.
L(w)=.5*mean(e^2)+.5*mean_k(w_k*|FFT_ortho(e)_k|^2), e=(yhat-y)/sigma.
G0는Parseval에의해그냥MSE와동일하다. 'uniform Fourier loss'를별도5번째fit으로세지않음.
G2와G3는가중치multiset 동일; 어떤frequency에놓는지가다름.
가중치가모두1이면G2/G3=G0로alias. 가짜고유후보fit을실행하지않음.

주지표: 원래시간영역normalizedRMSE 전체.
부가: TRAIN에서고정한저에너지/예측가능bin의오차,큰성분손해,MAPE사용안함.
개선 주장은bin몇개오차만좋아진것으로성립하지않음.
G2의직접상대는G1/G3및G0.
추가CPU대조: 위ridge성분예측을시간영역으로역변환한예측; 사용정보동일.

강한정식선행 Fredformer/MSFT는이번결과와직접재현비교하지않음.
후속주제로남으면그누락을우선보고하고, 주파수bias전체가원인이라고확정하지않음.
필수검사: FFT Parseval, 실수역변환,weight평균/대칭,TRAIN-only skill,
E label변경으로weights불변, 일정한weight의별도학습미실행.

======================================================================
12. R08 LEAD — 다른 채널의 선행 정보가 관측된 예측 구간
======================================================================

데이터: Traffic4채널, C336/H48.
문제: 선행채널의관측을직접쓸수있는초기구간과,그채널도예측해야하는뒷구간을
다르게조정하는것이단순시차정렬·일반horizon조건보다유용한가?
선행: LIFT(ICLR2024). 동적선행지표/지연정렬을처음제안한다고쓰지않는다.
이전Gaussian정렬실험의단순회귀우위는보존. 실제Traffic으로바꾼새조건이다.

TRAIN만으로:
타깃c마다다른채널j, lag l=1..24의corr(z_c[t],z_j[t-l])를계산.
같은donor의최대절대corr lag를선정하고,서로다른상위2donor선정.
동률donor index작은것/lag작은것. TRAIN finite공통시점만.
고정donor/lag에ridge(상수포함) 회귀 g_c=b_c+sum_j beta_cj*z_j[t-l_j].
alpha .1/1/10의TRAIN forward-chaining fold만사용. 시험용합성비선형추가금지.

미래h=0..47에서:
s_j(h)=이미관측한 z_j[o+h-l_j] (h<l_j),
       현재model이예측한 donor j의 zhat_j[h-l_j] (h>=l_j).
절대시각으로조건검사. donor의실제미래정답을사용하지않는다.
g_c(h)=b_c+sum beta*s_j(h).
a_c(h)=sum |beta|*1[h<lag]/max(sum|beta|,1e-8).
base point zhat_c에 delta_c=g_c-zhat_c를더하는방식으로정의.
순환참조 방지: 모든g는보정전base zhat에서동시에계산,다른보정g를입력으로사용하지않음.

4학습군(모두동일LoRA기본point MSE):
H0 BASE: 보정0.
H1 STATIC: zhat+tanh(theta_c)*delta. theta0 초기.
H2 LINEAR_H: zhat+[(1-h/47)tanh(u_c)+(h/47)tanh(v_c)]*delta.
H3 LEAD_H: zhat+[a_c(h)tanh(u_c)+(1-a_c(h))tanh(v_c)]*delta.
H2/H3는동일8개scalar; 차이는선행정보의실제가용성반영여부.
H0/H1도같은donor관측과시간을접근가능하게한다. 표준model은원래모든4채널입력.
추가CPU기준: frozen모델과고정ridge로g만만든예측, H0/g의전역blend(V에서alpha선택).

주지표: 전체H48normalizedRMSE. h<lag정보가용비율구간은부가표.
기대: H3가H2와단순blend를넘을때 '실제가용구간에따른조정'의개발근거.
반례: H1/단순ridge/LIFT형구성으로충분 -> 새PEFT기여미확보.
선행정보가약한4채널에서결과가없으면그자료의한계로기록, E보고donor채널교체금지.
후속정식비교LIFT가남아있음을표시. H3를LIFT보다새롭다고전제하지않음.

필수검사: 모든관측donor timestamp<o, predicted donor index>=0,<H,
g루프의순환없음,초기모든학습군=base, h가시간조건을가진toy의정답정렬,
학습된g제거진단만으로처음부터g가없는H0보다좋다고결론내리지않음.

======================================================================
13. R09 MIXED — 상세/집계 레이블이 섞일 때의 보존
======================================================================

데이터: Electricity4채널. C336h/H24h. origin은완료된24h block경계(index mod24=0).
이주제는월합전용옛실험의재현이아니다. 24h집계+일부상세라는식별가능성을보완한새모사다.
목표: 집계자료를추가활용하면서상세예측패턴손해를줄이는가?
선행: Temporal Disaggregation of Time Series(The R Journal2013), Denton/Chow-Lin 계열.
관측null space보존/투영이라는수학자체를신규성으로주장하지않음.

가장중요한정보계약:
숨긴하루의시간별값이,뒤training origin의context에그대로다시나오면안된다.
관측권한은window label별이아니라 전체절대시간block별로정의한다.

1) 전체시간축을24h block으로구분.
2) TRAIN64 forecast origin을값과무관하게선정하고,그64 target blocks중
   hash가작은16개만detailed,48개는aggregate-only로고정.
3) 그외TRAIN/V/E과거로사용할block은hash modulo4==0인1/4만detailed.
   단V/E의미래ground truth는scorer가보유하며모델에미리공개하지않음.
4) detailed block은완료시시간별값을사용가능.
   aggregate-only block은완료시24개합계또는평균만사용가능.
   입력에서는완료된그block의관측평균을24번반복하고observed-resolution mask추가.
5) role선정과통계는이허용관측에만의존. 원시세부값의std/주파수를selection에쓰지않음.
6) TRAIN sigma는허용된상세값+집계평균복원sequence에서계산.
7) V_SELECT는설계상상세validation labels32일을모델선택기에허용하는것이며,
   '상세검증정답도전혀없는환경'을해결했다고주장하지않음.
8) 미래의집계량을추론때주지않는다. 알고있는합계로TEST예측을맞춰주는행위금지.

관측loss:
상세target이면 L_fine=mean_h[((yhat-y)/sigma)^2].
aggregate-only이면 L_agg=((mean(yhat)-observed_mean)/sigma)^2.
관측평균계산후rawfine label은training packet에서삭제.
q0는같은허용context를받은F0 point prediction.
delta=(yhat-q0)/sigma.
P(delta)=각24h블록평균을repeat, N(delta)=delta-P(delta).
N은집계평균으로관측되지않는상세변화다.

5개학습군:
I0 FINE_ONLY: 관측상세16target만반복해512updates.
   I1..I4와정보량/고유label수가다르므로이비교는집계정보의추가효과가섞임.
   loss를0으로만들어512updates인척하지않고실제상세16개반복횟수를기록.
I1 MIXED: 위관측loss,64target각8회.
I2 IMPUTE: aggregate target의pseudo detail=q0+(observed_mean-mean(q0)).
   detailedtarget은실제label. 이pseudo값으로MSE. 원형preserving disaggregation 대조.
   pseudo는TRAIN한정, 정답이라고부르지않음. 새예측때aggregate를받지않음.
I3 UNIFORM: MIXED + .1*mean(delta^2), aggregate-only block에만벌점.
I4 NULL: MIXED + .1*mean(N(delta)^2), aggregate-only block에만벌점.
I3/I4의동일계수는동일gradient총량을보장하지않음. loss/grad기여를기록한다.
입력resolution mask는모든방법이같이받는다. 상세target의추가보존벌점없음.

[수학적 동일성 — 구현 전 반드시 확인]
I2의 pseudo detail을 y_tilde=q0+(observed_mean-mean(q0))로 쓰면,
  mean[((yhat-y_tilde)/sigma)^2] = L_agg + mean[N(delta)^2]
가 정확히 성립한다. 즉 I2는 I4의 NULL 계수를 1로 둔 것과 같은 목적함수다.
현재 I4의 계수 .1과 차이가 있으므로 학습은 구분되지만, I4가 I2보다 좋아진 것만으로
새로운 보존 원리가 생겼다고 해석하지 않는다. 같은 계열의 보존 강도 차이일 수 있다.
I4 대 I3는 관측 가능한 평균 변화까지 억제하는지의 대조이고,
이 역시 알려진 projection/shrinkage 원리의 제한된 비교다.
공통 데이터 eligibility에서 원시 TRAIN 표준편차를 확인하는 일반 규칙은 R09에서 적용하지 않는다.
R09는 관측 지도부터 만든 뒤 허용된 상세/집계 값만으로 eligibility와 통계를 계산한다.

직접비교: I4 vs I3/I2/I1. I0는기본관측정보기준.
primary=전체실제시간별normalizedRMSE.
필수secondary=24h평균/합계RMSE,상세패턴오차(mean제거),허용context상태별결과.
기대: aggregate학습의유용한수준변화는허용하면서세부패턴손해를줄임.
반례: I2의단순보존복원으로충분하면특별한loss필요성약함.
aggregate오차만개선되고시간별오차가나빠지면기본목표달성아님.

필수검사:
숨긴fine target에합계0인perturbation을주어도입력/통계/optimizer label/pseudo불변,
미래test aggregate가입력에포함되지않음,
N^2=N, P^2=P, mean(Ndelta)=0, ||delta||²=||Pdelta||²+||Ndelta||²,
상수offset은NULL벌점0이지만UNIFORM벌점>0,
Fine-only의반복횟수와집계군의unique labels차이를정직하게보고.

======================================================================
14. 선행 비교와 결과 해석 — 이 배치에서 가능한 주장
======================================================================

각 후보의 TOPIC_ONEPAGE에는 다음 칸을 실제로채운다.
조건 / 사용가능정보 / 기존한계의근거 / 제안변경 / 가장가까운단순대조 /
예상결과 / 반례 / 정식선행과차이 / 아직미재현한강한대조.
문서작성후전부해시를고정하고나서어느track의E도채점한다.

태그:
[확인] 실제읽은코드/논문내용/실측값.
[설계] 여기서정한새실험조건.
[추정] 원인/일반화설명.
[미검증] 새조합의효과/신규성가설.
'기존연구와문자그대로같지않음'은새방법의최초성증거가아니다.

논문비교의역할:
- N01/N03: t-PatchGNN 및관측시각을쓰는강한단순baseline을확인.
- N02: RAFT의검색정보·후속관측사용조건과대조.
- N07: Fredformer/MSFT와단순주파수weightedloss를구분.
- R08: LIFT의동적지연·선행값활용과현재고정시차prototype의차이명시.
- R09: 기존분해/benchmarking과관측nullspace원리를명시.
- R04: static/dynamic forecast stability loss와원래FR을구분.
- R05: 기존vintage/dropout단순대조; 새PEFT아닌의사결정정책.
- N06: 이미알려진copula/postprocessing. 방법론신규성검증은미실행.
직접선행과동치이면 KNOWN_REPLICATION으로계속비교가능하지만새방법이라고보고하지않음.
공개원문을못읽었으면서지정보/초록까지만읽었다고적는다.
자료접근을못했다고그대로미검증새기법을확정하지않는다.

======================================================================
15. 평가, 통계, shortlist 규칙
======================================================================

정확도 track:
주 normalizedRMSE = mean_channel sqrt(mean_origin,h ((prediction-y)/sigma)^2).
타깃규모차이를원시MSE합으로몰아주지않음. RMSE를어느축에서sqrt하는지코드·표에명시.
N01같은condition평가는먼저condition/target별이정의를적용하고고정동일가중.
rawRMSE/MAE,각seed,각condition도별도표. 모든gain분모는비교군점수.

R04/R05/N06는각절의목적을따른다. 정확도와수정률/Brier/CRPS를같은합산점수로순위매기지않는다.
신규조건의주목적을바꿔서과거실패를성공으로재분류하지않는다.

불확실성:
2,000회paired7일calendar/index time-block bootstrap, seed73300 고정.
같은source시간구간/조건/target/방법에같은날짜블록index를사용.
ETTm1도실제7일=672개15분slot. hourly는168slot.
같은label이다른origin/case에서중복되면독립표본이라고세지않음.
군별로다른날짜를재표집하거나좋은seed만제외하지않음.
CI는관측한원천/채널/seed에조건부, 새도메인·모집단불확실성이나9주제다중선택보정을대체하지않음.
전체한개종합p-value로논문통과판정금지. 평균·구간·seed·조건별반전을같이읽음.

각track 상태는 다음3축:
EXECUTION: COMPLETE / PARTIAL / BLOCKED_DATA / BLOCKED_MODEL / INVALID_CONSTRUCT / NUMERICAL_ERROR.
EVIDENCE: SIMPLE_METHOD_SUFFICIENT / NO_SELECTED_ADAPTATION / POSITIVE_UNCERTAIN /
          PROMISING_WITHIN_SCOPE / NEGATIVE_WITHIN_SCOPE / NOT_MEASURED.
NOVELTY: KNOWN_CONTROL / UNVERIFIED_VARIANT / DIRECT_COLLISION / NOT_AUDITED.
EXECUTION_PASS를성능/논문PASS로섞지않음.

조건별사전핵심대비:
N01 AGE_RESID vs RESID와RIDGE.
N02 DELTA_ADAPT vs RETRIEVE, LONG, 단순blend.
N03 LEARN_KERNEL vs KERNEL, GRID.
R04 INNOV vs CORR/STABLE와단순평활화 (accuracy허용범위내).
R05 MONOTONE vs GLOBAL/BINARY (S0제약내).
N06 가장좋은coupling vs INDEPENDENT/AR1 (동일marginal).
N07 PRED vs ENERGY/SHUFFLE및BASE.
R08 LEAD_H vs LINEAR_H/STATIC.
R09 NULL vs UNIFORM/IMPUTE/MIXED.

모든dataset/seed에서승리할것을요구하지않는다.
반대로두개조건만사후선택해서scope를바꾸지않는다.
효과가0근처면동등성이입증됐다고쓰지않고불확실하다고기록.

최종 shortlist는최대2개. 다음정보를병렬제시해사람이판단할수있게한다:
- 실제문제와개선정보가남는가?
- 직접대조대비효과크기와불확실성은어떤가?
- 단순해결책으로충분한가?
- 정상조건/다른condition에어떤손해가있는가?
- 정식선행대비남은차이와다음필수비교는무엇인가?
'known_method가좋았다'만으로새방법론주제를확정하지않는다.
짧은리스트가0개여도정상결론이다. 다른후보를추가해2개를채우지않는다.

======================================================================
16. correctness와 예외 처리
======================================================================

공통최소검사:
1) role별supervisedtimestamp분리, paired/archive/aggregate특수권한.
2) 입력builder에E label을poison해도동일입력. 합법적인과거변화에는반응.
3) 같은전처리의LoRA0=nativeforward. 다른전처리끼리F0 identity를강제하지않음.
4) 추가파라미터/LoRA의실제finitegradient,의도한업데이트,head/backbone불변.
5) tau=.5 실제index를config에서읽음. 더짧은horizon의padding을loss분모에서제외.
6) 새인스턴스의선택checkpoint복원과예측일치.
7) TRAINstat/feature/lag/bank/weight/map의출처와hash.
8) 정보가같은alias/degenerate군의중복실행방지.
9) float64 scalar metric 대 vectorized rtol/atol1e-10.
10) loss/출력/metric의단위가표와일치.

FP32동일모델재생normalizedmaxerror<=1e-5, bitwise 여부도별도기록.
다른precision/batch를완전히같게만드는검사를새gate로가져오지않음.
추가파라미터가toy에서는영향을줘야하는데모델에서는항상0gradient면
native normalization/경로누락을진단하고숨기지않음. 차단된군만NUMERICAL/CONSTRUCT상태.

버그수정:
학습전: 명세의수식을올바르게구현하는수정은허용,로그/검사/새hash.
학습중: 같은seed/recipe의정확resume만허용. 반영여부모호한update자동재실행금지.
방법정의/정보/손실이바뀌는수정은새실험이므로이번queue에서자동허용하지않음.
완료후: 집계/문서오류는원기록보존후교정. 학습weights/원점/불리한seed를바꾸지않음.

한군오류시:
가능한나머지군은실행하되빠진직접대비는NOT_COMPARABLE.
그track을성능FAIL로쓰지않고문제와미실행부분표시.
공통loader오염이면그loader사용track만차단하고독립cachedtrack은계속가능.
worker는오류를무시하는blanket except로숫자를채우지않음.

======================================================================
17. 실행 orchestration — 명세 완료 후 스스로 순차 진행
======================================================================

먼저 audit-only로모든9개명세/데이터경로/예산/기존중복을확인.
MASTER_MANIFEST에9개ID,질문,군,원천,loss,seed,예산,직접대비,상태를저장.
미지정방법을LLM이임의발명하지않음. 전체specs를seal한뒤순차실행.

새CLI 인터페이스(실제구현후--help로확인):
  prepare-all   : 전체명세와입력계약/중복/예산확인, 성능채점하지않음
  run-all       : 고정queue순서대로구현검사->학습/CPU비교->선택->평가->검산->보고
  status        : 각track상태,완료fits/updates,현재단계,남은예산
  resume-all    : hash가맞는완전상태에서만이어가기
  verify-all    : 원점수/선택/예산/과거보존재계산
  report        : 모든완료/미실행결과를통합
옵션은이문서가새로요구하는인터페이스이며기존에있다고가정하지않는다.

개념적흐름:
for track in fixed_queue:
    if exact_completed_run_exists:
        audit_and_reference(); continue
    if track_preconditions_are_blocked:
        save_blocked_report(); continue
    try:
        test_construct_and_small_model()
        run_prespecified_comparisons_within_cap()
        seal_selection_and_any_calibrators()
        predict_then_score()
        verify_and_write_korean_report()
    except local_recoverable_failure:
        preserve_partial_and_continue_next_track()
    except shared_safety_or_global_budget_failure:
        save_global_paused_queue(); stop
write_master_decision(max_topics=2, allow_empty=True)

새훈련은선택2LR×seed73100에서방법별LR고정후반복2seed.
후보가E에서나빠보인다고뒤후보recipe를수정하지않음.
한후보완료마다receipt/status/한국어요약을갱신하고계속진행.
아무업데이트없이무기한대기하는driver를만들지않음.
CLI자체세션/실행시간한계가오면완전상태와정확한재개명령을남김.
완료전'백그라운드에서끝내겠다'는말로종료하지않음.

======================================================================
18. 산출물
======================================================================

루트:
MASTER_PROTOCOL.md (이문서의repo복사)
MASTER_MANIFEST.json
QUEUE_STATUS.json
BUDGET_LEDGER.csv
SOURCE_AND_EXPOSURE_LEDGER.md
MASTER_REPORT.md (한국어)
FINAL_DECISION.md (최대2개다음문제또는0개)

각track:
TOPIC_ONEPAGE.md
PROTOCOL.json + hash
LITERATURE_BOUNDARY.md
data_receipt.json / origins.csv / permissions.json
train_statistics.json / feature_or_transform_manifest
fit_manifest.csv / optimizer_log.csv / LR_selection.json / selections.json
predictions_manifest.json / raw_scores.csv / contrasts.csv / uncertainty.csv
resources.csv / verification.json / STATUS.json
REPORT.md (한국어), LIMITATIONS.md
CPU-only트랙은 neural_fit=0을명시하고CPU fitting/예측재생을별도표시.
큰raw data/weights/predictions는ignoredcache. git에는작은코드/원점수/해시/보고서.

REPORT 순서:
① 무슨문제와사용가능정보인가?
② 선행이무엇을보였고이번변경과어떻게연결되는가?
③ 어떤군/조건/정보/비용을맞췄는가?
④ 실제로얼마나실행했고무엇이막혔는가?
⑤ 원점수·효과·seed·조건손익은어떤가?
⑥ 가장단순한대안으로충분한가,미해결된정식선행대조는무엇인가?
⑦ 다음방법을정의할근거가있는가? 추정원인과관찰구분.

실측후그림최소2개/완료track:
- 주요대비 pairedgain(모든반복seed와조건포함)
- 조건/목표별손익(accuracy-stability,rate,delay,aggregate등).
없거나실행못한값을0으로그리지않음. CI없는대상에가짜errorbar금지.
새plot을만들기위해평가목표를바꾸지않음.

======================================================================
19. 참조 출처와 읽기 범위
======================================================================

[확인] 이번문서작성에서공식서지/초록과표시한원문부분을재확인했다.
모든공식방법을실행/재현한것은아니다. 정확한모델내부API는로컬설치코드검사필요.

[S1] 기준 repo와 native LoRA/data:
https://github.com/CanelE452/tsfm-peft-method-screen/tree/485b15b3990236d0372fc074f1c80c7f1df057e2
https://github.com/CanelE452/tsfm-peft-method-screen/blob/485b15b3990236d0372fc074f1c80c7f1df057e2/src/tsfm_peft_screen/lora.py
https://github.com/CanelE452/tsfm-peft-method-screen/blob/485b15b3990236d0372fc074f1c80c7f1df057e2/src/tsfm_peft_screen/data.py
https://github.com/CanelE452/tsfm-peft-method-screen/blob/485b15b3990236d0372fc074f1c80c7f1df057e2/research/overnight_20260913/data_receipt.json
[S2] 기존FR 계약 / 이전합계감독기록연결:
https://github.com/CanelE452/tsfm-peft-method-screen/blob/485b15b3990236d0372fc074f1c80c7f1df057e2/docs/CANDIDATE_04.md
https://github.com/CanelE452/mltimeseries/blob/main/README.md
[S3] t-PatchGNN: Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach (2024/ICML). 공식초록/서지확인.
https://proceedings.mlr.press/v235/zhang24bw.html
[S4] Retrieval Augmented Time Series Forecasting (2025/ICML). 공식초록/정보사용정의확인.
https://proceedings.mlr.press/v267/han25d.html
[S5] FlowState: Sampling-Rate-Equivariant Time-Series Forecasting (2026/ICML). 저자기관의채택정보/설계설명확인.
https://research.ibm.com/publications/flowstate-sampling-rate-equivariant-time-series-forecasting
[S6] Using dynamic loss weighting to boost improvements in forecast stability. 저자원문v2의pairedloss/안정성정의확인. 최종저널연도는CLI가출판사와재확인; 미확인시2024공개/2025v2로적음.
https://arxiv.org/html/2409.18267v2
[S7] Fredformer: Frequency Debiased Transformer for Time Series Forecasting (2024/KDD). 저자원문확인.
https://arxiv.org/html/2406.09009v2
[S8] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models (2025/NeurIPS). 저자원문,공식저장소의게재표기를확인.
https://arxiv.org/html/2506.14087v1
https://github.com/zqiao11/MSFT
[S9] Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators (2024/ICLR). 공식초록/공유대조범위확인.
https://proceedings.iclr.cc/paper_files/paper/2024/hash/b52b07a239a7afa155ca25cf17a55074-Abstract-Conference.html
[S10] Temporal Disaggregation of Time Series (2013/The R Journal). 공식서지/시간분해문제확인.
https://journal.r-project.org/articles/RJ-2013-028/
[S11] Efficiently Generating Correlated Sample Paths from Multi-step Time Series Foundation Models (2025/NeurIPS TSFM Workshop). 저자기관의초록/워크숍게재확인.
https://www.amazon.science/publications/efficiently-generating-correlated-sample-paths-from-multi-step-time-series-foundation-models
[S12] TACTiS-2: Better, Faster, Simpler Attentional Copulas for Multivariate Time Series (2024/ICLR). 공식 서지와 초록 확인.
https://proceedings.iclr.cc/paper_files/paper/2024/hash/63796148c99205adb0fcac069cc714d4-Abstract-Conference.html

모든구체숫자(4채널,64원점,512updates,LR,각kernel/가중규칙)는[설계]다.
논문의보편적PASS기준/정식bestrecipe라고쓰지않는다.
이문서는설계와작은수학검산을제공하며실제GPU학습을실행한결과가아니다.

======================================================================
20. CLI에 전달할 실행 문장
======================================================================

이 MASTER_CLI.txt 하나를사용해9개후보를고정한순서로실행해.
먼저모든명세를완성·봉인하고,의미있는직접비교를정해진예산안에서수행해.
후보하나가중단되거나성능이좋지않아도독립된다른후보는계속해.
기존결과를보존하고같은계산은중복실행하지마.
가설/원점/정보/seed/목표를결과에맞춰바꾸지마.
실행완료·문제의근거·단순대안의충분성·방법의추가가치·신규성을별도로판정해.
최종적으로다음에집중할문제최대2개를제안하되,근거가없으면0개라고적어.
새구조·새dataset·무제한후속탐색을자동으로추가하지마.


======================================================================
21. 동봉 검산 코드의 역할
======================================================================

reference_checks.py는 Python 3 + NumPy로 실행하는 합성 수학 검사다.
  python reference_checks.py

이 코드는 실제 데이터 다운로드, Chronos 연결, GPU 학습, 실험 runner를 수행하지 않는다.
PASS는 수식/정보 권한의 작은 예제 검산이 통과했다는 뜻일 뿐이다.
CPU 검산 코드와 실제 구현의 테스트를 분리하고, 실제 입력에서도 동일 조건을 재검사한다.
동봉 CHECKS_NOTE.md와 README.txt는 설명이고, 실행 조건의 권위는 이 MASTER_CLI.txt 한 파일이다.
압축본과 단독 TXT는 내용이 동일하며 서로 다른 실험 버전이 아니다.
