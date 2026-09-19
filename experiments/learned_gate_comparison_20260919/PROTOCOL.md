# 고정 MAG와 학습형 gate 비교 — 승인된 실행 계약

사용자 `자동시작해` 승인. 기준 e3def0f. MAG_ONLY를 변경 없이 후보로 유지하며 C3 지속성 규칙은 재튜닝하지 않는다. 새 두 군은 설명용 대조이며 새 논문 기여라고 미리 선언하지 않는다. 이번 단위 종료 후 새 후보·추가 학습을 시작하지 않는다.

## 수식·정보·선행

Chronos-Bolt-small의 source/seed별 학습된 B0를 동결하고 raw512 입력을 그대로 읽는다. patch32×embedding512에 기존 residual512→8 GELU→512, cap*tanh 잔차를 더한다. cap은 원래 embedding의 patch별 RMS의 중앙값, 최소1e-6, detached다. TOKEN_GATE의 g_j=sigmoid(W_g h_j+b_g)는 관측 embedding만 읽는 공유 Linear512→1이다. TOKEN_GATE_ENTROPY는 동일 gate에 평균 Bernoulli entropy/log(2)를 .01 곱해 normalized2pinball에 더한다. entropy 계산만 float eps에서 clamp한다. 두 군 모두 gate weight/bias0 (g=.5), residual up0, 같은 down 초기값(seed+200000)이다. 초기 예측=B0. residual8712+gate513=9225개로 MAG보다5.89% 많다. 미래/clean x/오류 위치/생성 state/true delta를 gate에 주지 않는다.

GateRA의 sigmoid token gate와 entropy 억제 원리를 같은 additive residual 위치에 통제 이식한다. 원 논문의 HiRA multiplicative weight adaptation 및 NLP backbone 전체 재현이 아니다. 공식 구현은 이전 검색에서 확인되지 않았다. entropy 계수 .01은 공식 계수라고 주장하지 않으며 이번에 단일 값으로 고정한다. https://ojs.aaai.org/index.php/AAAI/article/view/40538 . prior review의 확인 범위를 보존한다.

## 학습·선택·예산

Electricity/ETTm1 기존 TRAIN/V arrays 및 sigma/B0를 hash 재사용한다. 두 원천×두 군×두LR(1e-4/3e-4) selection seed81550=8fits. V평균 nMAE(REFERENCE/POINT8/BURST8/SHIFT4/SHIFT8)로 LR선택 후81551/81552=8fits. 총16fits/16384mainupdates +4설정×2smoke=8updates. 1024updates,32epochs×32, batch/micro32,FP32/TF32off/dropout0,AdamW(.9,.999)/eps1e-8/wd0/gradclip1. checkpoint0/256/512/768/1024. 동률 낮은LR/이른step. 순서rng(84100,source,seed,epoch). validation 목적에는 entropy를 더하지 않는다. 모든 군은 같은 draws/labels/optimizer 기회를 받는다. 유리하지 않은 결과로 다른 군을 생략하지 않는다.

CPU 식 검사와 실제 Chronos 초기B0동일/adapter off/동결 가중치·buffer보존/gate update/fresh restore를 분리한다. GPU RustDesk만 예외, 불명확한 update중단은 임의 재실행하지 않는다. 안전 차단·구현 오류·성능 부진을 구분한다.

## 평가와 노출

기존 electricity/electricity_transfer/ettm1 standard 및9shape의 B0/PLAIN/C3/MAG 96views hash재사용. 새 두 군의selected/fixed1024·두seed·세패널·두형식48views. 이전 POS_ONLY 및δ 비교는 기존 완료 보고서를 참조하며 재학습하지 않는다.

새 NESO는 이전 다운로드2026CSV의2026-07-01 이후, UTC의 **완전한 origin 날짜55개 모두**를 기존 select_days_reference(period24,count55,seed90301)로 선택한다. 마지막 부분 날짜(8/25)는 평가 성능을 읽기 전 제외한다. 이전 메타데이터 상한56일을64/128일이라고 하지 않는다. phase 차이≤1, target64시간 모두 유효, 기존H1 target과 불겹침을 검사한다. context512시간은 합법적인 이전 관측을 허용한다. sigma는2025H1기존값. 새 다운로드/새 TRAIN없음. ND결측으로55일 미달이면BLOCKED_DIVERSITY, 다른날짜로대체하지않음. 같은provider/ND계열의 시간 전이이며 독립source가 아니다. 파일은 이전부터 존재했고 제한된 프로젝트 기록으로 전세계 비노출을 증명할 수 없다.

standard10states×2draw 기존 transform/seed식 유지. 새shape는55origins전부×9형태×2sign; 기존shape는그대로64origins. measurementfault는x만, 지속변화는x/y일관변경. 이전4군과 새2군의 electricity-trained두seed selected/fixed1024를 새NESO에 추론:48views, 총192views. 학습·LR·checkpoint를 고정한 뒤 전체예측저장→정답채점. 정답은별도파일,미래값으로origin선택·gate·정규화하지않음.

## 사전 판단·보고

주family4: MAG 대 두학습gate 각각 × electricity_transfer 및 새NESO의 selected SHIFT8. index7일block bootstrap2000, 양측family4 Bonferroni95%구간, 공통두seed 방향을 보고한다. 사전에 정한 근거 문턱은 (i) 네 비교의 보정구간 하한>0, (ii) 각 두seed gain>0, (iii) 두패널에서MAG가B0/PLAIN보다REFERENCE와FAULT nMAE를 악화시키는 정도 각각≤1%. 이 문턱은 제한된 추가 가치 근거이며 논문PASS가 아니다. 학습gate가동등/우수하면이식위치에일반gate로충분할가능성, 부호혼재면불확실/전이제한으로기록한다. 임의조건제외나후보재정의없음. 새기간약8주/두seed, 짧은시계열block·기존개발E·사후MAG선정 때문에구간은탐색전체오류율/독립재현을보장하지않는다.

seed원점수, 모든조건/형태/불리한ETTm1, fixed1024, 학습gate평균/entropy, 자원 및파라미터, MAG대PLAIN의가치와C3대MAG의지속성가치를분리한다. 통계검산·원점수CSV·그림·한국어REPORT/FINAL_DECISION, scope commit/push. raw/weights/predictions는localmanifest로보존하며GitHub만으로완전재현가능하다고하지않는다.
