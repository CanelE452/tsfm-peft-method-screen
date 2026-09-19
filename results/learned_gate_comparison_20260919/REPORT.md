# 고정 MAG의 학습형 gate 대조 및 시간 전이

**실행·선택·192개 예측 view 저장·채점·독립 검산 완료. 판정: LIMITED_COMPONENT_EVIDENCE. 논문 PASS를 선언하지 않는다.**

## 실행과 검증

16/16 새 본학습,16384main+8smokeupdates. 미실행 승인 학습0. 기존96views 재사용, 새96views(동일 checkpoint alias 포함). 선택봉인→전체예측저장→E채점 순서를 검증했다. 실제 Chronos 초기 B0/adapter off 동일성, gate update, 동결 parameters/buffers 보존, fresh restore 검사 통과. checkpoint80개 SHA, 모든원점의 독립 계산476136개, 기존 원점수960행, 효과/구간 재계산을 확인했다. GPU 미승인외부compute 0, 최소여유8290MiB. RustDesk만 예외.

기존 B0/PLAIN/C3/MAG는 변경하지 않았다. 두 학습형 gate는 같은 잔차에 부착한 원리 통제이며 GateRA 공식 HiRA/NLP 전체의 재현이 아니다. entropy .01은 사전에 고정한 로컬값이다. 새gate9225 vsMAG8712(+513,5.89%). 초기 gate=.5, up=0; 학습초기 B0동일. 모든 군의 TRAIN/labels/V목적/LR/seed/1024updates 기회를 맞췄다. regularizer포함학습loss와 forecast loss는 UPDATE_LEDGER에 분리했다. gate사용량 평균은 실제 관측학습batch 기준이며 개입 인과기여를 뜻하지 않는다.

## 주 비교 — 양수는 MAG 이득

| panel | baseline | proposed_nmae | baseline_nmae | gain_pct | bonf4_low | bonf4_high | seed_gains |
| --- | --- | --- | --- | --- | --- | --- | --- |
| electricity_transfer | TOKEN_GATE | 0.363040 | 0.375400 | 3.292682 | 2.773252 | 3.789576 | {"81551": 0.9905263115159157, "81552": 5.462805460674447} |
| electricity_transfer | TOKEN_GATE_ENTROPY | 0.363040 | 0.373421 | 2.780113 | 2.268788 | 3.286957 | {"81551": 0.12289800444797994, "81552": 5.268282695287951} |
| neso_2026_jul_aug | TOKEN_GATE | 0.277940 | 0.283995 | 2.132216 | 1.020730 | 3.528824 | {"81551": 1.4031239713732102, "81552": 2.825987776417138} |
| neso_2026_jul_aug | TOKEN_GATE_ENTROPY | 0.277940 | 0.282803 | 1.719496 | 0.998562 | 2.281399 | {"81551": 0.7230802064301489, "81552": 2.6627275823706498} |

family4는 두패널×두학습gate의selected SHIFT8만이다. index7일block2000회, 두seed/고정채널에 조건부다. 예측horizon이겹치므로56개/55개 독립표본이라고하지않는다. 짧은새기간/약8block/두seed의불확실성과기존E를본후MAG선정은남는다. 이보정은역사전체탐색선택을제거하지않는다. 사전문턱(네구간하한>0,모든seed양수,기존B0/PLAIN대비REFERENCE/FAULT악화≤1%) 충족: True.

## 원점수·부정 조건

| panel | condition | B0 | C3 | MAG_ONLY | PLAIN | TOKEN_GATE | TOKEN_GATE_ENTROPY |
| --- | --- | --- | --- | --- | --- | --- | --- |
| electricity | FAULT | 0.171460 | 0.171995 | 0.172018 | 0.171823 | 0.171734 | 0.171725 |
| electricity | REFERENCE | 0.166524 | 0.166853 | 0.166890 | 0.166614 | 0.166511 | 0.166629 |
| electricity | SHIFT4 | 0.191307 | 0.189130 | 0.189160 | 0.189093 | 0.189147 | 0.189325 |
| electricity | SHIFT8 | 0.210032 | 0.201709 | 0.201424 | 0.204404 | 0.204271 | 0.202533 |
| electricity | SHIFT_POINT | 0.196221 | 0.193432 | 0.193520 | 0.193546 | 0.193779 | 0.194141 |
| electricity_transfer | FAULT | 0.255210 | 0.255619 | 0.255727 | 0.255428 | 0.255584 | 0.255194 |
| electricity_transfer | REFERENCE | 0.239856 | 0.240185 | 0.240273 | 0.239930 | 0.240011 | 0.239613 |
| electricity_transfer | SHIFT4 | 0.326449 | 0.319635 | 0.319098 | 0.318945 | 0.319229 | 0.317829 |
| electricity_transfer | SHIFT8 | 0.399762 | 0.364538 | 0.363040 | 0.377355 | 0.375400 | 0.373421 |
| electricity_transfer | SHIFT_POINT | 0.342813 | 0.335071 | 0.334295 | 0.334722 | 0.336123 | 0.334341 |
| ettm1 | FAULT | 0.423198 | 0.423214 | 0.423198 | 0.423652 | 0.423588 | 0.423193 |
| ettm1 | REFERENCE | 0.407336 | 0.407226 | 0.407336 | 0.407596 | 0.407482 | 0.407124 |
| ettm1 | SHIFT4 | 0.551405 | 0.549718 | 0.551405 | 0.548210 | 0.547819 | 0.548012 |
| ettm1 | SHIFT8 | 0.545368 | 0.547599 | 0.545368 | 0.540239 | 0.540690 | 0.545137 |
| ettm1 | SHIFT_POINT | 0.589807 | 0.588500 | 0.589807 | 0.586404 | 0.586056 | 0.586334 |
| neso_2026_jul_aug | FAULT | 0.204939 | 0.205647 | 0.205761 | 0.204852 | 0.205128 | 0.205280 |
| neso_2026_jul_aug | REFERENCE | 0.191119 | 0.191506 | 0.191583 | 0.191008 | 0.191050 | 0.191465 |
| neso_2026_jul_aug | SHIFT4 | 0.266394 | 0.265091 | 0.263951 | 0.264542 | 0.265078 | 0.265461 |
| neso_2026_jul_aug | SHIFT8 | 0.291422 | 0.278363 | 0.277940 | 0.283443 | 0.283995 | 0.282803 |
| neso_2026_jul_aug | SHIFT_POINT | 0.259068 | 0.255970 | 0.255654 | 0.256566 | 0.257173 | 0.257787 |

두 seed별 원점수는 RAW_SCORES.csv, 모든날짜/형태는 ORIGIN_SCORES.csv.gz, 채널은 CHANNEL_SCORES.csv다. fixed1024와selected를모두남겼다. 모든9shape와불리한ETTm1조건을제외하지않았다. C3의 지속성규칙과MAG의진폭마스크를혼동하지않는다. 일반adapter이득=B0대PLAIN,고정MAG추가가치=PLAIN대MAG,지속성추가가치=MAG대C3다. 두학습gate가좋다고모든강건PEFT가반증된것이아니며 MAG가좋아도gate일반론자체는기존방법이다.

## 시간 전이와 노출

새NESO origin은2026-07-01이후완전한55개UTC날짜로평가전에고정했다. 마지막부분날짜는사전제외했다. 표본수128/64를중복날짜로채우지않았다. 기존H1 target과불겹침,512문맥은합법적과거로겹칠수있음. 2025H1 sigma고정. 파일전체는이전다운로드됨. 제한된기록상새평가기간이며같은provider/ND계열이므로독립source라고하지않는다. 실제사건레이블없음:합성fault/shift만검증했다. 이기간손해를전세계센서문제실패로일반화하지않는다.

## 비용·한계

| arm | trainable_parameters | optimizer_seconds | peak_MiB |
| --- | --- | --- | --- |
| TOKEN_GATE | 9225.000000 | 36.559414 | 363.574585 |
| TOKEN_GATE_ENTROPY | 9225.000000 | 37.844852 | 363.707520 |

시간은이번측정의평균optimizer초,최대메모리평균이다. 전체검증/추론비용은별도receipt에있다. 기존MAG와다른시각에측정했으므로속도우위를확정하지않는다. gate513개/entropy계산추가비용과예측변화를분리한다.

기존δ/POS_ONLY직접비교는완료보고서를참조했으며재학습하지않았다. 정식GateRA전체/Time-PEFT동일backbone공정재현,다양한독립source,실제오류·변화레이블은여전히없다. 논문기여는관측입력의고정진폭gate가동결B0 위의잔차적응을어떤조건에서개선/제한하는지에국한해야한다. sigmoidgate자체신규성·범용PEFT우위·C3지속성기전성공을주장할수없다.

새후속학습은없다. 모델weights/raw/predictionarrays는localcache이며GitHub에는코드·해시·점수·검산·그림을게시한다. GitHub만으로완전수치재생가능하다고하지않는다.
