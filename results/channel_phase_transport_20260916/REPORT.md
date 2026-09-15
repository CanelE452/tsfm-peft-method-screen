# 같은 시간대 표현 집계의 입력 조건화 — 실행 결과

**한 후보(CONDITIONED)와 두 동일 용량 대조(POINTWISE/UNIFORM)의 유한 비교를 완료했다. 노출된 개발 E이며 독립 SCREEN_PASS가 아니다.**

실제12/12fits, 15240/15240 본학습updates, smoke12updates. 기존balanced LH4fits와head-only4fits는재학습하지않았다. 미완료fit0,추가후속학습0. 실행오류/재시도0.

[최종 판단](FINAL_DECISION.md)과 [실패 원인·남은 교란](INTERPRETATION.md)을 함께 읽을 수 있다. 이번 후보는 같은 용량의 단순 adapter를 넘지 못했고, LH 대비 정확도 손실도 사전 허용폭을 넘었다.

## 무엇을 왜 바꿨는가

동결한MOMENT encoder의마지막표현h에16차원bottleneck residual을더했다. POINTWISE는각patch를따로변환하고, UNIFORM은24시간간격의4patch를같게평균하며, CONDITIONED는같은위상patch의실제입력거리에따라가중평균한다. 후보의추가가치는일반비선형adapter와uniform집계를각각넘는지로판단한다. 세arm 모두606304개를학습하고head·초기예측·sample순열·학습률·평가를같게맞췄다. E의추가정답이나온라인적응은쓰지않았다. [완전한수식·사전정책](PROTOCOL.md).

## 원점수

| 원천 | recipe | POINTWISE MSE | UNIFORM MSE | CONDITIONED MSE | LoRA+head MSE | head-only MSE |
|---|---|---:|---:|---:|---:|---:|
| electricity | selected | 0.290063 | 0.290495 | 0.290178 | 0.280010 | 0.297230 |
| electricity | matched_old_epoch | 0.332379 | 0.333205 | 0.332934 | 0.312357 | 0.345150 |
| traffic | selected | 0.420738 | 0.421493 | 0.421276 | 0.386548 | 0.436876 |
| traffic | matched_old_epoch | 0.532362 | 0.532066 | 0.532220 | 0.507064 | 0.553986 |

각방법의V최저checkpoint비교와사전고정epoch의같은updates비교를모두공개했다. source안에서두seed MSE를평균하며서로다른source의raw단위를섞지않는다. [seed별MSE/MAE/rawMAE·자원](scores.csv), [개별효과](comparisons.csv), [모든채널](channel_scores.csv).

| 원천 | 후보의대조 | 평균MSE개선 | 두seed개선 | 사전1%조건 |
|---|---|---:|---|---|
| electricity | POINTWISE | -0.039% | False | False |
| electricity | UNIFORM | +0.109% | True | False |
| traffic | POINTWISE | -0.128% | False | False |
| traffic | UNIFORM | +0.051% | False | False |

참고 [paired block bootstrap 구간](descriptive_uncertainty.csv)은 두seed를같은원점에서평균하고, E원점을시간순8블록으로묶어2000회재표집했다(seed9018). 후보·대조에같은블록을적용했다. 노출된E의설명용민감도이며독립검증·선택편향보정이나새PASS gate가아니다. 이진조건이충족되지않아도효과의부재를확정하지않는다.

## 자원과예측을함께본판단

| 원천 | 후보/LH MSE비 | 학습파라미터절감 | 최대peak비 | 사전자원·정확도동시조건 |
|---|---:|---:|---:|---|
| electricity | 1.036312 | 20.428% | 0.2678 | False |
| traffic | 1.089843 | 20.428% | 0.2683 | False |

1%허용폭/파라미터20%절감/peak50%절감은본실험전에고정했다. 이를통계적동등성이나독립성공으로해석하지않는다. 작은메모리만으로성능손실을성공으로바꾸지않는다. peak는각fit실제학습최대allocated이며상주GPU사용량과다르다. 고정된LoRA모듈forward가남아있어최소LP실행의최적화벤치마크도아니다.

총 모델 파라미터는 LH 36,099,360개, 후보 36,115,744개로 후보가 16,384개 더 많다. 감소한 것은 학습 파라미터와 역전파 저장량이다. [실제 shape·학습 대상 목록](parameter_inventory.json). 실행 중 짧은 CPU 입력·파라미터 검사도 병행했으므로 wall 시간은 전용 성능 벤치마크로 일반화하지 않는다.

| fit | epochs | updates | peak MiB | 학습step합계초 |
|---|---:|---:|---:|---:|
| 00_electricity_41000_POINTWISE | 20 | 1280 | 286.561 | 27.95 |
| 01_electricity_41000_UNIFORM | 20 | 1280 | 288.826 | 28.00 |
| 02_electricity_41000_CONDITIONED | 20 | 1280 | 286.326 | 27.75 |
| 03_electricity_41001_POINTWISE | 20 | 1280 | 286.326 | 28.07 |
| 04_electricity_41001_UNIFORM | 20 | 1280 | 286.326 | 27.80 |
| 05_electricity_41001_CONDITIONED | 20 | 1280 | 286.326 | 27.59 |
| 06_traffic_41000_POINTWISE | 20 | 1260 | 286.326 | 27.49 |
| 07_traffic_41000_UNIFORM | 20 | 1260 | 286.326 | 27.56 |
| 08_traffic_41000_CONDITIONED | 20 | 1260 | 286.326 | 27.76 |
| 09_traffic_41001_POINTWISE | 20 | 1260 | 286.326 | 27.39 |
| 10_traffic_41001_UNIFORM | 20 | 1260 | 286.326 | 27.33 |
| 11_traffic_41001_CONDITIONED | 20 | 1260 | 286.326 | 27.63 |

Controller 1738.4초,최소GPU여유8723MiB,비승인compute0표본. RustDesk만허용했다. 총연구비용과선택checkpoint까지의적응비용은구분하며후자는scores.csv에있다. epoch상한도달은수렴의증명이아니다.

## 검산·한계·미실행

고유예측304개/304개 MSE·MAE·rawMAE기록을독립float64 scalar로검산했다. 최대MSE차1.11e-16. 선택checkpoint12개를새모델에서재생해차이0. FP64수식/gradient,초기identity,반례와uniform동치,학습대상실제변경·동결가중치/버퍼보존,미래값poison불변을확인했다. 기존2043개결과·연구파일보존. [검산](verification.json).

구성요소개발신호=False,자원·정확도동시신호=False. 독립SCREEN_PASS는NOT_EVALUATED,신규성은UNRESOLVED_PRIOR_COMPONENTS_KNOWN,새방법론주제는NOT_CONFIRMED다. 좋은seed/채널만선택하거나실패한조건을재튜닝하지않았다.

잔차분해는정답을쓴사후설명용이며배포가능한oracle가아니다. 한E시작위상·두공개source·두seed의개발비교라다른기간/데이터에대한일반화는미검증이다. Autoformer주기집계·일반kernel attention·bottleneck adapter의원리자체는알려져있고,TimePEFT/CoRA전체재현이나우위를본실험이확인하지않는다. [선행검토](RESEARCH_REVIEW.md). 독립미노출평가·추가backbone·신규성확정은미실행이다. 원자료/weights/예측배열은로컬cache에보존하고GitHub에는코드·원점수·hash·검산을보관한다.

봉인된 train의 첫64개 원점에서 입력 조건화 가중치는 모든 검사 행에서 uniform과 달랐다. [입력만 사용한 연산 검사](train_operator_probe.json). 이 활동성은 예측 이득의 증거가 아니다.
