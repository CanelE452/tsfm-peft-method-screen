# 방법론 주장과 완료된 실험 근거

**현재 판정: 고정 MAG의 제한된 추가 가치 근거가 확인됐다.** 승인된16fits/16384main+8smoke,192prediction views의 저장·채점·독립 검산을 완료했다. 사전에 정한 네 주 비교와 평균 손해 한도를 충족했다. 이는 방법론 논문을 뒷받침하는 한 근거의 완료이며, 사용자가 요청한 Time-PEFT와 같은 방법론 논문의 전체 완성·신규성·채택 가능성을 입증한 상태는 아니다.

## 지금 쓸 수 있는 핵심 주장

> 학습된 B0와 원래 관측을 유지하면서, 전체 관측창의 robust 진폭 통계로 추가 patch 잔차를 제한하는 고정 MAG 설계는 이 실험의 큰 합성 지속 변화(SHIFT8)에서 일반 잔차와 두 학습형 embedding gate보다 낮은 예측 오차를 보였다. 이득은 기존 전력 계열 전이와 사전에 고정한 NESO 후반 기간에서 두 공통 seed에 걸쳐 확인됐으나, 원자료·오류 조건의 소폭 손해와 긴 변화 형태·ETTm1의 한계가 남았다.

이 문장은 구체적인 적응 방법의 비교 결과를 말한다. ‘지속성을 알아냈기 때문에 좋아졌다’거나 ‘학습하지 않는 gate가 일반적으로 우월하다’는 주장으로 바꾸지 않는다. MAG를 후보로 채택한 결정은 기존 E 결과를 본 뒤 이루어졌음을 공개한다. 새로운 평가 기간의 긍정 결과로 과거의 사후 선택을 지우지 않는다.

## 주 비교와 단순 대안

| panel | baseline | proposed_nmae | baseline_nmae | gain_pct | bonf4_low | bonf4_high | seed_gains |
| --- | --- | --- | --- | --- | --- | --- | --- |
| electricity_transfer | TOKEN_GATE | 0.3630 | 0.3754 | 3.2927 | 2.7733 | 3.7896 | {"81551": 0.9905263115159157, "81552": 5.462805460674447} |
| electricity_transfer | TOKEN_GATE_ENTROPY | 0.3630 | 0.3734 | 2.7801 | 2.2688 | 3.2870 | {"81551": 0.12289800444797994, "81552": 5.268282695287951} |
| neso_2026_jul_aug | TOKEN_GATE | 0.2779 | 0.2840 | 2.1322 | 1.0207 | 3.5288 | {"81551": 1.4031239713732102, "81552": 2.825987776417138} |
| neso_2026_jul_aug | TOKEN_GATE_ENTROPY | 0.2779 | 0.2828 | 1.7195 | 0.9986 | 2.2814 | {"81551": 0.7230802064301489, "81552": 2.6627275823706498} |

네 보정구간의 하한이 모두0보다 크고, 개별두seed의 이득도모두양수다. 다만 전력전이의entropy대조 대비seed81551 이득은0.123%로작다. 구간은날짜index의7일block을resample한것이며, 훈련seed모집단의불확실성을충분히추정한구간이아니다. 2개seed만으로seed일반화가확립됐다고하지않는다.

![주 비교의 보정구간과 seed별 이득](primary_effects.png)

**단순 잔차와 B0 대비 SHIFT8:**

| panel | baseline | proposed_nmae | baseline_nmae | gain_pct |
| --- | --- | --- | --- | --- |
| electricity_transfer | B0 | 0.3630 | 0.3998 | 9.1860 |
| electricity_transfer | PLAIN | 0.3630 | 0.3774 | 3.7935 |
| neso_2026_jul_aug | B0 | 0.2779 | 0.2914 | 4.6263 |
| neso_2026_jul_aug | PLAIN | 0.2779 | 0.2834 | 1.9415 |

selected뿐아니라fixed1024에서도네학습gate대조의MAG이득은양수였다. 따라서이자료에서관찰한이득을checkpoint선택하나로만설명하기는어렵다. 그렇다고전체원인중진폭feature·초기gate·학습궤적의기여를분리한것은아니다. [공정성 감사](../../../research/learned_gate_comparability_20260919/COMPARABILITY_KO.md)를따른다.

## 보호 한도: 평균 기준과 비열등성 보장의 차이

| panel | condition | baseline | gain_pct | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| electricity_transfer | FAULT | B0 | -0.2026 | -0.3473 | -0.0676 |
| electricity_transfer | FAULT | PLAIN | -0.1173 | -0.2205 | -0.0239 |
| electricity_transfer | REFERENCE | B0 | -0.1739 | -0.3582 | -0.0077 |
| electricity_transfer | REFERENCE | PLAIN | -0.1429 | -0.2629 | -0.0334 |
| neso_2026_jul_aug | FAULT | B0 | -0.4010 | -1.0557 | 0.2010 |
| neso_2026_jul_aug | FAULT | PLAIN | -0.4434 | -0.8453 | 0.0077 |
| neso_2026_jul_aug | REFERENCE | B0 | -0.2425 | -1.0360 | 0.6300 |
| neso_2026_jul_aug | REFERENCE | PLAIN | -0.3008 | -0.7739 | 0.2066 |

사전1%한도는두seed평균점수의악화에적용됐다. 평균은모두한도안이지만, 새NESO에서B0대비REFERENCE와FAULT의일반95%구간하한은각각약−1.036%와−1.056%다. 따라서이결과로‘95%신뢰에서손해≤1%’라는비열등성을입증했다고하지않는다. 이구간은주family4와다른보조기술구간이며여기서새합격기준을만들지않는다.

## 남겨야 할 부정 결과

- ETTm1의MAG selected checkpoint는두seed에서초기0이며B0와같다. 학습이실패한것이아니라검증선택이추가적응을채택하지않은것이다. SHIFT8에서PLAIN보다약0.95%나쁘다.
- 전력전이의긴STEP12_D63에서MAG는B0보다약3.27%,PLAIN보다약2.65%나쁘다. 큰진폭이라는이유만으로모든지속변화에서유리하다고할수없다.
- 전력전이SHIFT4의PLAIN대비추가가치는거의0이고, 새NESO SHIFT4/SHIFT_POINT는seed방향이혼재한다. SHIFT8의강한이득을모든변화형태로확대하지않는다.
- C3의지속성규칙은여전히MAG대비독립적인추가가치가확립되지않았다. 이번긍정결과를C3지속성기전성공으로옮기지않는다.

## 논문 요건별 현재 증거

| 요건 | 현재 근거 | 판단 |
| --- | --- | --- |
| 실행 가능한 작은 추가 적응 방법 | 고정수식·실제Chronos연결·B0보존·실제update·복원 | 구현 완료 |
| 강한기존B0 위의 추가가치 | PLAIN대MAG 및학습gate직접대조,두seed·두주패널 | SHIFT8에서 지지 |
| 단순방법으로충분한가 | PLAIN,학습gate,기존POS_ONLY/δ보고서 | 현재대조들보다SHIFT8에추가이득;모든형태에우위아님 |
| 평가 재사용 통제 | 과거개발E와새NESO55일분리,모든선택후예측저장·채점 | 시간전이근거,독립source아님 |
| 효율 | MAG8712 vs학습gate9225추가params,동일updates | 파라미터차이확인;순수속도우위미입증 |
| 기존방법과의차이·신규성 | 고정robust-statistic gate와입력보존의구체적조합 | gating자체는알려짐;충분한방법론신규성미확정 |
| 정식선행비교 | δ XY와GateRA원리의통제이식,기존대조 | GateRA/Time-PEFT전체동일설정비교아님 |
| 현실적범위 | 합성fault/shift,한backbone,두학습원천 | 실제사건·독립source·더넓은일반화미검증 |

따라서 방법 절과조건부기여를작성할실험근거는강해졌지만, **전체방법론목표달성으로표시하지않는다.** 공식선행전체비교·더넓은독립검증·신규성검토가남아있다는사실과, 이미확인된좁은양성이득을함께보존한다. 이표의미완료항목을이유로이번실험을실행오류/전면성능실패로바꾸지않는다.

이번승인단위의미실행학습0. 새구조·threshold·LR·seed·dataset·후속학습은자동시작하지않았다. 실행계약을다시열지않고여기서종료한다. 아래원자료와코드로결론을검토할수있다.

[전체 REPORT](../../../results/learned_gate_comparison_20260919/REPORT.md) · [최종 결정](../../../results/learned_gate_comparison_20260919/FINAL_DECISION.md) · [독립 검산](../../../results/learned_gate_comparison_20260919/AUDIT.json) · [모든 seed 원점수](../../../results/learned_gate_comparison_20260919/RAW_SCORES.csv) · [자원 측정 범위](../../../research/learned_gate_comparability_20260919/RESOURCE_SCOPE_KO.md)
