# C3 약점 해결 대조 실험 — 한국어 결과

**핵심 결론:** C3의 전력 조건부 이득은 유지되지만, 단순 크기 가중(MAG_ONLY)을 넘어서는 연속성 고유의 추가 가치는 확인하지 못했다. [결과 해석과 논문 주장 수정](INTERPRETATION_KO.md)을 먼저 참고한다.

## 실행 범위와 완료

사용자 다운로드 문서 EXECUTION_CONTRACT.md를 실행 승인에 따라 구현했다. 결과 전 규약/코드 봉인 커밋은396de8d이다. C3/B0는 바꾸지 않고 POS_ONLY/MAG_ONLY/OUTPUT_CONTEXT만 **30/30 fits,30,720 main+12 smoke=30,732 updates**로 실행했다. 반복 가중치18개와 source/전력전이 **54개 새 prediction views**를 전체 저장한 뒤 E를 채점했다. 불리한 성능으로 취소하거나 추가 LR/seed/방법을 만들지 않았다.

각 source×대조는선택seed81550/LR1e-4·3e-4 두경로와선택LR의반복81551/52/53 세경로다.1024updates를모두실행한뒤V로0/256/512/768/1024중선택했다. step0선택은추가효과없음이며실행실패가아니다. C3와기존core/MEAN/ROTATE/RECENCY/F0는부모검증결과를재사용했다.

## 질문과 정보 권한

RECENCY도 입력별 지속성 값들을 정렬해 사용한다. 따라서 RECENCY와 비슷하다고 지속성 정보 전체가 무의미하다고 쓸 수 없다. POS_ONLY는 정적 학습 위치만, MAG_ONLY는 큰 값만 사용하며 OUTPUT_CONTEXT는 동결 B0의예측과관측평균만으로출력을보정한다. 모두원래입력을보존하고미래y/state/clean x/true delta를feature로주지않는다.

POS_ONLY는8712adapter+32logits=8744params, gate LR.01고정이다. MAG_ONLY는8712params. OUTPUT_CONTEXT는72→64 affine+gamma=4673params이며W N(0,.01),b0,gamma0에서시작했다. 초기/off B0 동일성을실제모델에서확인했다. 모든군의TRAIN/V/loss/정답노출/updates는동일하지만파라미터수·optimizer난이도·확률출력표현력까지같다고주장하지않는다. OUTPUT_CONTEXT는quantile간격을보존한다.

OUTPUT_CONTEXT는COSA에서동기를얻은offline구조대조이며TAFAS/COSA의온라인정답도착·갱신을재현하지않았다. [BASELINE_MAPPING.md](BASELINE_MAPPING.md)의공식선행전체와의차이를유지한다. 일반adapter·learnedgate·identity시작자체는신규성이아니다.

## 자료 및 범위

Electricity4계열,동일날짜의transfer16계열,ETTm1 4계열을그대로사용했다. TRAIN256일/V64일/E128일,두E draw,기존9개shape를유지했다. 모두반복사용한개발E다. transfer16은과거노출확인4/불명12이며독립source가아니다. 실제오류/소비변화사건label은없다. 새NESO2026평가,LCL준비/다운로드,ETTm2기전학습은하지않았다. SAME/DIFFERENT의이전분석을새핵심결과로다시계산하지않았다.

## 주 비교: SHIFT8

양수는C3가대조보다낮은nMAE,음수는대조가유리하다.각계열/원점/draw동가중오차의비율이며개별origin백분율평균이아니다.7일paired block2000회,고정seed평균에조건부인구간.transfer는계열+시간구간도보존한다.각 패널 안의 3개 사전 대조에 대한 Bonferroni 구간이다. 전체 패널·조건의 다중성이나 누적 연구 선택 편향을 제거하지 않는다.

| panel | baseline | new_nmae | baseline_nmae | gain_pct | ci_type | ci_low_pct | ci_high_pct | bonferroni3_low_pct | bonferroni3_high_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| electricity | POS_ONLY | 0.200891 | 0.201520 | 0.312083 | time | 0.018920 | 0.565042 | -0.060332 | 0.618824 |
| electricity | MAG_ONLY | 0.200891 | 0.200736 | -0.077494 | time | -0.136881 | -0.015409 | -0.146942 | 0.005050 |
| electricity | OUTPUT_CONTEXT | 0.200891 | 0.203909 | 1.480085 | time | 1.024052 | 1.971624 | 0.924738 | 2.120092 |
| electricity_transfer | POS_ONLY | 0.358660 | 0.361168 | 0.694312 | time | 0.518499 | 0.856241 | 0.479936 | 0.891022 |
| electricity_transfer | POS_ONLY | 0.358660 | 0.361168 | 0.694312 | series_and_time | 0.377340 | 1.082671 | 0.311426 | 1.175803 |
| electricity_transfer | MAG_ONLY | 0.358660 | 0.357760 | -0.251519 | time | -0.324375 | -0.178174 | -0.338086 | -0.160706 |
| electricity_transfer | MAG_ONLY | 0.358660 | 0.357760 | -0.251519 | series_and_time | -0.430722 | -0.103690 | -0.470086 | -0.072983 |
| electricity_transfer | OUTPUT_CONTEXT | 0.358660 | 0.369368 | 2.898901 | time | 2.055435 | 3.710056 | 1.859884 | 3.876774 |
| electricity_transfer | OUTPUT_CONTEXT | 0.358660 | 0.369368 | 2.898901 | series_and_time | 1.475726 | 4.735510 | 1.188639 | 5.267671 |
| ettm1 | POS_ONLY | 0.547346 | 0.539636 | -1.428716 | time | -1.874919 | -0.980050 | -1.956867 | -0.861630 |
| ettm1 | MAG_ONLY | 0.547346 | 0.546065 | -0.234657 | time | -0.447594 | -0.027569 | -0.480196 | 0.005558 |
| ettm1 | OUTPUT_CONTEXT | 0.547346 | 0.544575 | -0.508765 | time | -1.223069 | 0.215971 | -1.382525 | 0.309849 |

electricity: POS_ONLY 대비 +0.3121%, MAG_ONLY 대비 -0.0775%, OUTPUT_CONTEXT 대비 +1.4801%
electricity_transfer: POS_ONLY 대비 +0.6943%, MAG_ONLY 대비 -0.2515%, OUTPUT_CONTEXT 대비 +2.8989%
ettm1: POS_ONLY 대비 -1.4287%, MAG_ONLY 대비 -0.2347%, OUTPUT_CONTEXT 대비 -0.5088%

## 세 seed와 실제 선택

| panel | baseline | seed | new_nmae | baseline_nmae | gain_pct |
| --- | --- | --- | --- | --- | --- |
| electricity | POS_ONLY | 81551 | 0.201225 | 0.201912 | 0.340347 |
| electricity | POS_ONLY | 81552 | 0.202194 | 0.203148 | 0.469692 |
| electricity | POS_ONLY | 81553 | 0.199255 | 0.199501 | 0.122987 |
| electricity | MAG_ONLY | 81551 | 0.201225 | 0.201079 | -0.072714 |
| electricity | MAG_ONLY | 81552 | 0.202194 | 0.201770 | -0.210173 |
| electricity | MAG_ONLY | 81553 | 0.199255 | 0.199359 | 0.051969 |
| electricity | OUTPUT_CONTEXT | 81551 | 0.201225 | 0.204679 | 1.687728 |
| electricity | OUTPUT_CONTEXT | 81552 | 0.202194 | 0.204583 | 1.168056 |
| electricity | OUTPUT_CONTEXT | 81553 | 0.199255 | 0.202465 | 1.585464 |
| electricity_transfer | POS_ONLY | 81551 | 0.361402 | 0.364375 | 0.816100 |
| electricity_transfer | POS_ONLY | 81552 | 0.367675 | 0.371666 | 1.073838 |
| electricity_transfer | POS_ONLY | 81553 | 0.346904 | 0.347462 | 0.160632 |
| electricity_transfer | MAG_ONLY | 81551 | 0.361402 | 0.360709 | -0.191989 |
| electricity_transfer | MAG_ONLY | 81552 | 0.367675 | 0.365370 | -0.630873 |
| electricity_transfer | MAG_ONLY | 81553 | 0.346904 | 0.347202 | 0.085841 |
| electricity_transfer | OUTPUT_CONTEXT | 81551 | 0.361402 | 0.371880 | 2.817717 |
| electricity_transfer | OUTPUT_CONTEXT | 81552 | 0.367675 | 0.380617 | 3.400103 |
| electricity_transfer | OUTPUT_CONTEXT | 81553 | 0.346904 | 0.355607 | 2.447349 |
| ettm1 | POS_ONLY | 81551 | 0.538887 | 0.538887 | 0.000000 |
| ettm1 | POS_ONLY | 81552 | 0.556311 | 0.545952 | -1.897467 |
| ettm1 | POS_ONLY | 81553 | 0.546841 | 0.534070 | -2.391137 |
| ettm1 | MAG_ONLY | 81551 | 0.538887 | 0.538887 | 0.000000 |
| ettm1 | MAG_ONLY | 81552 | 0.556311 | 0.551850 | -0.808349 |
| ettm1 | MAG_ONLY | 81553 | 0.546841 | 0.547457 | 0.112653 |
| ettm1 | OUTPUT_CONTEXT | 81551 | 0.538887 | 0.538887 | 0.000000 |
| ettm1 | OUTPUT_CONTEXT | 81552 | 0.556311 | 0.551850 | -0.808349 |
| ettm1 | OUTPUT_CONTEXT | 81553 | 0.546841 | 0.542990 | -0.709214 |

| source | arm | seed | lr | step |
| --- | --- | --- | --- | --- |
| electricity | POS_ONLY | 81551 | 0.000300 | 768 |
| electricity | POS_ONLY | 81552 | 0.000300 | 1024 |
| electricity | POS_ONLY | 81553 | 0.000300 | 1024 |
| electricity | MAG_ONLY | 81551 | 0.000300 | 768 |
| electricity | MAG_ONLY | 81552 | 0.000300 | 768 |
| electricity | MAG_ONLY | 81553 | 0.000300 | 1024 |
| electricity | OUTPUT_CONTEXT | 81551 | 0.000300 | 512 |
| electricity | OUTPUT_CONTEXT | 81552 | 0.000300 | 1024 |
| electricity | OUTPUT_CONTEXT | 81553 | 0.000300 | 256 |
| ettm1 | POS_ONLY | 81551 | 0.000300 | 0 |
| ettm1 | POS_ONLY | 81552 | 0.000300 | 1024 |
| ettm1 | POS_ONLY | 81553 | 0.000300 | 768 |
| ettm1 | MAG_ONLY | 81551 | 0.000100 | 0 |
| ettm1 | MAG_ONLY | 81552 | 0.000100 | 0 |
| ettm1 | MAG_ONLY | 81553 | 0.000100 | 1024 |
| ettm1 | OUTPUT_CONTEXT | 81551 | 0.000100 | 0 |
| ettm1 | OUTPUT_CONTEXT | 81552 | 0.000100 | 0 |
| ettm1 | OUTPUT_CONTEXT | 81553 | 0.000100 | 0 |

## 원점수와 손익

| panel | arm | REFERENCE | FAULT | SHIFT4 | SHIFT8 | SHIFT_POINT |
| --- | --- | --- | --- | --- | --- | --- |
| electricity | C0 | 0.167110 | 0.172251 | 0.191848 | 0.207544 | 0.196577 |
| electricity | C2 | 0.167338 | 0.172812 | 0.189463 | 0.202896 | 0.193753 |
| electricity | C3 | 0.167496 | 0.172893 | 0.189495 | 0.200891 | 0.193630 |
| electricity | F0 | 0.169059 | 0.174216 | 0.265069 | 0.321226 | 0.285354 |
| electricity | MAG_ONLY | 0.167546 | 0.172935 | 0.189563 | 0.200736 | 0.193723 |
| electricity | M_RECENCY | 0.167396 | 0.172720 | 0.189575 | 0.201176 | 0.193765 |
| electricity | OUTPUT_CONTEXT | 0.167146 | 0.172289 | 0.191473 | 0.203909 | 0.196394 |
| electricity | POS_ONLY | 0.167230 | 0.172697 | 0.189326 | 0.201520 | 0.193719 |
| electricity_transfer | C0 | 0.242005 | 0.257960 | 0.328048 | 0.386121 | 0.343452 |
| electricity_transfer | C2 | 0.242189 | 0.258294 | 0.318977 | 0.367478 | 0.335079 |
| electricity_transfer | C3 | 0.242293 | 0.258406 | 0.319916 | 0.358660 | 0.335089 |
| electricity_transfer | F0 | 0.231110 | 0.247261 | 0.444540 | 0.560445 | 0.505777 |
| electricity_transfer | MAG_ONLY | 0.242373 | 0.258534 | 0.319662 | 0.357760 | 0.334655 |
| electricity_transfer | M_RECENCY | 0.242048 | 0.258003 | 0.319778 | 0.359308 | 0.335394 |
| electricity_transfer | OUTPUT_CONTEXT | 0.241916 | 0.257889 | 0.323917 | 0.369368 | 0.340994 |
| electricity_transfer | POS_ONLY | 0.242209 | 0.258272 | 0.318824 | 0.361168 | 0.335583 |
| ettm1 | C0 | 0.406544 | 0.421938 | 0.549134 | 0.544575 | 0.587048 |
| ettm1 | C2 | 0.407354 | 0.423055 | 0.544939 | 0.537419 | 0.581408 |
| ettm1 | C3 | 0.406513 | 0.422099 | 0.546446 | 0.547346 | 0.585155 |
| ettm1 | F0 | 0.405979 | 0.417559 | 0.920601 | 1.029181 | 0.990499 |
| ettm1 | MAG_ONLY | 0.406647 | 0.422135 | 0.548062 | 0.546065 | 0.586468 |
| ettm1 | M_RECENCY | 0.406469 | 0.422187 | 0.545493 | 0.547476 | 0.584625 |
| ettm1 | OUTPUT_CONTEXT | 0.406544 | 0.421938 | 0.549134 | 0.544575 | 0.587048 |
| ettm1 | POS_ONLY | 0.407221 | 0.422976 | 0.544991 | 0.539636 | 0.581695 |

개별fault6종,shapes9종,모든seed와기존대조는RAW_SCORES.csv에있다. C3/C2·C3/RECENCY 및각신규대조/C0·C2를MATCHED_CONTRASTS.csv에함께보고한다. 원자료손해를임의허용오차로삭제하지않고평균양수/CI0포함을동등성또는논문PASS로변환하지않는다. shape유리조건을새주조건으로승격하지않았다.

C3와각대조의주요상태차이:

| panel | condition | baseline | gain_pct | ci_low_pct | ci_high_pct |
| --- | --- | --- | --- | --- | --- |
| electricity | FAULT | POS_ONLY | -0.113734 | -0.213357 | -0.004691 |
| electricity | FAULT | MAG_ONLY | 0.024098 | -0.001438 | 0.048660 |
| electricity | FAULT | OUTPUT_CONTEXT | -0.350788 | -0.565653 | -0.159053 |
| electricity | REFERENCE | POS_ONLY | -0.158856 | -0.244546 | -0.066469 |
| electricity | REFERENCE | MAG_ONLY | 0.030016 | 0.000710 | 0.055321 |
| electricity | REFERENCE | OUTPUT_CONTEXT | -0.209373 | -0.408702 | -0.031171 |
| electricity | SHIFT4 | POS_ONLY | -0.089701 | -0.245196 | 0.077309 |
| electricity | SHIFT4 | MAG_ONLY | 0.035431 | -0.057098 | 0.116241 |
| electricity | SHIFT4 | OUTPUT_CONTEXT | 1.032720 | 0.715481 | 1.382688 |
| electricity | SHIFT8 | POS_ONLY | 0.312083 | 0.018920 | 0.565042 |
| electricity | SHIFT8 | MAG_ONLY | -0.077494 | -0.136881 | -0.015409 |
| electricity | SHIFT8 | OUTPUT_CONTEXT | 1.480085 | 1.024052 | 1.971624 |
| electricity | SHIFT_POINT | POS_ONLY | 0.045743 | -0.147901 | 0.226512 |
| electricity | SHIFT_POINT | MAG_ONLY | 0.048150 | -0.024760 | 0.120575 |
| electricity | SHIFT_POINT | OUTPUT_CONTEXT | 1.407493 | 1.112122 | 1.706834 |
| electricity_transfer | FAULT | POS_ONLY | -0.051853 | -0.109255 | 0.003202 |
| electricity_transfer | FAULT | MAG_ONLY | 0.049471 | 0.032571 | 0.066984 |
| electricity_transfer | FAULT | OUTPUT_CONTEXT | -0.200600 | -0.328261 | -0.079303 |
| electricity_transfer | REFERENCE | POS_ONLY | -0.034738 | -0.092418 | 0.026844 |
| electricity_transfer | REFERENCE | MAG_ONLY | 0.033069 | 0.014159 | 0.052750 |
| electricity_transfer | REFERENCE | OUTPUT_CONTEXT | -0.156039 | -0.308233 | -0.023723 |
| electricity_transfer | SHIFT4 | POS_ONLY | -0.342498 | -0.466527 | -0.227109 |
| electricity_transfer | SHIFT4 | MAG_ONLY | -0.079581 | -0.138293 | -0.022110 |
| electricity_transfer | SHIFT4 | OUTPUT_CONTEXT | 1.235211 | 0.883735 | 1.550693 |
| electricity_transfer | SHIFT8 | POS_ONLY | 0.694312 | 0.518499 | 0.856241 |
| electricity_transfer | SHIFT8 | MAG_ONLY | -0.251519 | -0.324375 | -0.178174 |
| electricity_transfer | SHIFT8 | OUTPUT_CONTEXT | 2.898901 | 2.055435 | 3.710056 |
| electricity_transfer | SHIFT_POINT | POS_ONLY | 0.147270 | 0.018386 | 0.279548 |
| electricity_transfer | SHIFT_POINT | MAG_ONLY | -0.129439 | -0.198661 | -0.064630 |
| electricity_transfer | SHIFT_POINT | OUTPUT_CONTEXT | 1.731943 | 1.401676 | 2.058221 |
| ettm1 | FAULT | POS_ONLY | 0.207267 | 0.089840 | 0.331751 |
| ettm1 | FAULT | MAG_ONLY | 0.008489 | -0.096068 | 0.120745 |
| ettm1 | FAULT | OUTPUT_CONTEXT | -0.038247 | -0.227910 | 0.162864 |
| ettm1 | REFERENCE | POS_ONLY | 0.174030 | 0.027351 | 0.337678 |
| ettm1 | REFERENCE | MAG_ONLY | 0.032977 | -0.069867 | 0.143380 |
| ettm1 | REFERENCE | OUTPUT_CONTEXT | 0.007674 | -0.190093 | 0.205445 |
| ettm1 | SHIFT4 | POS_ONLY | -0.266883 | -0.574905 | 0.044836 |
| ettm1 | SHIFT4 | MAG_ONLY | 0.294884 | 0.034036 | 0.565471 |
| ettm1 | SHIFT4 | OUTPUT_CONTEXT | 0.489416 | 0.150872 | 0.874968 |
| ettm1 | SHIFT8 | POS_ONLY | -1.428716 | -1.874919 | -0.980050 |
| ettm1 | SHIFT8 | MAG_ONLY | -0.234657 | -0.447594 | -0.027569 |
| ettm1 | SHIFT8 | OUTPUT_CONTEXT | -0.508765 | -1.223069 | 0.215971 |
| ettm1 | SHIFT_POINT | POS_ONLY | -0.594745 | -0.871787 | -0.316641 |
| ettm1 | SHIFT_POINT | MAG_ONLY | 0.224003 | 0.034134 | 0.412488 |
| ettm1 | SHIFT_POINT | OUTPUT_CONTEXT | 0.322438 | -0.008174 | 0.644135 |

## 학습이 실제로 움직였는가

전체30경로에서동결B0와buffer를보존했다. 파라미터변화·POS_ONLY gate·OUTPUT_CONTEXT gamma 및component gradient를PARAMETER_DYNAMICS.json/UPDATE_LEDGER.jsonl에남겼다. 모든실행경로의추가가중치가실제로변했고,선택step0은학습미실행과구분했다. 초기POSgate .5는입력변형이아니며adapter출력0으로B0와동일하게시작한다. gamma와W를동시에0으로초기화하지않았다.

## 자원과 검산

새training compute 14.34분,validation 2.79분,checkpointIO 2.09초,run wall 28.56분. 과거B0학습비용은부모COST_ACCOUNTING.json을별도참조한다. 추가파라미터만으로전체비용이같은비율감소했다고해석하지않는다.

| panel | arm | parameters | retained_adaptation_parameters | median_seconds | inference_peak_allocated_mib |
| --- | --- | --- | --- | --- | --- |
| electricity | MAG_ONLY | 8712.000000 | 303624.000000 | 0.056310 | 234.556641 |
| electricity | OUTPUT_CONTEXT | 4673.000000 | 299585.000000 | 0.054434 | 234.541504 |
| electricity | POS_ONLY | 8744.000000 | 303656.000000 | 0.053399 | 234.536133 |
| electricity_transfer | MAG_ONLY | 8712.000000 | 303624.000000 | 0.058416 | 234.556641 |
| electricity_transfer | OUTPUT_CONTEXT | 4673.000000 | 299585.000000 | 0.060406 | 234.541504 |
| electricity_transfer | POS_ONLY | 8744.000000 | 303656.000000 | 0.054291 | 234.557129 |
| ettm1 | MAG_ONLY | 8712.000000 | 303624.000000 | 0.056335 | 234.556641 |
| ettm1 | OUTPUT_CONTEXT | 4673.000000 | 299585.000000 | 0.053738 | 234.541504 |
| ettm1 | POS_ONLY | 8744.000000 | 303656.000000 | 0.053577 | 234.557129 |

본학습 1,024 updates 경로당 평균 시간과 메모리(선택 및 반복 경로 포함):

| source | arm | optimizer_seconds | validation_seconds | train_peak_allocated_mib |
| --- | --- | --- | --- | --- |
| electricity | MAG_ONLY | 35.064409 | 5.749051 | 362.108887 |
| electricity | OUTPUT_CONTEXT | 18.327679 | 5.789936 | 234.605469 |
| electricity | POS_ONLY | 34.272531 | 5.512212 | 363.669434 |
| ettm1 | MAG_ONLY | 34.540489 | 5.675019 | 362.108887 |
| ettm1 | OUTPUT_CONTEXT | 17.457646 | 5.523850 | 234.605469 |
| ettm1 | POS_ONLY | 32.473250 | 5.206169 | 363.669434 |

기존 C3/C2/B0의 동일 크기 추론 측정과 함께 보기. 기존 결과는 이전 측정 시점의 값을 재사용했으며, 원시 반복 시간은 RESOURCE_COMPARISON.csv 및 부모 RESOURCE_REPORT.csv에 남겼다.

| panel | arm | source | median_seconds | inference_peak_allocated_mib |
| --- | --- | --- | --- | --- |
| electricity | C0 | reused parent profile; earlier measurement | 0.049449 | 234.459961 |
| electricity | C2 | reused parent profile; earlier measurement | 0.050761 | 234.497559 |
| electricity | C3 | reused parent profile; earlier measurement | 0.054192 | 234.497559 |
| electricity | MAG_ONLY | new matched-size profile | 0.056310 | 234.556641 |
| electricity | M_RECENCY | reused parent profile; earlier measurement | 0.052599 | 234.497559 |
| electricity | OUTPUT_CONTEXT | new matched-size profile | 0.054434 | 234.541504 |
| electricity | POS_ONLY | new matched-size profile | 0.053399 | 234.536133 |
| electricity_transfer | C0 | reused parent profile; earlier measurement | 0.051020 | 234.459961 |
| electricity_transfer | C2 | reused parent profile; earlier measurement | 0.050450 | 234.497559 |
| electricity_transfer | C3 | reused parent profile; earlier measurement | 0.058037 | 234.497559 |
| electricity_transfer | MAG_ONLY | new matched-size profile | 0.058416 | 234.556641 |
| electricity_transfer | M_RECENCY | reused parent profile; earlier measurement | 0.053708 | 234.497559 |
| electricity_transfer | OUTPUT_CONTEXT | new matched-size profile | 0.060406 | 234.541504 |
| electricity_transfer | POS_ONLY | new matched-size profile | 0.054291 | 234.557129 |
| ettm1 | C0 | reused parent profile; earlier measurement | 0.049658 | 234.459961 |
| ettm1 | C2 | reused parent profile; earlier measurement | 0.051654 | 234.497559 |
| ettm1 | C3 | reused parent profile; earlier measurement | 0.053783 | 234.497559 |
| ettm1 | MAG_ONLY | new matched-size profile | 0.056335 | 234.556641 |
| ettm1 | M_RECENCY | reused parent profile; earlier measurement | 0.052659 | 234.497559 |
| ettm1 | OUTPUT_CONTEXT | new matched-size profile | 0.053738 | 234.541504 |
| ettm1 | POS_ONLY | new matched-size profile | 0.053577 | 234.557129 |

같은RTX3080/FP32/micro32/128입력의warmup후3회측정과seed별rawtime을공개했다. 작은시간차이를보편속도우위라고해석하지않는다. 기존C3 등은부모RESOURCE_REPORT의동일크기측정치를참조하며다른측정시점의noise는남는다. OUTPUT_CONTEXT는본체를통과하는추가gradient가필요없는단순출력대조다.

GPU 최소여유 8174MiB,승인RustDesk외compute0건. CPU부품검사5개와실제smoke6×2updates를구분했다.54예측복원/offidentity/상태불변,150checkpoint hash,update30732상한,선택규칙,전체원점점수재집계,scalar 2052항목×2metric 및주효과재검산을통과했다. Scalar범위는모든새view/조건의첫·마지막원점/채널과두draw이며모든원점을scalar로재생했다고쓰지않는다.

## 미실행·판정 경계

이번승인범위의미실행fit/필수평가는없다. LCL실제사건자료의연결감사,정식온라인TAFAS/COSA전체재현,독립source와다른backbone은미실행이며이번설명대조로대체완료했다고쓰지않는다. 후속새구조·추가학습은자동시작하지않는다.

보고서수나그림수가방법근거가되는것은아니다. 효과·단순대안·신규성·현실타당성을별도로해석한다. 최종근거에맞는결론은FINAL_DECISION.md에있다. 원자료/가중치/전체predictions는ignored로컬cache이며GitHub에는코드·봉인·원점수·검산·그림을공개한다.

## 그림

- [세 주요 대조와seed](figures/01_primary_controls.png)
- [원자료·오류·변화의절충](figures/02_tradeoffs.png)
- [선택된위치gate](figures/03_selected_position_gates.png)
- [전체변화형태](figures/04_all_shapes.png)
