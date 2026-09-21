# Rollout uncertainty PEFT 결과

실행·채점 산출물입니다. 과학적 최종 범주는 검토 대기이며 자동 PASS 판정은 하지 않습니다. 선택 seed 92120은 모든 반복 평균에서 제외했습니다. 고정 F0/Chronos-2는 한 번 측정한 값이며 반복 seed로 복제해 평균하지 않았습니다.

## 1. 현재 공식 branching

현재 Chronos-Bolt 공식 장기 예측은 중앙값-only가 아니라 9개 분위수 경로와 9×9 축약을 사용합니다. 다음 표의 양수는 baseline−candidate 점수 개선입니다.

### 공식 branching과 F0 중앙값 경로

```text
     source baseline_variant  effect_a_minus_b  relative_gain_percent  ci95_low  ci95_high
Electricity              raw          0.003540               1.920880  0.003041   0.003964
Electricity          ordered          0.003536               1.918984  0.003031   0.003975
Electricity           affine          0.000958               0.531087  0.000551   0.001304
      ETTh1              raw          0.022244               5.243401  0.019069   0.025188
      ETTh1          ordered          0.022244               5.243401  0.019162   0.025056
      ETTh1           affine          0.008126               2.033954  0.005354   0.010767
```

### 2. 일반 rollout LoRA 학습 효과

```text
     source baseline_variant  effect_a_minus_b  relative_gain_percent  ci95_low  ci95_high
Electricity              raw          0.005134               2.786335  0.003818   0.006380
Electricity          ordered          0.005156               2.798011  0.003870   0.006480
Electricity           affine          0.001483               0.821646  0.000360   0.002494
      ETTh1              raw          0.037076               8.739517  0.025487   0.049069
      ETTh1          ordered          0.037077               8.739630  0.025320   0.048882
      ETTh1           affine          0.013226               3.310468  0.001649   0.025296
```

### 3. 생성 상태 정보의 효과: S 대 R

```text
     source baseline_variant  effect_a_minus_b  relative_gain_percent  ci95_low  ci95_high
Electricity              raw          0.000544               0.303922  0.000197   0.000835
Electricity          ordered          0.000544               0.303818  0.000212   0.000833
Electricity           affine          0.000281               0.156922 -0.000046   0.000572
      ETTh1              raw         -0.003322              -0.857964 -0.006683  -0.000153
      ETTh1          ordered         -0.003322              -0.858092 -0.006336   0.000071
      ETTh1           affine         -0.004727              -1.223798 -0.008869   0.000366
```

### 3. 예측 폭 정보의 추가 효과: U 대 S

```text
     source baseline_variant  effect_a_minus_b  relative_gain_percent  ci95_low  ci95_high
Electricity              raw          0.000031               0.017498  0.000007   0.000062
Electricity          ordered          0.000031               0.017546  0.000008   0.000061
Electricity           affine          0.000028               0.015438  0.000005   0.000057
      ETTh1              raw          0.000003               0.000815 -0.000032   0.000043
      ETTh1          ordered          0.000003               0.000815 -0.000032   0.000041
      ETTh1           affine          0.000002               0.000471 -0.000033   0.000041
```

### U 대 R 전체 차이

```text
     source baseline_variant  effect_a_minus_b  relative_gain_percent  ci95_low  ci95_high
Electricity              raw          0.000576               0.321367  0.000238   0.000881
Electricity          ordered          0.000575               0.321310  0.000237   0.000854
Electricity           affine          0.000308               0.172336  0.000014   0.000598
      ETTh1              raw         -0.003319              -0.857142 -0.006655  -0.000091
      ETTh1          ordered         -0.003319              -0.857270 -0.006560  -0.000021
      ETTh1           affine         -0.004725              -1.223322 -0.008904   0.000513
```

## 4. 단순 출력 보정

각 모델에 동일한 CAL 전용 35점 grid를 적용했습니다. 보정값은 rollout에 되먹임하지 않았으며, conformal 보장을 뜻하지 않습니다. ordered 대비 affine 차이는 아래와 같습니다.

```text
     source             candidate  effect_a_minus_b  ci95_low  ci95_high
Electricity             F0_MEDIAN          0.003800  0.003158   0.004490
Electricity             F0_NATIVE          0.001222  0.000652   0.001806
Electricity       CHRONOS2_DIRECT          0.000000  0.000000   0.000000
Electricity        R_ROLLOUT_LORA          0.000127  0.000024   0.000242
Electricity       S_STATE_ADAPTER         -0.000137 -0.000260  -0.000010
Electricity U_UNCERTAINTY_ADAPTER         -0.000140 -0.000263  -0.000010
Electricity              R_NATIVE         -0.000002 -0.000121   0.000133
Electricity                R_MC16         -0.000056 -0.000164   0.000067
      ETTh1             F0_MEDIAN          0.024730  0.021231   0.028558
      ETTh1             F0_NATIVE          0.010611  0.007168   0.014213
      ETTh1       CHRONOS2_DIRECT          0.000000  0.000000   0.000000
      ETTh1        R_ROLLOUT_LORA          0.000879 -0.002281   0.005228
      ETTh1       S_STATE_ADAPTER         -0.000526 -0.007287   0.007816
      ETTh1 U_UNCERTAINTY_ADAPTER         -0.000528 -0.007221   0.007627
      ETTh1              R_NATIVE          0.000997 -0.009477   0.013336
      ETTh1                R_MC16         -0.000221 -0.000952   0.000421
```

## 5. 공식 branching·MC16·직접 장기 예측 대비

```text
     source        baseline baseline_variant  effect_a_minus_b  relative_gain_percent  ci95_low  ci95_high
Electricity       F0_NATIVE              raw          0.002170               1.200937  0.000936   0.003353
Electricity        R_NATIVE              raw          0.000750               0.418409  0.000122   0.001419
Electricity          R_MC16              raw          0.008473               4.530436  0.007570   0.009434
Electricity CHRONOS2_DIRECT              raw         -0.005306              -3.062500 -0.007443  -0.003410
Electricity       F0_NATIVE          ordered          0.002195               1.214656  0.000908   0.003333
Electricity        R_NATIVE          ordered          0.000766               0.426940  0.000124   0.001444
Electricity          R_MC16          ordered          0.008500               4.544538  0.007589   0.009464
Electricity CHRONOS2_DIRECT          ordered         -0.005280              -3.047698 -0.007506  -0.003464
Electricity       F0_NATIVE           affine          0.000833               0.463944 -0.000363   0.001876
Electricity        R_NATIVE           affine          0.000627               0.349568  0.000012   0.001253
Electricity          R_MC16           affine          0.008415               4.497979  0.007557   0.009371
Electricity CHRONOS2_DIRECT           affine         -0.005421              -3.128652 -0.007484  -0.003594
      ETTh1       F0_NATIVE              raw          0.011513               2.864058  0.002753   0.020946
      ETTh1        R_NATIVE              raw         -0.000604              -0.155001 -0.004135   0.003974
      ETTh1          R_MC16              raw          0.010016               2.500812  0.005213   0.015039
      ETTh1 CHRONOS2_DIRECT              raw         -0.005379              -1.396740 -0.016809   0.003856
      ETTh1       F0_NATIVE          ordered          0.011513               2.864056  0.003088   0.020809
      ETTh1        R_NATIVE          ordered         -0.000605              -0.155123 -0.004250   0.004021
      ETTh1          R_MC16          ordered          0.010016               2.500812  0.005381   0.015205
      ETTh1 CHRONOS2_DIRECT          ordered         -0.005379              -1.396741 -0.016264   0.003516
      ETTh1       F0_NATIVE           affine          0.000374               0.095636 -0.009689   0.012430
      ETTh1        R_NATIVE           affine         -0.002130              -0.547636 -0.007001   0.002100
      ETTh1          R_MC16           affine          0.009709               2.422875  0.003972   0.018398
      ETTh1 CHRONOS2_DIRECT           affine         -0.005907              -1.533771 -0.019317   0.006327
```

단일 경로/공식 branching/MC16의 조건부 context 계산량은 예제당 4/28/49회입니다. 시간 배수로 해석하지 않습니다. 아래는 실제 batch1/8 자원 수치입니다. 각 모델 3회 추론 중앙값을 구한 뒤, 학습 모델만 두 seed의 중앙값을 평균했습니다. peak는 seed 중 최대값입니다. loading은 추론 시간에 포함하지 않았습니다. MC16은 분위수 범위 밖 tail을 고정한 16개 유한 경로이며 완전한 joint sampling을 주장하지 않습니다. Chronos-2는 다른 크기·사전학습·구조의 직접 장기 모델입니다.

```text
     source                method variant  batch_size  scaled_pinball  measured_wall_seconds  peak_allocated_bytes
Electricity       CHRONOS2_DIRECT ordered           1        0.173254               0.021596             548239360
Electricity       CHRONOS2_DIRECT ordered           8        0.173254               0.023967             563017728
Electricity       CHRONOS2_DIRECT  affine           1        0.173254               0.019305             548239360
Electricity       CHRONOS2_DIRECT  affine           8        0.173254               0.023082             563017728
Electricity             F0_MEDIAN ordered           1        0.184266               0.053422             260568064
Electricity             F0_MEDIAN ordered           8        0.184266               0.055841             271279104
Electricity             F0_MEDIAN  affine           1        0.180466               0.053678             260568064
Electricity             F0_MEDIAN  affine           8        0.180466               0.054983             271279104
Electricity             F0_NATIVE ordered           1        0.180730               0.054781             272656896
Electricity             F0_NATIVE ordered           8        0.180730               0.080050             367998464
Electricity             F0_NATIVE  affine           1        0.179507               0.056640             272656896
Electricity             F0_NATIVE  affine           8        0.179507               0.088275             367998464
Electricity                R_MC16 ordered           1        0.187034               0.071518             286513664
Electricity                R_MC16 ordered           8        0.187034               0.160348             357882880
Electricity                R_MC16  affine           1        0.187090               0.072715             286513664
Electricity                R_MC16  affine           8        0.187090               0.160139             357882880
Electricity              R_NATIVE ordered           1        0.179300               0.070332             273836544
Electricity              R_NATIVE ordered           8        0.179300               0.098248             369178112
Electricity              R_NATIVE  affine           1        0.179301               0.071535             273836544
Electricity              R_NATIVE  affine           8        0.179301               0.095841             369178112
Electricity        R_ROLLOUT_LORA ordered           1        0.179110               0.068168             261747712
Electricity        R_ROLLOUT_LORA ordered           8        0.179110               0.070289             272458752
Electricity        R_ROLLOUT_LORA  affine           1        0.178983               0.067903             261747712
Electricity        R_ROLLOUT_LORA  affine           8        0.178983               0.070827             272458752
Electricity       S_STATE_ADAPTER ordered           1        0.178566               0.067442             261783552
Electricity       S_STATE_ADAPTER ordered           8        0.178566               0.070940             272494592
Electricity       S_STATE_ADAPTER  affine           1        0.178702               0.069525             261783552
Electricity       S_STATE_ADAPTER  affine           8        0.178702               0.070765             272494592
Electricity U_UNCERTAINTY_ADAPTER ordered           1        0.178534               0.069759             261783552
Electricity U_UNCERTAINTY_ADAPTER ordered           8        0.178534               0.070826             272494592
Electricity U_UNCERTAINTY_ADAPTER  affine           1        0.178675               0.069772             261783552
Electricity U_UNCERTAINTY_ADAPTER  affine           8        0.178675               0.071496             272494592
      ETTh1       CHRONOS2_DIRECT ordered           1        0.385099               0.020000             548239360
      ETTh1       CHRONOS2_DIRECT ordered           8        0.385099               0.022965             563755008
      ETTh1       CHRONOS2_DIRECT  affine           1        0.385099               0.019715             548239360
      ETTh1       CHRONOS2_DIRECT  affine           8        0.385099               0.022920             563755008
      ETTh1             F0_MEDIAN ordered           1        0.424236               0.053320             260568064
      ETTh1             F0_MEDIAN ordered           8        0.424236               0.054538             271279104
      ETTh1             F0_MEDIAN  affine           1        0.399506               0.053743             260568064
      ETTh1             F0_MEDIAN  affine           8        0.399506               0.056799             271279104
      ETTh1             F0_NATIVE ordered           1        0.401992               0.054688             272656896
      ETTh1             F0_NATIVE ordered           8        0.401992               0.080566             367998464
      ETTh1             F0_NATIVE  affine           1        0.391380               0.055145             272656896
      ETTh1             F0_NATIVE  affine           8        0.391380               0.079915             367998464
      ETTh1                R_MC16 ordered           1        0.400494               0.075120             286513664
      ETTh1                R_MC16 ordered           8        0.400494               0.160760             357882880
      ETTh1                R_MC16  affine           1        0.400715               0.074105             286513664
      ETTh1                R_MC16  affine           8        0.400715               0.161499             357882880
      ETTh1              R_NATIVE ordered           1        0.389873               0.069927             273836544
      ETTh1              R_NATIVE ordered           8        0.389873               0.093379             369178112
      ETTh1              R_NATIVE  affine           1        0.388876               0.069693             273836544
      ETTh1              R_NATIVE  affine           8        0.388876               0.093326             369178112
      ETTh1        R_ROLLOUT_LORA ordered           1        0.387159               0.069457             261747712
      ETTh1        R_ROLLOUT_LORA ordered           8        0.387159               0.070648             272458752
      ETTh1        R_ROLLOUT_LORA  affine           1        0.386281               0.069329             261747712
      ETTh1        R_ROLLOUT_LORA  affine           8        0.386281               0.069592             272458752
      ETTh1       S_STATE_ADAPTER ordered           1        0.390481               0.068956             261783552
      ETTh1       S_STATE_ADAPTER ordered           8        0.390481               0.069866             272494592
      ETTh1       S_STATE_ADAPTER  affine           1        0.391008               0.068918             261783552
      ETTh1       S_STATE_ADAPTER  affine           8        0.391008               0.070983             272494592
      ETTh1 U_UNCERTAINTY_ADAPTER ordered           1        0.390478               0.069363             261783552
      ETTh1 U_UNCERTAINTY_ADAPTER ordered           8        0.390478               0.071346             272494592
      ETTh1 U_UNCERTAINTY_ADAPTER  affine           1        0.391006               0.069826             261783552
      ETTh1 U_UNCERTAINTY_ADAPTER  affine           8        0.391006               0.073402             272494592
```

## 6. 자료별·seed별 반례와 한계

```text
     source                method variant  seed_count  scaled_pinball      pce  coverage80  scaled_width80  scaled_mae   raw_mae  scaled_rmse
Electricity       CHRONOS2_DIRECT ordered           1        0.173254 0.028669    0.761034        0.624329    0.217045 72.003199     0.351758
Electricity       CHRONOS2_DIRECT  affine           1        0.173254 0.028669    0.761034        0.624329    0.217045 72.003199     0.351758
Electricity             F0_MEDIAN ordered           1        0.184266 0.054655    0.666596        0.484724    0.223195 74.757387     0.351641
Electricity             F0_MEDIAN  affine           1        0.180466 0.029344    0.793062        0.668835    0.223195 74.757387     0.351641
Electricity             F0_NATIVE ordered           1        0.180730 0.037825    0.719246        0.548458    0.223916 74.930790     0.351598
Electricity             F0_NATIVE  affine           1        0.179507 0.029216    0.803852        0.680176    0.223916 74.930790     0.351598
Electricity                R_MC16 ordered           2        0.187034 0.033945    0.755721        0.655661    0.233085 78.083043     0.363965
Electricity                R_MC16  affine           2        0.187090 0.032255    0.753530        0.653968    0.233085 78.083043     0.363965
Electricity              R_NATIVE ordered           2        0.179300 0.038693    0.794641        0.681174    0.223454 74.409541     0.352568
Electricity              R_NATIVE  affine           2        0.179301 0.035952    0.791824        0.680378    0.223454 74.409541     0.352568
Electricity        R_ROLLOUT_LORA ordered           2        0.179110 0.034668    0.802199        0.691850    0.222689 74.347441     0.352692
Electricity        R_ROLLOUT_LORA  affine           2        0.178983 0.032482    0.799156        0.690658    0.222689 74.347441     0.352692
Electricity       S_STATE_ADAPTER ordered           2        0.178566 0.031203    0.796813        0.681400    0.222400 74.241269     0.353283
Electricity       S_STATE_ADAPTER  affine           2        0.178702 0.030887    0.805941        0.703292    0.222400 74.241269     0.353283
Electricity U_UNCERTAINTY_ADAPTER ordered           2        0.178534 0.031420    0.797470        0.682568    0.222377 74.243246     0.353197
Electricity U_UNCERTAINTY_ADAPTER  affine           2        0.178675 0.031180    0.806620        0.704495    0.222377 74.243246     0.353197
      ETTh1       CHRONOS2_DIRECT ordered           1        0.385099 0.046477    0.751472        1.371388    0.491418  1.912391     0.701177
      ETTh1       CHRONOS2_DIRECT  affine           1        0.385099 0.046477    0.751472        1.371388    0.491418  1.912391     0.701177
      ETTh1             F0_MEDIAN ordered           1        0.424236 0.107811    0.464414        0.723418    0.490618  1.899143     0.693902
      ETTh1             F0_MEDIAN  affine           1        0.399506 0.049557    0.678712        1.276886    0.490618  1.899143     0.693902
      ETTh1             F0_NATIVE ordered           1        0.401992 0.061481    0.612686        1.001363    0.492579  1.902148     0.693929
      ETTh1             F0_NATIVE  affine           1        0.391380 0.032578    0.749603        1.386960    0.492579  1.902148     0.693929
      ETTh1                R_MC16 ordered           2        0.400494 0.041652    0.725783        1.351727    0.505352  1.924352     0.697957
      ETTh1                R_MC16  affine           2        0.400715 0.044709    0.697921        1.257956    0.505352  1.924352     0.697957
      ETTh1              R_NATIVE ordered           2        0.389873 0.057504    0.758869        1.425641    0.492200  1.885168     0.680504
      ETTh1              R_NATIVE  affine           2        0.388876 0.028228    0.710727        1.287464    0.488392  1.862582     0.672326
      ETTh1        R_ROLLOUT_LORA ordered           2        0.387159 0.048570    0.765270        1.445823    0.486934  1.863389     0.676923
      ETTh1        R_ROLLOUT_LORA  affine           2        0.386281 0.033292    0.766875        1.487944    0.485609  1.857970     0.674140
      ETTh1       S_STATE_ADAPTER ordered           2        0.390481 0.047912    0.779001        1.544770    0.487815  1.869323     0.676786
      ETTh1       S_STATE_ADAPTER  affine           2        0.391008 0.037899    0.735896        1.395481    0.488996  1.870195     0.674329
      ETTh1 U_UNCERTAINTY_ADAPTER ordered           2        0.390478 0.047933    0.779349        1.545656    0.487843  1.869399     0.676817
      ETTh1 U_UNCERTAINTY_ADAPTER  affine           2        0.391006 0.037835    0.736212        1.396373    0.489008  1.870226     0.674331
```

probability score, PCE, coverage와 width는 서로 다른 지표입니다. coverage 상승만으로 개선을 선언하지 않습니다. full/tail_129_256의 point 지표는 SCORES_SUMMARY.csv에, 개별 seed 효과는 SEED_EFFECTS.csv에 보존했습니다.

선택 step0 모델 수: 0/12. step0이면 추가 학습이 선택되지 않은 결과입니다. 해당 모델: 없음.

두 반복 seed는 고정되어 있으며 512 updates가 충분한 최적화였다고 주장하지 않습니다. CI는 7개 연속 원점 moving-block bootstrap 2,000회로, 고정 seed·선택 계열 아래 원점 불확실성만 반영합니다. horizon256과 stride24로 인접 target이 232시간 겹칩니다. 블록 길이7은 전체 256시간 의존을 완전히 제거한다는 보장이 없으며, 과거 탐색이나 optimizer 모집단 불확실성을 보정하지 않습니다.

Electricity는 타임스탬프 없는 hourly slot이며 달력·UTC를 검증한 자료가 아닙니다. ETTh1은 온도 OT와 부하 6개입니다. 동일 24시간 시작 phase만 평가했습니다. 두 공개 자료는 개발·사전학습에 노출됐을 수 있으며 독립 확증이나 ETT 표준 12/4/4개월 재현이 아닙니다.

## 7. 제한된 후속 투자 판단

과학적 범주는 수치와 실측 비용을 검토해 FINAL_DECISION.md에 확정해야 합니다. 신규성·논문 PASS 또는 rollout PEFT 전체 반증은 선언하지 않습니다. 이번 예산 이후 자동 후속 실험은 0입니다.

GitHub에는 코드·집계·검산·manifest를 남깁니다. 원자료, weights, 예측 cache는 로컬 보관이므로 저장소만으로 모든 수치를 재생할 수 있다고 주장하지 않습니다.
