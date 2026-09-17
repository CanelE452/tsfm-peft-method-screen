# 기존 실패 기전 진단 — optimizer update 0회

기존 TRAIN/V/E 입력과 선택 A5 네 모델을 그대로 사용했다. E는 이미 사용한 개발 기간의 서술 감사이며 새 선택에 쓰지 않는다.

| source | role | state | clipped_fraction | normalized_mass | longest_run | shift_clipped_fraction | shift_nonzero_residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| electricity | E_DISCOVERY | SHIFT4 | 0.015261 | 0.052879 | 3.067383 | 0.036102 | 0.243164 |
| electricity | E_DISCOVERY | SHIFT8 | 0.067236 | 0.145942 | 23.480469 | 0.869965 | 0.986328 |
| electricity | E_DISCOVERY | SHIFT_POINT | 0.018322 | 0.055583 | 3.578125 | 0.038269 | 0.299805 |
| electricity | TRAIN | BURST | 0.018258 | 0.191547 | 4.027100 | nan | nan |
| electricity | TRAIN | POINT | 0.016958 | 0.188935 | 2.353149 | nan | nan |
| electricity | TRAIN | SHIFT | 0.012139 | 0.157548 | 1.713013 | 0.029289 | 0.188599 |
| electricity | V_SELECT | SHIFT4 | 0.015320 | 0.093660 | 3.699219 | 0.048401 | 0.265625 |
| electricity | V_SELECT | SHIFT8 | 0.068050 | 0.185744 | 25.583984 | 0.891602 | 0.982422 |
| electricity | V_SELECT | SHIFT_POINT | 0.018261 | 0.096394 | 4.164062 | 0.043945 | 0.279297 |
| ettm1 | E_DISCOVERY | SHIFT4 | 0.007219 | 0.010653 | 2.146484 | 0.039673 | 0.132812 |
| ettm1 | E_DISCOVERY | SHIFT8 | 0.060671 | 0.101505 | 27.344727 | 0.896088 | 0.997070 |
| ettm1 | E_DISCOVERY | SHIFT_POINT | 0.010454 | 0.016542 | 2.724609 | 0.037140 | 0.198242 |
| ettm1 | TRAIN | BURST | 0.010293 | 0.022451 | 3.736084 | nan | nan |
| ettm1 | TRAIN | POINT | 0.009037 | 0.019882 | 1.986328 | nan | nan |
| ettm1 | TRAIN | SHIFT | 0.005265 | 0.010790 | 1.642578 | 0.036629 | 0.125244 |
| ettm1 | V_SELECT | SHIFT4 | 0.009155 | 0.011821 | 2.517578 | 0.024536 | 0.111328 |
| ettm1 | V_SELECT | SHIFT8 | 0.063461 | 0.093273 | 26.615234 | 0.889526 | 1.000000 |
| ettm1 | V_SELECT | SHIFT_POINT | 0.012756 | 0.017871 | 3.359375 | 0.036865 | 0.208984 |

## V에서 기존 A5 사용량

| source | seed | state | examples | adapter_delta_rms | native_embedding_rms | adapter_over_native | discarded_feature_rms | discarded_coordinate_rms | adapter_discarded_correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| electricity | 81501 | BURST8 | 512 | 0.002212 | 0.212173 | 0.010118 | 0.092696 | 0.224640 | 0.828693 |
| electricity | 81501 | POINT8 | 512 | 0.002169 | 0.208994 | 0.010222 | 0.080789 | 0.212099 | 0.828740 |
| electricity | 81501 | SHIFT4 | 512 | 0.002172 | 0.210002 | 0.009989 | 0.057069 | 0.163953 | 0.800975 |
| electricity | 81501 | SHIFT8 | 512 | 0.002586 | 0.211781 | 0.011038 | 0.125876 | 0.262940 | 0.860316 |
| electricity | 81502 | BURST8 | 512 | 0.002241 | 0.212173 | 0.010297 | 0.092696 | 0.224640 | 0.797665 |
| electricity | 81502 | POINT8 | 512 | 0.002207 | 0.208994 | 0.010386 | 0.080789 | 0.212099 | 0.812955 |
| electricity | 81502 | SHIFT4 | 512 | 0.002179 | 0.210002 | 0.010061 | 0.057069 | 0.163953 | 0.801510 |
| electricity | 81502 | SHIFT8 | 512 | 0.002592 | 0.211781 | 0.011159 | 0.125876 | 0.262940 | 0.793623 |
| ettm1 | 81501 | BURST8 | 512 | 0.001631 | 0.214863 | 0.007603 | 0.048726 | 0.071821 | 0.395810 |
| ettm1 | 81501 | POINT8 | 512 | 0.001688 | 0.212453 | 0.007900 | 0.037402 | 0.055601 | 0.353755 |
| ettm1 | 81501 | SHIFT4 | 512 | 0.001687 | 0.213089 | 0.007810 | 0.012199 | 0.016745 | 0.355395 |
| ettm1 | 81501 | SHIFT8 | 512 | 0.001633 | 0.214694 | 0.007433 | 0.073313 | 0.102913 | 0.587200 |
| ettm1 | 81502 | BURST8 | 512 | 0.001527 | 0.214863 | 0.007142 | 0.048726 | 0.071821 | 0.464965 |
| ettm1 | 81502 | POINT8 | 512 | 0.001566 | 0.212453 | 0.007350 | 0.037402 | 0.055601 | 0.386701 |
| ettm1 | 81502 | SHIFT4 | 512 | 0.001567 | 0.213089 | 0.007289 | 0.012199 | 0.016745 | 0.377651 |
| ettm1 | 81502 | SHIFT8 | 512 | 0.001535 | 0.214694 | 0.007050 | 0.073313 | 0.102913 | 0.695450 |

TRAIN은 4r0 shift만, E는 4/8r0를 포함한다. clipped fraction과 shift-affected fraction을 구분한다. nonzero_residual은 전체 입력, shift_nonzero_residual은 실제 shift 위치만 세었다. residual_mass_over_abs_delta는 shift 위치에서 버린 절대량 합/|delta|이며 위치 수로 나눈 비율은 아니다. last64_same_sign은 양/음 중 큰 count다.

연속 clip 길이는 같은 부호 제약 없는 최장 run이다. 어댑터 RMS 비율은 patch별 비율의 평균이며 상관계수는 scenario 안의 example×patch 쌍이다. 기존 zero/permutation V 진단에서 경로 의존성은 관찰됐지만, 대조군보다 좋은 방법이라는 증거는 아니다. 노출 차이는 가능한 설명이지 단독 인과 증명이 아니다.

{
  "electricity": {
    "train_shift_nonzero": 0.1885986328125,
    "E_shift8_nonzero": 0.986328125,
    "train_shift_fraction": 0.0292892456054687,
    "E_shift8_fraction": 0.869964599609375,
    "train_normalized_mass": 0.1575480320106885,
    "E_shift8_normalized_mass": 0.1459418310333444,
    "descriptive_tag": "TRAIN_EXPOSURE_GAP",
    "hard_clip": "HARD_CLIP_INFORMATION_LOSS: quantitative exposure and historical shift error direction; not a causal isolation",
    "underutilization": "No arbitrary near-zero threshold; actual adapter ratios reported, no categorical zero-use claim"
  },
  "ettm1": {
    "train_shift_nonzero": 0.125244140625,
    "E_shift8_nonzero": 0.9970703125,
    "train_shift_fraction": 0.0366287231445312,
    "E_shift8_fraction": 0.896087646484375,
    "train_normalized_mass": 0.010790444180731,
    "E_shift8_normalized_mass": 0.1015053282955436,
    "descriptive_tag": "TRAIN_EXPOSURE_GAP",
    "hard_clip": "HARD_CLIP_INFORMATION_LOSS: quantitative exposure and historical shift error direction; not a causal isolation",
    "underutilization": "No arbitrary near-zero threshold; actual adapter ratios reported, no categorical zero-use claim"
  }
}

이 진단의 크기를 성능 gate로 쓰지 않는다. B0–B5는 동일 고정 예산으로 직접 비교한다.
