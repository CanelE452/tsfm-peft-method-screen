# Anchoring / DualClock 재진단 결과

**44/44 fits, 46,080 training updates 완료. 이번 결과는 train/V 진단이며 새 E 평가나 논문 PASS가 아니다.**

실행 commit `f174a612895607bb2a4d59ee3a33ff0b416d8687`. 기존 270 fits에 이번44 fits를 추가해 누적314 fits다. Stream 시도는9회(8완료/과거1중단) 그대로다. 별도 GPU smoke14 updates는 fits에 포함하지 않는다.

저장 예측 345개 독립 재계산, 최대 primary 오차 1.67e-16. 기존 결과 700개 파일 불변. 실제 parameter update와 frozen weights 검사를 통과했고 각 fit의 선택 checkpoint를 다시 읽어 V 예측이 정확히 일치함을 확인했다.

**Anchoring: 같은 목적함수 내 보존항 효과**

| 데이터 | Native | Native+anchor | Raw | Raw+anchor | Native 보존 이득 | Raw 보존 이득 |
|---|---:|---:|---:|---:|---:|---:|
| etth1 | 0.286498051 | 0.284772972 | 0.288063013 | 0.287286318 | +0.6021% | +0.2696% |
| traffic | 0.107570426 | 0.108443329 | 0.108612089 | 0.108907773 | -0.8115% | -0.2722% |

각 arm의 두 seed V-selected loss 평균이다. 양수 이득은 보존항을 넣은 쪽의 오차 감소다. 목적함수 간 보존항의 상대적 gradient 강도가 같다고 가정하지 않는다. 고정 LR·step의 비교는 comparisons.csv에 모두 포함했다.

| 데이터 | Seed | Arm | LR | 선택 step | V loss |
|---|---:|---|---:|---:|---:|
| etth1 | 33000 | native | 3e-05 | 450 | 0.285201169 |
| etth1 | 33000 | native_anchor | 3e-05 | 450 | 0.283392366 |
| etth1 | 33000 | raw | 3e-05 | 150 | 0.287912198 |
| etth1 | 33000 | raw_anchor | 3e-05 | 450 | 0.285872247 |
| etth1 | 33001 | native | 3e-05 | 150 | 0.287794933 |
| etth1 | 33001 | native_anchor | 3e-05 | 450 | 0.286153577 |
| etth1 | 33001 | raw | 3e-05 | 150 | 0.288213828 |
| etth1 | 33001 | raw_anchor | 3e-05 | 450 | 0.288700389 |
| traffic | 33000 | native | 0.0001 | 450 | 0.108207900 |
| traffic | 33000 | native_anchor | 0.0001 | 900 | 0.108690765 |
| traffic | 33000 | raw | 3e-05 | 450 | 0.109269458 |
| traffic | 33000 | raw_anchor | 3e-05 | 900 | 0.109318340 |
| traffic | 33001 | native | 0.0001 | 450 | 0.106932951 |
| traffic | 33001 | native_anchor | 0.0001 | 900 | 0.108195893 |
| traffic | 33001 | raw | 0.0001 | 450 | 0.107954721 |
| traffic | 33001 | raw_anchor | 0.0001 | 450 | 0.108497206 |

**DualClock: 추가 예산과 비교군**

| Seed | Standard | Summary | DualClock | vs Standard | vs Summary |
|---|---:|---:|---:|---:|---:|
| 30000 | 0.463186267 | 0.463211648 | 0.463189700 | -0.0007% | +0.0047% |
| 30001 | 0.463152211 | 0.463100700 | 0.463081786 | +0.0152% | +0.0041% |

| Seed | Arm | LR | 선택 step | V loss |
|---|---|---:|---:|---:|
| 30000 | standard | 0.0001 | 1080 | 0.463186267 |
| 30000 | summary | 0.0001 | 1080 | 0.463211648 |
| 30000 | dualclock | 0.0001 | 1080 | 0.463189700 |
| 30001 | standard | 0.0001 | 360 | 0.463152211 |
| 30001 | summary | 0.0001 | 360 | 0.463100700 |
| 30001 | dualclock | 0.0001 | 360 | 0.463081786 |

| Seed | Arm / 비교 | 구분 | 360 또는 기준 loss | 1440 또는 비교 loss | 개선율 |
|---|---|---|---:|---:|---:|
| 30000 | standard_long_vs_short | V_selected | 0.463587285 | 0.463186267 | +0.0865% |
| 30000 | standard_long_vs_short | fixed_endpoints_lr_3e-05 | 0.463992512 | 0.463576200 | +0.0897% |
| 30000 | standard_long_vs_short | fixed_endpoints_lr_0.0001 | 0.463587285 | 0.463495495 | +0.0198% |
| 30000 | summary_long_vs_short | V_selected | 0.463490021 | 0.463211648 | +0.0601% |
| 30000 | summary_long_vs_short | fixed_endpoints_lr_3e-05 | 0.463926892 | 0.463576156 | +0.0756% |
| 30000 | summary_long_vs_short | fixed_endpoints_lr_0.0001 | 0.463490021 | 0.463439746 | +0.0108% |
| 30000 | dualclock_long_vs_short | V_selected | 0.463459940 | 0.463189700 | +0.0583% |
| 30000 | dualclock_long_vs_short | fixed_endpoints_lr_3e-05 | 0.463897538 | 0.463595181 | +0.0652% |
| 30000 | dualclock_long_vs_short | fixed_endpoints_lr_0.0001 | 0.463459940 | 0.463599895 | -0.0302% |
| 30001 | standard_long_vs_short | V_selected | 0.463152211 | 0.463152211 | +0.0000% |
| 30001 | standard_long_vs_short | fixed_endpoints_lr_3e-05 | 0.463719495 | 0.463578995 | +0.0303% |
| 30001 | standard_long_vs_short | fixed_endpoints_lr_0.0001 | 0.463152211 | 0.464195510 | -0.2253% |
| 30001 | summary_long_vs_short | V_selected | 0.463100700 | 0.463100700 | +0.0000% |
| 30001 | summary_long_vs_short | fixed_endpoints_lr_3e-05 | 0.463630354 | 0.463528849 | +0.0219% |
| 30001 | summary_long_vs_short | fixed_endpoints_lr_0.0001 | 0.463100700 | 0.464216423 | -0.2409% |
| 30001 | dualclock_long_vs_short | V_selected | 0.463081786 | 0.463081786 | +0.0000% |
| 30001 | dualclock_long_vs_short | fixed_endpoints_lr_3e-05 | 0.463617814 | 0.463546190 | +0.0154% |
| 30001 | dualclock_long_vs_short | fixed_endpoints_lr_0.0001 | 0.463081786 | 0.464253014 | -0.2529% |

V-selected long-vs-short 비교는 후보 checkpoint 수가 늘면 최솟값이 나빠질 수 없다는 선택 효과를 포함한다. 고정 LR의360→1440 endpoint 변화, 최종 선택 위치, 비교군과의 간격을 함께 해석해야 한다.

역사적 seed30000의 첫360-step V trajectory 최대 차이: 0. 이 수치는 historical_prefix.json에서 모든 LR·arm·checkpoint별로 확인할 수 있다.

**Event branch 진단**

| Seed | Arm | 개입 | 원래 대비 V loss 변화 |
|---|---|---|---:|
| 30000 | summary | summary_zero_events | +0.00413% |
| 30000 | summary | summary_rotate_events | -0.00001% |
| 30000 | summary | summary_drop_adapter | +0.06983% |
| 30000 | dualclock | dualclock_zero_events | +0.00146% |
| 30000 | dualclock | dualclock_rotate_events | +0.00007% |
| 30000 | dualclock | dualclock_drop_adapter | +0.06802% |
| 30001 | summary | summary_zero_events | +0.00108% |
| 30001 | summary | summary_rotate_events | -0.00011% |
| 30001 | summary | summary_drop_adapter | +0.03673% |
| 30001 | dualclock | dualclock_zero_events | +0.00134% |
| 30001 | dualclock | dualclock_rotate_events | -0.00010% |
| 30001 | dualclock | dualclock_drop_adapter | +0.06177% |

양수는 개입으로 손실이 증가했다는 뜻이다. Zero-event는 event embedding만0, rotate는32-item 평가 batch 안에서 event 정보를 회전, drop은 hidden residual adapter 전체 제거다. OOD 개입 가능성이 있으므로 event의 고유 인과 기여나 재학습 대조의 대체물로 해석하지 않는다. 이 결과로 모델을 다시 선택하지 않았다.

| Seed | 대조 | 이긴 item 비율 | 평균 차이 | Median 차이 | 상위26개 효과 제외 평균 | Zero-heavy 평균 |
|---|---|---:|---:|---:|---:|---:|
| 30000 | standard | 50.0% | -0.00000343 | -0.00000228 | -0.00011213 | +0.00004144 |
| 30000 | summary | 56.6% | +0.00002195 | +0.00002295 | -0.00007358 | +0.00001851 |
| 30001 | standard | 66.4% | +0.00007042 | +0.00008971 | -0.00000862 | +0.00013130 |
| 30001 | summary | 53.1% | +0.00001891 | +0.00000836 | -0.00004836 | -0.00000357 |

item 효과는 비교군−DualClock, 양수가 유리하다. 기존 train-only item 선택 및 zero-heavy 정의를 사용했다. V 관측의 사후 진단이며 유의성·독립 재현 주장은 아니다.

**계산량과 한계**

| 작업 | Fits | Updates | Active seconds | Fit wall seconds | Peak MiB |
|---|---:|---:|---:|---:|---:|
| anchor | 32 | 28800 | 3148.2 | 3272.5 | 1045.2 |
| dualclock | 12 | 17280 | 1626.9 | 1984.3 | 679.5 |

Active time은 실제 학습 step이며 V 평가·checkpoint·GPU 대기·teacher cache 준비는 포함하지 않는다. 총 queue wall은 queue_status.json, 각 F0 cache 준비 시간은 각 작업 baselines.json에 있다. 동일 updates와 minibatch를 제공했으며 동일 end-to-end 비용을 주장하지 않는다.

두 실험 모두 기존 train/V를 재사용했다. Anchor는 두 데이터셋의 네 채널, DualClock은 한 M5 subset이다. Seed는 optimizer 반복이고 독립 데이터 표본이 아니다. 성능이 좋은 조건만 선택해 새 PASS를 만들지 않았다. 후속 방법/적용 조건의 변경에는 별도 미노출 평가와 사전에 고정한 비교가 필요하다.

이 실행은 끝났으며 새 E 평가나 또 다른 연구 주제를 자동 실행하지 않는다. 과거 FAIL/STOP은 유지한다.
