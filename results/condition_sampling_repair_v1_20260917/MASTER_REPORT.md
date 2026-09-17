# 날짜 다양성 교정 — 전체 보고서

단일 [실행 계약](MASTER_PROTOCOL.md), [저장소 감사](REPOSITORY_AUDIT.md), [GPU 전 원점 감사](ORIGIN_REPAIR_AUDIT.md).

| track | execution | fits | evidence | tags |
| --- | --- | --- | --- | --- |
| N01 | BLOCKED_DIVERSITY | 0 | NOT_MEASURED |  |
| N02 | COMPLETE | 16 | POSITIVE_UNCERTAIN | POSITIVE_UNCERTAIN,SAMPLING_SENSITIVE |
| N03 | COMPLETE | 16 | POSITIVE_UNCERTAIN | POSITIVE_UNCERTAIN,SAMPLING_SENSITIVE |
| R04 | BLOCKED_DIVERSITY | 0 | NOT_MEASURED |  |
| N07 | BLOCKED_DIVERSITY | 0 | NOT_MEASURED |  |
| R08 | BLOCKED_DIVERSITY | 0 | NOT_MEASURED |  |

| track | main_fits | main_updates | smoke_updates |
| --- | --- | --- | --- |
| N01 | 0 | 0 | 0 |
| N02 | 16 | 8192 | 8 |
| N03 | 16 | 8192 | 8 |
| R04 | 0 | 0 | 0 |
| N07 | 0 | 0 | 0 |
| R08 | 0 | 0 | 0 |

본학습 상한96fits/49,152updates, smoke 상한48updates, 총49,200이다. 다양성 미충족 트랙의 예산은 다른 학습에 사용하지 않는다. 실행 완료를 논문 성공으로 해석하지 않는다.

## UNAFFECTED_REFERENCE

R05/N06는 기존 저장 예측 CPU 분석이고 이번 selector와 무관하다. R05의 알려진 혼합 정책은 재사용 E 및 두 모델 비용을 포함한다. N06의 AR1 합계 CRPS 개선은 알려진 단순 결합이며 새 PEFT 근거가 아니다. R09는 기존 E64가64 index-days에 분산되어 대상 밖이다. NULL 대 MIXED의 부정적 결과와 FINE_ONLY의 고유 상세 정답16 대 MIXED64의 정보 차이를 유지한다. 세 결과를 이번 새 증거나 우승 점수로 합산하지 않았다.

## 후보별 보고서

- [N01](N01/REPORT.md)
- [N02](N02/REPORT.md)
- [N03](N03/REPORT.md)
- [R04](R04/REPORT.md)
- [N07](N07/REPORT.md)
- [R08](R08/REPORT.md)

[최종 결정](FINAL_DECISION.md). 큰 원자료·예측 배열·가중치는 로컬 ignored cache에 보존하며 GitHub만으로 수치 재생이 된다고 주장하지 않는다.

## 최종 답변 — 무엇이 달라졌는가

**적격 두 트랙은 학습·평가·교차 평가·검산을 완료했고, 네 트랙은 GPU 전 BLOCKED_DIVERSITY로 종료했다.** 전체96경로 상한 중32경로를 실행했고64경로는 미실행이다. R05/N06/R09는 재실행하지 않았다. 이번 repair에서도 새 PEFT 구성요소 미확보, 다음 투자 후보0개다. 동시에 기존 부정적 수치를 일반적인 방법 실패로 확대할 근거도 약해졌다.

### 1. 기존 표본의 문제와 교정 결과

기존 알고리즘은 phase별 quota를 먼저 배정했기 때문에 여러 phase가 같은 첫·중간·마지막 날짜를 반복 선택했다. N02는 원점 span만 보면 기간 전체에 걸쳐 있었지만 실제 날짜는6개, target timestamp 고유 비율은6.93%였다. N03는64원점이23.75시간 안에 몰려 target timestamp 고유 비율이4.65%였다. 단순한 원점 수나 전체 span만으로 다양성을 확인할 수 없었다.

새 selector는 날짜를 먼저 고르고 한 날짜에서 한 origin만 선택한다. N02/N03의 TRAIN/E64일·V32일, span ratio1.0, week/phase 기준을 모두 충족했다. 아래 timestamp 수는 채널을 곱한 독립 표본 수가 아니라 시간 index의 고유/총 참조 수다.

| track | version | distinct_days | origin_span_hours | seven_day_blocks | target_timestamps_unique | target_timestamps_total | target_uniqueness_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N02 | old | 6 | 3461.000000 | 3 | 213 | 3072 | 0.069336 |
| N02 | repaired | 64 | 3457.000000 | 22 | 3026 | 3072 | 0.985026 |
| N03 | old | 2 | 23.750000 | 2 | 143 | 3072 | 0.046549 |
| N03 | repaired | 64 | 3456.250000 | 22 | 3072 | 3072 | 1.000000 |

N01/R04/N07/R08은 날짜64개·phase24개를 얻었으나, 마지막 부분 날짜에서 circular distance상phase0이 선택되어 phase별 개수가1~4개(차이3)가 됐다. 계약의 한도2를 넘었으므로 사전 차단했다. 숫자를 채우려고 날짜를 중복하거나 phase 규칙·split을 바꾸지 않았다. [원점 감사](ORIGIN_REPAIR_AUDIT.md), [결과를 보기 전 해석 한계](DESIGN_LIMITATIONS.md).

### 2. 기존 효과가 유지됐는가

| track | old_gain_percent | repaired_gain_percent | CI_low | CI_high | evidence |
| --- | --- | --- | --- | --- | --- |
| N02 | -2.012361 | 0.013565 | -0.006485 | 0.036592 | POSITIVE_UNCERTAIN + SAMPLING_SENSITIVE |
| N03 | -1.035942 | 0.001136 | -0.038884 | 0.040342 | POSITIVE_UNCERTAIN + SAMPLING_SENSITIVE |

두 주 비교 모두 ROBUST_NEGATIVE는 아니다. 가까운 대조에 대한 평균 부호가 바뀌어 SAMPLING_SENSITIVE를 병기했으나 CI가0을 포함한다. N02는 두 seed 양수, N03는 seed 부호가 다르다. REOPEN_CANDIDATE는0개이며, 네 사전 차단 트랙에는 repaired evidence 태그를 부여하지 않는다. "부호가 유의하게 뒤집혔다"거나 "차이가 없다"고 입증한 것이 아니다.

### 3. 평가 원점 변화인가, 재학습까지 해야 달라졌는가

| track | OLD_MODEL_OLD_E_gain | OLD_MODEL_NEW_E_gain | NEW_MODEL_NEW_E_gain | NEW_MODEL_OLD_E_gain |
| --- | --- | --- | --- | --- |
| N02 | -2.012361 | 3.391018 | 0.013565 | 0.054016 |
| N03 | -1.035942 | -0.425677 | 0.001136 | -0.814853 |

N02는 옛 가중치를 새 E로 옮기는 것만으로 가까운 대조 대비−2.012%가+3.391% [1.621%,5.226%]로 바뀌었다. 재학습 없이 평가 분포에서 이미 변화가 나타났다. 새 TRAIN/V로 적응·선택한 후에는 B3뿐 아니라 B2의 새 E 오차도 낮아져 상대적인 추가 이득은+0.014%로 줄었다.

N03는 평가만 바꾸면−1.036%가−0.426%로 약해지고 CI에0이 포함됐다. 새 가중치를 옛 E에 적용하면 여전히−0.815%였고, 새 가중치/새 E에서만 약0인 양의 평균이 나타났다. TRAIN/V·선택·E가 함께 바뀐 복합 효과를 완전한 인과 분해라고 부르지 않는다.

**N02 모든 군의 네 조합 원점수 — 두 반복 평균, 낮을수록 좋음**

| arm | OLD_MODEL_OLD_E | OLD_MODEL_NEW_E | NEW_MODEL_NEW_E | NEW_MODEL_OLD_E |
| --- | --- | --- | --- | --- |
| B0 | 0.371624 | 0.630879 | 0.608899 | 0.375896 |
| B1 | 0.378546 | 0.605353 | 0.566402 | 0.395858 |
| B2 | 0.379895 | 0.662370 | 0.604291 | 0.395349 |
| B3 | 0.387540 | 0.639909 | 0.604209 | 0.395135 |

[모든 seed와 점수 변화](N02/four_way_score_decomposition.csv), [같은 E 위 가중치 비교 CI](N02/same_E_weight_effects.csv). 서로 다른 E의 절대 점수 차이에는 paired CI를 붙이지 않았다.

**N03 모든 군의 네 조합 원점수 — 두 반복 평균, 낮을수록 좋음**

| arm | OLD_MODEL_OLD_E | OLD_MODEL_NEW_E | NEW_MODEL_NEW_E | NEW_MODEL_OLD_E |
| --- | --- | --- | --- | --- |
| C0 | 0.893154 | 0.724521 | 0.687344 | 0.825264 |
| C1 | 0.688727 | 0.737559 | 0.668426 | 0.652408 |
| C2 | 0.641412 | 0.734150 | 0.673544 | 0.503911 |
| C3 | 0.648056 | 0.737275 | 0.673536 | 0.508017 |

[모든 seed와 점수 변화](N03/four_way_score_decomposition.csv), [같은 E 위 가중치 비교 CI](N03/same_E_weight_effects.csv). 서로 다른 E의 절대 점수 차이에는 paired CI를 붙이지 않았다.

### 4. 단순 대안과 새 구성요소의 필요성

N02의 LONG0.566402가 제안 B3 0.604209보다 낮고 두 seed에서 우세했다. 긴 이력은 입력량·메모리 비용이 더 드는 대조이므로 무비용 대안이라고 부르지 않는다. 그래도 검색 후 보정의 필요성을 입증하지 못했다. N03의 학습 커널0.673536과 고정 커널0.673544의 차이는 불확실했고 HOLD 평균0.668426도 함께 공개했다. HOLD의 우위가 모든 seed에서 유지된 것은 아니므로 엄격한 반복 일관성의 SIMPLE_METHOD_SUFFICIENT를 새로 확정하지 않았다. 고정 커널·HOLD를 넘는 학습 τ의 추가 가치는 미확보다.

### 5. 실행·자원·검산

| track | fits | optimizer_seconds | selection_included_fit_seconds | main_E_prediction_seconds | cross_prediction_seconds | peak_allocated_MiB | contaminated_updates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N02 | 16 | 786.951538 | 984.020040 | 34.861486 | 34.614963 | 783.712402 | 0 |
| N03 | 16 | 811.891600 | 1064.412031 | 121.411692 | 139.547983 | 748.025879 | 0 |

본학습 16,384/49,152 updates, smoke 16/48, 총 16,400/49,200회다. 미실행 본학습32,768updates와smoke32회는 사전 차단의 결과이며 여유 예산을 전용하지 않았다. native forward는 30,416/200,000회이고 TRAIN16,400·V/E/교차13,952·검증64 호출이 [독립 장부](FORWARD_LEDGER.csv)와 일치했다. cross 추가optimizer0회다.

GPU guard의 실제 경과는 42.24분, 최소 여유 VRAM 7892MiB, peak CPU RSS 2.380GiB였다. 새 cache는 0.911GiB다. 위 표의 peak allocated는 PyTorch 할당이며 장치 총사용량과 다르다. guard wall의 external_compute_samples는 허용된 RustDesk 관측도 포함하는 원시 표기다. 다른 학습에 의한 오염 update는0회이고 사전 허용된 RustDesk 외의 compute 작업을 공유하지 않았다.

optimizer 시간, V/체크포인트를 포함한 경로 시간, E 예측 시간, 교차 예측 시간을 분리했다. 예측 루프에는 guard/Python 비용이 포함된다. CPU 검산을 일부 병행했으므로 작은 시간 차이를 엄격히 격리된 속도 우위로 해석하지 않는다.

[최종 검산](verification.json)은 기존 파일 2,952개 보존, 모든32경로의 정확한 원점 순서·optimizer/RNG/최종resume, V scalar96개와 모델 선택24건, 주 지표672개 scalar 및 교차36개 scalar를 확인했다. [모델·수식 동일성](implementation_parity.json), [원점 회귀 검사](origin_selector_verification.json), [실제 입력 poison 검사](input_verification.json), [검색/주파수 통계 독립 검사](supplemental_input_audit.json)를 분리했다. N02 a/b는 모든 업데이트에서 비영 gradient였고 포화율0, N03 θ는 dense epoch를 제외한 절반의 업데이트에서 비영 gradient였다. 실제 학습 오류를 성능 부진으로 분류하지 않았다.

### 6. 무엇을 남기고 어디서 종료하는가

기존 결론의 표본 의존성을 확인했지만 새 PEFT 구성요소의 채택 근거는 확보하지 못했다. 다음 투자 후보는0개다. 정식 선행 비교와 독립 source 검증은 남았고 이번에 수행하지 않았다. 날짜 우선 선택에도 날짜 순서와 목표 phase의 결정적 연결, 동일 공개 원천·개발 기간 재사용, 두 seed·네 채널, 다중 주제 탐색의 한계가 있다. 이번32fits를 아홉 후보 탐색의 독립 확증으로 재명명하지 않는다. [최종 결정](FINAL_DECISION.md)에 따라 새 후보·seed·LR·dataset·후속 학습 없이 종료한다.
