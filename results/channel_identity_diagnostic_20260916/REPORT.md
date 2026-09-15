# 표현 보존 채널 PEFT 진단 — 한국어 결과

**이번 실행은 알려진 ReZero 대조의 원인 진단이다. 새 PEFT 방법론 주제나 독립 SCREEN_PASS를 확보했다는 결과가 아니다.**

실행 상태 COMPLETE. 신규 본학습 4/4 fits, optimizer updates 2030/5080. smoke 4/4 updates. 기존 LH·SHARED_BUDGET 8 fits를 다시 학습하지 않고 해시 검산 후 재사용했다.

## 무엇을 확인했나

사전학습 표현 h를 새 채널 모듈 출력 u로 교체하던 경로에 h+a·u를 적용했다. a는0에서 학습하는 스칼라이고 나머지 설정은 기존 R2와 같다. 초기 CPU 두 seed 및 실제 BF16 smoke에서 LH와 출력이 정확히 같았다. 이후 gate·주파수·채널·head·LoRA가 업데이트되고 동결 파라미터/버퍼가 보존됐다.

이는 잔차 경로와0 초기화를 함께 바꾼 대조다. 두 요인의 개별 인과효과나 과거 모든 실패의 원인을 증명하지 않는다. 공개 Time-PEFT 전체 재현도 아니다.

## 원점수와 같은 seed 비교

지표는 기존과 같은 정규화된 채널-macro MSE다. 개선율은100×(대조−진단군)/대조. 양수일수록 진단군이 좋다. 기존 E는 이미 여러 번 노출됐으며 이번에도 개발 자료(DISCOVERY_REUSED_E)다.

| 원천 | seed | 선택 epoch | 진단군 MSE | 대조 | 대조 MSE | 개선율 |
|---|---:|---:|---:|---|---:|---:|
| electricity | 41000 | 1 | 0.471337 | LH | 0.465861 | -1.176% |
| electricity | 41000 | 1 | 0.471337 | SHARED_BUDGET | 0.489744 | +3.758% |
| electricity | 41000 | 1 | 0.471337 | SEASONAL_NAIVE | 0.525609 | +10.325% |
| electricity | 41001 | 3 | 0.458071 | LH | 0.472899 | +3.136% |
| electricity | 41001 | 3 | 0.458071 | SHARED_BUDGET | 0.488276 | +6.186% |
| electricity | 41001 | 3 | 0.458071 | SEASONAL_NAIVE | 0.525609 | +12.849% |
| traffic | 41000 | 1 | 1.243977 | LH | 1.220259 | -1.944% |
| traffic | 41000 | 1 | 1.243977 | SHARED_BUDGET | 1.703336 | +26.968% |
| traffic | 41000 | 1 | 1.243977 | SEASONAL_NAIVE | 0.981449 | -26.749% |
| traffic | 41001 | 7 | 1.321253 | LH | 1.250465 | -5.661% |
| traffic | 41001 | 7 | 1.321253 | SHARED_BUDGET | 1.598655 | +17.352% |
| traffic | 41001 | 7 | 1.321253 | SEASONAL_NAIVE | 0.981449 | -34.623% |

normalized MAE와 원 단위 channel-macro MAE, 모든 비용은 [comparisons.csv](comparisons.csv)에 함께 보관했다. 두 데이터의 단위가 다르므로 raw MAE를 원천 간 단일 성능으로 합산하지 않는다.

## 같은 원천 두 seed의 평균

| 원천 | 진단군 MSE | LH MSE | SHARED MSE | 계절 반복 MSE |
|---|---:|---:|---:|---:|
| electricity | 0.464704 | 0.469380 | 0.489010 | 0.525609 |
| traffic | 1.282615 | 1.235362 | 1.650995 | 0.981449 |

## 추가 가치와 연구 판단

- electricity: SHARED 대비 두 seed 모두 양수=True; LH 대비 두 seed 모두 양수=False.
- traffic: SHARED 대비 두 seed 모두 양수=True; LH 대비 두 seed 모두 양수=False.

이 기록은 봉인 프로토콜의 서술 규칙에 따른 개발 신호다. 기존 실험의1%/CI/seed 기준을 소급 변경하지 않는다. 같은 데이터에서 다른 이름·gate 초기값·학습률로 재시도하지 않았다.

## 자원과 실제 실행

학습 파라미터는 진단군1,319,712개, SHARED1,319,711개, LH761,952개다. SHARED보다1개 많고 LH보다 약73.20% 많으므로 파라미터 절감은 없다. 아래 시간은 실제 완료 trajectory의 optimizer-step 누적 시간이며 선택 checkpoint까지의 배포 비용과 다르다. 기존 실험과 현재 측정의 실행 시점이 달라 인과적인 속도 우위로 해석하지 않는다.

| fit | epochs | updates | 선택 epoch | peak MiB | active seconds |
|---|---:|---:|---:|---:|---:|
| 00_electricity_41000_REZERO_SHARED | 6 | 384 | 1 | 1111.438 | 20.49 |
| 01_electricity_41001_REZERO_SHARED | 8 | 512 | 3 | 1110.938 | 27.40 |
| 02_traffic_41000_REZERO_SHARED | 6 | 378 | 1 | 1114.688 | 20.44 |
| 03_traffic_41001_REZERO_SHARED | 12 | 756 | 7 | 1113.438 | 40.80 |

실행 wall 337.7초, 관측 최소 GPU 여유 7570MiB. GPU 원장에는 승인된 RustDesk와 자체 작업을 구분해 남겼다.

## 검증·미실행·한계

저장 예측 62개 MSE 및 62개 MAE를 독립 float64 scalar로 재계산했다. 최대 MSE 절대차2.22e-16. 선택 체크포인트 4개를 새 모델에서 재생해 예측 차이0을 확인했다. 두 원천의 모든 학습 origin에서 V 이후 값을 독으로 바꿔도 학습 batch가 변하지 않았다. 기존 결과·연구 1932개 파일의 해시는 보존됐다.

미완료 본학습은 0 fits. 계획에 없는 다른 후보·추가 seed·재튜닝은 실행하지 않았다. 독립 미사용 데이터 평가와 새 방법론 비교는 이 진단 범위에 없으며 미실행이다.

ReZero는 알려진 방법이며 이번 적용 자체의 신규성은 DIRECT_EQUIVALENCE다. 채널별 저랭크, hypernetwork, 변하는 채널 수의 TSFM 적응에도 가까운 선행이 있다. [선행·연구 검토](RESEARCH_REVIEW.md), [봉인 프로토콜](PROTOCOL.md), [검증](verification.json), [판단](decision.json)을 함께 읽어야 한다.

현재 새 방법론 주제는 미확보다. 논문 주제를 정하려면 실제 남은 오류 구조와 강한 단순 대조로 설명되지 않는 이득, 가까운 선행과의 구체적 차이가 더 필요하다. GPU 실행이 정상이라는 사실을 예측·신규성 PASS로 바꾸지 않는다. 원시 자료·가중치·예측 배열은 로컬 cache에 남고 GitHub에는 해시와 검증 결과만 보관한다.
