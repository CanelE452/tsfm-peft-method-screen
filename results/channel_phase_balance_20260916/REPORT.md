# 시작 위상 균형 LoRA 대조 결과

**단일 sampling 변경의 개발 대조를 완료했다. 알려진 표본 설계이므로 결과가 좋아도 새로운 PEFT 방법론 PASS는 아니다.**

신규4/4 fits, 4632/5080 optimizer updates, smoke4 updates. 기존 LH4fits는 재학습하지 않았다. 모든 E 평가는 이미 노출된 같은 구간이다. 모델·원자료·채널·split·정규화·V/E 원점·학습률·epoch 상한은 유지했다.

[최종 원인 해석과 남은 목표](INTERPRETATION.md). 대상은 MOMENT-small의32채널96→96 예측이며, 단일건물 Chronos-2 실험의 원인 검증은 아니다.

## 검증한 변경

기존 날짜별 원점 a_i에 i mod24 시간을 더하고, train 끝을 넘는 경우24시간을 뺐다. 기존 시점에서 최대23시간 이동하며512/504개 창 수를 유지한다. 모델은 같은761952개 trainable parameters의 LH다. 각seed의 초기 학습 파라미터는 기존 epoch0 checkpoint와 동일했고, BF16 smoke의 초기 V 예측도 기존 저장값과 정확히 같았다.

선택 checkpoint 비교와 별개로 기존 V 선택 epoch에 맞춘 비교를 사전에 고정했다. 후자는 optimizer updates 수가 정확히 같다. 창의 개별 값과 순서도 바뀌므로 전체 sampling 변경의 효과이며 시간 위상 하나의 순수 인과효과를 증명한 것은 아니다.

## 원점수와 효과

| 원천 | seed | 평가 recipe | epoch / updates | 새 MSE | 기존 LH MSE | LH 대비 개선 |
|---|---:|---|---:|---:|---:|---:|
| electricity | 41000 | selected | 16 / 1024 | 0.278283 | 0.465861 | +40.265% |
| electricity | 41000 | matched_old_epoch | 1 / 64 | 0.321743 | 0.465861 | +30.936% |
| electricity | 41001 | selected | 8 / 512 | 0.281737 | 0.472899 | +40.423% |
| electricity | 41001 | matched_old_epoch | 3 / 192 | 0.302970 | 0.472899 | +35.933% |
| traffic | 41000 | selected | 17 / 1071 | 0.385908 | 1.220259 | +68.375% |
| traffic | 41000 | matched_old_epoch | 1 / 63 | 0.509221 | 1.220259 | +58.269% |
| traffic | 41001 | selected | 18 / 1134 | 0.387187 | 1.250465 | +69.037% |
| traffic | 41001 | matched_old_epoch | 1 / 63 | 0.504907 | 1.250465 | +59.622% |

| 원천 | recipe | 새 평균 MSE | 기존 평균 MSE | 평균 개선율 | 두 seed 모두 개선 |
|---|---|---:|---:|---:|---|
| electricity | selected | 0.280010 | 0.469380 | +40.345% | True |
| electricity | matched_old_epoch | 0.312357 | 0.469380 | +33.453% | True |
| traffic | selected | 0.386548 | 1.235362 | +68.710% | True |
| traffic | matched_old_epoch | 0.507064 | 1.235362 | +58.954% | True |

계절 반복과의 비교, normalized MAE, 원단위MAE, 자원 사용량은 [comparisons.csv](comparisons.csv)에 있다. 효과는같은원천의 macro에서100×(기존−새값)/기존으로 계산했다. 서로 다른 원천의 raw MAE를 섞지 않는다. 이전 online 일별 잔차 보정은 E 정답을 추가 사용하므로 이 표의 동일정보 대조에 포함하지 않았다.

## 실제 자원·검산·미실행

| fit | epochs | updates | peak MiB | optimizer active seconds |
|---|---:|---:|---:|---:|
| 00_electricity_41000_BALANCED_LH | 20 | 1280 | 1069.600 | 61.90 |
| 01_electricity_41001_BALANCED_LH | 13 | 832 | 1069.100 | 40.37 |
| 02_traffic_41000_BALANCED_LH | 20 | 1260 | 1067.100 | 61.11 |
| 03_traffic_41001_BALANCED_LH | 20 | 1260 | 1069.850 | 61.05 |

Controller wall 685.3초, 최소GPU여유 7587MiB, 비승인compute 표본 0개. 모델 파라미터 수는 변하지 않았으므로 파라미터 절감은0이다. 20epoch 상한 도달을 수렴으로 간주하지 않는다. 선택된epoch와 실제 종료epoch가 달라 총연구비용과 배포 적응비용을 구분해야 한다.

고유예측cache 99개/99개 MSE·MAE 기록을 독립 scalar로 검산했다. 최대 MSE차 2.22e-16. 선택 checkpoint4개를 새 모델에서 재생해 차이0, 같은epoch 비교의updates 일치, 기존 1974개 결과 파일 보존을 확인했다. 모든 새 train 창은 train 경계 안에 있고 미래값 오염의 영향을 받지 않았다.

미완료 fits0. 추가 후보·추가seed·학습률 조정·평가원점 교체0. 다른 시작 위상/새 원천의 독립 확인과 새로운 PEFT 방법의 추가 가치는 이 실행에서 검증하지 않았다. 동일 sampler를 사용한 Time-PEFT 채널 모듈 재평가도 아직 하지 않았다.

## 연구 판단의 범위

실행은 COMPLETE, 근거는 EXPOSED_DEVELOPMENT_SAMPLING_CONTROL, 신규성은 KNOWN_SAMPLING_CONTROL이다. 두 원천·두 seed 모두 sampling 대조에서 개선됐다. 이전 제한된 학습 시작점에서 얻은 모듈 순위를 일반적인 학습 조건으로 넓히지 않아야 한다. 모든 조건의 원점수를 공개했다. 새 sampling의 고유 target 시점은 기존 target 시점의 부분집합으로, 새로운 정답 시점 추가는0이었다. [정답 노출 감사](target_exposure_audit.json). 새 알고리즘의 유용성은 이 강한 baseline과 같은 정보량의 단순 대조를 넘을 때 별도로 검토해야 한다.

[봉인 프로토콜](PROTOCOL.md), [표본 감사](sampler_audit.json), [검산](verification.json), [판단](decision.json). 원자료·가중치·예측 배열은 로컬cache에 남고 GitHub에는 source·원점수·hash·검증 기록을 보관한다.
