# 예보 vintage 참조 — 실제 예측 결과

[확인] 판정은 **NO_ADDED_VALUE_IN_THIS_REFERENCE**다. 새 PEFT 주제는 아직 확보하지 못했다. 신경망 학습 0fits·업데이트0, Chronos-2 target forecast **233개 / pipeline 128호출**을 완료했다. 동일 모델의 통계 보정7건과 폭 후보 비교30개를 별도로 수행했다.

## 질문과 고정 범위

네 개의 적법한 과거 예보 버전을 여러 미래 경로로 처리하는 것이 최신 경로 및 같은 보정 기회의 단순 대조보다 나은지 확인했다. OS Gorredijk 한 타깃의 C12원점(3~5월)에서 보정을 고정하고 D16원점(6~9월)을 평가했다. 10~12월은 점수 계산에 쓰지 않았다. 데이터 가용성 점검 이후 고정한 탐색 구간이며 독립 외부 검증으로 주장하지 않는다. [사전 프로토콜](PROTOCOL.md), [입력·코드·모델 봉인](seal.json).

기상 경로는 원점에 가용한 최신 네 행을 각 valid time에서 모은 것으로, 같은 발행 cycle의 확률적 ensemble이 아니다. 실제 기상 정답을 입력하거나 보정에 쓰지 않았다. 과거 기상도 versioned 파일의 모사된 가용시각으로 선택했다. 부하 및 기상 미래 행의 오염에 28개 원점 입력이 모두 불변이었다. 시간별 평균은 정확히 네 15분 값으로 만들었고 결측 때문에 원점을 교체하지 않았다.

Chronos-2의 알려진 공변량 입력 기능을 사용했다. CDF 혼합과 편향·폭 보정도 알려진 참조 연산이며, 이번에 새 PEFT 모듈을 구현하거나 학습하지 않았다. [공식 Chronos 코드](https://github.com/amazon-science/chronos-forecasting).

## 개발 원점수

Primary는 21개 native 분위수의 평균2-pinball / 각 원점의 과거336h 표준편차다. 낮을수록 좋다. 연속 CRPS 정확값이 아니다. 모두 같은 원점·target·scale로 비교했다. CAL은 C만으로 추정한 편향과 폭이다. 계절 대조는 C잔차 분위수를 더했다. 원래 분위수 crossing을 정렬하는 정책도 모두 동일하다.

| 방법 | Primary | raw 2-pinball | median RMSE | 80% 포함률 |
|---|---:|---:|---:|---:|
| F0 | 0.181721253 | 77400.0736 | 176242.1900 | 0.8021 |
| PAST | 0.195957419 | 82953.6423 | 185352.8233 | 0.8333 |
| LATEST | 0.149478679 | 64334.8364 | 140815.2333 | 0.7995 |
| MEAN_INPUT | 0.151565625 | 65191.0077 | 138364.0945 | 0.7812 |
| MIXTURE | 0.147815075 | 63430.4337 | 139073.0984 | 0.8438 |
| F0_CAL | 0.185938368 | 79083.7599 | 174404.4601 | 0.9427 |
| PAST_CAL | 0.195648074 | 82758.3673 | 184705.2298 | 0.9089 |
| LATEST_CAL | 0.150457303 | 64727.9932 | 140528.8786 | 0.8698 |
| MEAN_INPUT_CAL | 0.151790342 | 65294.2919 | 138305.5656 | 0.8385 |
| MIXTURE_CAL | 0.149799309 | 64252.8662 | 139118.7995 | 0.9010 |
| SEASONAL24 | 0.310022730 | 131266.2151 | 257077.9325 | 0.8359 |
| SEASONAL168 | 0.277288800 | 120418.8296 | 228586.3053 | 0.9089 |

MIXTURE_CAL 대비 가장 낮은 비-mixture 대조는 **LATEST**다. 후보 상대 개선율은 **-0.214%**, 개선한 달은 **2/4**다. 월 단위 탐색적 bootstrap95% 구간은 [-7.086, 6.438]%다. 네 달뿐이므로 정밀한 불확실성 추정이나 모집단 유의성으로 해석하지 않는다. [전체 원점수](scores.csv), [모든 대조 효과](effects.csv), [월별](monthly.csv), [C 보정 기록](calibration.json).

MIXTURE는 분포의 CDF를 혼합한 뒤 분위수를 역산했다. 각 native 분위수 함수를 선형 보간하고 양끝은 고정한 유계 분포라는 가정이 있다. 단순 분위수 평균과 다르다. 여러 경로를 처리해도 부하의 joint trajectory distribution을 얻었다고 주장하지 않는다.

## 비용과 검산

GPU controller 45.81초(시작 안정 대기 포함), 최대 allocated 503.61MiB, 최저 여유 8507MiB다. 실행 원장에 외부 compute 및 오염 여부를 기록했다. 아래 시간은 동일 batch_size16, 첫 C원점에서 warmup1회 뒤5회 측정한 pipeline 호출이다. 4PATHS는 네 경로를 배치로 처리한다.

| 처리 | 중간값 ms | 최대 allocated MiB |
|---|---:|---:|
| LATEST | 22.545 | 492.44 |
| PATHS | 23.539 | 503.61 |

CDF 혼합 CPU 총 1.618초/28원점, 보정 설정 선택 0.010초는 별도다. 전처리·입출력·모델 적재를 전부 포함한 서비스 latency가 아니며, 학생을 실행하지 않았으므로 PEFT 자원 이득은 미평가다. [호출 원장](calls.json), [GPU controller](predict_wall.json).

모든 336개 원점/방법 점수를 독립 scalar float64로 확인했고 최대 오차는 2.44e-15다. 혼합 분위수 14112개의 CDF 조건을 별도 스칼라 코드로 점검했다. 첫 C원점 7개 예측을 같은 shape로 재생했고 검산 결과는 [verification.json](verification.json)에 있다. 파라미터·버퍼 hash와 기존 2221개 결과 파일 hash도 그대로다.

## 다음 결정과 남은 한계

이번 사전 teacher 신호 조건의 충족 여부는 False다. 이 결과만으로 모든 불확실성 처리 방법을 기각하거나 새 PEFT의 성공을 선언하지 않는다. 동일 가중의 나이 다른 예보 경로는 보정된 미래 입력 분포가 아니며, 한 지역의 여름 개발 구간이다. 일반 LoRA·동일 정보와 감독의 증류·학생 PEFT·독립 원천 검증은 미실행이다. 원점수 확인 뒤 입력·기간·경로수·문턱을 바꾸지 않았다.

예보 버전 자체의 가치와, 그 가치를 학생으로 옮기는 새 적응 규칙의 가치는 별도다. 강한 단순 대조가 이 참조를 대체한다면 해당 참조를 teacher로 삼아 학생부터 학습할 근거가 부족하다. 개선이 확인돼도 알려진 ensemble distillation 이상의 차별점을 따로 입증해야 한다. [최종 상태](decision.json). 자동 학생 학습은 실행하지 않았다.

## 재현

코드·provenance·원점수·검증은 Git에 보관하고 입력/target/예측 npz·모델 가중치는 로컬 캐시에 둔다. 준비/예측은 같은 run을 덮어쓰지 않으며 기존 완료 결과를 다시 학습하지 않는다.

```bash
scripts/with_cuda.sh .venv/bin/python experiments/covariate_vintage_reference_20260916/run.py prepare
scripts/with_cuda.sh .venv/bin/python experiments/covariate_vintage_reference_20260916/run.py predict
scripts/with_cuda.sh .venv/bin/python experiments/covariate_vintage_reference_20260916/report.py
```

[추가 해석](INTERPRETATION.md)에는 미래 기상 자체의 이득, 보정 전 혼합의 양성 관찰, 보정 전달 실패와 배치 비용을 별도로 설명했다. 사전 판정은 변경하지 않았다.

공개 전에는 최종 캐시 28개에서 primary 및 보조 지표를 포함한 336행·2,016개 수치를 별도 스칼라 루프로 다시 검산하고 macro 집계와 실제 호출 횟수를 확인했다. [공개 검증](publication_audit.json).
