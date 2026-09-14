# Censor tail controlled v1 — 완료 검토

본학습 **6/6 fits, 2,160 updates**, 실제 모델 smoke **6 updates**를 완료했다. 모든 본학습 update가 실제 파라미터를 변경했다. TAIL은 DROP보다 두 seed 모두 작게 개선했지만 평균 +0.001729%이며, NAIVE보다 -0.675272%다. 사전 판정은 **현재 설정에서 추가 가치 미확보**이고, 추가 학습 없이 종료했다.

## 1. 무엇을 고쳤고 무엇을 비교했는가

유한 지지범위와 survival clipping으로 큰 벌점에도 gradient가 0이 되던 꼬리 계산을 지수 꼬리와 직접 log-survival로 완성했다. 원래 함수와 과거 FAIL은 보존했다. 새 어댑터와 anchor 없이 동일 LoRA에서 NAIVE(관측 판매량 감독), DROP(검열 target 제외), TAIL(DROP + 하한 사건 벌점)을 비교했다. 이 수정은 연속 분포의 꼬리 가정이며 새 PEFT 구조나 정확한 이산 count likelihood가 아니다.

## 2. 데이터와 선택 권한

기존 M5 256개 계열의 원판매량에 동일한 인위적 상한을 적용했다. 학습 데이터에는 capped 관측과 검열 비트만 전달했고, scale도 capped train에서 계산했다. 숨은 초과 크기를 바꾸어도 허용 입력과 비트가 같음을 검사했다. clean V 원판매량은 모든 arm의 checkpoint 선택에 동일하게 허용했다. 여섯 선택과 GPU disk reload를 봉인한 뒤 reused E_dev에 접근했다. 이는 실제 품절 수요의 정답도, 연구자에게 새로운 독립 test도 아니다.

## 3. 전체 성능과 불확실성

| V-selected 비교 | seed41000 개선율 | seed41001 개선율 | seed 평균 개선율 | 기술적 paired-series 95% CI |
| --- | --- | --- | --- | --- |
| TAIL 대 DROP | +0.001774% | +0.001684% | +0.001729% | [+0.001559%, +0.001886%] |
| TAIL 대 NAIVE | -0.673457% | -0.677087% | -0.675272% | [-0.890398%, -0.469013%] |
| TAIL 대 F0 | +0.021423% | +0.028839% | +0.025131% | [-0.059775%, +0.116342%] |

개선율은 100×(baseline−TAIL)/baseline이다. 두 seed 평균 원 primary는 F0 0.813549085, NAIVE 0.807889185, DROP 0.813358694, TAIL 0.813344631이다. DROP 대비 절대 감소는 0.000014063743이다. CI가 0을 포함하지 않아도 효과 크기는 매우 작다. 상품/상점과 미래 날짜의 상관을 완전히 처리하지 않은 2,000회 series bootstrap이므로 일반화 확증으로 해석하지 않는다. F0 대비 개선은 CI가 0을 포함한다.

NAIVE는 두 seed 모두 step360, DROP과 TAIL은 모두 step15가 V에서 선택됐다. 모두 학습된 상태이며 step0 선택 사례는 없다. 사전 고정한 step360 secondary에서는 TAIL이 DROP보다 0.014371%, NAIVE보다 3.271792%, F0보다 2.553325% 나빴다. primary와 secondary를 바꾸어 성공으로 보고하지 않았다.

## 4. 하위집단과 과소예측

TAIL은 DROP 대비 검열 위치의 전체 오차 기여도를 약 0.000018199 줄였고, 비검열 기여도는 약 0.000004135 늘렸다. 따라서 전체의 작은 순이득을 확인할 수 있다. NAIVE의 전체 우세는 전체 분모로 계산한 비검열 기여도에서 주로 나온다. 조건부 subgroup primary와 전체 위치 분모의 기여도는 서로 다른 가중 방식이며 혼동하지 않는다.

TAIL의 검열 위치 median signed bias는 -3.062596, F0는 -3.025517로 여전히 강한 과소예측이다. TAIL의 검열 위치 80% interval coverage는 0.248347%이며, 전체 coverage는 78.849284%다. 추가 하한 정보를 넣었다고 숨긴 판매량 분포를 충분히 복원한 것은 아니다. 모든 arm의 MAE·bias·coverage·width와 subgroup 기여도는 [seed 평균 표](seed_mean_metrics.csv), [seed별 전체 표](metrics.csv)에 있다.

## 5. 수치 수정은 작동했는가, 효과는 왜 작았는가

CPU 검사에서 기존 강한 upper-tail 예제의 0 gradient와 수정식의 음수 common-shift gradient를 재현했다. 실제 TAIL 학습에서는 강한 상한 위반이 각각 337/339회 있었고, 해당 출력 텐서의 공통 이동 미분은 모두 유한한 음수였다. 이는 gradient 소실 문제의 수치 수정을 지지한다. 성능 향상이나 과거 실패 전체의 원인을 입증하지는 않는다.

고정한 lambda는 0.002429179111이며 clipping 경계에 걸리지 않았다. 처음 16개 F0 train batch의 survival 평균은 221.009143이지만 중앙값은 0.156678이다. 가장 큰 한 batch가 합의 70.6563%를 차지했다. 이 관측은 평균 크기 보정이 소수의 큰 꼬리 벌점에 민감했음을 보여준다. 학습 중 weighted survival/task 비율의 중앙값은 각각 약 0.008708%/0.008781%로 작았고, 평균 weighted survival은 약 0.2943이었다. gap floor가 적용된 간격 비율은 약 6.33%/6.37%였다.

따라서 추가 항이 보통 batch에서는 작고 일부 batch에서 매우 컸다는 제약이 관측된다. 이것이 작은 성능 차이의 인과적 원인이라고 확정할 수는 없다. 이번 예산에는 lambda 보정 방식 또는 꼬리 가정 ablation이 없었고, 결과를 보고 lambda를 변경하지 않았다. 단순히 clamp만 제거한 것과 동일한 실험도 아니다.

## 6. 실행 비용과 검증

실모델 preflight와 본 실행의 합은 **451.52초(약 7.53분)**였다. 본 실행에는 startup 대기, V 검사와 E_dev scoring이 포함되며 별도 CPU 검증/보고와 구현 시간은 포함하지 않는다. 본학습 fit 최대 allocated GPU는 674.53MiB였다. 각 fit의 wall/training/V/I/O/대기와 allocated/reserved/RAM은 [fits.json](fits.json)에 구분되어 있다. 자원 효율 우위는 주장하지 않는다.

원래 결과·자료·체크포인트·모델 파일 1,131개 해시가 보존됐다. V prediction 60개와 E prediction 13개를 독립 재계산했고, primary 최대 오차는 1.11e-16이었다. 선택된 6개 checkpoint의 GPU V reload 오차는 0이다. 본 실행 model forward 5,424회, model backward 2,160회, 출력 텐서만의 추가 backward 192회를 구분했다. CPU 회귀 검사 110개가 통과했다. [기본 검증](verification.json)과 [완료 교차 검산](completion_audit.json)에 범위를 기록했다.

RustDesk 272MiB 예외만 사용자 학습 우선 지시에 따라 허용했다. 실행 중 허용되지 않은 compute sample은 0, 최소 GPU free는 8121MiB였다. 다른 프로세스를 종료하지 않았다. 원시 데이터·모델·prediction/checkpoint 캐시는 로컬에 있고, GitHub에는 코드·프로토콜·표·해시·검증 기록을 보관한다.

## 7. 다음 결정

수치적으로 타당한 검열 대조군을 확보했지만, 이 고정 설정에서 NAIVE를 넘는 추가 가치는 확보하지 못했다. 작은 DROP 대비 이득은 보존하고, 이를 큰 학술 성과나 모든 검열 PEFT의 반증으로 확대하지 않는다. **이번 통제 파일럿은 여기서 종료하며 추가 fit, 새 어댑터 또는 다른 트랙을 자동으로 실행하지 않는다.** 실제 잠재 수요·신규성·외부 일반화와 과거 실패 전체의 원인은 별도 검토가 필요하다.

원 계획과 자동 생성 표/그림은 [REPORT.md](REPORT.md), [PROTOCOL.md](PROTOCOL.md)에 있다.
