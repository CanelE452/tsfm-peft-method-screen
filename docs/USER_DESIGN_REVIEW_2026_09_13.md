# 7개 TSFM PEFT 후보 — 실행·결과·방법 설계 검토

## 기준과 범위

- 저장소: `CanelE452/tsfm-peft-method-screen`
- 고정 검토 커밋: `1f7ac730bdff95108815261b0f23f35ba0876a20`
- 기존 연구를 덮어쓰거나 재학습하거나 GitHub를 수정하지 않았다.
- [확인] README, 7개 결과, 공통 학습·평가·데이터·LoRA 코드, 후보 모듈, 선택 기록, 05 복구 결과를 대조했다.
- [독립 계산] 공개된 집계 손실과 05의 원점별 손실을 읽어 6개 후보의 대비를 재계산했다. 보고 대비의 최대 절대차는 `8.881784197001252e-16 %F0`였다.
- [독립 계산] 후보 01의 동기식 마스크, 03의 위상 입력 지지집합, 07의 CDF 꼬리 gradient를 작은 CPU 예제로 검증했다.
- [한계] 원래 `.cache`의 모델·예측 NPZ를 내려받아 재검산한 것이 아니다. 원래 GPU 학습, CUDA 환경, 모든 테스트를 다시 실행하지 않았다. 이 문서의 독립 계산은 집계 산술 및 코드 수식 재현에 한정한다.
- 본 문서에서 [확인]은 저장소의 관찰 사실 또는 읽은 코드, [독립 계산]은 이번 별도 계산, [판정]은 검토 의견, [추정/미검증]은 아직 검증하지 않은 설명·수정안이다.

## 1. 최상위 판정

[판정] 바로 Round 2로 승격할 승자는 없다. 그러나 “7개 아이디어가 모두 과학적으로 틀렸음”은 잘못된 해석이다.

현재는 다음이 섞여 있다.

1. 제안한 추가 부품이 강한 단순 대조보다 유리하지 않았음: 01, 02, 03, 07.
2. 검증 선택이 모든 학습 경로를 학습 전 모델로 되돌림: 04.
3. 실제 온라인 비교가 성능·비용 모두 미통과: 복구된 05.
4. 선행 중복 위험으로 학습하지 않음: 06.

[판정] 기존 실패 결과는 보존한다. 7개를 전부 확장하지 않는다. 새로운 방법 개발은 후보 01의 문제 정의·조건부 계산을 한 번만 구체적으로 수정한 소규모 v2에 집중하는 것을 추천한다. 이는 Round 1 통과 선언이 아니라 별도 수정 후보다.

## 2. 실행 완료와 점수

[확인] 실제 실행은 표준 fit 34회와 누적 스트림 시도 9회(완료 8, 최초 중단 1)다. 05 복구는 승인 후 별도 폴더에서 5개 스트림을 다시 실행했다. Round 2는 실행하지 않았다. 기존 38-fit 상한에서 06의 4개 학습이 선행 중복 판단으로 빠져 34회가 됐다.

아래 차이는 모두 다음 단위다.

`gain = 100 * (L_baseline - L_proposed) / L_F0`

양수면 제안법이 해당 대조보다 좋다. 기준선 손실을 분모로 쓰는 일반 상대 개선율과는 다르다. 서로 다른 데이터·태스크의 수치를 바로 합산해 후보 순위를 만들면 안 된다.

| 후보 | 강한 대조 손실 | 제안 손실 | 대비 %F0 | 저장소 판정 |
|---|---:|---:|---:|---|
| 01 Freshness | 0.724028413 | 0.729543100 | -0.734523 | FAIL |
| 02 DualClock | 0.484154233 | 0.484085464 | +0.014029 | WEAK |
| 03 PatchPhase | 0.334411945 | 0.334463661 | -0.013717 | FAIL |
| 04 FR | 0.583857107 | 0.583857107 | 0 | FAIL: 모두 step 0 |
| 05 Maturity, 복구 | 0.630833796 | 0.655929350 | -4.298236 | FAIL |
| 06 Joint Path | 미측정 | 미측정 | 미측정 | NOVELTY_COLLISION |
| 07 Censor | 0.520622658 | 0.520791463 | -0.032158 | FAIL |

01은 unseen corruption 4개 평균, 03은 phase 4/12 평균, 05는 issued forecast 전체 평균이다. 각 열의 비교 기준은 해당 후보 내에서만 같다.

### 실행 무결성의 의미

[확인] 공통 기록의 step0 F0 identity, 선택 체크포인트 재로드, 지표 replay는 통과했다. 실행 코드에서는 같은 후보 안에 동일 샘플 순서·LR 후보·저장 시점이 적용되고 E 전에 선택이 봉인된다.

[판정] 이는 “저장과 계산이 일관적이다”는 근거이지 다음을 보장하지 않는다.

- 연구하려던 현상이 실제로 생성됐는가.
- 새 조건부 분기가 학습할 입력 정보를 받았는가.
- 새 모듈이 충분히 학습됐는가.
- 정칙화가 손실·gradient 수준에서 유효했는가.
- 소량의 차이가 독립 원천에 일반화되는가.

## 3. 후보 01 — 문제는 존재하지만 현재 gate의 추가 가치가 없다

### 3.1 확인된 결과

[확인] F0는 관측 손상에서 악화됐다. Round0의 block6는 +15.746%, block12는 +26.796%, refresh4는 +18.266%의 손실 증가였다. stale carry는 +95.370%였지만 이는 Round0만의 조건이며 본학습·최종 E의 stale 성능을 보여주는 값은 아니다.

[독립 계산] unseen corruption 평균:
- F0: 0.750784540
- Standard LoRA: 0.729510001
- Feature LoRA: 0.724028413
- Freshness-gated LoRA: 0.729543100

Feature의 F0 대비 이득은 3.563756%F0, Gated는 2.829233%F0다. Gated는 Standard와 거의 같으며 Standard 대비 -0.004409%F0다. Feature보다 -0.734523%F0, clean에서는 Feature보다 0.988742%F0 나쁘다.

### 3.2 중요한 설계 불일치: 비동기 관측을 시험하지 않았다

[확인] `freshness.py`는 다음처럼 만든다.

- block: 모든 채널의 마지막 n개 시점을 동시에 마스킹.
- refresh: 모든 채널에 같은 `t % n == 0` 갱신 규칙.
- 실제 채널별 상이한 갱신 간격/offset을 생성하지 않음.

[독립 계산] 원래 결측이 없는 4채널 입력에서 block6/block12/refresh2/refresh8의 서로 다른 채널 마스크 수는 모두 1이다.

[판정] 현재 결과는 동기식 관측 손상에 대한 결과다. “각 센서의 신선도가 서로 다를 때 조건부 적응이 유효한가”라는 원래 넓은 문제를 충분히 시험한 것은 아니다. 자연 결측이 있으면 일부 상태 차이는 남을 수 있지만, 인위적 손상 자체는 동기식이다.

### 3.3 더하기와 곱하기의 차이

[확인] 구현은 다음과 같다(공통 상수 배율 생략).

- Feature: `W0h + B[Ah + r(state)]`
- Gated: `W0h + B[g(state) * Ah]`, `g=2 sigmoid(r)`

[추정] 관측 상태가 특정 feature 방향에 대한 이동을 요구한다면, 스케일만 바꾸는 gate보다 additive 보정이 유리할 수 있다. 이 해석은 현재 결과와 양립하지만 아직 원인 증명은 아니다. 훈련된 gate 값·모듈별 gradient·상태를 제거한 영향은 공개되어 있지 않으므로 “gate가 학습되지 않았다”고 단정할 수 없다.

### 3.4 권고

[판정] 원래 gate를 Round2에 올리지 않는다. 이 문제에 한해 additive 강한 대조를 유지하고, 실제 채널별 관측 비동기와 조건부 affine 저랭크 보정을 검사하는 v2를 한 번만 개발할 가치가 있다.

## 4. 후보 02 — +0.014%를 승자로 고르면 안 된다

[확인] F0 0.490207190, Standard 0.484232330, Summary 0.484154233, Dual 0.484085464다.

[독립 계산] Dual의 F0 대비 이득은 약 1.249%F0지만, 이미 Standard가 대부분을 얻었다. 사건 순서를 넣은 추가 이득은 Summary 대비 0.014029%F0다.

[확인] 256계열 중 약 63.7%에서 작은 양의 차이가 있었지만, 가장 많이 개선된 10%를 제외하면 평균 이득이 사라졌다. RMSSE는 F0 1.0580에서 Dual 1.0649로 나빠졌다. 주지표와 보조 지표는 동일한 결론이 아니다.

[확인] 새 GRU 사건 인코더는 학습에 포함되어 있다. optimizer에서 빠진 것은 아니다. 하지만 Standard와 함께 전체 attention LoRA를 학습하고, 사건 어댑터는 마지막 미래 토큰의 표현에 붙는다.

[확인] 모든 선택 모델은 최대 LR 1e-4, 마지막 step 360이다.

[판정] 현재 관찰을 “충분히 수렴시켜도 사건 시간이 무용하다”로 일반화하면 안 된다. 동시에 단순히 더 오래 학습하면 해결된다고 할 근거도 없다. 상한에서 선택됐다는 것은 수렴 판단이 제한된다는 뜻이다.

[판정] 지금 즉시 Round2로 확장하지 않는다. 이 후보를 다시 시도한다면 사건 전용 branch가 실제 예측에 기여할 수 있게 학습됐는지와 단순 사건 모델 대조가 우선이며, 현재 tiny win을 논문 기여로 승격하지 않는다. 다음 후보 01 v2와 병렬로 확대하지 말고 보류한다.

## 5. 후보 03 — 학습하지 못한 위상 방향을 평가에 요구했다

### 5.1 구현과 결과

[확인] 원래 336개 값, 시간 인코딩, 관측 마스크를 보존하면서 masked padding으로 구획을 바꾼다. phase0은 21패치, nonzero phase는 22패치다. 따라서 같은 관측 정보이지만 패치 개수·부분 패치도 함께 달라진다.

[확인] Standard LoRA의 phase 증강만으로 unseen-phase F0 대비 약 11.29%F0 이득을 얻었다. 조건부 어댑터는 증강 어댑터 대비 -0.013717%F0, 위상 분산 추가 감소는 0.275888%로 목표 30%에 미달했다.

### 5.2 구체적인 입력 지지집합 문제

[확인] 훈련 phase는 {0,8}, 평가 phase는 {4,12}. gate의 입력은
`[sin(2*pi*s/16), cos(2*pi*s/16)]`이다.

이상적인 수학 값은 다음과 같다.

| phase | sin | cos | 역할 |
|---|---:|---:|---|
| 0 | 0 | 1 | 학습 |
| 8 | 0 | -1 | 학습 |
| 4 | 1 | 0 | 평가 |
| 12 | -1 | 0 | 평가 |

[독립 계산] 학습 행렬 [sin, cos, bias]의 rank는 2다. sine 계수는 훈련 지지집합에서 식별되지 않는다. 부동소수점 sin(pi)는 매우 작은 값이므로 코드에서는 완전 0이 아닐 수 있지만 유의미한 학습 신호가 되는 설계는 아니다.

CPU 예:
- 두 gate 가중치 [0, .2, .1]과 [10, .2, .1]
- 학습 logits는 둘 다 [.3, -.1]
- 평가 logits는 [.1,.1] 대 [10.1,-9.9]

[판정] 전체 Transformer가 절대 일반화할 수 없다는 뜻은 아니다. 그러나 추가한 sine/cos gate의 핵심 자유도를 훈련에서 규정하지 않은 채 unseen 일반화를 요구한 테스트다.

### 5.3 위치도 문제와 바로 대응하지 않는다

[확인] 어댑터는 입력 패치 이전이 아니라 `enc.last_hidden_state[:, -3:]` 이후, 즉 미래 토큰 출력 직전에 붙는다.

[판정] 이는 원래 패치 구획을 바로잡는 모듈이 아니라 이미 처리된 결과를 사후 보정하는 모듈이다. 늦은 보정이 불가능하다는 뜻은 아니지만, “패치 경계 처리 개선”이라는 설명에는 추가 근거가 필요하다.

### 5.4 권고

현재 결과를 보존하고 “현 구현·현 phase support 미통과”로 기록한다. 다음 시도는 full-rank phase support와 개입 위치를 먼저 고쳐야 한다. 단, 실제 배포에서 canonical re-alignment로 해결 가능한 인공 변형인지도 확인해야 한다. 새 GPU 확장 우선순위는 후보 01 뒤다.

이 문제는 이전 지시문에서 {0,8}/{4,12}를 먼저 정하고 표현 식별 가능성을 확인하지 않은 내 설계에도 책임이 있다.

## 6. 후보 04 — 모든 방법이 step0로 돌아간 비교

[확인] 8회 학습을 했지만 Standard/Raw Stability/Anchor/FR의 V 선택 모델이 모두 step0다. 모든 E 예측과 점수는 F0와 동일하다.

[판정] 실제 배포·선택 절차의 추가 이득은 0이므로 이번 파일럿 실패 판정은 적절하다. 그러나 “4개 학습식이 전혀 다르게 작동하지 않았다”거나 “regularizer가 학습에서 꺼져 있었다”는 뜻은 아니다.

[확인] 정칙화 계수 0.190773은 첫 train 배치의 raw-stability 항을 native task의 약1%로 맞춰 정했다. 이것은 초기값이 0인 Anchor/FR의 gradient 크기까지 맞춘 절차가 아니다. 모듈별 실제 regularizer/task gradient ratio는 저장되지 않았다.

[판정] 일부 학습식을 더 잘 조정할 여지는 있지만, 선행 forecast-stability와의 거리도 가깝다. 이 결과만 보고 FR λ 대규모 탐색을 시작하지 않는다. 현재 FR은 보류한다.

## 7. 후보 05 — 복구 후에도 실제 온라인 비교는 실패

### 7.1 복구와 무결성

[확인] 원래 실행의 NaN-gradient 오류를 수정한 뒤 동일 recipe로 5개 스트림을 다시 실행했다. 원래 완료된 F0/Immediate/WaitFull 예측은 정확히 재현됐다고 기록한다. 미도착 정답을 읽지 않고 issued forecast를 변경하지 않으며, preservation은 실제로 활성화됐다.

TAFAS_LIKE는 GCM만 재현하고 고정 주기와 native 확률 손실을 사용하는 제한 대조다. 완전한 공식 TAFAS 재현 결과라고 부르면 안 된다.

### 7.2 성능

| 방식 | 전체 손실 |
|---|---:|
| F0 | 0.583857107 |
| Immediate | 0.646770200 |
| WaitFull | 0.662482790 |
| TAFAS-like | 0.630833796 |
| Maturity | 0.655929350 |

Maturity는 TAFAS-like 대비 -4.298236%F0, F0 대비 -12.344158%F0다. Immediate보다도 나쁘다. TAFAS-like보다 adaptation 시간은 103.676% 크다. 이는 서로 다른 구조의 총 적응 시간 비교이며, preservation 연산 자체만의 추가 비용으로 읽으면 안 된다.

### 7.3 후반이 실패를 주도한다

[독립 계산] 공개 원점별 점수로 재구성:

| 구간 | F0 | Immediate | Maturity | TAFAS-like |
|---|---:|---:|---:|---:|
| 처음25 | 0.590883 | 0.590846 | 0.596079 | 0.625768 |
| 마지막5 | 0.548728 | 0.926390 | 0.955182 | 0.656163 |

[판정] 앞 구간에서는 TAFAS-like보다 좋지만, 후반 실패가 전체 결과를 뒤집었다. 마지막5를 제외해서 성공으로 바꿀 수 없다.

[코드 해석] 보호 기준이 최초 F0가 아니라 각 업데이트 직전 모델이다. 작은 변화마다 기준이 이동하므로, 한 번의 업데이트를 억제하는 것이 장기 누적 변화를 제한한다는 보장은 없다. 이것은 실패와 양립하는 설명이지 현재 로그로 확정한 원인은 아니다.

[판정] 현 Maturity 후보는 중단한다. 반복적인 온라인 recipe 조정으로 전환하지 않는다.

## 8. 후보 06 — 점수가 나쁜 것이 아니라 미실험

[확인] 0 fits. 핵심 아이디어와 가까운 prior는 frozen multi-step TSFM marginals에 context-conditioned neural copula를 붙여 joint paths를 생성한다. 기존 문헌에는 MLP/TCN/GRU로 상관 파라미터를 예측하는 경우도 포함된다.

[판정] 우리 제안의 TSFM hidden 활용과 full cross-channel/horizon low-rank covariance는 완전히 같은 수식은 아니다. 저장소도 이를 인정하며, 핵심 메커니즘 차별화가 약하다고 판단해 계산을 중단했다.

[판정] 이 종료는 비용을 고려한 연구 우선순위 결정으로 타당하다. “성능이 반증됐다”, “어떤 변형도 독창적일 수 없다”는 뜻이 아니다. 지금은 보류한다.

## 9. 후보 07 — 손실 설계의 꼬리 영역을 먼저 고쳐야 한다

### 9.1 결과

[확인] Censor-only 0.520622658, Censor-Preserve 0.520791463으로 보존 항이 추가 이득을 내지 못했다. 검열 위치 손실도 제안법이 더 나쁘고, bias의 아주 작은 개선만으로 성공이라고 부를 수 없다.

### 9.2 큰 벌점인데 gradient가 0인 경우

[확인] `censor.py`는 마지막 두 분위수 사이 간격 하나를 넘어선 곳에 CDF=1인 끝점을 만들고, survival을 `1e-6`으로 clamp한다.

[독립 계산] 분위수 값을 0..20으로 둔 작은 CPU 예에서:
- 검열 하한10: CDF0.5, loss0.6931, gradient L1=0.1
- 하한20.5: CDF0.995, loss5.2983, gradient L1=4
- 하한22: CDF1, loss13.8155, gradient L1=0

[판정] 실제 수요가 현재 분포보다 더 크다는 관측이 주어졌을 때, 해당 꼬리에서 gradient를 받지 못할 수 있다. 이는 함수 형태의 확인된 한계다. 실제 M5 train에서 그 영역에 얼마나 자주 들어갔는지는 원시 예측이 없어 확인하지 못했으므로 이번 실패의 주원인이라고 확정하지 않는다.

단조 quantile spline과 명시적 꼬리를 다루는 기존 분포 방법을 활용할 수 있다. 꼬리 함수를 고치는 것 자체를 새 PEFT 기여로 주장해서는 안 된다.

### 9.3 검증에는 검열 전 타깃을 사용했다

[확인] candidate07의 입력은 판매 상한으로 잘라지지만 공통 validation 함수의 y는 원래 저장된 판매다. 즉 controlled recensoring에서 가려지기 전 검증 정답으로 모델을 고른다.

[판정] 모든 arm에 같은 원래 정답을 썼으므로 이 통제 비교 자체를 바로 test leakage라고 부르는 것은 부정확하다. 하지만 실제로는 검열된 판매만 관측된다는 배포 시나리오의 모델 선택 절차까지 검증한 것은 아니다. Oracle validation 조건으로 명시해야 한다.

또 M5 원래 판매를 실제 잠재 수요 전체라고 보장할 수 없다. 이 실험은 인공 검열 이전의 관측 판매를 회복하는 대조다.

[판정] 현재 결과로 검열 적응의 가능성을 부정할 수는 없지만, 빠른 방법 개발 우선순위에서는 후순위다.

## 10. 공통 스크리닝 설계에서 바꿀 것

### 10.1 같은 update 수가 충분한 학습을 보장하지 않음
[확인] 신규 gate/GRU와 LoRA에 같은 작은 LR 후보와 360회 상한을 사용했다. 02/03은 모두 최대 LR·마지막 step을 선택했다.

[판정] 이는 고정 계산 예산의 실용적 screen으로는 의미가 있다. 그러나 서로 다른 신규 모듈을 충분히 학습한 대결의 증거는 아니다. 새 모듈을 오래 돌리면 반드시 성공한다는 뜻도 아니다.

### 10.2 실행 전 identity 외에 조건부 branch의 기능 테스트 필요
- 학습 대상 파라미터가 optimizer에 포함되는가.
- train-only 짧은 smoke 후 조건부 파라미터 gradient/update가 유한·비영인가.
- 상태가 변할 때 실제 보정 출력도 변하는가.
- 정칙화가 실제 task gradient 대비 어느 정도 비중인가.
이것은 GPU 대규모 분석 프로젝트가 아니라 새 방법 구현의 기능 테스트로 포함한다.

### 10.3 step0 선택은 보존하되 해석을 구분
[판정] 선택된 최종 시스템의 이득이0이면 배포 방법으로는 실패다. 그러나 학습된 내부 후보의 차이를 보려면 전체 V 학습 경로도 함께 읽어야 한다. E를 보고 좋은 checkpoint를 다시 골라서는 안 된다.

### 10.4 단일 seed의 1%F0는 탐색 중단 규칙
통계적 유의성이나 논문 채택 기준이 아니다. 후보2의0.014%가 양수라고 1위로 확정할 수 없고, 다른 데이터의0.7%와 정밀한 크기 비교도 할 수 없다.

### 10.5 새로 보인 후보라도 같은 E에서 계속 수정하면 개발 자료
스크리닝 E는 모두 향후 개발 근거다. 수정 v2를 평가하더라도 독립 증명이라고 부르지 않는다. 최종 구조/선택 절차는 새로운 원천·기간을 열기 전에 고정한다.

## 11. 다음 방법 개발은 후보01 한 개의 제한된 v2로

### 목표

[제안·미검증] 채널별 갱신 간격과 지연이 다른 조건에서, 관측 상태에 따른 저랭크의 이동과 크기 조절을 결합한다. 새 백본, 선택기, 동결 정책은 추가하지 않는다.

### 왜01인가

- 관측 손상에 따른 F0 손해가 확인됐다.
- 같은 관측 상태를 사용하는 additive Feature 대조가 Standard보다 좋았다.
- 원래 의도한 채널별 비동기 조건을 아직 직접 테스트하지 않았다.
- 수정이 작은 module/data change이며 기존 runner를 재사용할 수 있다.

이는 성공 확률을 측정한 결과가 아니라 비용과 남은 구체적 질문을 고려한 우선순위 판단이다.

### 실제 계산 후보

```text
r = 관측 마스크 / 경과시간 / 최근 가용률
r_clean = 모두 정상 관측된 참조 상태

a_delta(r) = a(r) - a(r_clean)
b_delta(r) = b(r) - b(r_clean)

output = W0 h + B[(1+a_delta(r))*A h + b_delta(r)]
```

[제안·미검증] 정상 관측 상태에서는 조건부 추가항이 사라져 standard LoRA 형태로 돌아간다. 이것은 F0 성능을 보장하지 않는다. 공유 LoRA A/B가 여전히 바뀌므로 정상 관측의 실제 성능을 따로 평가해야 한다.

조건부 affine modulation 자체는 FiLM 등 기존 연구의 원리다. 이 수식만으로 독창성을 주장할 수 없고, 시계열 관측 상태와 제약·효용의 구체성이 추가로 입증되어야 한다. 부족하면 방법 논문으로 승격하지 않는다.

### 비교와 상한

3개 arm:
1. 동일 증강 Standard LoRA
2. 동일 r을 사용하는 기존 additive Feature LoRA
3. 조건부 affine 제안 v2

공통:
- 원래 시간 정보와 관측 마스크를 모든 arm에 공정하게 제공.
- 채널마다 다른 갱신 간격·offset·block 시작점을 학습 전에 생성 규칙으로 고정.
- 미래 타깃은 손상시키지 않고 origin 이후 관측도 보지 않는다.
- 정상 관측을 train mixture에 포함.
- conditional module LR는 LoRA와 별도 group으로 명세할 수 있지만 feature와 proposed에 같은 탐색 기회를 제공.
- gate 출력 범위·gradient·update norm을 학습 로그의 작은 항목으로 추가.

12회 상한:
`3 arms × 2 predefined optimizer recipes × 2 seeds = 12 fits`

이는 사용자 합의 후 수행할 수정 개발안이며 이번 검토에서 실행하지 않았다.

### 판단

- strongest additive 대조 대비 양의 효과가 두 seed에서 유지되는가.
- 기존 운영 기준으로 평균 1%F0 이상 이득과 clean 손해0.5%F0 이하를 목표로 한다.
- 단순 Feature로 같은 이득이 나오면 추가 조건부 계산의 이유가 없다.
- 고정된 v2가 실패하면 이 분기를 멈춘다. LR/threshold를 끝없이 늘리지 않는다.
- 통과했을 때만 규칙을 고정하고 새 원천으로 넘어간다. 이번 E 재사용을 final holdout이라고 부르지 않는다.

## 12. 후보별 후속 최종 분류

| 후보 | 후속 |
|---|---|
|01|원래 gate는 종료, 명시적 비동기 조건과 additive+scale v2만 한 번 제안|
|02|보류. tiny win으로 Round2 승격하지 않음|
|03|설계 수정 필요. 위상 지지집합·개입 위치·실사용 필요성 해결 전 확대 금지|
|04|현 FR 보류. step0와 weak regularizer 비교를 성공으로 포장하지 않음|
|05|현 온라인 보호 방식 종료|
|06|핵심 선행 중복으로 보류; 미실험임을 유지|
|07|CDF·검증 관측 계약 수정 전 확대 금지|

## 13. 근거 파일

모든 경로는 위 고정 커밋 기준이다.

- `README.md`
- `results/screening_summary/latest_review.md`
- `results/screening_summary/common_integrity.json`
- `results/screening_summary/compute_summary.json`
- `docs/MASTER_SCREEN_PROTOCOL.md`
- `docs/NOVELTY_BOUNDARY.md`
- `results/candidate_01/RESULT.md`, `selections.csv`
- `results/candidate_02/RESULT.md`, `selections.csv`
- `results/candidate_03/RESULT.md`, `selections.csv`
- `results/candidate_04/RESULT.md`, `contract.json`
- `results/candidate_05_repaired/RESULT.md`, `origin_losses.json`
- `results/candidate_06/RESULT.md`
- `results/candidate_07/RESULT.md`
- `src/tsfm_peft_screen/backbone.py`, `lora.py`, `data.py`, `metrics.py`
- `src/tsfm_peft_screen/runners/common_fit.py`, `common_eval.py`, `streaming_eval.py`
- `src/tsfm_peft_screen/candidates/freshness.py`, `dualclock.py`, `patchphase.py`, `fr_lora.py`, `maturity.py`, `censor.py`

## 14. 외부 선행 대조

여기의 외부 지식은 저장소가 직접 입증한 사실과 분리한다.

- FiLM: Visual Reasoning with a General Conditioning Layer (2018/AAAI).
  조건부 feature affine modulation 원리는 기존 것이다.
  `https://ojs.aaai.org/index.php/AAAI/article/view/11671`
- LoRA+: Efficient Low Rank Adaptation of Large Models (2024/ICML).
  LoRA A/B의 최적화에 같은 LR가 항상 적절하지 않음을 분석한다.
  이 논문이 현재 새 gate/GRU의 실패 원인을 입증해 주는 것은 아니다.
  `https://proceedings.mlr.press/v235/hayou24a.html`
- Learning Quantile Functions without Quantile Crossing for Distribution-free Time Series Forecasting (2022/AISTATS).
  단조 quantile function과 inter/extrapolation을 위한 기존 접근.
  `https://www.amazon.science/publications/learning-quantile-functions-without-quantile-crossing-for-distribution-free-time-series-forecasting`
- Efficiently Generating Correlated Sample Paths from Multi-step Time Series Foundation Models (2025/arXiv 본문으로 확인).
  frozen marginal forecasts + conditional neural copula의 가까운 선행.
  `https://arxiv.org/html/2510.02224v1`

## 15. 재현 번들

- `six_candidate_gain_recalculation.csv`: 집계 수치 기준 대비 재계산.
- `candidate05_temporal_decomposition.csv`: 처음25/마지막5 분해.
- `reproduce_structural_checks.py`: 01 마스크 동기성, 03 phase gate 식별성, 07 CDF gradient의 CPU 테스트.
- `reproduced_checks.json`: 이번 테스트 출력.
- `arithmetic_audit.json`: 집계 산술 대조의 범위와 오차.

필요 패키지: Python 3.10+, numpy, torch. GPU/모델/개인 캐시는 필요 없다.

```bash
python reproduce_structural_checks.py --output local_checks.json
```

이 번들은 전체 TSFM 학습 재현 패키지가 아니다. 실제 원시 예측에서 07의 zero-gradient 영역 빈도나 01 gate 활성값을 확인하려면 로컬 `.cache` 자료가 추가로 필요하다.