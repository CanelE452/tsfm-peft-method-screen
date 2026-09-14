품절·검열 관측 PEFT: 꼬리 손실 수정 및 최대 6-fit 통제 파일럿
작성: 2026-09-14
검토 저장소: CanelE452/tsfm-peft-method-screen
검토 기준 commit: e07ae8cd724fd8877734d730ec8cf3649c2d7591

이 파일 전체를 CLI에 전달한다.
CLI는 사용자가 이 지시문의 실행을 요청한 경우 아래 계획의 구현·검증·제한된 학습·보고를 수행한다.
지시문을 검토하거나 보관하라는 요청만 받은 경우에는 학습을 시작하지 않는다.
새 어댑터 발명, 후속 학습 자동 추가, 다른 연구 트랙 재개는 범위 밖이다.

표기:
[확인] = 아래 고정 commit의 코드/문서 또는 명시한 원문으로 확인한 내용.
[제안] = 이번에 고정할 실행 설계. 이미 존재하거나 실행된 결과가 아니다.
[가설·미검증] = 결과가 아직 없는 예측·해석.
[한계] = 이번 실험으로 주장할 수 없는 범위.
문서에 없는 현재 파일·변수·데이터 컬럼은 존재한다고 가정하지 않는다.

======================================================================
0. 먼저 읽을 결론: 무엇을 할 것인가
======================================================================

[제안] M5의 기존 256개 계열과 인위적 판매 상한을 사용하는 통제 환경에서,
동일한 Chronos-2/LoRA를 다음 3가지 감독 방식으로 각각 2개 seed만 학습한다.

A. NAIVE: 관측된 판매량을 그대로 정답으로 학습한다.
B. DROP: 검열된 목표 위치는 제외하고 나머지만 학습한다.
C. TAIL: DROP의 동일한 손실에, 수요가 관측 하한을 넘는 사건의 벌점을 추가한다.
         단, 기존의 유한 지지범위/확률 clipping 때문에 gradient가 사라지지
         않는 꼬리 완성을 사용한다. 보존항(anchor)은 넣지 않는다.

F0: 아무 적응도 하지 않은 모델. 동일하게 상한이 적용된 과거를 입력한다.
    예측만 수행하며 별도 fit으로 세지 않는다.

최대 본학습: 3 arms × 2 seeds × 1 learning rate = 6 fits.
fit당 360 optimizer steps, 최대 2,160 본학습 updates.
별도 실제 모델 smoke: 3 arms × 2 updates = 최대 6 updates, 학습 상태는 폐기한다.
출력 텐서만 사용하는 CPU autograd 검사는 fit이나 모델 학습으로 세지 않는다.

핵심 비교: TAIL 대 DROP. 부차 비교: TAIL 대 NAIVE와 F0.
“검열 정보를 버리는 것보다 하한 정보를 사용하는 편이 유리한가?”에 답한다.
TAIL이 좋아도 새 PEFT 구조나 논문의 신규성이 확보됐다고 부르지 않는다.

======================================================================
1. 목적과 예상 결과를 먼저 고정한다
======================================================================

최상위 연구 방향:
  검열된 판매 관측으로 적응하는 시계열 모델에서 기존 접근의 한계를 찾고,
  그 한계가 남을 경우에만 새로운 시계열 특화 PEFT를 연구한다.

이번 실행의 직접 목표:
  수치적으로 타당한 검열 대조군을 확보하고, 동일 예산의 단순 제외보다
  추가 정보 활용의 이득이 있는지 판단하여 다음 연구 투자를 결정한다.

소비처:
  사용자가 “이 방향을 계속할지, 단순 수정으로 종료할지” 판단하는 보고서.
  이번 보고서를 방법론 논문의 최종 성능표로 사용하지 않는다.

행동 → 필요한 이유 → 설계 조건:
  1) 꼬리 손실 수정
     - 손실은 큰데 학습 신호가 0인 수치 문제를 제거한다.
     - 약하거나 고장 난 대조군을 상대로 새 방법의 우위를 주장하지 않는다.
     조건: 정상 확률분포, 안정적인 log-survival, 실제 autograd 검사.
  2) NAIVE/DROP/TAIL 비교
     - 검열을 무시한 감독과 검열 정보를 버린 감독을 구분한다.
     - 동일 모델에서 하한 정보 사용의 추가 효과를 본다.
     조건: 같은 입력·LoRA·초기화·샘플링·step·선택 기회.
  3) 전체 오차 및 검열 위치 오차를 함께 평가
     - 숨긴 값에 대한 예측 회복을 측정한다.
     - 모든 예측을 무작정 높여 검열 위치만 개선하는 편법을 잡는다.
     조건: 보관된 원판매량으로 평가, 전체와 두 하위집단을 모두 보고.

예상 결과 / 목적 지지 / 연구 방향과의 연결:
  - CPU에서 기존 upper-tail gradient=0, 수정식 gradient가 타당하게 유지됨:
    구현 목적 지지. 성능 향상·방법 신규성은 아직 지지하지 않는다.
  - TAIL이 DROP과 NAIVE보다 전체 미래 오차를 줄임:
    제한된 통제 환경에서 하한 정보를 쓰는 가치 지지. 다음 연구 검토 가능.
  - 검열 위치만 좋아지고 전체가 나빠짐:
    일부 효과만 지지. 전체 예측 개선 주장은 불가; 과대예측 위험 보고.
  - TAIL이 DROP과 비슷하거나 나쁨:
    이 설정/예산/꼬리 가정의 추가 가치 미확보. 같은 실험 자동 확장 금지.
  - 세 학습 방식 모두 F0보다 나쁨:
    해당 조건의 적응이 불리했다고 보고. 모든 PEFT의 실패로 일반화하지 않는다.

가지치기:
  - CENSOR_PRESERVE의 anchor, 새 분포 head, 새 rank 정책은 이번에 넣지 않는다.
    이유: 꼬리 수정·하한 정보 활용과 별개 효과를 섞지 않기 위해서다.
  - 실제 품절 수요 복원, 실재고 이익, 타 데이터셋 일반화는 이번에 주장하지 않는다.

======================================================================
2. 저장소에서 실제로 확인한 내용과 읽어야 할 파일
======================================================================

[확인] 기존 Candidate07은 실제 품절 레이블이 아니라 M5 원판매량에 인위적인
상한을 적용한다. 원판매량도 실제 잠재 수요가 검열 없이 관측된 값이라는
보장은 없다. 따라서 이번 대상은 “추가로 인위적으로 가린 판매량의 미래 예측”이다.

[확인] 기존 CDF는 예측 분위수 양 끝에서 인접 간격 하나만큼 연장한 뒤
확률을 0/1로 제한하고, survival을 1e-6 아래로 제한한다.
두 선택 상태의 기존 진단에서 검열 관측의 약 2.73~2.79%가 큰 벌점을
받으면서 gradient=0이었다. 해당 위치는 검열 손실 항의 약 18.25~18.55%다.
이 숫자는 전체 손실 비율도, 과거 실패 원인의 기여율도 아니다.

실행 전에 다음 파일을 현재 checkout에서 읽고 기준 commit과 차이를 확인한다.

  docs/RESULTS_INDEX.md
  docs/CANDIDATE_07.md
  research/reopen_review_20260914/CENSOR_TAIL_DIAGNOSTIC.md
  results/reopen_censor_diagnostic_20260914/summary.json
  src/tsfm_peft_screen/candidates/censor.py
  src/tsfm_peft_screen/runners/common_fit.py
  src/tsfm_peft_screen/data.py
  src/tsfm_peft_screen/backbone.py
  src/tsfm_peft_screen/lora.py
  src/tsfm_peft_screen/metrics.py
  scripts/run_reopen_fr_censor.py

로컬에서 존재와 내용을 추가 확인할 기존 자료:
  results/candidate_07/sampling_manifest.json
  results/candidate_07/contract.json
  results/candidate_07/selection.json
  data/processed/m5/manifest.json
  data/processed/m5/fit.npz
  data/processed/m5/evaluation.npz

이 경로들은 검토 코드에서 참조됨을 확인했지만, 실행 컴퓨터에서 존재하는지와
현재 해시까지 이 지시문 작성자가 확인한 것은 아니다.
없는 파일·키를 상상해서 대체하지 말고 필요한 자료와 영향 범위를 보고한다.

원격 최신 내용이 바뀌었다면 현재 commit과 본 계획의 의미상 차이를 먼저 기록한다.
이미 같은 6-fit 연구가 완료됐다면 재학습하지 말고 기존 결과를 검증한다.
과거 branch로 강제 reset하거나 작업 중 변경을 지우지 않는다.

======================================================================
3. 선행 근거와 이번 구현의 정확한 지위
======================================================================

[확인] Learning Quantile Functions without Quantile Crossing for
Distribution-free Time Series Forecasting(2022/AISTATS)은 분위함수의
보간·외삽을 다룬다. GluonTS 공식 I(S)QF 문서는 지수형 꼬리를 명시한다.

[제안] 이번에는 그 계열의 “분위수 사이 보간 + 무한 지지범위 꼬리” 아이디어를
참조하되, Chronos의 21개 출력 분위수에 아래의 고정된 완성 규칙을 붙인다.
ISQF 전체 모델·학습법 재현이라고 부르지 않는다. 새 어댑터도 아니다.

[중요한 한계]
  - 유한한 21개 분위수만으로 극단 꼬리 분포가 유일하게 정해지지 않는다.
    지수 꼬리를 택하는 것은 모델링 가정이며, clamp 제거만 하는 사소한
    산술 수정과 동일하지 않다.
  - 이번 목적함수는 비검열 native pinball + 검열 survival 벌점의 혼합이다.
    전체 관측의 엄밀한 censored maximum likelihood, 검열 보정의 일관성,
    또는 실제 잠재 수요의 식별 가능성이 증명됐다고 쓰지 않는다.
  - 고정 상한 위에서는 초과 사건만 관측하므로, 초과 크기의 분포는 이 정보만으로
    비모수적으로 식별되지 않는다. 사전학습과 꼬리 가정에 의존한다.
  - 결과가 좋아도 “기존 꼬리 구현만이 과거 실패의 원인이었다”고 결론 내리지 않는다.
    과거 buggy arm의 같은 조건 재학습은 이번 6 fits에 포함하지 않기 때문이다.

선행과 코드 출처는 마지막 참고 자료에 고정 URL로 기록했다.

======================================================================
4. 데이터 계약: 기존 통제 조건을 사용하되 숨긴 값의 유입을 막는다
======================================================================

[확인] 기존 M5 선택:
  원자료 d_1..d_1941, 일 단위.
  eligibility는 days337..1400에서 zero_fraction>=0.5, positive days>=40,
  nonzero standard deviation. id의 SHA256 순서에서 256개를 선택한다.

[제안] 이번에는 기존 256개 id와 그 순서를 그대로 사용한다.
성능을 보고 새 id를 고르거나, 다른 데이터셋으로 교체하지 않는다.

[확인] 기존의 시간 인덱스는 0-based다.
  L=336: 과거 336개 일별 관측.
  H=48: 미래 48일 예측.
  학습 구간: [336, 1400).
  학습 origin: range(336, 1400-47, 24).
  V origin: [1400, 1450, 1500, 1550, 1600].
  평가 origin: [1650, 1698, 1746, 1794, 1842, 1890].

원자료 일 번호와 배열 인덱스를 혼동하지 않는다.
manifest와 위 값이 다르면 이유를 확인하고 조용히 맞춰 쓰지 않는다.
각 origin에서 context=[o-336,o), target=[o,o+48)임을 검사한다.

[제안] 각 계열 i의 상한은 기존 규칙 그대로다.
  c_i = max(1, positive original train sales의 60th percentile)
  observed_sale = min(original_sale, c_i)
  is_censored = original_sale > c_i
  original_sale == c_i는 기존과 동일하게 비검열이다.

상한 생성은 “실험용 관측 생성기”가 원판매량의 학습 부분만 이용해 수행한다.
이 상한을 실제 재고의 추정치라고 부르지 않는다. 미래 원판매량으로 상한을 정하지 않는다.
검열 여부도 생성기가 제공하는 정확한 사건 비트이며, 실제 품절 탐지 실험이 아니다.
미래 검열 비트는 loss/evaluation에만 사용하고 예측 입력으로 주지 않는다.

학습 프로세스에 허용되는 것:
  상한이 적용된 context, 관측 판매량, 검열 비트, 고정 상한,
  관측된 학습 판매량에서만 계산한 정규화 정보.
학습 프로세스에 금지되는 것:
  검열 위치의 원판매량, 원판매량에서 계산한 숨은 크기 특징,
  원판매량을 다른 경로로 읽는 캐시 또는 우회 target.

[제안: 기존 코드와 다른 점]
  기존 Panel.scale은 상한 적용 전 학습 판매량에서 계산된다.
  이번에는 학습 보조 계산 및 주 평가 스케일을
    s_obs_i = std(observed_sale_i[336:1400], ddof=0, float64)
  로 새로 만든다. 모든 값은 유한하고 >0이어야 한다.
  원래 Panel.scale/caps 또는 기존 데이터 파일은 수정하지 않는다.
  zero scale이 있으면 제외하지 말고 데이터 계약 문제로 보고한다.
  이번 지표를 과거 uncapped-scale 수치와 같은 수치처럼 직접 비교하지 않는다.

Chronos 내부 loc/scale은 동일한 capped context에서 계산한다.
Censor의 과거 context에서 censored 위치를 NaN으로 바꾸지 않는다.
DROP은 “학습 target 위치를 제외”하는 arm이지 입력 문맥까지 지우는 arm이 아니다.

V와 평가의 원판매량:
  별도 평가 경로에 보관한다. V 원판매량은 아래처럼 체크포인트 선택에 허용한다.
  이것은 모든 arm에 동일하게 주어진 clean-validation 권한이다.
  실제 품절 데이터만 있고 깨끗한 V가 없는 배포 절차를 검증하는 실험이 아니다.
  깨끗한 V 없이 선택하는 정책은 이번에 새로 만들지 않는다.

노출 경계:
  이 id와 V/평가 구간은 이미 이전 연구에서 보았다.
  이번에 학습 전 선택을 봉인해도 연구자 관점의 새로운 독립 test가 되지 않는다.
  보고서에서는 평가를 E_dev(재사용 개발 평가 구간)로 표시한다.
  기존 자료를 덜 봤다고 주장하거나 E_dev를 최종 외부 검증으로 재분류하지 않는다.

======================================================================
5. 공통 모델·학습 예산
======================================================================

[확인] 기준 구현:
  model: amazon/chronos-2
  model revision: 29ec3766d36d6f73f0696f85560a422f50e8498c
  standard rank-8 LoRA, 96개 attention q/k/v/o projection.
  LoRA 출력 스케일 2, 원래 backbone와 output head 고정.
  trainable tensors=192, trainable parameters=1,179,648.
  추가 학습 head와 조건부 네트워크 없음.
  dropout=0, 모델/파라미터 FP32.

[제안]
  seeds: 41000, 41001. 이는 이번 설계값이며 과거 실행값이 아니다.
  learning rate: 1e-4 하나만 사용. 이번 결과를 본 뒤 변경하지 않는다.
  optimizer: AdamW, betas=(0.9,0.999), eps=1e-8, weight_decay=0.
  gradient clipping: 전체 학습 파라미터 L2 norm 1.0.
  본학습: 각 fit 360 steps. 조기 중단해 fit 예산을 추가하는 방식 금지.
  effective batch: 8개 series-origin 쌍.
  각 쌍은 독립 group으로 처리해 평가 batch에서 다른 상품이 정보를 주지 않게 한다.
  mixed precision은 이번에는 사용하지 않는다. 기존 Candidate07과 동일 FP32 경로.
  작은 꼬리 확률 연산만 FP64를 사용하고 그 비용도 측정한다.

sampling:
  기존 results/candidate_07/sampling_manifest.json의 360개 배치를 고정해 사용한다.
  manifest의 origins/channels/variant를 실제로 읽고 유효성을 검사한다.
  두 seed의 차이는 초기화에 한정하고, 6개 fit 모두 같은 배치 순서를 사용한다.
  같은 seed에서는 세 arm의 초기 LoRA state가 byte-identical이어야 한다.

[한계] 동일 trainable parameters와 updates는 동일 wall-clock/메모리를 뜻하지 않는다.
TAIL은 추가 확률 연산이 있으므로 더 느릴 수 있다. 효율 우위를 주장하지 않는다.
1개 학습률·짧은 예산의 제한된 파일럿이며 각 방법의 최적 성능을 증명하지 않는다.

======================================================================
6. 세 학습 손실과 공통 정규화
======================================================================

기호:
  z: Chronos native 공간의 21개 분위 예측.
  q_raw: 원 단위의 21개 분위 예측.
  s: 관측된 판매량. c: 관측 하한(검열 위치에서는 s와 같음).
  m: 검열 비트. 1이면 원판매량 > 상한.
  N: 현재 batch의 전체 유효 series×horizon 위치 수.

A. NAIVE
  기존 native_loss(z, s, loc, scale).

B. DROP
  검열 위치를 NaN으로 바꾼 s를 동일 native_loss에 넣는다.
  현재 native_loss의 reduction, 즉 horizon mean → quantile sum → series mean을 유지한다.
  검열 위치를 제외했다고 남은 위치 수로 새롭게 재정규화하지 않는다.

C. TAIL
  L_TAIL = L_DROP + lambda_c * L_survival
  L_survival = sum_{m=1}[-log S_q(c)] / N.

TAIL과 DROP의 비검열 task 항은 값과 gradient가 정확히 같아야 한다.
추가 anchor, 검열 비트 입력 feature, 재표본화, loss별 별도 optimizer는 금지한다.

[제안] lambda_c는 학습 전에 F0와 동일한 첫 16개 training minibatch로 한 번 정한다.
  T0 = 첫 16개 batch의 DROP task 평균.
  C0 = 첫 16개 batch의 수정된 survival 항 평균.
  lambda_c = clip(0.1 * T0 / C0, 0.001, 100).
  T0>0, C0>0, 모두 유한인지 검사한다.
  두 seed/모든 steps에서 동일한 lambda를 쓴다.
  C0=0이면 임의의 작은 분모로 대신하지 말고 구성 문제로 중단한다.
  clipping 경계에 걸렸다면 기록하되 V/E를 보고 다시 조절하지 않는다.

이는 기존 첫 1개 batch 기준 보정과 다르다. 첫 16개를 쓰는 이유는 한 batch의
검열 구성에만 상대 손실 크기가 좌우되는 것을 줄이기 위한 고정된 파일럿 결정이다.
초기 손실 크기를 약 10%로 맞춘다는 뜻이지 gradient 크기가 같다는 뜻은 아니다.

======================================================================
7. 수정 log-survival: 수식을 하나로 고정한다
======================================================================

[제안·성능 미검증] exponential-tail completion을 사용한다.
이는 다음 수치 사양을 갖는 분포 완성이다. 원 논문 전체의 구현 재현은 아니다.

각 sample/horizon의 raw 분위 예측을 오름차순 정렬하여 p_1..p_K를 얻는다.
분위 수준 tau_1..tau_K는 backbone.QUANTILES를 읽고 0<tau_1<...<tau_K<1을 검사한다.

중복 knot를 수치적으로 처리:
  eps_i = 1e-4 * max(1, s_obs_i)
  x_1 = p_1
  x_j = x_{j-1} + max(p_j - p_{j-1}, eps_i)
이 미세한 엄격 단조화는 survival 계산의 proxy knots에만 적용한다.
원래 native task 출력이나 저장·평가 예측을 변경하지 않는다.
중복 처리 빈도와 최대 knot 이동을 기록한다.

중간 구간 x_j <= y < x_{j+1}:
  F(y) = tau_j + (tau_{j+1}-tau_j)*(y-x_j)/(x_{j+1}-x_j)
  log S(y) = log1p(-F(y)).

오른쪽 꼬리:
  b_R = (x_K-x_{K-1})*(1-tau_K)/(tau_K-tau_{K-1}) > 0
  y >= x_K 이면
  log S(y) = log(1-tau_K) - (y-x_K)/b_R.

왼쪽 꼬리:
  b_L = (x_2-x_1)*tau_1/(tau_2-tau_1) > 0
  y <= x_1 이면
  log F(y) = log(tau_1) + (y-x_1)/b_L
  log S(y) = log1p(-exp(log F(y))).

b_L/b_R는 경계에서 중간 CDF 기울기와 꼬리의 기울기가 연결되게 한 고정 규칙이다.
새 학습 파라미터는 없으며, knot와 tail scale의 gradient를 임의로 detach하지 않는다.

구현 주의:
  - 확률 S를 먼저 계산해서 작은 값으로 clamp한 뒤 log를 취하지 않는다.
  - log S를 직접 계산한다. 아주 작은 S가 화면상 0으로 반올림돼도
    log S의 gradient가 정상일 수 있으니 확률 표시값만으로 실패를 판단하지 않는다.
  - torch.where의 비활성 branch에서 overflow/NaN이 생기지 않도록 안전하게 구현한다.
    참조 구현처럼 active mask별로 해당 수식만 평가하는 것이 가능하다.
  - 비검열·missing 위치에서 쓸모없는 log 계산을 해 NaN×0을 만들지 않는다.
  - 검열 위치가 없는 batch의 추가 손실은 정확한 0으로 반환한다.
  - 강한 upper-tail 위반에서 모든 분위수를 함께 올리는 방향의 -log S 미분은
    음수여야 한다. 모든 개별 knot의 미분이 음수여야 한다고 요구하지 않는다.
  - 비선택 분위수·비검열 위치·이미 만족한 하한에서 gradient=0은 정상일 수 있다.
  - 이는 연속 확률 proxy다. M5의 이산 count를 위한 정확한 count likelihood가 아니다.
    이번에 0.5 보정, 정수 반올림, 음수 clipping 등을 추가하지 않는다.

======================================================================
8. 학습 전 검사: 구현 합격과 예측 성공을 분리한다
======================================================================

CPU synthetic tests — 모두 통과해야 본학습 가능:
  a) knot에서 F(x_j)=tau_j, CDF 단조성, 확률 범위, 양끝 극한.
  b) 구간 연결의 연속성과 log S의 유한성.
  c) upper-tail threshold를 멀리 보내도 loss가 계속 증가하고,
     전체 knot를 이동시키는 미분이 사라지지 않음.
  d) 기존 cdf+clipped-survival은 같은 강한 위반 예제에서 gradient=0을 재현.
  e) 중복·교차 knot, 없는 검열, 전부 검열, missing 입력의 처리를 각각 테스트.
  f) 경계/동률이 아닌 점에서 torch.autograd.gradcheck.
  g) scalar reference와 vectorized 결과 비교.
  h) lambda=0이면 TAIL과 DROP의 loss/gradient/한 번 update가 일치.
  i) 검열이 없으면 NAIVE/DROP/TAIL의 task 및 update가 일치.

수치 허용치 [제안]:
  FP64 scalar vs vector loss: abs <= 1e-10 + 1e-8*abs(reference).
  gradcheck: eps=1e-6, atol=1e-5, rtol=1e-4.
  같은 graph인 lambda=0 비교는 가능하면 exact, FP32 비교 허용치는
  rtol=1e-5, atol=1e-7로 먼저 고정하며 실패 후 몰래 완화하지 않는다.

모델 smoke — 최대 6 optimizer updates, 이후 초기화:
  동일 입력에서 F0와 초기 LoRA 예측 일치. 기존 FP32 경로 허용 오차 <=1e-6.
  실제 intended parameters에서 유한한 gradient와 update 확인.
  frozen weights/output head 변경 0 확인.
  trainable parameter count가 세 arm에서 완전히 같은지 확인.
  validation/reload 경로 동작과 금지된 원판매량 target 유입 검사.

[중요] LoRA는 B=0 초기화 때문에 첫 backward에서 A-gradient가 0일 수 있다.
모든 A/B tensor가 첫 step부터 nonzero여야 한다는 잘못된 검사로 중단하지 않는다.

자료 누출 검사:
  준비 이후에는 학습 worker에 숨은 원판매량 배열을 전달하지 않는다.
  상한과 검열 비트를 고정한 상태에서 숨은 초과 크기만 바꿔도 학습 입력,
  초기 lambda, loss와 gradient가 같아야 한다. V/E scoring은 이 검사와 분리한다.
  “숨은 값 변경 후 상한을 재계산”하면 다른 시나리오가 되므로 그렇게 검사하지 않는다.

이미 알려진 720개 historical batch 진단을 전부 자동 재실행하지 않는다.
필요하면 기존 결과를 읽고, 수치 문제의 재현은 CPU synthetic test로 먼저 확인한다.
원래 실제 checkpoint에서의 추가 gradient 재생은 최대 두 상태×16개 고정 train batch,
optimizer update 0이며 선택적 검산으로 별도 계수한다. 캐시 부재는 정직하게 보고한다.

======================================================================
9. 체크포인트 선택과 E_dev 접근
======================================================================

[제안] 모든 fit의 V 검사 step:
  [0, 4, 8, 15, 30, 60, 120, 180, 240, 360].

V primary:
  동일한 capped context → 원 단위 분위 예측 → 보관된 원판매량과
  equal-series scaled 2-pinball을 계산한다. 스케일은 s_obs다.

가장 작은 V primary를 선택한다. 정확한 동률은 더 이른 step을 선택한다.
step0도 후보이며, 선택되면 최종 예측은 F0와 같음을 그대로 기록한다.
학습을 안 했다는 뜻이 아니라 “학습 후 모델이 선택되지 않았다”는 뜻이다.

동일한 clean V truth 권한을 세 arm 모두 갖는다. 이것은 통제 환경의
checkpoint 선택이며, 실제 품절 환경의 deployable selector라고 주장하지 않는다.

6 fits의 예정 학습·V 선택이 모두 끝난 후:
  1) 여섯 선택 checkpoint와 실제 선택 step, 설정·데이터·소스 hash 봉인.
  2) disk reload 후 선택 당시 V prediction/primary 재현.
  3) 그 후 E_dev 원판매량을 scoring 경로에서 연다.

Primary는 V-selected model의 E_dev 점수다.
최종 step360 모델의 E_dev 점수도 사전에 정한 secondary로 함께 기록한다.
둘 중 결과가 더 좋은 쪽을 사후 primary로 바꾸지 않는다.
E_dev의 모든 중간 checkpoint를 훑어 가장 좋은 모델을 고르지 않는다.

======================================================================
10. 평가: 정확히 무엇에 대한 오차인가
======================================================================

[제안] 정답 y는 인위적으로 상한을 적용하기 전 M5 원판매량이다.
모든 방법은 인위적으로 제한된 과거만 입력받고 미래 H=48의 y를 예측한다.
이미 본 미래 값을 복원하는 imputation 실험과 혼동하지 않는다.

주 지표:
  E_m = (1/I) * sum_i { mean_{origin,h,tau}[2*rho_tau(y-q_m)] / s_obs_i }
  rho_tau(u)=max(tau*u,(tau-1)*u).
  계열을 같은 비중으로 평균한다. 분위수는 기존 metrics.score처럼 정렬한다.
  낮을수록 좋다. CRPS/WQL/M5 competition RMSSE라고 임의로 바꿔 부르지 않는다.

상대 개선율:
  Gain(TAIL vs B) = 100*(E_B-E_TAIL)/E_B, B=DROP, NAIVE, F0.
  양수는 TAIL 우세, 음수는 TAIL 열세.
  분모 E_B=0이면 비율은 undefined로 표시하고 절대 차이를 보고한다.

필수 산술 테스트:
  E_B=0.5,E_TAIL=0.5 → 0%.
  E_B=0.5,E_TAIL=0.45 → +10%.
  E_B=0.5,E_TAIL=0.55 → -10%.
  100*E_TAIL/E_B를 개선율로 출력하지 않는다.

F0 정규화 비율을 추가하면 “E_model/E_F0”라고 명시한다.
F0가 1이 되는 이유는 정규화이지 실제 예측 오차가 1이어서가 아니다.
원 primary와 비율과 %개선율은 서로 다른 열에 둔다.

필수 결과:
  - seed별/seed평균 전체 primary: F0, NAIVE, DROP, TAIL.
  - y>c인 synthetic-censored 위치와 y<=c인 나머지 위치 primary 각각.
  - 두 집단의 전체 오차 기여도: 전체 위치 수를 분모로 유지해 합이 전체가 되게 함.
  - median forecast의 MAE와 signed bias, 특히 검열 위치의 과소/과대예측.
  - 80% interval coverage 및 width. q10/q90을 실제 quantile 배열에서 확인한다.
  - 각 학습 arm의 선택 step과 nonzero update count.
  - TAIL의 censor-loss 크기, tail 구간 사용 빈도, gap-floor 적용 빈도,
    강한 upper-tail 위반에서의 gradient, LoRA gradient norm, clipping 빈도.
  - fit wall time, 전체 실행 시간, V/scoring/I/O, 최대 GPU allocated/reserved,
    CPU RAM, GPU 대기 시간을 분리 기록.

분위수의 단순 평균을 조건부 평균이라고 부르지 않는다.
현재 metrics.py의 qmean_mse를 출력하더라도 그 한계를 명시하고 주 지표로 쓰지 않는다.

통계 표시:
  2,000회 paired series bootstrap, seed=41002.
  한 계열의 모든 origin/horizon과 두 optimizer seed 결과를 같이 유지한다.
  resampling 인덱스는 모든 arm에 동일하게 적용한다.
  seed 반복을 독립 데이터 표본으로 세지 않는다.
  같은 상품/상점·동일 미래 날짜의 상관을 완전히 처리한 CI가 아니므로
  이 파일럿의 기술적 불확실성 표시로만 보고한다. 유의성·일반화 확증은 아니다.

======================================================================
11. 사전 판정과 중단 규칙
======================================================================

구현 판정:
  CPU/smoke/누출 검사가 실패하면 본학습 시작 금지.
  “예측 가설 FAIL”이 아니라 “구현 또는 데이터 계약 미해결”로 기록한다.
  NaN/Inf/비인가 파라미터 변경/평가 선택 위반은 즉시 중단하고 부분 결과를 보존한다.

예측 판정 [제안: 후속 투자용, 논문 통과선 아님]:
  1) 두 seed 각각 TAIL의 전체 E_dev가 DROP보다 작고,
     seed평균에서 NAIVE보다도 작으면 “제한된 후속 검토 신호”로 기록한다.
     차이가 매우 작거나 CI가 0을 포함하면 그 크기와 불확실성을 반드시 함께 쓴다.
     자동 PASS나 새 아키텍처 진입 승인이 아니다.
  2) 평균은 좋아도 seed 부호가 다르면 “불안정한 개발 신호”.
  3) 검열 위치만 좋아지고 전체가 나빠지면 “회복/과대예측 간 절충”.
  4) DROP과 NAIVE를 넘지 못하면 “현재 설정에서 추가 가치 미확보”.
  5) TAIL이 비교군보다 좋아도 F0보다 나쁘면 “적응 손해 완화”와
     “적응 자체의 실용적 이득 미확보”를 동시에 적는다.
  6) 모두 step0이면 “학습된 수정이 선택되지 않음”. 수치 수정 실패와 동일시하지 않는다.

다음 행동:
  위 결과 어느 쪽이든 6 fits 이후 자동 추가 학습은 하지 않는다.
  재학습을 권고하려면 바꿀 변수 하나와 그 이유, 이미 시험한 것과의 차이를 보고한다.
  단순 loss 구현 보완으로 끝났다면 그렇게 종료한다.
  새 PEFT 방법으로 이어가려면 다음 사용자 판단에서 별도 신규성·필요성 검토가 필요하다.

이 실험이 모든 데이터셋·seed에서 F0를 이겨야 PEFT 연구가 가능하다는 규칙은 없다.
또한 tiny positive gain을 큰 학술 성과로 바꾸기 위해 기준을 사후에 완화하지 않는다.

======================================================================
12. 실행 안전과 재현성
======================================================================

[제안] 단일 GPU 직렬 실행. 기존 다른 프로젝트의 프로세스는 종료하지 않는다.
GPU 사용 여부는 실제 compute PID와 GPU 메모리로 판단한다.
pgrep로 모든 python 프로세스/자기 runner를 외부 작업으로 오인하지 않는다.
자신과 자신의 자식 PID는 구분하고, shell/모니터만 있다고 학습을 영구 대기하지 않는다.

상한은 예상 소요시간이 아니라 안전용 중단 기준이다.
  - 전체 작업 wall cap: 2시간.
  - 개별 fit wall cap: 30분.
  - 누적 GPU 대기 cap: 15분.
  - 시작: 외부 compute 없음 + free GPU>=4 GiB 상태 30초.
  - 실행 중 외부 compute 또는 free GPU<1 GiB이면 step 경계에서 대기.
  - resource 확인 간격 최대 5초. 총 대기 한도를 넘으면 보존 후 종료.
  - RAM은 기존 guard를 유지하고 최소 available RAM 2 GiB 미만이면 중단.

한도를 통과하기 위해 batch/모델/rank/precision을 자동으로 바꾸지 않는다.
실행 불가능은 환경 제약으로 보고한다. 다른 프로젝트·시스템 드라이버·전역 패키지 변경 금지.
GPU가 다르면 해당 기기의 메타데이터를 기록하고 수치 검사부터 통과한다.

Fit accounting:
  시작된 본학습 시도도 6개 한도에 포함한다. 실패를 삭제하고 공짜 재시도로 세지 않는다.
  exact checkpoint/optimizer/RNG에서의 재개와 처음부터 다시 학습한 시도를 구분한다.
  안전 중단 시 부분 결과를 남기고 추가 시도는 자동 승인하지 않는다.
  smoke state나 historical trained state를 새 fit 초기값으로 재활용하지 않는다.

과거 결과 불변:
  새로운 output 경로를 제외한 기존 tracked 결과/원자료/체크포인트 hash를 보존한다.
  기존 candidate_07 폴더, cdf 함수, 과거 FAIL 판정은 덮어쓰지 않는다.
  새 helper/module과 별도 runner로 구현하고 필요한 공통 함수만 import한다.
  기존 prepare_common_data나 screen_all을 실행해 과거 결과를 재생성하지 않는다.

======================================================================
13. 새로 만들 산출물과 CLI 인터페이스
======================================================================

아래는 이번에 새로 만들 경로다. 이미 존재하는 파일/명령이라고 가정하지 않는다.
  docs/CENSOR_TAIL_CONTROLLED_V1_PROTOCOL.md
  scripts/run_censor_tail_controlled_v1.py
  src/tsfm_peft_screen/candidates/censor_tail_v1.py
  tests/test_censor_tail_controlled_v1.py
  results/censor_tail_controlled_v1/
  .cache/censor_tail_controlled_v1/

동일 이름의 실험이 이미 존재하면 의미와 상태를 읽고 중복 실행하지 않는다.
임의로 v2를 붙여 재시도 예산을 늘리지 않는다.

runner에 구현할 단계:
  prepare: 입력·노출·모델·샘플링·설정 audit 및 봉인된 프로토콜 작성.
  preflight: CPU tests, 실제 모델 smoke, lambda 고정.
  run: 6 fits → 6개 V 선택 봉인 → E_dev 평가.
  verify: 독립 수치 재계산, checkpoint reload, source/data hash 점검.
  report: 전체 표/그림/한계/최종 판정.
  status: 현재 진행 단계, 완료/시도 fits, 종료 원인.

runner 구현과 --help 확인 후에만 아래 명령을 사용한다.
실행 예(위 인터페이스를 만든 후):
  scripts/with_cuda.sh .venv/bin/python scripts/run_censor_tail_controlled_v1.py prepare
  scripts/with_cuda.sh .venv/bin/python scripts/run_censor_tail_controlled_v1.py preflight
  scripts/with_cuda.sh .venv/bin/python scripts/run_censor_tail_controlled_v1.py run
  scripts/with_cuda.sh .venv/bin/python scripts/run_censor_tail_controlled_v1.py verify
  scripts/with_cuda.sh .venv/bin/python scripts/run_censor_tail_controlled_v1.py report

필수 산출물:
  PROTOCOL.md: 목적, 바뀐 점, 가정, 선택/평가 역할, 하드 상한.
  manifest/config: 원자료·모델·소스·id·상한·scale·샘플링 hash 및 환경.
  preflight.json: CPU/실모델 검사, 실패 내용, smoke updates.
  fits.json / trajectories.csv: 모든 fit, step별 train/V, 선택 모델.
  selection_seal.json: 선택 전에 정한 계약과 E_dev 접근 시각.
  predictions: F0 및 6개 V-selected, secondary endpoint 예측과 정답/마스크.
  metrics.csv: raw primary, 명시한 분모의 개선율, subgroup, 자원.
  verification.json: scalar 재계산, source/data 불변, reload, 총 counts.
  REPORT.md: 다음 절의 질문 순서에 맞춘 보고서.

저장 NPZ/CSV/JSON의 내부 키는 실제 코드와 함께 정의하고 문서화한다.
다른 파일에서 못 본 키가 있을 것이라고 추측해 읽지 않는다.

보고서 흐름:
  1) 무슨 문제인가: 큰 벌점인데 gradient가 없는 검열 손실.
  2) 왜 이 비교인가: 무시/버림/하한 활용을 같은 LoRA에서 구분.
  3) 무엇을 고쳤나: 꼬리 가정과 log-domain 계산, 과거 대비 변경표.
  4) 데이터와 정답은 무엇인가: M5 synthetic censoring, clean V 권한, reused E_dev.
  5) 실제로 좋아졌나: 전체→하위집단→불확실성→F0/비용 순서.
  6) 무엇을 주장하지 못하나: 실제 잠재 수요·신규성·독립 test·과거 실패 전체의 원인.
  7) 다음 결정: 종료/불안정/제한 신호 중 해당 판정. 후속 실행 자동 시작 없음.

그림은 표와 검산이 끝난 후 최대 2개만 만든다.
전체/검열 위치의 비교 하나, TAIL 손실·gradient 진단 하나면 충분하다.
그림 수나 검증 항목 수를 연구 기여로 세지 않는다.

Git 보관:
  이번에 만든 코드·프로토콜·작은 결과표·보고서만 범위를 지정해 commit한다.
  대용량 데이터/모델/예측 캐시는 기존 .gitignore 정책을 따른다.
  push는 사용자에게 기존에 승인된 현재 저장소 workflow가 확인될 때만 따른다.
  force-push·다른 branch 변경·실험 결과를 본 뒤 과거 기록 재작성은 금지한다.
  push 권한/승인을 확인하지 못하면 local commit과 미업로드 상태를 보고한다.

최종 사용자 보고의 첫 줄에는 실제 수행한 본학습 fits, smoke updates,
핵심 비교 결과, 현재 종료 상태를 적는다. 계획 숫자를 완료 숫자로 바꾸어 쓰지 않는다.

======================================================================
14. 출처와 점검 범위
======================================================================

저장소 근거(읽기 기준 commit 고정):
[1] https://github.com/CanelE452/tsfm-peft-method-screen/blob/e07ae8cd724fd8877734d730ec8cf3649c2d7591/research/reopen_review_20260914/CENSOR_TAIL_DIAGNOSTIC.md
[2] https://github.com/CanelE452/tsfm-peft-method-screen/blob/e07ae8cd724fd8877734d730ec8cf3649c2d7591/src/tsfm_peft_screen/candidates/censor.py
[3] https://github.com/CanelE452/tsfm-peft-method-screen/blob/e07ae8cd724fd8877734d730ec8cf3649c2d7591/docs/CANDIDATE_07.md
[4] https://github.com/CanelE452/tsfm-peft-method-screen/blob/e07ae8cd724fd8877734d730ec8cf3649c2d7591/src/tsfm_peft_screen/runners/common_fit.py
[5] https://github.com/CanelE452/tsfm-peft-method-screen/blob/e07ae8cd724fd8877734d730ec8cf3649c2d7591/src/tsfm_peft_screen/data.py
[6] https://github.com/CanelE452/tsfm-peft-method-screen/blob/e07ae8cd724fd8877734d730ec8cf3649c2d7591/src/tsfm_peft_screen/backbone.py
[7] https://github.com/CanelE452/tsfm-peft-method-screen/blob/e07ae8cd724fd8877734d730ec8cf3649c2d7591/src/tsfm_peft_screen/lora.py
[8] https://github.com/CanelE452/tsfm-peft-method-screen/blob/e07ae8cd724fd8877734d730ec8cf3649c2d7591/src/tsfm_peft_screen/metrics.py

선행 근거:
[9] Learning Quantile Functions without Quantile Crossing for Distribution-free
    Time Series Forecasting(2022/AISTATS).
    https://proceedings.mlr.press/v151/park22a.html
    이번 확인 범위: 공식 학회 페이지의 초록 및 공식 구현 문서.
[10] GluonTS 공식 I(S)QF 문서: exponential tails/parameterize_tail.
    https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.distributions.isqf.html
    이번 고정 tail-scale 규칙이나 혼합 검열 loss 전체가 이 문헌의 방법이라는 뜻이 아니다.
[11] FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent
    Demand Recovery and Forecasting in Fresh Retail(2025 공개/arXiv).
    https://arxiv.org/html/2505.16319v5
    실제 품절 레이블·인위적 재검열·평가 정답의 차이를 확인하기 위한 자료.
    이번에는 이 데이터 다운로드/학습/잠재수요 정답 사용을 수행하지 않는다.

======================================================================
부록 A. 손실 참조 구현: 합성 출력 텐서로만 검산됨
======================================================================

다음 함수는 CPU에서 knot 일치, 단조성, 매우 먼 upper-tail의 유한 log-loss와
nonzero gradient, 중복/교차 knot, FP32 gradient 전달, FP64 gradcheck를 확인했다.
이는 사용자 모델·실데이터·GPU 통합 검증이 아니다. 8절의 실모델 검사는 여전히 필요하다.
이 참조 함수만 복사해서 전체 실험이 준비됐다고 보고하지 않는다.

"""CPU-checked reference, not integrated with the user's forecasting repository."""
from __future__ import annotations
import torch


def log_survival_from_quantiles(
    prediction: torch.Tensor,
    threshold: torch.Tensor,
    levels: torch.Tensor,
    observed_train_scale: torch.Tensor,
    *,
    gap_floor_relative: float = 1e-4,
) -> torch.Tensor:
    """Return log P(Y > threshold) for a continuous, exponential-tail proxy.

    prediction: [batch, quantile, horizon]; threshold: [batch, horizon]
    levels: increasing quantile levels in (0, 1)
    observed_train_scale: [batch], derived from capped training observations only.
    This is a distribution-completion assumption, not an identified count model.
    """
    if prediction.ndim != 3 or threshold.shape != (prediction.shape[0], prediction.shape[2]):
        raise ValueError('Expected prediction [B,K,H] and threshold [B,H]')
    if levels.ndim != 1 or len(levels) != prediction.shape[1] or len(levels) < 2:
        raise ValueError('Quantile levels must have shape [K], K >= 2')
    if observed_train_scale.shape != (prediction.shape[0],):
        raise ValueError('Expected observed_train_scale [B]')
    if not (gap_floor_relative > 0):
        raise ValueError('Gap floor must be positive')
    # Deliberate small-tensor FP64 computation, also for a FP32 forecasting model.
    p = prediction.transpose(1, 2).to(torch.float64).sort(dim=-1).values
    y = threshold.to(device=p.device, dtype=torch.float64)
    tau = levels.to(device=p.device, dtype=torch.float64)
    scale = observed_train_scale.to(device=p.device, dtype=torch.float64)
    if not all(bool(torch.isfinite(v).all()) for v in (p, y, tau, scale)):
        raise ValueError('Inputs must be finite; filter missing observations first')
    if not (bool((scale > 0).all()) and bool((tau > 0).all())
            and bool((tau < 1).all()) and bool((tau[1:] > tau[:-1]).all())):
        raise ValueError('Invalid scale or quantile levels')
    eps = gap_floor_relative * scale.clamp_min(1.0)[:, None, None]
    gap = torch.maximum(p[..., 1:] - p[..., :-1], eps)
    knots = torch.cat((p[..., :1], p[..., :1] + gap.cumsum(dim=-1)), dim=-1)
    b_left = gap[..., 0] * tau[0] / (tau[1] - tau[0])
    b_right = gap[..., -1] * (1 - tau[-1]) / (tau[-1] - tau[-2])
    left = y <= knots[..., 0]
    right = y >= knots[..., -1]
    middle = ~(left | right)
    out = torch.empty_like(y)
    # Compute only active branches: avoid overflow in an unused torch.where branch.
    if bool(left.any()):
        log_cdf = tau[0].log() + (y[left] - knots[..., 0][left]) / b_left[left]
        out[left] = torch.log1p(-torch.exp(log_cdf))
    if bool(right.any()):
        out[right] = torch.log1p(-tau[-1]) - (y[right] - knots[..., -1][right]) / b_right[right]
    if bool(middle.any()):
        k = knots[middle].contiguous()
        t = y[middle]
        hi_idx = torch.searchsorted(k, t[:, None].contiguous(), right=True).squeeze(-1)
        lo_idx = hi_idx - 1
        lo = k.gather(-1, lo_idx[:, None]).squeeze(-1)
        hi = k.gather(-1, hi_idx[:, None]).squeeze(-1)
        frac = (t - lo) / (hi - lo)
        cdf = tau[lo_idx] + frac * (tau[hi_idx] - tau[lo_idx])
        out[middle] = torch.log1p(-cdf)
    return out

======================================================================
부록 B. 작성 시 참조 수식 검산 결과(실제 모델 학습 결과 아님)
======================================================================
{
  "knots_max_abs": 3.469446951953614e-18,
  "grid_monotonic_finite": true,
  "far_tail_loss": 3999924.605170186,
  "far_tail_shift_gradient": -4.0,
  "gradcheck": true,
  "tied_finite_nonzero": true,
  "crossed_finite_nonzero": true,
  "upper_join_difference": 7.999999249719281e-07,
  "fp32_autograd": true,
  "scope": "Synthetic output tensors only; no user model/data and no optimizer steps."
}
