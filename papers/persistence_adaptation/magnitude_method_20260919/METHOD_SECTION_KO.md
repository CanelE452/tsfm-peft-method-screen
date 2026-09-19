# 방법 절 초안: 관측 진폭에 따른 추가 잔차 적응의 제한

상태: 고정된 실제 구현의 방법 기술. 새 방법 이름을 만들거나 기존 MAG를 변경하지 않았다. 성능·신규성·논문 채택 가능성은 이 절의 작성으로 입증되지 않는다. `learned_gate_comparison_20260919`의 전체 평가·독립 검산은 완료됐다. 고정한 제한적 구성요소 기준은 충족했으며, 구체적인 이득·손해·미완료 논문 요건은 [주장과 근거](PAPER_CLAIM_EVIDENCE_KO.md)에 분리했다.

![고정 MAG 구현의 정보 흐름](method.png)

**그림 설명.** 관측 입력을 그대로 유지한 상태에서 동결된 B0의 patch embedding에 작은 잔차를 더한다. MAG의 고정 gate는 관측창의 robust 통계를 이용해 patch별 잔차의 크기를 제한한다. 파란색은 동결된 B0, 주황색은 학습되는 잔차, 보라색은 학습 파라미터가 없는 gate다. 정답은 학습 손실에만 들어간다. 동결은 해당 가중치의 업데이트를 막는 것으로, 추가 잔차를 학습하기 위한 encoder/decoder 경유 역전파까지 막는 것은 아니다. 아래 대조군은 같은 잔차 위치에서의 차이를 보이며, 학습형 gate에는 추가513개 파라미터가 있다. 그림의 주 경로는 MAG이고, 학습형 대조에서는 잔차와 gate 파라미터를 함께 업데이트한다.

[벡터 PDF](method.pdf) · [편집 가능한 SVG](method.svg) · [재현 코드](draw_method.py) · [구현·그림 hash](FIGURE_MANIFEST.json)

## 문제 설정과 B0

관측창은 $x\in\mathbb R^{512}$, 예측 길이는64이며0.1부터0.9까지9개 분위수를 예측한다. B0는 원래 입력과 합성 학습 조건을 이용해 LoRA로 학습한 Chronos-Bolt-small이다. 추가 적응에서는 source와 seed가 일치하는 B0 checkpoint를 고정한다. B0에 이미 학습된 LoRA294,912개와 본체를 공유하므로, 이하의8,712개는 B0 구축 비용을 포함하지 않는 **추가** 학습 파라미터 수다.

B0의 원래 instance normalization과 patch embedding을 유지한다. 길이16의32개 patch에서 얻은 embedding을 $h_j\in\mathbb R^{512}$로 둔다. 관측값을 clipping하거나 정답 상태에 따라 교체하지 않는다. 정규화의 위치·scale과 출력의 역변환도 B0 경로를 따른다.

## 고정 진폭 gate

관측창의 중앙값과 robust scale을 다음과 같이 계산한다.

$$m(x)=\operatorname{median}_t x_t,\qquad
s(x)=\max\{1.4826\operatorname{median}_t|x_t-m(x)|,\;0.1\sigma_{\mathrm{TRAIN}}\}.$$

여기서 $\sigma_{\mathrm{TRAIN}}$은 사전에 고정한 TRAIN population 표준편차다. gate에는 평가 미래나 합성 과정의 정답 상태를 사용하지 않는다. patch $P_j$에서 robust 절댓값3을 넘는 관측 비율을 이용한다.

$$g_j(x)=1-\frac1{16}\sum_{t\in P_j}
\mathbf1\!\left[\frac{|x_t-m(x)|}{s(x)}>3\right].$$

따라서 $g_j\in[0,1]$이고, 해당 patch의 극단값 비율이 높을수록 추가 잔차를 줄인다. 이 gate는 측정 오류와 실제 지속 변화를 구별하는 분류기가 아니다. 두 경우가 같은 관측 과거를 만들면 숨겨진 원인을 알아내는 기능은 없다.

## 동결 모델 위의 잔차

$W_d\in\mathbb R^{8\times512}$, $W_u\in\mathbb R^{512\times8}$ 및 두 bias로 잔차를 정의한다.

$$a_\theta(h_j)=c(h)\tanh\!\left[W_u\operatorname{GELU}(W_dh_j+b_d)+b_u\right],$$
$$h'_j=h_j+g_j(x)a_\theta(h_j).$$

$c(h)$는 원래 patch embedding RMS들의 중앙값을 최소$10^{-6}$으로 제한한 값이며, 이 값 자체에는 gradient를 흘리지 않는다. 잔차의 차원은512→8→512, 추가 파라미터는$512\times8+8+8\times512+512=8,712$개다. up weight와 bias를0으로 초기화하므로 초기 예측은 동일한 B0와 정확히 일치한다. 이후 $h'$는 B0의 동결된 encoder·decoder·head와 원래 역정규화를 거쳐 분위수 예측을 만든다.

이 수식은 각 좌표의 잔차 절댓값이 $g_jc(h)$ 이하임을 보장한다. 그러나 한 patch의 gate가0이라고 최종 예측이 불변인 것은 아니다. 다른 patch의 잔차가 attention과 readout을 통해 영향을 줄 수 있다. 전체 gradient norm 감소나 예측 손해 방지의 보장으로 확대하지 않는다.

## 학습 목적과 직접 대조

기존 TRAIN draws·labels를 유지하고 normalized2pinball을 최소화한다. 손실은 모든 batch·9분위수·64 horizon의 $2\rho_q(y-\hat y_q)/\sigma_{\mathrm{TRAIN}}$ 평균이다. 입력512, batch32, FP32,1024updates와 checkpoint0/256/512/768/1024를 고정했다. selection seed81550에서 LR1e-4/3e-4를 같은 검증 목적에 적용한 뒤81551/81552에서 반복한다. 전체 정의는 [봉인 계약](../../../experiments/learned_gate_comparison_20260919/PROTOCOL.md)을 따른다.

PLAIN은 같은 잔차에 $g=1$을 사용한다. C3는 기존 지속성 규칙을 사용하며 수정하지 않는다. TOKEN_GATE는 원래 patch embedding에서 $\operatorname{sigmoid}(W_gh_j+b_g)$를 계산하고, TOKEN_GATE_ENTROPY는 동일 gate에 평균 Bernoulli entropy/log2의0.01배를 손실에 더한다. 후자의 검증 목적에는 entropy를 더하지 않는다. 두 학습형 gate는 weight/bias0으로 초기화되어 처음에는0.5이고, 잔차 up은 공통으로0이다. gate가 추가하는 파라미터는513개다.

이 학습형 gate는 [GateRA 원문 §Method](https://arxiv.org/html/2511.17582v1#S3)의 gating 원리를 현재 additive residual에 통제 이식한 것으로 공식 HiRA/NLP 전체 재현이 아니다. MAG는 전체 관측의 robust 통계를, 학습형 gate는 pre-attention embedding을 읽는다. 따라서 관측 권한은 같지만 feature 표현은 다르다. 비교 차이를 ‘학습 가능 여부 하나’의 인과효과라고 해석하지 않는다. [공정성 감사](../../../research/learned_gate_comparability_20260919/COMPARABILITY_KO.md)에 확인 범위를 기록했다.

## 평가와 주장 범위의 연결

기존 Electricity·ETTm1·전력16계열은 이미 노출된 개발 평가다. 새 NESO55일은 이전 H1 정답 구간과 겹치지 않는 시간적 전이 점검이지만 같은provider/ND계열이고 CSV는 이전에 다운로드돼 있었다. 독립 source 또는 완전히 새로운 원자료라고 부르지 않는다.

모든 학습률·체크포인트를 먼저 고정하고 전체 예측을 저장한 뒤 평가 정답을 채점한다. 기본·오류·SHIFT4·SHIFT8·SHIFT_POINT 및 모든9개 형태, 두 seed, selected/fixed1024를 보존한다. MAG의 일반 잔차 대비 추가 가치, 학습형 gate 대조, C3 지속성 규칙의 추가 가치, 시간 전이, 비용을 따로 판단한다. [자원 측정 범위](../../../research/learned_gate_comparability_20260919/RESOURCE_SCOPE_KO.md)의 timer 차이 때문에 과거/현재 시간비를 순수 계산 속도 향상으로 주장하지 않는다.

이 방법 절의 평가 결과는 [완료된 주장·근거표](PAPER_CLAIM_EVIDENCE_KO.md)와 [주 비교 그림](primary_effects.pdf)에 연결한다. 단순 gate 자체의 신규성, 실제 센서 오류 해결, Time-PEFT/GateRA 전체에 대한 우위, 범용 PEFT 우위는 이 구현 기술로 증명되지 않는다. 최종 주장과 제목은 완료된 비교의 실제 효과에 맞춰야 하며, 불리한 결과를 빼거나 기준을 고쳐 방법론 성공으로 만들지 않는다.

구현 근거: [MAG gate와 실제 forward](../../../experiments/c3_weakness_controls_20260918/model.py), [잔차 모듈](../../../experiments/additive_persistence_validation_v1_20260917/model.py), [robust scale 및 손실](../../../experiments/outlier_signal_peft_v1_20260917/model.py), [학습형 gate 대조](../../../experiments/learned_gate_comparison_20260919/model.py).
