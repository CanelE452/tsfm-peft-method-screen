# 방법론 주장 재검토: 지속성 규칙과 관측 기반 보정 제한

작성 2026-09-19. 기준 `f78422a`. 현재 코드·기존 결과를 보존한 검토다. 새 학습·Chronos 추론·새 후보는0개다. **방법론 논문 목표는 아직 달성되지 않았다.** 이번 검토는 분석 원고의 제목을 바꾸어 완료 처리하는 작업이 아니다.

## 현재 결론과 연구 방향

C3의 지속성 계산을 새 핵심 기여로 유지할 근거는 약하다. 기존 세 seed 전력16계열 SHIFT8에서 C3는 일반 C2보다2.3994% 좋았지만 MAG_ONLY보다0.2515% 나빴다. 새 δ 직접 비교의 공통 두 seed에서는 C3가 두δ구성보다7.9857~8.3531% 좋았으나 MAG보다0.4128% 나빴다. 서로 다른 seed집합의 수치를 섞거나 최신 두 seed로 과거 세 seed 결론을 대체하지 않는다.

그렇다고 작은 추가 PEFT의 효용이 없다는 뜻은 아니다. **검토할 만한 방법 문제는 관측을 훼손하지 않으면서 추가 어댑터가 어느 구간을 얼마나 수정할지 제한하는 것**이다. 다만 이미 구현된 MAG를 사후에 정식 후보로 승격하는 것은 별도의 연구 결정이며, 지금까지의 E 이득은 발견 근거로만 취급해야 한다. MAG를 새 이름으로 포장하거나 threshold/rank/초기값을 고쳐 우승시키지 않는다. 범위 변경은 사용자에게 확인 중이며 아직 새 학습을 시작하지 않았다.

## 새로 확인한 가까운 선행

| 선행 | 확인한 실제 내용 | 현재 구현과의 구분 및 남은 비교 |
| --- | --- | --- |
| GateRA, AAAI2026 | token embedding으로 sigmoid gate를 만들고 PEFT 분기를 조절한다. entropy penalty와 gradient masking 설명을 포함한다. | 입력별 PEFT 강도 조절 자체는 신규 주장 불가. MAG는 관측값의 robust 크기로 정하는 비학습 gate이고 현재 잔차 위치·수식도 다르다. 동일 방법이라는 뜻은 아니며, 학습형 gate와 직접 비교는 미실행이다. |
| δ-Adapter, ICLR2026 | frozen forecaster에 bounded input/output correction을 붙이며 별도의 입력 선택 mask도 제시한다. | 현재 완료한 것은 공개 XY cell의 Chronos 통제 이식이다. 전체 방법·feature selector를 이겼다는 근거가 아니다. MAG는 입력값 자체 대신 latent residual을 조절한다. |
| COSA, ICLR2026 | frozen output에 문맥과 gate를 쓰는 잔차 보정, 관측된 최근 정답으로 온라인 갱신한다. | 원래 입력을 유지하는 출력 보정도 이미 존재한다. 현재 오프라인 synthetic TRAIN과 정보·갱신 권한이 다르다. 기존 OUTPUT_CONTEXT는 공식 COSA 재현이 아니다. |
| Time-PEFT, 공식 코드 | frequency adapter와 channel adapter를 사용한다. | 시계열 특화 모듈의 비교 예시다. 현재 C3/MAG가 Time-PEFT와 같은 논문 기여를 갖췄다거나 성능에서 이겼다는 근거는 없다. |
| STAR, preprint | state 변수에 조건부인 bottleneck adaptation으로 이상 탐지를 다룬다. | 조건부 TSFM adapter 자체는 새 범주가 아니다. MAG는 별도 state label을 받지 않으며 forecasting 과제다. STAR를 같은 정보권한의 직접 forecasting 대조로 취급하지 않는다. |

근거는 각각 [GateRA 출판 페이지](https://ojs.aaai.org/index.php/AAAI/article/view/40538), [GateRA 최종 PDF의 Method 및 Table5](https://ojs.aaai.org/index.php/AAAI/article/download/40538/44499), [δ 원문](https://arxiv.org/html/2601.20280v1), [COSA 공식 구현 설명](https://github.com/bigbases/COSA_ICLR2026), [Time-PEFT 공식 저장소](https://github.com/kaist-dmlab/TimePEFT), [STAR §3.3](https://arxiv.org/html/2510.16014v1)이다. GateRA의 공식 저자 코드 연결은 이번 원문/웹검색으로 확인하지 못했다. 코드가 존재하지 않는다고 단정하지 않는다. Time-PEFT OpenReview 원문은 접근 challenge가 있어 이번 추가 검토에서는 공식 코드와 설명 범위를 사용했다. 모든 선행을 망라했다는 주장은 하지 않는다.

GateRA 원문의 gate는 `sigmoid(W_g h+b_g)`다. 이를 현재 nonlinear residual의 앞에 붙인 통제군을 만든다면 **GateRA 원 논문의 HiRA 전체 재현이 아니라 gating 원리를 이식한 대조**라고 명시해야 한다. 원문에 명시되지 않은 세부 계수나 초기화를 임의로 정하고 공식 설정이라고 부르지 않는다. 이는 직접 비교 구현 전에 해결해야 할 명세 항목이다.

## 현재 MAG의 정확한 정의

입력512개에서 median m과 `s=max(1.4826 MAD, 0.1 sigma_TRAIN)`을 계산한다. patch j에 속한16개 값에서 `|x-m|/s > 3`인 비율을 u_j라 하고 `g_j=1-u_j`로 둔다. Chronos 정규화와 원래 관측값은 그대로 유지한다. patch embedding h_j에 다음을 더한다.

`h'_j = h_j + g_j c(h) tanh(W_up GELU(W_down h_j+b_down)+b_up)`

`c(h)`는 patch embedding RMS의 patch 간 중앙값을 최소1e-6으로 제한한 detached 값이다. down512→8/up8→512와 bias의 합은8,712개이며 B0 LoRA294,912개와 본체는 별도 유지한다. source 코드: [MAG](../../experiments/c3_weakness_controls_20260918/model.py), [기존 residual](../../experiments/additive_b0_adapter_v1_20260917/model.py).

현재 gate는 측정 오류인지 지속 변화인지 식별하지 않는다. 극단값 비율을 쓰는 고정 규칙이다. 이 점은 C3를 뺀 뒤에도 해결한 문제라고 주장할 수 없다. 작은 원자료 손해·긴 변화 형태 손해·ETTm1 부진도 그대로 남는다.

## 실제로 보장하는 것과 보장하지 않는 것

별도 NumPy식과 현재 MAG 코드의 출력이 CPU에서 일치했다. 현재 residual 모듈을 그대로 사용해 좌표별 `|h'_j-h_j| <= g_j c(h)`와 g_j=0인 patch의 embedding 불변성을 검사했다. θ와 독립적인 gate에서는 고정 h의 국소 parameter Jacobian에 g_j가 곱해진다. 같은 upstream cotangent를 고정한 두 계산의 gradient 차이는0이었다. positive affine 입력 변환과 sigma의 동일 배율 변환에 따른 gate 불변성 예도 확인했다. 이들은 표준 연산에서 따르는 성질이지 새로운 일반 정리라고 주장하지 않는다.

**patch의 보정을0으로 두어도 최종 예측은 바뀔 수 있다.** 다른 patch가 바뀌면 frozen attention/readout이 정보를 섞는다. 합성 두 token 반례에서 첫 token은 그대로인데 frozen 합 readout은3.0에서3.5로 변했다. 또한 token별 gradient 기여가1과−1이면 ungated 합은0이지만 두 번째 기여를 끄면 합은1이다. 따라서 local gate가0~1이라는 사실만으로 전체 gradient norm 감소나 예측 손해 방지를 증명할 수 없다.

이는 실제 Chronos 성능을 측정한 새 실험이 아니다. [실행 코드](reference_checks.py), [CPU 검산 결과와 코드 hash](REFERENCE_CHECKS.json)에 범위를 명시했다. optimizer update0, 실제 forecaster inference0이다.

## 방법론 논문으로 쓰려면 채워야 할 부분

1. **고정한 기여:** 기존 MAG의 관측 기반 residual 제한을 후보로 채택할지 결정해야 한다. C3/MAG 교차 가중치의 E 최고 조합을 고르거나 새 threshold를 탐색하는 방향은 제외한다.
2. **가장 가까운 대조:** 같은 frozen B0·residual·TRAIN·초기값·선택 기회에서 고정 MAG와 학습형 token gate를 비교해야 한다. 기존 PLAIN·C3·POS_ONLY와 이미 완료한δ결과도 보존한다. GateRA식 gate만 이식한 결과와 원 논문 전체 재현을 구분한다.
3. **새 평가의 실체:** 평가에 사용하지 않은 기간인지 기록을 감사하고, 모델·선택을 먼저 고정해야 한다. 같은 NESO의 다른 기간은 시간적 전이이며 독립 source가 아니다. 자료가 과거에 내려받아졌다는 것과 모델 선택·평가에 사용됐다는 것은 구분하되, 기록 밖의 노출 부재를 보장하지 않는다.
4. **성능·비용·신규성의 분리:** 고정 gate가 더 적은 학습 계수로 유리한 절충을 주는지 확인한다. 단순 대조로 충분하면 그 사실을 인정한다. 모든 데이터에서 이겨야 한다는 기준은 사용하지 않지만, 새로 주장하는 구성요소의 추가 가치는 필요하다.

이번 문서는 방법론의 연구 방향을 구체화한 검토이며, 위 비교가 이미 실행됐거나 논문 성공이 보장됐다는 뜻이 아니다. 미승인 후속 학습은 자동 실행하지 않는다. [후속 범위 검토안](FOLLOWUP_SCOPE_KO.md)에 중복 회피와 미확정 항목을 분리했다.
