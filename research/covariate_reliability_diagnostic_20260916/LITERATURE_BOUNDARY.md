# 신뢰도 기반 PEFT의 선행 경계

2026-09-16에 1차 원문에서 아래 범위를 읽었다. 공개 코드를 실행하거나 논문 결과를 재현한 것은 아니다. 문헌에서의 이득과 로컬 실험 이득을 분리한다. '최초' 주장은 하지 않는다.

| 선행 | 직접 확인 범위 | 이미 알려진 부분 | 현재 문제와 남는 차이 |
|---|---|---|---|
| CoRA: Covariate-Aware Adaptation of Time Series Foundation Models, arXiv v1, 2025 | §3.1–3.2, 식2–7, §4.1 설정 | 동결 embedding, 학습된 covariate weight의 softmax, adaLN shift/scale, zero-init | 식6의 선택 weight는 covariate별 학습 파라미터다. forecast vintage의 원점별 신뢰도 추정/실제 공개시각 검증은 여기서 확인한 설계와 다르다. 그렇다고 단순 동적 gate의 신규성이 입증되는 것은 아님 |
| UniCA: Unified Covariate Adaptation for Time Series Foundation Model, ICLR2026 | 공식 proceedings, §4.2 pre/post fusion, §I limitations | 과거 공변량 GLU 및 future-known 공변량의 conditional attention pooling/후속 attention | 입력의 noisy/conflicting 관계를 개선하는 방향을 한계로 명시한다. 따라서 '미래 공변량을 adapter로 골라 쓴다'만으로 차별점은 부족 |
| TFMAdapter: Lightweight Instance-Level Adaptation of Foundation Models for Forecasting with Covariates, arXiv v1, 2025 | §3–4, Algorithm1 전반의 pseudo-forecast 생성과 두 단계 설명 | 소수 TSFM 호출로 pseudo-forecast를 만들고 과거 context의 공변량과 출력 보정기를 학습 | 가벼운 출력 보정·instance 적응도 직접 비교 대상. 이번에 전체 optimizer/GP 구현까지 검증하지 않았으며 모든 구조와의 동치 판단은 미완료 |
| Exogenous Dropout: A Simple, Strong Baseline for Corruption-Robust Time Series Forecasting with Covariates, arXiv v1, 2026 | §3.1–3.5, §4.1–4.5, §5–6 | 전체 채널의 train-only inverted dropout, gated FiLM/fallback BoundEx, 동일 dropout 대조 | 같은 채널의 과거·미래를 같은 mask로 없앤다. 실제 forecast vintage 품질 이동이나 frozen Chronos-2 LoRA와 동일하지 않다. 그래도 단순 dropout은 반드시 강한 대조로 남아야 함 |

출처:
- [CoRA v1](https://arxiv.org/html/2510.12681v1#S3).
- [UniCA 공식 논문](https://proceedings.iclr.cc/paper_files/paper/2026/file/0b5eb45a22ff33956c043dd271f244ea-Paper-Conference.pdf).
- [TFMAdapter v1](https://arxiv.org/html/2509.13906v1#S4).
- [Exogenous Dropout v1](https://arxiv.org/html/2607.05452v1#S3).

Exogenous Dropout의 mask는 채널 k별 Bernoulli(1-p)를 뽑고 m_k x_k/(1-p)를 쓴다. 연구의 주 비교는 p=.3이며 이것을 이번 로컬 학습 설정으로 채택/실행한 것은 아니다. 논문 내에서는 bounded foil보다 dropout을 적용한 DAG가 강했다. 인위적 corruption과 native-vintage 오차는 다르므로 이 결과를 현재 데이터로 그대로 이전하지 않는다. 논문의 표현공간 bound는 gate 비례 제어이며 무조건적인 예측 손실 개선 보장이 아니다.

추가 탐색: [Model-Agnostic Online Certificate-Driven Calibration for Time Series Forecasting Under Distribution Shift, UAI2026](https://proceedings.mlr.press/v337/huang26b.html)의 공식 초록과 출판 메타데이터만 확인했다. gated Bayesian residual head와 predict-then-update를 다룬다고 명시한다. PDF 접근 실패로 정리·가정·구현은 검증하지 못했다. 위험 인지 잔차 gate가 새롭다고 주장하기 전 추가로 읽어야 할 가까운 선행이며, 이번 네 논문 수준의 검토를 완료했다고 세지 않는다.

이번 문헌 판단은 KNOWN_COMPONENTS / DISTINCT_METHOD_NOT_SPECIFIED다. 단순 gate, FiLM, 미래 변수 선택, dropout, 일반 LoRA 증류를 재명명해 새 PEFT라고 하지 않는다. 가까운 대조가 강하다는 이유만으로 새로운 메커니즘이 불가능하다고 결론내리지도 않는다.
