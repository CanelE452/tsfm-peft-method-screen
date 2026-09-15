# 선행 경계 — 1차 자료6편

[확인] 2026-09-16. 지정3편, 직전 PROTOCOL 관련1편, 변경에 가까운2편만 본문을 확인했다. 검색에 함께 나온 다른 논문은 이 깊은 검토의 근거로 추가하지 않았다. 최초성·논문 채택을 확정하는 목록이 아니다.

| 논문·공개 상태 | 문제·데이터 권한 | 수식·학습 대상 | 대조·실제 평가 근거 | 이번 경계 |
| --- | --- | --- | --- | --- |
| **LoRA+: Efficient Low Rank Adaptation of Large Models (2024/ICML)** | 저랭크 행렬 A/B의 같은 LR가 넓은 모델에서 비효율적인가; task labeled train | W=W0+BA, etaB/etaA 고정 비대칭; A/B 학습 | 본문§5, GLUE의 GPT2/Roberta 및 Llama; 같은 LR LoRA와 비교; 비율은 모델·task·초기화에 민감 | LR 비대칭 자체는 알려진 방법. 이번 후보는 양쪽 동일 LR를 유지하므로 LoRA+의 재명명이 아님. |
| **In-Context Fine-Tuning for Time-Series Foundation Models (2025/ICML)** | 관련 시계열 예시를 추가 context로 제공; 이를 쓰도록 지속 사전학습 | f(관련 예시들, target history)→future, TimesFM-ICF 모델과 separator 학습 | 원문§4–6/부록A; Monash·ETT, base/긴 context/full FT/LP 비교 | 다른 시계열·추가사전학습 권한을 본 target-only LoRA와 혼동하지 않는다. |
| **On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation (2010/JMLR)** | 작은 검증 표본의 criterion 최적화도 과적합; 선택과 최종 평가 데이터 분리 | KRR/ARD 등의 train/selection 수준 구분; bias와 variance | 본문§2–6, nested evaluation과 단순 split의 비교; 선택 편향이 알고리즘 차이만큼 클 수 있음 | 현재 tune에서 고른 것과 dev의 역전은 이 위험과 일치할 수 있지만 본 자료만으로 원인이 확정되지 않음. |
| **Learning to Adapt Cross-Domain Preferences via Meta-LoRA for LLM Personalization (2026/arXiv v1)** | source task로 meta-init/prior 학습, target support로 적응 | KL=Σ||theta-theta0||²/(2 sigma_group²); support 수·entropy·Fisher로 penalty 조절 | 본문§3, 식1–5 및 부록C/H; 개인화 benchmark와 ablation; 무작위 predictor 이론과 실제 deterministic surrogate 범위를 구분 | 데이터 의존 비등방 prior 자체가 새 원리는 아님. 이번에는 source meta-init/Fisher 대신 target 과거 forecast residual의 레벨·형상 subspace를 사용한다는 구체 차이만 검토 가능. |
| **Explicit Inductive Bias for Transfer Learning with Convolutional Networks (2018/ICML)** | pretrain 지식을 적은 target sample에서 보존 | L2-SP: 초기 가중치와의 제곱거리; Fisher 가중형도 있음; CNN 가중치/새 head 학습 | 식3–4, 표1–3; MIT Indoor/Dogs/Caltech,5 runs. Fisher가 단순 L2-SP보다 유의한 이득을 보이지 않는 반례도 보고 | 보존항과 비등방 가중은 알려진 원리. 직접 parameter 거리와 normalized forecast function의 시간축 subspace 거리는 다른 수식이나, 일반 정규화의 재사용이라는 한계는 남는다. |
| **TILDE-Q: A Transformation Invariant Loss Function for Time-Series Forecasting (2022/arXiv, v2 2024)** | 시계열의 크기 이동/phase/증폭 불변 형태 학습; supervised target | softmax 잔차, Fourier, correlation 세 항의 조합(식5–8); 모델 가중치 학습 | 본문§4–5, 표1, 실제 여러 시계열·모델의 MSE/shape metric 비교 | 레벨과 형태를 구분하는 loss는 선행이다. 본 후보는 정답과의 task loss를 그대로 두고 frozen teacher의 보존 기하만 target residual 일관성으로 정한다. phase invariant task loss로 바꾸지 않는다. |

[확인] 접근한 원문 및 출판 확인:
- [LoRA+ 출판](https://proceedings.mlr.press/v235/hayou24a.html), [읽은 arXiv v2 본문](https://arxiv.org/html/2402.12354v2). PMLR PDF 경로는 web 도구 내부 오류여서 본문은 이 버전으로 확인했다.
- [In-Context 출판](https://proceedings.mlr.press/v267/faw25b.html), [읽은 arXiv PDF](https://arxiv.org/pdf/2410.24087). PMLR PDF 접근 오류로 대체 원문 사용; 출판 최종본과의 문장/숫자 전체 일치는 미확인.
- [Cawley–Talbot 원문](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf).
- [Meta-LoRA 본문](https://arxiv.org/html/2608.12389v1). 여기서 학회 게재를 주장하지 않는다.
- [L2-SP 출판 원문](https://proceedings.mlr.press/v80/li18a/li18a.pdf).
- [TILDE-Q v2 원문](https://arxiv.org/pdf/2210.15050), [공개 상태](https://arxiv.org/abs/2210.15050). 페이지에는 ICML2024 under review로 적혀 있어 채택 또는 VLDB 게재라고 쓰지 않는다.

[추정] 검토하려는 차이는 **target-only 과거 forecast residual의 반복성과 변동으로 레벨/형상 출력 보존의 상대 precision을 고정하고, uniform 보존과 평균 precision을 같게 하는 것**이다. 균일 anchoring, OLS, LR 조정, 출력 mixture를 새 이름으로 부르지 않는다. 서로 다른 수식이라는 사실만으로 충분한 신규성이 증명되는 것은 아니다. 같은 geometry의 선행을 완전히 배제하지 않았으므로 최초성은 미확인이다. 이번 실험은 이 구체 가설의 추가 가치를 직접 확인한다.
