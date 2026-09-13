# 최신 PEFT 결과 검토와 다음 방법 후보

검토 기준: CanelE452/tsfm-peft-method-screen, main, a1bf744988fdc456ba4f117da56bc3496159ac7f.
검토 범위: GitHub의 결과·프로토콜·핵심 구현 대조, Freshness v2 공개 집계 재계산, CPU 수식 검증. 사용자 GPU 학습·로컬 예측 캐시·메모리 계측을 재실행하지 않았다. 새 방법은 설계 후보이며 성능·독창성은 미검증이다.

## 1. 최신 결과 판정

### Freshness v2

채널별 관측 주기와 offset이 다른 설정, conditional LR 분리, affine scale/shift의 gradient와 상태 민감도 확인까지 수행했다. 12 fits, 두 seed 모두 강한 단순 대조를 넘지 못했다.

- seed30000: Standard 0.6404601079155101, Affine 0.6410889355905157, 차이 -0.0947336172%F0.
- seed30001: Feature 0.6525542672661115, Affine 0.6541930968211311, 차이 -0.2468915696%F0.
- 평균 -0.1708125934%F0. F0 손실 분모 0.6637851417747509.

이 분기는 종료한다. 학습되지 않은 gate라는 설명으로 다시 v3를 만들 근거는 현재 없다. 모든 관측 상태 어댑터가 불가능하다는 일반 명제가 아니라, 현재 후보의 추가 투자 근거가 부족하다는 판정이다.

### 메모리 feasibility

context4096, float32, batch8, horizon48, warmed fixed state:

|방법|peak MiB|ETTm2 전체 gradient 상대오차|Electricity 상대오차|ETTm2 step ms|
|---|---:|---:|---:|---:|
|exact|1981.0|0%|0%|199.86|
|checkpoint|763.7|0%|0%|283.99|
|FP16 saved storage|1335.7|0.0895%|0.1299%|238.72|
|INT8 hidden + FP16 other|1140.8|0.8968%|1.0392%|250.74|
|pair means + FP16 other|1140.9|66.2432%|77.0991%|235.14|

이는 보고서의 반올림 값이다. Stage B는 optimizer states를 모든 방법에서 제외했고 optimizer update는 0회다. 세 반복은 같은 배치의 계측 반복이다. gradient 오차는 forecasting loss 악화율이 아니다. 현재 forward가 정확해도 다음 update 이후 모델이 같다는 보장은 없다.

동일 bytes의 INT8가 훨씬 낮은 gradient error를 내므로 현재 pair-average primitive를 더 학습할 이유는 부족하다. 이 결과를 모든 시계열 압축의 불가능성으로 해석하지 않는다.

## 2. 계산상 원인: 평균은 잔차와 gradient의 결합항을 삭제한다

열벡터 표기를 사용한다. 토큰 입력 h_t∈R^d, LoRA A∈R^{r×d}, B∈R^{o×r}, frozen W∈R^{o×d}:

    y_t = W h_t + B A h_t
    g_t = dL/dy_t
    u_t = B^T g_t

스케일 alpha/r는 간단히 생략한다. 실제 구현에는 동일하게 적용해야 한다.

    grad_A = Σ_t u_t h_t^T
    grad_B = Σ_t g_t (A h_t)^T
    grad_h_t = W^T g_t + A^T u_t

두 인접 토큰에 대해:

    mean_h = (h1+h2)/2
    mean_u = (u1+u2)/2
    delta_h = h1-h2
    delta_u = u1-u2

정확한 항등식:

    u1 h1^T + u2 h2^T
      = 2 mean_u mean_h^T + 0.5 delta_u delta_h^T

둘 다 mean_h로 복원하면 오른쪽 두 번째 항을 잃는다. h가 비슷하다는 것만으로 이 항이 무시 가능하다고 결론낼 수 없다.

CPU 예제 h=(1.01,0.99), u=(1,-1): activation 상대오차 약 1%, 실제 gradient 0.02, mean 복원 gradient 0, gradient 상대오차 100%.

이는 합성 선형 계산 예제다. 실제 Chronos hidden tensor의 통계나 성능을 측정한 결과가 아니다.

## 3. 다음 방법 후보: 시간 구조를 이용한 잔차 보정형 LoRA backward

핵심: forward와 층 사이 gradient는 유지하고, 저장량 때문에 근사해야 하는 local grad_A만 대상으로 삼는다. 시간 평균을 저장한 뒤 잔차를 완전히 삭제하지 않고, 일부 잔차를 확률적으로 보존해 빠진 기여를 보정한다.

### 연산

시간축을 배치·채널별로 독립 처리한다. REG/future token은 따로 정확히 보존한다. batch나 채널을 시간축으로 오인해 섞지 않는다.

1. forward에서 Z_t=A h_t를 보존한다. grad_B에 필요하다.
2. 입력의 coarse temporal components를 보존한다.
3. high-frequency/detail components 중 일부를 p_j>0인 확률로 추출해 저장한다. 현재 backward gradient를 아직 알 수 없으므로 sampling은 forward 정보와 사전 고정 규칙으로만 한다.
4. backward에서 incoming g와 u=B^Tg는 정확히 계산한다.
5. coarse contribution은 계산하고, 보존한 detail contribution에는 역확률 보정한다.
6. grad_h와 grad_B는 표준 LoRA와 같은 계산을 사용한다. 근사 reconstruction을 normalization/attention backward 전체에 공급하지 않는다.

pair detail contribution R_j=0.5 delta_u_j delta_h_j^T이고, m개 독립 추출 J_s~p라면:

    grad_A_hat = coarse_grad + (1/m) Σ_s R_{J_s}/p_{J_s}

조건부로 정확한 incoming gradients, p_j>0, 정확한 산술을 가정하면:

    E[grad_A_hat | h,u] = grad_A.

하지만:
- 한 번의 추정 gradient는 오차가 클 수 있다.
- float16/저비트 저장은 별도의 오차를 만든다.
- clipping/Adam update는 비선형이므로 gradient 불편성이 update 불편성이나 학습 성능 보장을 뜻하지 않는다.
- coarse component가 효과적이지 않으면 random residual variance가 커진다.

### 메모리에서 가장 중요한 실패 위험

local X를 안 저장해도 같은 backing storage가 다른 autograd operation에 남아 있으면 실제 peak는 줄지 않는다. full X나 full reconstructed X를 숨겨서 보존하면 실패다.

모든 비용을 세야 한다: Z, coarse coefficients, sampled details, indices/probabilities, normalization states, gradients, optimizer, reconstruction temporaries. local compression ratio를 total CUDA peak 절감으로 바꾸어 말하지 않는다.

### 무엇이 새롭고 무엇은 이미 있나

새로움 확정 아님. 기존 원리를 명시해야 한다.
- GACT (2022/ICML): gradient sensitivity를 고려하는 activation-compressed training.
- ActNN (2021/ICML): 저비트 확률적 activation compression.
- VeLoRA (2024/NeurIPS): rank-1 sub-token activation projection.
- PRAC: Principal-Random Subspace for LLM Activation Compression and Memory-Efficient Training (2026/ICML 공식 목록 확인): principal component + random tail로 불편 추정.
- CARE-LoRA (2026/arXiv에서 확인, 정식 채택 미확인): XA와 decoder를 저장하고 local A-gradient만 근사하며 B/input gradient는 정확히 계산.
- Activation Compression in LLMs: Theoretical Analysis and Efficient Algorithm (2026/arXiv에서 확인): linear/nonlinear operator의 오차 전파 구분.
- Fast Monte Carlo Algorithms for Matrices I (2006/SIAM Journal on Computing): 역확률 scaling을 사용하는 무작위 행렬곱.

따라서 '주요 성분 + 무작위 잔차'나 'A만 근사' 자체를 새 기여로 주장할 수 없다. 남는 후보는 expensive SVD 없이 시간 구조로 낮은 분산을 얻는 실용적인 codec, exact inter-layer propagation과의 통합, 동등 메모리에서 검증된 품질/속도 개선이다. 이 추가 가치는 아직 미측정이다.

## 4. 대안 후보

### 대안 A: Forecast-query side adaptation

큰 backbone은 no_grad로 실행하고, 소수의 미래 query token만 작은 trainable side network로 업데이트한다. 각 layer의 고정된 context features를 query가 읽는다. LST (2022/NeurIPS)와 중복되는 큰 틀은 인정해야 한다. 단순히 작은 side net을 붙이는 것으로 새로운 방법이 되지 않는다.

원래 WIDE head와 다른 점은 마지막 한 표현만 읽지 않고, 여러 깊이의 과거 표현을 미래 horizon별 query가 읽도록 하는 점이다. learned key/value projection이 full hidden activation을 다시 저장하는 함정에 주의한다. frozen low-dimensional projection이나 cache/recompute 비용을 명시한다. full backbone no_grad라고 total activation cost가 자동으로 없어지는 것은 아니다.

### 대안 B: Mean/detail를 모두 남기는 quantization

시간축 변환 후 mean과 detail을 각각 양자화하고 복원한다. pair mean만 남기는 방법과 달리 detail을 삭제하지 않는다. 구현은 간단한 편이지만 wavelet coding·mixed precision·GACT와 가까워 novelty risk가 크다. 주방법이 아니라 직접 대조 또는 낮은 비용의 탐색 후보로 둔다.

## 5. 실제 다음 개발 범위

합의 전 GPU 학습이나 Git 변경은 하지 않는다. 다음 실행안은 최대 하나만 채택한다.

### 최소 구현
- 기존 repo에 별도 실험 디렉터리와 immutable contract 사용. 기존 결과 수정하지 않음.
- custom LoRA backward 구현. global saved-tensor pooling을 계속 바꾸는 방식은 피한다.
- 초기 B=0 때문에 A-gradient가 0인 무의미한 검사 금지: 기존 warm state 사용.
- forward, grad_B, grad_input exact parity.
- 잔차를 모두 저장하면 grad_A까지 exact.
- 작은 경우의 모든 sampling 선택을 열거하면 estimator 평균이 exact.
- 실제 peak에서 저장량이 줄지 않으면 큰 학습은 중단.

### 직접 비교
필수 profiler: standard LoRA, exact checkpointing, generic local compression, CARE-style control, PRAC-style control, proposed. 공식 방법 전체를 재현하지 않았으면 primitive라고 명시한다. 동일 dtype·batch·context·optimizer state·metric·validation 기회 유지.

전체 정확도 pilot은 모든 profiler arm을 대규모 실행하는 대신 가장 가까운 generic/CARE-or-PRAC 대조와 proposed + exact reference로 제한한다. 두 데이터 × 두 seed × 네 방식 = 최대16 fits를 다음 단계 상한으로 제안한다. 이는 승인 전 초안이며 품질/시간/메모리 목표와 recipe를 실행 전에 고정해야 한다.

### 성공 정의
같은 품질에서 peak 또는 time/throughput의 실용적 이득. gradient L2≤1% 자체를 논문 성공으로 보지 않는다. 현재 checkpoint control을 무시하지 않는다. 실용적 BF16/AMP baseline도 필요하다.

새 방법이 더 큰 microbatch를 허용하면, '더 큰 batch' 자체의 최적화 차이를 분리해서 fixed-update와 fixed-wall-clock 결과를 함께 보고한다.

## 6. 최신 코드의 추가 유용한 관찰

context4096 inventory에서 frozen MLP activation 12개는 각각 25,559,040 bytes, 합 292.5MiB다. 단순 ReLU derivative만 필요할 때는 부호 mask를 보존하는 exact backward 최적화 여지가 있다. 하지만 saved bytes 합은 peak 감소가 아니며, 해당 입력이 다른 연산에 필요한지 확인해야 한다. 이 engineering 개선 자체는 새 PEFT 연구 기여가 아니라 강한 메모리 baseline이다.

## 7. 소스 경로

- results/candidate_01_v2/RESULT.md
- results/candidate_01_v2/metrics.csv
- results/memory_feasibility/RESULT.md
- results/memory_feasibility/stage_b_metrics.json
- results/memory_feasibility/verification.json
- docs/MEMORY_FEASIBILITY_PROTOCOL.md
- src/tsfm_peft_screen/memory/compression.py
- src/tsfm_peft_screen/memory/common.py
- scripts/compare_memory_feasibility.py
- results/memory_feasibility/inventory_ettm2_4096.json

동봉 check_math.py는 CPU NumPy만 사용한다. 수식 항등식·평균 삭제 반례·샘플링 기댓값·공개 metric 집계만 검증한다. 실제 메모리 절감이나 TSFM forecasting 향상을 입증하지 않는다.