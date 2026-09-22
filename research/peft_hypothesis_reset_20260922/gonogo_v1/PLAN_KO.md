# 두 PEFT 가설의 빠른 GO / NO-GO 설계

작성: 2026-09-22. 상태: **DESIGN_COMPLETE_AWAITING_AGREEMENT**. 이번 산출물은 사용자가 요청한 실험 설계다. 새 모델 추론·학습·성능 검사는 아직 실행하지 않았다. 이 문서와 `DESIGN.json`의 일치 여부만 CPU에서 검사했다. 실행 runner가 이미 있다는 뜻이 아니다.

## 목적과 판정의 의미

목적은 제한된 환경에서 기존 방법보다 추가 가치가 있는 PEFT 후보를 싸게 선별하는 것이다. 소비처는 사용자의 다음 연구 투자 결정이다. **GO_SCREEN은 작은 재검증에 투자할 신호**이며 통계적 확증·배포 승인·신규성·논문 PASS가 아니다. 한 데이터셋, 두 반복 seed의 개발 선별이다. 기존 연구에서 Electricity를 사용했으므로 새로운 독립 벤치마크라고 부르지 않는다.

- H1: 짧은 이력의 개인화에서, 불안정한 계수만 공유값 쪽으로 축소하는 것이 같은 계수의 단순 축소보다 유리한가?
- H2: 부분적으로 누락된 patch의 실제 관측 위치를 attention에 반영하는 것이 일반 LoRA·보간·단순 시간 편향보다 유리한가?

두 후보는 독립이다. H1 성능 실패가 H2 실행을 막지 않는다. 공통 데이터·수치·자원 문제가 생기면 영향을 받는 작업만 중단한다. 이전 HIER/rollout/continuation/F 실험을 재개하거나 변경하지 않는다.

## 선택과 근거

[확인] 로컬에 검증 이력이 있는 Electricity 원본과 Chronos-Bolt-small weights가 있다. GPU는 RTX 4070 12GB다. 원본 SHA-256은 이번 설계에서도 다시 확인했다. 기존 실행에서 512 updates가 약 35–44초였지만 다른 학습 방식이므로 이번 실행시간을 보장하지 않는다.

[설계] **한 원천, context 192시간 / horizon 24시간**으로 고정한다. 192는 주간 이력을 포함하고 native 16시간 patch에 맞으며, 24는 native 64 이내이므로 rollout 혼선을 제거한다. 데이터는 hourly Electricity이며 건물 유형 metadata나 검증된 달력을 가정하지 않는다. 이력은 총 40일(32일 개인화 + 8일 보정)이다. 1주일 few-shot 결과라고 부르지 않는다.

모델은 `amazon/chronos-bolt-small`, revision `772f3d25d38aec6d914c8949dab4462e2d46f5d8`, FP32/eval mode다. native 9 quantiles의 첫 24개 horizon을 사용한다. 공통 q/v LoRA는 rank 8, alpha 16, dropout 0, bias none. 대상 모듈 이름을 실제로 열거해 저장한다. 새 설치·모델 변경·양자화·rank/LR 탐색은 없다.

설계 근거는 [MixFT](https://arxiv.org/html/2603.02840v1)의 이질적 TSFM 적응 문제, [LiFT](https://proceedings.iclr.cc/paper_files/paper/2025/hash/27e5626cabdbb6cd5c56ce4114ff93e4-Abstract-Conference.html)의 task별 Bayesian PEFT, [STAR-Set](https://arxiv.org/html/2603.06605v1)의 시간 attention bias다. MixFT/STAR-Set은 여기서는 arXiv 원문으로 확인한 선행이며, 본 후보의 재현 성능 근거가 아니다. 새로운 후보 추가는 하지 않는다. 비정형 대조는 H1의 gate 순환 교환, H2의 patch에 정확히 맞춘 결측이다. 각각 제안 메커니즘이 필요 없는 반례를 의도적으로 만든다.

## 공통 데이터 분할과 누수 방지

기존 audited Electricity 32계열 목록을 그대로 모집단으로 사용한다. `sha256('peft-gonogo12-v1|'+원본열번호)` 순서로 donor 8, DEV 4, EVAL 8을 고정하고 나머지 12개는 사용하지 않는다. 결과를 보고 계열·원천을 바꾸지 않는다. 구체 ID는 `DESIGN.json`에 있다. 기존 실험 노출 및 사전학습 중복 가능성을 명시한다.

전체 26,304시간에서 D=15,768, E=D+336=16,104로 고정한다. 모든 범위는 zero-based 반열림 구간이다.

```text
donor TRAIN     [0,     15576)       공통 LoRA / H2 학습
DEV ADAPT       [14808, 15576)       32일 개인화
DEV CAL         [15576, 15768)        8일 출력 보정
DEV VALIDATION  [15768, 16104)       14일 선택, 성능 사전고정
EVAL ADAPT      [15144, 15912)       32일 개인화
EVAL CAL        [15912, 16104)        8일 출력 보정
EVAL TEST       [16104, 16440)       14일 봉인 평가
```

DEV 선택이 끝난 시점 E에서 EVAL의 허용 과거 이력만 개인화·보정에 쓴다. EVAL의 과거를 공통 기저·tau·LR·구조 선택에 사용하지 않는다. TEST 중 모델과 보정은 갱신하지 않고, 각 forecast origin 직전까지 실제 관측된 context만 갱신한다. 최종 TEST target에 접근하기 전 모든 DEV 선택과 EVAL 계수를 hash 봉인한다.

H1 ADAPT는 두 개의 384시간 블록으로 나눈다. 각 블록의 첫 192시간은 context 준비, 다음 192시간에 target이 완전히 들어가는 origin만 사용한다. 학습 stride 6시간(블록당 29 origin), 평가/CAL stride 24시간. 두 블록의 전체 context+target support도 겹치지 않는다. donor 학습 origin도 stride 6, target은 TRAIN 안이다. 겹치는 학습 window를 독립 관측으로 세지 않는다. TEST는 계열당 14개의 겹치지 않는 24시간 target이다.

schedule은 학습 전 저장한다. donor의 1,024개 학습 예시는 계열당 정확히 128개이며 seed별 shuffle 후 각 계열의 적법한 origin을 균등 복원추출한다. H1 각 블록도 적법한 29개 origin에서 128개를 균등 복원추출한다. 역할·계열·블록을 RNG key에 포함한다. mask는 `peft-gonogo12-v1|role|series|origin|repeat|kind`의 SHA-256 앞 8바이트를 little-endian 정수로 읽어 NumPy default_rng seed로 쓴다. 학습 mask key에는 fit seed도 포함하고, CAL/DEV/TEST mask는 fit seed와 무관하게 고정한다. 실제 tuple·mask 목록과 SHA를 보관한다.

스케일 sigma는 해당 계열의 허용 ADAPT 768시간 표준편차(ddof=0), donor는 donor TRAIN 표준편차다. 비유한 값·0분산·schema 불일치는 DATA_FAILURE로 기록하고 임의 대체하지 않는다. 원본은 행 순서의 시간축이며 달력·timezone을 만들어 붙이지 않는다.

## P0: 가장 먼저 하는 작은 사전 검사

공통: raw/model/source SHA, 환경, 실행 충돌, native prediction parity(유한 입력과 결측 입력), 실제 trainable parameter 목록을 검사한다. dropout을 끄고 normalized forecast에서 atol=rtol=1e-5 parity를 요구한다. 실패하면 학습하지 않는다. raw/model weights는 계속 ignored cache에 둔다.

H1 문제 검사에는 **DEV ADAPT만** 사용한다. 두 블록의 일별 F0 예측에 각각 affine bias/scale을 맞춘 뒤, sigma로 나눈 hour-of-horizon 평균 잔차 24개를 만든다. 각 블록에서 4계열 평균 profile을 빼서 공통 패턴을 제거한다. 최소 2/4계열에서 두 profile의 cosine>0이고 양쪽 RMS>=0.02이면 통과한다. 그렇지 않으면 `STOP_SETUP_H1_NO_STABLE_HETEROGENEITY`. 이 값은 비용 선별용 임계값이며 모든 개인화 필요의 필요조건이나 통계적 유의성 기준이 아니다. 거짓 기각 가능성을 보고한다.

H2 문제 검사는 동일 DEV ADAPT의 F0 예측으로 수행한다. 아래 고정 BLOCK48에서 clean 대비 점수가 평균 1% 이상 나빠지고, 안전한 보간으로도 clean 대비 오차 차이가 0.5% 이상 남을 때만 학습한다. 그 외는 `STOP_SETUP_H2_NO_UNRESOLVED_MASK_GAP`. 평균적으로 부분 관측 patch가 1개 미만인 경우도 현재 구조의 적용 조건 부족으로 중단한다. clean 예측은 사전 문제 진단용이며 실제 결측 forecast의 입력으로 쓰지 않는다.

이후 최대 18 smoke updates로 학습 경로만 검사한다. H1 shared/coefficient/ridge/local 4 code path ×2, H2 5 arms ×2. 별도 폐기 상태에서 실행하고 본학습을 새로 초기화한다. finite loss, 실제 nonzero gradient, frozen weight 보존, optimizer 대상 일치를 검증한다. 다른 layer·LR·수식으로 바꾸는 것은 구현 수정이 아니므로 자동 실행하지 않는다.

## H1: 두 블록 개인화와 불안정성 축소

반복 seed는 92251/92252. seed마다 donor 8계열로 shared LoRA를 256 updates 학습한다. AdamW LR=1e-4, batch=4, wd=0, clip=1, betas=(.9,.999), eps=1e-8, scheduler 없음. 그 밖의 가중치는 동결한다. checkpoint는 0/128/256을 기록하되 **256을 고정 사용**한다. 중간 checkpoint는 학습 진행 진단용이다.

각 q/v 업데이트의 rank-8 SVD에서 U,V,s를 고정한다. c=||s||2/sqrt(8)로 두고 `W_i=W0+U diag(s+c*d_i) V^T`로 개인화한다. c=0인 전체 기저는 식별 불가능한 상태로 중단한다. 전형적으로 모듈당 8개 계수이며 실제 전체 수를 기록한다. 시작 d=0의 예측이 shared LoRA와 일치해야 한다.

DEV 4 + EVAL 8계열 각각에 아래 **4개의 실제 micro-fit**을 수행한다. 모델은 메모리에서 재사용하되 매 fit 가중치/optimizer를 정확히 복원한다.

- BLOCK_A: 첫 블록만, d=0부터 32 updates.
- BLOCK_B: 둘째 블록만, d=0부터 32 updates.
- RIDGE: 두 블록의 같은 전체 256개 학습 예시 노출, 64 updates. loss에 `0.01*mean(d^2)` 추가.
- LOCAL_LORA: shared A/B에서 출발해 독립 local q/v LoRA를 64 updates. 같은 전체 예시 노출.

coefficient/ridge LR=0.01, LOCAL LoRA LR=1e-4, 나머지는 shared optimizer와 같다. 차원·크기가 다른 파라미터의 고정 초기 recipe이며 최적이라고 주장하지 않는다. 블록별 128개 예시 순서를 봉인하고, RIDGE/LOCAL은 두 순서의 합집합 256개를 고정 interleave해 쓴다. seed가 같으면 대조군의 데이터 노출이 같다.

`dbar=(dA+dB)/2`, `v_i=mean((dA-dB)^2)/4`. DEV 계수만으로 `tau2=max(mean_DEV(mean(dbar^2)-v_i),0)`를 추정한다. `g_i=tau2/(tau2+v_i)`이며 둘 다 0이면 g=0. 제안 U_SHRINK는 `g_i*dbar`를 적용한다. 블록 차이는 추정 잡음과 실제 변화가 섞인 **불안정성 proxy**이며 정확한 posterior variance가 아니다.

필수 비교는 F0, SHARED, LOCAL_LORA, RIDGE, UNSHRUNK(dbar), FIXED_SHRINK(g*dbar), U_SHRINK다. FIXED_SHRINK의 전역 g는 {0,.25,.5,1} 중 DEV 평균으로 고르고 동률은 작은 g다. U는 추가 search 없이 위 추정식 하나를 쓴다. 이는 제안보다 단순 대조에 더 강한 선택 기회를 주는 보수적 비교다. 모든 대조와 U에 아래 동일한 affine 보정 기회를 준다.

추가 fit 없이 고정 순환 교환 대조를 만든다. EVAL hash 순서에서 각 계열의 g를 다음 계열 g로 바꾸되 dbar는 유지한다. g의 올바른 계열 대응이 필요한지 검사한다. TEST 효과를 보고 순열을 선택하지 않는다. g 분포가 거의 같으면 그 사실도 보고하고 uncertainty 메커니즘이 식별됐다고 말하지 않는다.

## H2: 관측 위치 집합으로 마지막 encoder attention만 조정

반복 seed는 92261/92262. 모두 같은 pretrained F0에서 시작한다. 학습은 donor TRAIN, 256 updates, batch 4다. origin/data/mask schedule은 같은 seed의 모든 arm에서 동일하다. 25% clean, 75% IID48(192개 중 정확히 48개 누락)의 고정 schedule을 쓴다. 자연 결측을 흉내 냈다는 일반화는 하지 않는다.

5개 학습 arm을 비교한다.

- QV_LORA: 위 rank-8 q/v LoRA, LR=1e-4.
- SET_BIAS: 마지막 encoder self-attention logits에만 관측집합 보정. head별 3개 alpha, 총 8×3=24개, LR=0.01.
- CENTROID_BIAS: 같은 24개 alpha로 평균 관측시각 거리 RBF와 완전관측 평균시각 거리 RBF의 차이를 조합.
- KEY_BIAS: 같은 24개 alpha로 key별 관측비율-1, centroid 이동/16, 마지막 관측시각 이동/16을 조합. full patch에서 0.
- GENERIC_BIAS: 같은 24개 alpha로 완전 격자의 patch centroid 거리 RBF 3개를 조합하고, 입력에 결측이 있을 때만 활성화.

bias arm은 pretrained weights 전체를 동결하고 alpha=0으로 시작한다. 마지막 encoder layer 외에 native relative bias와 mask를 바꾸지 않는다. tau는 {16,64,192}시간으로 고정한다. SET의 정의는 아래와 같으며 log-space에서 계산한다.

`K_tau(S,T)=mean_{u in S,v in T} exp(-(u-v)^2/(2*tau^2))`

`delta_b[h,i,j]=sum_k alpha[h,k]*(log K_tau(O_i,O_j)-log K_tau(G_i,G_j))`

O는 실제 관측 위치, G는 원래 16개 격자다. REG 행/열 및 완전 결측 query/key의 보정은 0으로 둔다. 완전 결측 key는 기존 native attention mask가 제외한다. decoder는 전체 encoder hidden states에 cross-attend하므로 patch 보정의 예측 경로가 있으나 실제 nonzero gradient를 smoke에서 확인해야 한다. masked key를 유한 보정으로 다시 살리거나 위치 편향을 다른 layer에 중복 더하면 구현 실패다.

중요한 반례: **누락이 patch 단위에 정확히 정렬되면 남은 O=G이므로 SET은 F0와 같다.** 이 구조는 모든 결측을 해결하지 못한다. 아래 ALIGNED48에서 학습 후 native F0와 parity를 검사하고 효과 0을 정상적인 구조 한계로 보고한다. clean에서도 SET/CENTROID/KEY/GENERIC은 정확히 F0를 보존해야 한다.

비학습 대조는 F0_NATIVE와 F0_INTERP. 보간은 forecast origin 이전의 남은 값만 써서 interior linear interpolation, 양끝은 가장 가까운 관측값으로 연장한다. INTERP는 채운 값을 실제 입력으로 사용하도록 native input mask를 all-observed로 준다. 원래 mask는 감사용으로 보관한다. native 구현은 원래 mask가 0인 값 자체를 지우므로 '보간+원래 mask'라고 잘못 구현하지 않는다. 별도 mask embedding을 추가하지 않으며, 정보 원천은 같은 관측 context다.

H2 VALIDATION/CAL은 IID48와 clean만 사용하고 두 조건에 각 1/2 가중치를 준다. IID48 내부 mask 반복은 균등 가중한다. TEST primary는 BLOCK48: [0,144] 중 균등한 시작점에서 48시간 연속 누락, series/origin별 사전고정 hash의 2개 mask 반복을 평균한다. 시작점을 patch 비정렬로 골라 유리하게 만들지 않는다. secondary는 IID48, 마지막 48시간 누락(ALIGNED48), clean. 미래 target은 항상 같은 완전관측 truth이며 context의 가려진 truth는 모델·보간·normalization 입력에서 제외한다. 실제 자연 결측 자료 검증은 후속 별도 작업이다.

## 손실·보정·선택·평가

학습 loss와 primary score는 `mean_q,h(2*pinball(y-qhat))/sigma`이며 계열을 균등 가중한다. loss는 native quantile-labelled output을 쓰고, 채점은 모든 arm에 동일한 분위수 정렬을 적용한다. crossing 비율을 따로 남긴다. median MAE/sigma, raw MAE, 80% coverage, interval width도 보조로 보고한다. 후보1과 후보2의 점수를 서로 합쳐 한 승자로 만들지 않는다.

모든 forecast arm에 raw 또는 affine-CAL 선택을 준다. CAL의 median과 truth로 OLS slope를 계산하고 a를 [.5,1.5]로 제한한 뒤 b=mean(y)-a*mean(median)로 맞춘다. 모든 quantile에 a*q+b를 적용한다. median 분산이 0이면 a=1이다. H1은 계열별 8일 clean CAL, H2는 계열별 8일 IID48/clean CAL을 사용한다. mask 반복이 독립 target 수를 늘리지는 않는다. TEST에서는 a,b를 갱신하지 않는다.

raw/CAL 여부는 arm별 DEV 평균(두 반복 seed 포함)으로 한 번 선택하고 EVAL 전체에 적용한다. 동률이면 raw. H1 FIXED는 4개 g와 raw/CAL의 8개 조합을 DEV에서 공동 선택하며 동률이면 작은 g, 그다음 raw다. candidate를 제외한 baseline의 DEV winner를 후보별 한 번 고정하고 이름을 봉인한다. baseline 동률이면 위 나열 순서(H2는 F0_NATIVE, F0_INTERP, QV_LORA, CENTROID, KEY, GENERIC)로 정한다. 별도 선택 전용 seed는 없으며, 두 seed 모두 EVAL 반복이다. 모든 EVAL baseline 점수도 공개해 DEV winner가 TEST 최강은 아니었을 가능성을 숨기지 않는다. 모든 calibration·g·baseline 선택 전에 EVAL forecast score에 접근하지 않는다.

cached Chronos-2 official direct 24-hour forecast도 같은 CAL 기회의 비학습 실용 참조로 기록한다. 학습 arm 메커니즘 대조와 분리하며, 없거나 native missing parity를 확인하지 못하면 그 범위의 실용 우위는 UNVERIFIED다. 이를 이유로 새 모델을 다운로드하거나 구현을 임의 변형하지 않는다.

효과 `Delta=100*(score_baseline-score_candidate)/score_baseline`를 seed별·계열별로 남긴다. TEST 14개 날짜의 길이 3 noncircular moving-block bootstrap 2,000회, RNG=92270으로 95% 구간을 계산한다. 모든 계열·seed·mask를 같은 날짜 index로 묶어 뽑는다. CI는 이 패널과 두 학습에 조건부이며, 독립 8계열×14일로 N을 부풀리지 않는다. seed 두 개로 학습 변동의 모집단 신뢰구간을 주장하지 않는다.

## 사전고정 GO / NO-GO

GO_SCREEN 조건은 다음을 모두 만족하는 것이다. 임계값은 작은 후속 검증의 자원 배분 기준이며 이론적 최소 유효 효과가 아니다.

1. 선택된 Bolt 계열 강한 baseline 대비 두 seed 모두 Delta>0, seed 평균 Delta>=1%.
2. 8 EVAL 계열 중 최소 5개에서 seed 평균 점수가 그 baseline보다 좋음.
3. H1은 FIXED_SHRINK와 UNSHRUNK 각각 대비 평균 >=0.5%, 순환 교환 g 대비 평균 >0. H2는 KEY/CENTROID/GENERIC 각각 대비 평균 >=0.5%.
4. 후보보다 평균 1% 넘게 좋은 다른 사전지정 Bolt baseline이 없어야 함. TEST로 winner를 다시 고르는 것이 아니라 사전고정한 지배 여부 검사다.
5. 아래 UNDERTRAINED flag가 없고, 수치·누수·parameter 검사가 통과함.

95% CI가 0을 포함한다고 자동 기각하지 않는다. 그 경우 GO라도 반드시 '불확실성이 큰 탐색 신호'로 표시한다. H1은 shared보다 나쁜 계열 비율도 보고하되, 이를 줄이지 못하면 개인화 안전성 개선 주장을 하지 않는다. Chronos-2보다도 평균 1% 이상 좋고 두 seed 방향이 양수인 경우에만 `PRACTICAL_REFERENCE_GAIN`을 붙인다. 그 외 GO는 `MECHANISM_SIGNAL_ONLY`이며 실용 우위 확보로 발표하지 않는다.

건강한 실행에서 primary baseline 또는 필수 메커니즘 대조에 두 seed 모두 Delta<=0이면 `NO_GO_CURRENT_RECIPE`. 그 외 GO 조건을 충족하지 못한 경우(양수지만 작음, seed 부호 혼재, 메커니즘 기준 미달 등)는 `HOLD_INCONCLUSIVE`로 구분하고 자동 확장하지 않는다. CI 폭 단독으로 GO 조건을 취소하지 않는다.

UNDERTRAINED: 후보와 DEV 최강 학습 대조의 DEV score가 0→중간→끝에서 모두 엄격 감소하고, 각각 마지막 구간 개선이 >=1%이면 `HOLD_BUDGET_LIMITED`를 우선한다. H1은 각 block의 0/16/32 조합, RIDGE/LOCAL은 0/32/64, H2는 0/128/256을 사용한다. TEST를 보고 더 학습하지 않는다. 이 조건이 없다고 수렴이 증명되는 것은 아니다.

IMPLEMENTATION_FAILURE, DATA_FAILURE, RESOURCE_BLOCK은 위 과학적 판정과 분리한다. 학습 시작 후 구현 오류를 찾으면 해당 결과는 무효로 표시하고 실패 update도 장부에 남긴다. 남은 예산 내 의미 보존 버그 수정만 허용하며 데이터/선택 오염이 생기면 중단한다. 후보·LR·rank·loss·mask family를 바꾸어 실패를 복구하지 않는다.

## 예산과 종료

```text
H1 shared       2 fits × 256                         =   512 updates
H1 micro        12계열 × 2seed × (32+32+64+64)        = 4,608 updates / 96 fits
H2              5 arms × 2seed × 256                 = 2,560 updates / 10 fits
합계            108 actual main fit attempts          = 7,680 main updates
smoke           9 code paths × 2                    =    18 별도 updates
```

개별 계열의 짧은 fit을 하나의 job으로 뭉쳐 fit 수를 축소 보고하지 않는다. 실패·재시도도 상한에 포함한다. H1의 variance 추정은 두 block-fit 안에 포함되며 숨은 bootstrap 학습은 없다. 모든 optimizer.step 호출 전에 예약 장부를 갱신한다. 사전 검사 탈락 또는 실행 중단 시 남은 예산을 다른 후보에 넘기지 않는다.

후보별 GPU 단계 wall 상한 45분, 합계 90분이다. 이는 구현 시간을 제외한 학습·추론·측정 상한이지 ETA가 아니다. 실제 smoke에서 초당 처리량과 평가 호출 수를 측정해 상한 내 완료 가능성을 기록한다. 초과 예상이면 RESOURCE_BLOCK으로 멈추고 sample/arm을 자동 줄이지 않는다. GPU 작업은 한 번에 하나만 수행하며 사용자 프로세스를 종료하지 않는다.

예상 산출물은 사전 봉인/환경/데이터/파라미터 parity, fit/update 장부, raw·CAL·seed·series·mask·origin 점수표, 자원표(공통 기저와 개인화·분산 추정 비용 포함), CI와 learning trace, REPORT_KO.md/FINAL_DECISION.md다. 도표는 seed 효과와 단순 대조 대비 효과, 비용을 보여준다. raw·weights·예측 배열은 ignored cache에 두고 SHA manifest만 push한다. 주요 단계마다 scoped commit/push하며 두 후보 판정 후 자동 후속 없이 종료한다.

현재는 설계·기존 cache checksum·예산/분할 산술 검사만 완료했다. 학습 구현이나 실제 GO/NO-GO 결과는 아직 없다. 기존 후보 검토에서 실행 가능한 작은 설계로 구체화한 것이며, 성공 시에도 MixFT/LiFT/시간 bias 선행과의 차이를 별도로 확인해야 한다.
