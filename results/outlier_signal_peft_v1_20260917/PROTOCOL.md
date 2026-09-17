입력 오류 강건성 / 지속 변화 보존 PEFT — 단일 실행 계약
버전: outlier_signal_peft_v1_20260917
작성일: 2026-09-17 KST
대상 저장소: CanelE452/tsfm-peft-method-screen
확인한 기준 commit: 72eda9650a121d484f5246da725d4b6a6ed5733a

0. 범위와 최상위 목표

[설계] 이번에는 직전 추천의 1순위만 수행한다.
“큰 입력 오류가 미래 예측을 훼손하는 것을 줄이면서, 관측된 지속적 변화에
대응할 정보를 버리지 않는 작은 적응이 표준 LoRA와 강건한 전처리보다 유용한가?”

소비처: 사용자와 지도교수가 이 문제에서 추가 방법론 개발을 할 근거가 있는지 판단한다.
시점별 미래 예측이 주목적이다. 이상탐지 정확도, 최저 메모리, rollout calibration으로
중간에 목표를 바꾸지 않는다. 정확도와 강건성의 손익을 함께 보고한다.

이번에 하지 않는 것:
- 2순위 자기회귀 rollout·불확실성 전파, 일회성 사건 텍스트 연구.
- 이전 날짜 다양성 4개 차단 트랙 재개.
- PRIOR/SIDE/BASIS/Query, 검색·이력압축·센서지연·관측간격 연구의 변형 추가.
- 새 데이터셋이나 백본을 성능이 나올 때까지 교체.
- 본 파일을 이전 CLI 계약과 합쳐 실행.

[확인] 직전 주장은 Chronos와 Chronos-Bolt의 입력 표현 편향을 분석한 선행에서 왔다.
따라서 처음부터 Chronos-2에 그대로 옮겨 적용하지 않는다. 본 파일럿은 선행의 공개
Chronos-Bolt-small을 사용한다. 기존 Chronos-2 결과·가중치를 같은 초기 모델로 재사용하지 않는다.

1. 실행 전 기대와 반례

[미검증 가설]
강건한 전처리로 극단적 값의 영향을 줄이되, 제거된 차이를 제한된 크기의 표현으로
남기면, 전처리가 버린 유용한 지속 변화 정보를 일부 회수할 수 있다.
이것은 이미 확인한 결과가 아니며 새 알고리즘의 신규성도 확정되지 않았다.

예상 결과 / 목적과의 연결 / 반례:
A) 오류 입력에서만 좋아지고 무변형·지속 변화에서 나빠진다.
   -> 강건성-정확도 손익. “실제 변화를 보존했다”는 목적은 지지하지 못한다.
B) 증강 LoRA나 Hampel+LoRA가 같은 이득을 준다.
   -> 문제는 있을 수 있으나 새 경로의 필요성은 약하다.
C) 같은 크기의 일반 어댑터와 차이가 없다.
   -> 제거된 정보의 복원이 아니라 추가 용량으로 설명될 수 있다.
D) 제거된 정보 경로가 가까운 단순 대안을 넘고 지속 변화의 손해도 줄인다.
   -> 한정된 개발 근거. 실제 오류/사건 레이블·독립 원천·정식 선행 비교는 여전히 필요하다.
E) 공개 모델에서 선행의 민감성을 재현하지 못한다.
   -> 원문 설정과 차이를 기록한다. 거짓 재현 성공으로 쓰지 않는다.
   -> 참조 모델·입력이 정상임이 확인되면 실제 데이터 직접 비교는 수행할 수 있다.
      간접 민감도 점수를 본학습의 성능 입장권으로 쓰지 않는다.

2. 먼저 알아야 하는 식별성 한계

값만 보아 “측정 오류”와 “실제 상태 변화”를 항상 구별할 수는 없다.
동일한 관측 과거에 다른 미래 정답을 붙이면 어떤 결정적 예측기도 둘을 구별하지 못한다.
예: 직전 32시점에 동일한 +delta가 관측되었으나,
(1) 측정 편향은 예측 원점에서 끝나고 실제 미래는 원래 값인 경우,
(2) 실제 레벨 변화가 미래에도 이어지는 경우.
관측 과거가 같으면 같은 모델 출력이어야 한다. 이를 oracle case_type으로 구분하면 누출이다.

이번 직접 비교는 지속 길이·주변 관측 등에 통계적 단서가 있는 모사 조건이다.
reference_core의 동일 과거/다른 미래 예시는 정보 한계 검사로만 사용한다.
이 예시의 두 정답을 모두 맞추라는 PASS 기준을 만들지 않는다.

[설계] 실제 오류·급변의 전문가 레이블을 이번에 확보한 것은 아니다.
- 원자료 그대로 = UNMODIFIED_REFERENCE. 완벽하게 깨끗하다고 부르지 않는다.
- 입력에만 주입한 오류 = SYNTHETIC_MEASUREMENT_FAULT.
- 과거 끝과 미래에 함께 더한 지속 변화 = SYNTHETIC_PERSISTENT_SHIFT.
- 원자료의 큰 과거 변화로 분류한 부분 = HISTORY_TRIGGERED_SUBSET.
  실제 고장/사건의 확정 레이블이 아니다.
이 네 가지를 한국어 REPORT에서도 그대로 구분한다.

3. 선행·소스 경계

읽고 출처·commit·변경점을 literature_boundary.md에 기록한다.

[S1] Understanding the Implicit Biases of Design Choices for Time Series Foundation Models
     (2026 / ICLR)
https://proceedings.iclr.cc/paper_files/paper/2026/hash/753d9584b57ba01a10482f1ea7734a89-Abstract-Conference.html
https://arxiv.org/html/2510.19236v1
https://github.com/amazon-science/TSFM-Biases
공개 노트북: notebooks/outlier-bias.ipynb
확인 blob SHA: 723583c1be813c54d90886e73837a543d1772580
[확인] 노트북에는 amazon/chronos-t5-small, amazon/chronos-bolt-small과
별도 pretrained patch-size-1 모델 placeholder가 있다.
[설계] 본 작업은 공개 두 모델의 제한된 outlier 비교만 재현한다.
patch-size-1 모델을 임의로 초기화하거나 새로 사전학습해 채우지 않는다.
전체 Figure 8·논문 전체 재현이라고 부르지 않는다.

[S2] Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared
     Time Series Foundation Models (2026 / ICLR)
https://proceedings.iclr.cc/paper_files/paper/2026/hash/48c5226582f41254026748c7e35d4ac2-Abstract-Conference.html
https://arxiv.org/html/2603.00629v1
https://github.com/thulab/TATO
[확인] 문맥·정규화·이상값 보정의 데이터 변환 검색이라는 직접 선행이다.
[설계] 아래 CLIP/HAMPEL은 명시된 단순 대조이며 TATO 전체 재현이 아니다.
TATO 공식 검색·전체 파이프라인을 실행하지 않고 “TATO를 이겼다”고 쓰지 않는다.
공식 코드의 현재 모델 연결·검색 계약을 조사하고 남은 정식 비교를 기록하되,
이 파일럿 도중 수십 가지 TATO 변환이나 새로운 backbone을 자동 추가하지 않는다.

[S3] 모델 카드·실제 forward
https://huggingface.co/amazon/chronos-bolt-small
https://huggingface.co/amazon/chronos-bolt-small/blob/main/config.json
https://github.com/amazon-science/chronos-forecasting/blob/4dbf163c2734c089cdf7da2b86fde48862ff9c6f/src/chronos/chronos_bolt.py
[확인] 본문 작성 시 확인한 config: context 최대2048, patch16/stride16,
prediction_length64, quantiles 0.1~0.9의9개, d_model512,
encoder6/decoder6층, heads8, d_kv64, use_reg_token=True.
런타임에 실제 config와 revision을 다시 읽어 고정한다.

새 경로는 기존의 residual adapter·강건 전처리·bounded representation과 가까울 수 있다.
“최초의 이상값 PEFT”, “독창성 확보”, “논문 PASS”를 전제하지 않는다.
분류·이상탐지 논문의 효과를 미래 예측 효과로 옮기지 않는다.

4. 저장소·환경 감사

git HEAD, dirty files, 원격 이후 변경, 동일 실험 존재 여부를 먼저 확인한다.
이전 결과·사용자 변경·원자료·모델 cache를 삭제하거나 reset하지 않는다.
최신 작업이 같은 계약을 이미 수행했으면 해시와 정보 권한을 확인하고 중복을 피한다.
단순히 제목이 비슷하다는 이유로 재사용하지 않는다.

새로 만들 경로(기존에 있다고 가정하지 않는다):
experiments/outlier_signal_peft_v1_20260917/
results/outlier_signal_peft_v1_20260917/
.cache/outlier_signal_peft_v1_20260917/
scripts/run_outlier_signal_peft.py

원래 condition_sampling_repair/history_compression은 read-only reference다.
기존 GPU guard, journal, hash, causal split 검사만 계약을 읽고 재사용한다.
Chronos-2 전용 21분위수·group_ids·세 미래 패치 가정은 Bolt에 이식하지 않는다.

모델·라이브러리:
- main = amazon/chronos-bolt-small.
- reference-only = amazon/chronos-t5-small.
- 실제 모델 snapshot revision, safetensor hash, chronos-forecasting/transformers/torch
  버전을 기록한다. CLI가 설치된 안정 버전을 먼저 검토하고 변경은 별도 환경에서 한다.
- 이 문서는 관찰한 upstream source commit을 참고점으로 제공한다.
  무조건 main 최신 버전으로 pip upgrade하지 않는다.
- main 모델의 네이티브 예측 길이64와 patch16이 다르면 모델을 바꾸지 말고
  BLOCKED_MODEL_CONTRACT로 명확히 기록한다.
- FP32, TF32 off. 추론 eval mode. 적응 시 모든 dropout을0으로 고정한다.
  이는 이번 비교의 선택이며 원문 학습 recipe를 재현했다는 뜻이 아니다.

5. 데이터 역할과 표본 선정

[설계] 두 source를 사용한다.
1) 기존 Electricity 시간별 원자료.
2) 기존 ETTm1 15분 원자료.
저장소의 현재 data receipt와 loader를 읽고 정확한 파일 경로·hash를 확인한다.
보지 않은 컬럼 이름을 추측하지 않는다. 입력 오류를 만들려고 원자료 파일 자체를 수정하지 않는다.
source별로 학습하며 두 source의 데이터를 한 모델에 pooled training하지 않는다.

각 source:
- numeric target 중 TRAIN 구간 유한·표준편차>1e-6인 첫4개를 원본 컬럼 순서로 선택.
- 해당 기준은 magnitude 성능 선별이 아니라 기본 데이터 적격성이다. 임의 이상값 제거 금지.
- timestamp 컬럼이 있으면 단조·격자·중복을 확인한다.
- Electricity에 실제 시간대 정보가 없으면 index-day/phase라고 쓴다.
- 모든 계열은 단변량 예제로 batch 처리한다. Bolt는 이번 실험에서 채널 간 정보를 읽지 않는다.

시간 순서 split:
TRAIN=[0, floor(.6N))
V_SELECT=[floor(.6N), floor(.8N))
E_DISCOVERY=[floor(.8N), N)
경계는 모든 군에서 같다. 길이512 과거, 길이64 미래.
과거 문맥은 원점 이전에 관측된 값을 사용할 수 있고,
각 예측 정답은 자신의 role 안에 완전히 들어가야 한다.
이전 역할의 관측이 나중 시점의 입력에 들어오는 것은 합법이다.

관측512/미래64의 실제 시간:
Electricity = 과거512시간 / 미래64시간.
ETTm1 = 과거128시간 / 미래16시간.
동일한 물리 예측 길이라고 묶지 않는다.

후보 origin은 네 선택 채널의 입력·정답이 유한한 완전한 창만 포함한다.
정답의 “결측 여부”는 complete-case eligibility에만 사용하며 값의 크기·예측오차로
날짜를 선택하지 않는다. 변형 뒤의 값으로 후보 origin을 다시 선정하지 않는다.

원점 목표:
TRAIN256일, V_SELECT64일, E_DISCOVERY128일; 하루에 origin 하나.
source/role별 다음 알고리즘을 reference_core와 일치하게 구현한다.
1) 모든 합법적 origin을 만든다.
2) 하루의 모든 phase가 가능한 full eligible day만 남긴다.
   경계의 일부만 합법적인 날짜는 제외한다. 최대/최소 phase를 원형거리로 대체하지 않는다.
3) full eligible day 정렬목록을 목표 n개 구간으로 나눠 각 구간 중간의 날짜 하나를 선택.
   index=floor((k+.5)*D/n). 날짜는 서로 다르다.
4) phase_k=floor(k*P/n)의 균형 roster를 만든다. P=24 또는96.
5) source/role별 미리 고정한 seed로 roster를 한 번 섞어 날짜에 대응한다.
   날짜 순서와 phase가 같이 증가하는 이전 문제를 반복하지 않는다.
6) 실제 origin=day*P+phase. 모든 값이 후보목록에 들어가는지 검사한다.

이 절차는 full day만 써서 phase count max-min<=1을 구성상 보장한다.
신뢰성 없는 1~4회 경계 문제를 검출만 하고 멈추는 것이 아니라 발생하지 않게 한다.
실제 유한한 full days가 목표보다 적으면 INSUFFICIENT_FULL_DAYS로 source 범위를 기록한다.
같은 날짜의 여러 창으로 숫자를 채우거나 평가를 보고 다른 source로 교체하지 않는다.
두 원자료의 명목 길이26304/69680에서 목표 수가 가능한 것은 합성 metadata 검사로 확인했다.
실제 데이터 missingness까지 확인했다는 뜻은 아니다.

학습 전 보고/봉인:
origin 목록, distinct days/weeks, phase histogram, 날짜 구간별 count,
origin span, 정답 시각 unique/total, overlap histogram, day-phase correlation.
이때 미래 수치나 모델 점수는 사용하지 않는다.
샘플을 보고 임의의 정밀 균형 문턱을 새로 만들지 않는다.

6. 정보 권한과 스케일

sigma_c = 원자료 TRAIN 전체 허용 값의 population std(ddof=0), 채널별로 고정.
모든 loss/metric의 분모는 이 sigma_c를 사용한다. 오염된 창마다 달라지는 native scale을
loss 가중치로 사용하면 전처리 효과와 loss 재가중 효과가 섞이므로 이번에는 피한다.

모사 생성기의 r0 = max(1.4826*median(|x0-median(x0)|), .1*sigma_c).
x0는 그 원점의 변형 전 과거 문맥. 미래 y는 r0 계산에 사용하지 않는다.
모사 severity를 정하는 r0와 원래 오류 위치는 model input에 전달하지 않는다.

실제 전처리의 r = 같은 수식을 “모델이 받은 관측 문맥 x_obs”에서 다시 계산한다.
원래 오류 없는 x0, 오류 mask, scenario ID, true shift amplitude는 전처리에 넘기지 않는다.
native Bolt instance normalization은 전처리된 실제 입력에서 계산하고 출력도 그것으로 복원한다.

7. 선행 제한 재현 — 새로운 학습0회

먼저 S1 공개 notebook의 실제 코드 셀을 읽고 제한 재현을 한다.
- cos(t+.3), period160, context512, future64의 공개 예를 기준으로 한다.
- 오류 수 {1,16}, 상대 amplitude {1,10,100}, 난수8개: 48개 + 무변형1개.
- 오류는 과거에만 주입, 미래 cos 정답 유지.
- Bolt-small은 config의0.5 분위수, Chronos-t5-small은 고정 seed의20개 sample 중앙값.
  sample 수 변경 등 공개 notebook과 다른 설정은 재현 차이표에 기록한다.
- 가능하면 공식 magnitude sweep 구현과 동일한 위치/진폭 난수를 사용하고,
  정확히 동일하지 않으면 “reduced replication”이라고 표기한다.
- 별도 patch-size-1 모델 placeholder는 SKIPPED_UNAVAILABLE_REFERENCE로 기록한다.
- Chronos-t5-small 로더가 불가능하면 그 참조만 차단하고 본체를 바꾸지 않는다.
- model/pipeline 요청은 최대98개 series-forecast, optimizer0회.
  내부 autoregressive step은 별도이며 하나의 sample forecast와 혼동하지 않는다.
- 대표16개 입력에서 raw/clip6의 patch norm max/median을 기록해도 된다.
  이런 관찰을 이상값의 원인 전체에 대한 인과 증명으로 쓰지 않는다.

예측이 nonfinite하거나 계약을 잘못 연결했으면 구현 문제부터 고친다.
민감도나 개선율이 기대와 다르다는 이유만으로 실제 데이터 방법 비교를 막지는 않는다.

8. 학습 입력 4종과 평가 입력

기본 (x0,y0) 원점·채널을 모든 군이 공유한다.
각 epoch에서 각 예제는 다음4종을 순환한다.
state = [reference, point, burst, shift][(epoch+example_index)%4].
32epochs이면 각 예제마다 각 state8회다.
모든 방법·LR·optimizer seed가 같은 epoch/example의 모사 난수를 공유한다.
데이터 변형 seed와 optimizer seed는 분리한다.

학습 변형:
REFERENCE: x=x0, y=y0.
POINT: x0 중1/2/4개 위치에 ±8*r0를 더한다. y=y0.
BURST: 길이2/4의 연속 구간에 같은 부호의8*r0를 더한다. y=y0.
SHIFT: 문맥 마지막24/48개와 미래 전체에 같은 ±4*r0를 더한다.
       y=y0+delta. 단순 측정 오류가 아닌 지속 변화 모사다.
모든 위치·부호·길이는 과거와 사전 난수로 정한다. 미래값으로 변형 세기를 조정하지 않는다.

부호가 음수인 shift로 전력값이 음수가 되는 사례도 발생할 수 있다.
그것은 수학적 레벨변화 stress일 뿐 물리적으로 유효한 부하 사건이라고 부르지 않는다.
미래 최소값을 보고 delta를 바꾸거나 음수를 잘라 정답을 유리하게 바꾸지 않는다.
범위 밖 물리적 사례 수를 표시하고, 원자료 실증과 분리한다.

평가/검증의 고정 패널:
REFERENCE 1종.
POINT amplitude {4,8,16}, count2: 3종.
BURST amplitude {4,8,16}, duration8: 3종.
SHIFT amplitude {4,8}, duration32: 2종.
SHIFT_POINT: shift4/duration32 + point8/count2: 1종.
총10개 상태. 원점/채널/상태마다 독립 모사 seed2개를 고정한다.
REFERENCE는 한 번만 계산하고 논리적 반복을 참조해도 된다.
검증/평가 모사 seed는 TRAIN과 다르며, 평가용 seed로 설정을 선택하지 않는다.
원점마다 같은 모사 입력을 모든 방법과 optimizer seed에 제공한다.
숫자가 더 크다는 것을 실제 배포 발생률로 해석하지 않는다.

SHIFT와 오류를 섞은 state의 미래 정답은 shift가 적용된 y이며,
measurement fault를 추가로 주입해도 정답은 더 바뀌지 않아야 한다.

9. 비교군 6개 — 하나의 문제, 하나의 후보 변경

A0 NO_FAULT_AUG:
표준 LoRA. TRAIN point/burst slot에는 x0를 사용하고 나머지는 공통 REFERENCE/SHIFT 사용.
따라서 A1~A5와 정답 및 진짜 변화 모사의 노출 횟수는 같다.
측정 오류 증강만 없다는 참조군이다. 완전한 무증강 학습이라고 부르지 않는다.

A1 AUG:
표준 LoRA. 네 TRAIN 변형 모두 사용. 입력 전처리 없음.

A2 CLIP_AUG:
A1과 같은 변형. x_obs를 global median ± 6*r로 제한한 뒤 native model 입력.

A3 HAMPEL_AUG:
A1과 같은 변형. 길이25의 국소 window, median±4*local_scale을 벗어나면 local median으로 대체.
local_scale=max(1.4826*local MAD, .1*sigma_c).
원래 x_obs로 한 번만 검사한다. 한 점 대체 결과를 다음 점의 판정에 재사용하지 않는다.
중앙 window는 예측 원점 이전의 문맥 안에서만 구성한다.
과거 위치t의 오른쪽 이웃도 현재 예측 원점 이전이면 합법이다. 미래 정답을 붙이지 않는다.

A4 CLIP_GENERIC:
A2와 동일한 clip6. native input patch embedding 직후 작은 bounded residual adapter 추가.
아래 A5와 같은 크기이며, 깨끗하게 제한된 값에서 만든 일반 특징만 제공한다.

A5 CLIP_RESIDUAL:
A4와 동일한 모델·LoRA·adapter 크기·학습 조건.
다만 adapter에 전처리로 제거한 차이의 크기·부호 정보를 제공한다.
일부 값을 무조건 복원하는 것이 아니라, 제한된 임베딩 크기로 정보를 전달한다.
이 A5가 이번 [미검증 방법 후보]다.

무학습 대조(새 optimizer0):
FROZEN_RAW, FROZEN_CLIP6, FROZEN_HAMPEL, 원자료 간격에 맞는 직전 하루 반복.
같은 오류 입력에서 계산한다. 계절반복이 틀려도 대조군을 삭제하지 않는다.

10. A4/A5의 정확한 수식과 파라미터

P=16, native embedding dim d=512, side rank r=8.
clip된 문맥을 native instance norm한 patch 값 z_j ∈ R^P.
e_j = 기존 frozen input_patch_embedding([z_j, mask_j]) ∈ R^d.
이번 적격 입력은 유한해서 mask_j는1이지만 native mask 경로 자체는 보존한다.

discarded_j = patch((x_obs - clip6(x_obs))/r_observed).
r_observed는 x_obs에서 계산한 robust scale이며 모사 r0가 아니다.

A4 feature v_j = concat(clip(asinh(z_j),-6,6), tanh(z_j)).
A5 feature v_j = concat(clip(asinh(z_j),-6,6), clip(asinh(discarded_j),-6,6)).
공간은 모두 R^(2P)이다. A4는 정보가 제거된 값만,
A5는 raw 입력에 원래 있었으나 전처리가 지운 차이도 활용한다.

u_j = W_up GELU(W_down v_j + b_down) + b_up.
c = stop_gradient(max(median_j RMS(e_j), 1e-6)).
e'_j = e_j + c*tanh(u_j).
각 좌표의 추가 크기는 c 이하이고 RMS(delta_j)<=c다.
raw residual이 커질수록 embedding norm이 무한히 커지는 경로는 만들지 않는다.

W_down:8×32, bias8. W_up:512×8, bias512.
추가 파라미터=4,872개. W_up/b_up=0 초기화해서 step0에서 A4=A5=A2가 되어야 한다.
W_down은 같은 seed의 동일 초기값; generic/residual로 달라지는 입력 특징만 비교한다.
초기0에서는 down gradient가0일 수 있다. 한 번의 update에서 모든 gradient가 비영이어야
한다는 잘못된 gate를 만들지 말고 두 번 이상 후의 실제 파라미터 변화를 확인한다.

REG token에는 adapter를 적용하지 않는다.
모델 forward에서 raw x와 clipped x를 local arguments로 전달한다.
전역 mutable buffer/hook가 다음 batch의 raw residual을 잘못 참조하지 않도록 한다.
가능하면 encode를 명시적으로 감싼다. 매 forward에 patch count/mask/배치 일치 검사.

추가 진단은 저장된 선택 A5에 대해서만 V에서 수행:
- residual 특징을0으로 변경.
- 같은 문맥의 patch 순서로 residual만 고정 순열 변경.
새 optimizer0. 제거 시 나빠지는 것은 A5의 의존성이지 A4보다 우월하다는 증거가 아니다.
E에서 진단 결과를 보고 설정·선택을 변경하지 않는다.

11. LoRA와 학습 손실

LoRA 대상은 실제 모듈을 enumerate해서 encoder self-attention,
decoder self-attention/cross-attention의 q/v Linear만 선택한다.
rank8, alpha16, LoRA dropout0, bias학습없음. k/o/FFN/head/원래 embedding은 동결.
Q,V가512×512이고 각 attention을 포함하면36개 projection,
기본 LoRA 294,912개가 예상된다. 실제 모듈 이름/shape/count를 봉인한다.
모델 config가 달라 count가 맞지 않으면 무시하지 말고 연결을 조사한다.
A0~A3=294,912개, A4/A5=299,784개 예상.
A4/A5는 서로 정확히 동수. A1보다 추가 파라미터가 없는 방법이라고 주장하지 않는다.

학습 목적:
J = mean_{b,q,h} 2*rho_q(y_bh - pred_bqh) / sigma_train(channel_b).
rho_q(e)=max(q*e,(q-1)*e).
9개 native quantile levels를 config에서 읽고, 원단위로 복원된 예측으로 계산한다.

이것은 native와 같은 quantile loss 형태지만 native가 문맥 scale로 나누는 가중 방식과 다르다.
본체 model.loss를 그대로 호출해서 arm마다 scale weight가 달라지게 하지 않는다.
이 차이를 기록하고 이전 MSE 실험과 원점수를 비교하지 않는다.
CLIP 임계값6과 HAMPEL window25/임계값4는 이번 고정 대조 설정이다.
이를 해당 전처리의 최적 설정으로 주장하지 않는다. TATO 전체 검색과의 우열은 미검증이다.
학습 중 quantile을 정렬하지 않는다. 평가에서도 native0.5 위치를 point forecast로 쓰며,
quantile crossing 빈도를 별도 기록한다. 정렬 보정은 주결과에 추가하지 않는다.

12. 고정 학습 budget와 선택

source마다 TRAIN256 origins ×4채널 =1,024개의 기본 예제.
예제마다 epoch에 정해진 변형 하나만 사용한다.
effective batch32, 32updates/epoch, 32epochs=1,024updates/fit.
각 예제는 4종 state를8회씩 경험한다.
A0의 point/burst 입력만 원래 과거로 되돌리며 y와 예제 순서는 공유한다.

optimizer = AdamW, betas(.9,.999), eps1e-8, weight_decay0.
LR 후보 = {1e-4,3e-4}. scheduler없음, 전체 trainable global clip_norm1.
선택용 seed=81500; 반복 seed=81501,81502.
추가 adapter 초기화 때문에 공통 LoRA 초기값이나 배치 순서가 달라지면 안 된다.

checkpoint = {0,256,512,768,1024}.
각 source/arm의 LR는 선택용 seed V에서 고른다.
V 목적 = mean(nMAE[REFERENCE], nMAE[POINT8], nMAE[BURST8], nMAE[SHIFT4]).
이 비중은 stress 개발 선택용이며 실제 오류 발생률이 아니다.
tie: 더 작은 LR, 더 이른 step.
반복 seed에는 고른 LR만 사용하고 checkpoint는 같은 V 목적로 선택한다.
어떤 모델이 손해라고 조기중단해 다른 군보다 적게 학습시키지 않는다.
선택용 seed의 E를 최종 평균에 넣지 않는다.
최대 step에서 계속 좋아지면 OPTIMIZATION_LIMITED를 기록하되 자동 연장하지 않는다.

합계:
source2 × arm6 × (선택seed LR2 + 반복seed2) =48fits.
48×1024 =49,152 본학습 optimizer updates.
source2×arm6×2 =24 smoke updates. 총 optimizer 상한49,176.
추가 profiler/diagnostic에서 optimizer를 몰래 쓰지 않는다.
현재 값은 입력 오류 문제의 직접 비교를 위한 [설계]이며 문헌의 최적 학습량이 아니다.

메모리 문제:
effective batch32는 유지하고 필요시 microbatch16/8/4 중 실행 가능한 최대값을
동일source 모든 군에 공통 적용한다. GPU 적격성 확인에 full effective batch 한 번만 비교한다.
loss를 microbatch sample 수/effective batch 수로 가중한 후 한 optimizer update.
계산 shape를 바꿨다는 사실을 기록하되 fullbatch와 bitwise 같아야 한다는 Query gate를
가져오지 않는다. batch norm은 없음을 확인하고, 손실/gradient 누적 의미를 시험한다.
모든 군의 microbatch를 봉인한 뒤 학습한다. nonfinite/OOM은 실행 오류이지 성능 패배가 아니다.

13. 비교를 완료하는 순서

(1) source와 model 연결 감사 + 모든 source의 날짜/조건 명세 봉인.
(2) 참조 cosine 비교. 결과가 약해도 정상 구현이면 본 비교는 그대로 진행 가능.
(3) 전체 source/arm smoke와 미니배치 크기 고정.
(4) 모든 source의 선택용24fits.
(5) LR 고정 후 모든 source의 반복24fits.
(6) 선택·비교군·후처리없음 정책을 GLOBAL_EVALUATION_SEAL에 고정.
(7) E 예측 파일을 먼저 저장; 이후 scorer가 E 수치 정답을 채점.
(8) 동일선택 모델로 자원 측정, 독립 scalar 검산, 한국어 보고.

날짜 다양성·label권한·실제gradient 오류는 본학습 전에 해결해야 한다.
반면 F0 대비 성능, 논문재현 개선율, V효과 부호 같은 성능 gate로 군을 건너뛰지 않는다.
한 source가 입력 권한/데이터 부족으로 불가능하면 그 범위만 명시하고 나머지를 수행한다.
다른 데이터셋으로 자동 대체하지 않는다.

14. 평가할 것

주 관심: 센서오류 상태6종(POINT4/8/16, BURST4/8/16)의 nMAE 평균.
source별로 각 채널 평균 ->4채널 동가중 ->조건6종 동가중 -> optimizer seed2 동가중.
값이 커서 모든 실제 업무에서6조건이 동일빈도라는 뜻은 아니다.
원자료REFERENCE, SHIFT4/8, SHIFT_POINT 결과는 따로 제시한다.

nMAE = mean |median_prediction-y| / 고정TRAIN sigma_c.
secondary: raw MAE, 채널별 NRMSE, 고정TRAIN scale 2-pinball, quantile crossing.
원자료 nMAE와 오류 nMAE를 사후 유리한 비율로 섞어 우승자를 정하지 않는다.

필수 직접 대비:
A5/A4: 같은 크기 일반 경로를 넘는 residual 정보의 추가 가치.
A5/A2: clip으로 버린 정보의 복원 가치.
A5/A3: sustained change를 덜 없애는 국소 전처리보다도 필요한가?
A5/A1: 오류 증강 표준 LoRA보다도 유용한가?
A1/A0: 알고리즘이 아니라 오류를 학습한 것만의 효과.
A2,A3/A1: 전처리 자체의 효과.
FROZEN 계열과 계절반복은 참조. 그것만 이겨서 새 방법 성공으로 하지 않는다.

[설계] 원자료 HISTORY_TRIGGERED_SUBSET:
past score=abs(median(x[-32:])-median(x[-64:-32])) / robust_scale(x).
TRAIN 원점의 점수90분위수를 기준으로 정하고, E의 과거로만 subset 태그를 부여한다.
이 subset의 미래를 보고 태그·문턱·채널·날짜를 고르지 않는다.
작은 subset이면 원점수와 n만 보고; 새 가설을 확증했다고 쓰지 않는다.
이것은 실제 “정상 급변” 레이블이 아니므로 synthetic shift와 합산하지 않는다.

조건별 같은 base origin에 대응하는 paired error 차이/비율을 보고한다.
한 origin의 여러 오류seed·상태·채널은 독립 날짜로 세지 않는다.
7일 시간block을 함께 재표집하여2,000회 paired bootstrap.
Electricity block168slots, ETTm1 block672slots.
두 optimizer seed를 각 draw 안에서 평균한다. seed 모집단 CI를 확보했다고 말하지 않는다.
두 source 결과를 합친 하나의 작은 p-value를 main evidence로 쓰지 않는다.
서로 다른 source의 물리적 horizon 차이와 과거 공개자료 사용을 함께 적는다.

15. 가능한 결론과 그 한계

EXECUTION: COMPLETE / PARTIAL_DATA / BLOCKED_MODEL / INVALID_IMPLEMENTATION.
EVIDENCE는 수치 부호·구간·조건별 손익을 분리해 쓴다.
- METHOD_SIGNAL: A5가 가까운 대조보다 오류 예측을 개선하고,
  REFERENCE/SHIFT/NATURAL_SUBSET의 손해도 작거나 개선하는 일관된 개발 관찰.
- PREPROCESSING_OR_AUGMENTATION_SUFFICIENT: A1/A2/A3으로 같은 효과를 얻음.
- ROBUSTNESS_SIGNAL_WITH_SHIFT_COST: 오류는 좋지만 지속 변화 예측이 나빠짐.
- UNCERTAIN: 평균은 작게 좋거나 나쁘고 구간·source·seed가 불확실.
- NO_ADDED_METHOD_EVIDENCE: 후보의 특별한 경로가 직접 대조를 넘지 못함.
이 태그들은 1%를 넘으면 무조건승리 같은 숫자 자동 판정기가 아니다.

최종 보고에 소형 trade-off 표를 반드시 포함:
source별 A5 대 A1/A3/A4의 fault gain, reference gain,
shift gain, 자연 과거변화 subset gain, TRAIN/INFERENCE 비용.
“오류 감소”와 “변화 보존” 중 하나가 반대면 숨기지 않는다.
실제 사건 보호에 대한 주장은 이 파일럿만으로 확정할 수 없다.

어떤 결과든 신규성은 별도 판정한다.
좋으면 정식 TATO/다른 강건 방법 재현과 실제 품질 레이블이 있는 자료가 다음 단계다.
나쁘면 이 고정된 bounded residual 구현을 내리는 것이지,
모든 outlier 연구/모든 PEFT를 반증한 것이 아니다.
자동적인 2순위 rollout 전환, side rank 확대, beta 조정, threshold 탐색 확장 금지.

16. 최소 실제 correctness 검사

reference_core/test_reference는 CPU 예제다. 실제 모델·자료로 아래를 추가 확인한다.
- raw/original file hashes와 train/validation/evaluation 역할의 절대시간 분리.
- 모든 지표 스케일 TRAIN-only; 오류정도·시나리오·원래x0가 모델로 새지 않음.
- future y를 바꿔도 input/scenario transform/날짜/전처리/adapter feature가 동일.
- fault target 그대로, persistent shift는 x와y 일관 변경, mixed fault는 shifted y 유지.
- 모든 군의 label stream hash 동일(A0 포함), A1~A5의 augmentation seed 동일.
- native forward vs wrapper forward 같은입력 FP32 근접성.
- 추가 module step0 = CLIP 입력의 native+LoRA step0.
- 같은 입력에서 generic/residual 두 module의 초기 parameter hash 동일.
- q/v LoRA 모듈 이름/shape·원래가중치 불변·출력 head 불변.
- A4/A5 추가 파라미터의 gradient 유한·실제 변경. 초기zero의 정상적인 gradient를 구분.
- residual delta의RMS bound 검사; generator mask를 꺼내보지 않고 해석.
- 미니배치 순서·checkpoint 복원·seed RNG stream 보존.
- 입력8개에서 clip/hampel/feature를 독립 NumPy/torch로 대조.
- loss의scalar2pinball과native출력단위확인. 평균/합계 차이는 명시적으로 고정.
- 같은 과거/다른미래의 ambiguity 검사에서 모델 입력·출력은같음.

허용오차를 단계마다 무조건1e-10으로 적용하지 않는다.
CPU float64 scalar지표 rtol1e-10/atol1e-12, 동일FP32연산 parity는
TRAIN sigma로 정규화한 출력 maxdiff1e-5 또는 rtol1e-4를 사전에 사용한다.
failed numerical parity는 원인·연산을 조사하고, 단순 허용치 확대 반복 금지.
연산구현정상화와 성능비교의 실패를 분리한다.

17. 자원·재개

1GPU/1worker. 기존 외부작업 종료금지. RustDesk 등 사용자 허용 예외만 기존규칙대로.
48fits/49,152main updates/24smoke가 상한이다. inference reference는 별도.
원자료·가중치 download는공식공개범위, credential을소스나보고서에저장하지않음.
모델별학습peak allocated/reserved, optimizer time, validation time, checkpoint I/O를 분리한다.
실행 전체시간을optimizer시간이라고부르지않는다.
checkpoint는0/256/512/768/1024, 정확 resume은epoch끝32updates마다 보관한다.
매 update의 scalar journal은 append하고, 매 update 전체모델을디스크에저장하지않는다.
완료 업데이트의 정확한 복원근거가 없으면조용히재실행해학습예산을누락하지않는다.
cache상한50GiB, 새작업 hard wall safety24h. 예측완료시간약속이아니다.
장시간외부GPU점유/예산소진이면 PARTIAL과resume명령을남긴다.
미사용예산을새후보에사용하지않는다.

18. 산출물·runner

코드 경로는 source를 읽고 구현한다. 없는옵션/함수가 이미 있다고가정하지마라.
새runner action:
prepare / check / reproduce / train-select / train-repeat / evaluate / verify / report / status / run-all.
run-all은 유효한입력에서는 전체 승인비교까지실행한다. 계획서만쓰고종료하지않는다.

필수 산출물:
PROTOCOL.md + MACHINE_CONTRACT.json + SOURCE_MANIFEST.json
literature_boundary.md + reproduction_differences.md
DATA_MANIFEST.json + origins.csv + origin_audit.json
CONDITION_MANIFEST.json (generator-only audit와 model-input 파일 분리)
MODEL_RECEIPT.json + PARAMETER_RECEIPT.json
FIT_LEDGER.csv + optimizer.jsonl + LR_SELECTION.json + MODEL_SELECTION.json
GLOBAL_EVALUATION_SEAL.json + predictions_manifest.json
scores_by_origin.csv + scores_by_condition.csv + paired_effects.csv
resources.csv + implementation_checks.json + verification.json
REPORT.md + FINAL_DECISION.md

한국어REPORT의질문순서:
1) 과거입력의오류와유용한지속변화를왜구분하는가?
2) 선행에서확인된것과본파일럿이새롭게시험한것은무엇인가?
3) 원자료와모사변화·진짜사건레이블의차이는무엇인가?
4) 전처리·증강·일반adapter와후보의차이를어떻게분리했는가?
5) 오류예측·무변형·지속변화의원점수와손익은어떤가?
6) 실제업데이트·미실행부분·같은정보·연산검산이맞는가?
7) 추가방법개발근거가있는가, 단순대안으로충분한가?
8) 정식TATO/실제오류레이블/다른backbone/독립원천중어떤확인이남는가?

그림은 필요할 때별도plot으로작성:
- magnitude별fault error 곡선.
- reference/fault/shift tradeoff.
- 같은origin의원자료·오류관측·전처리·예측사례(성능으로best case선정금지).
- 날짜원점분포.

원자료나모델가중치는push하지않는다. 원래승인된scopedcommit/push만적용한다.
모든기존결과를보존하고새결과가좋아도이전FAIL을PASS로고쳐쓰지않는다.

19. 첨부 코드의 의미

reference_core.py:
- full-day stratified sampling + 독립phase roster.
- causal-to-origin 전처리, synthetic fault/persistent-shift 생성.
- 같은 목표의2pinball.
- A4/A5 bounded residual adapter의정확한작은Torch module.

이것은 pretrained model 연결/실제 학습 runner를 완성한 코드가 아니다.
모델 어댑터 삽입·LoRA 부착·데이터 loader·배치·예측 복원은 본문에 맞춰CLI가구현하고
실제모델검사를통과해야한다. test_reference.py 통과를GPU학습완료로보고하지않는다.

20. 최종 실행 문장

이 MASTER_CLI.txt 하나를 이번 작업의 실행 계약으로 사용하라.
실행 목적은 입력 오류가 있어도 지속 변화 정보를 버리지 않는 작은 적응의
추가 가치를 직접 비교하는 것이다. 센서지연·검색·관측간격·PRIOR·압축을 재개하지 마라.

현재 저장소/선행의 실제 코드를 확인하고, 모든 날짜·입력변형·모델·대조·예산을 봉인하라.
Chronos-Bolt-small에서6개군×2sources의최대48fits를 수행하라.
부정적인중간성능때문에다른군의직접비교를건너뛰지마라.
모든선택을고정한뒤평가하고, 입력오류와지속변화의손익을따로보고하라.

실제오류·사건레이블을확보한것처럼쓰거나신규성·성능을보장하지마라.
특별한경로가전처리/증강/동일크기adapter를넘지못하면그대로결론내려라.
반대로작은개선이있어도그것을논문PASS로바꾸지마라.
결과·원점수·정확한미실행범위·다음필수검증을한국어REPORT.md와FINAL_DECISION.md에남겨라.
자동후속학습과다른후보로의전환은수행하지마라.
