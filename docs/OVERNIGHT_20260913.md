# 세 PEFT 방향의 야간 개발 스크리닝

실행 요청에 따라 새로 고정한 파일럿이다. 과거 FAIL/STOP 및 원래 7개 후보의 기준을 바꾸지 않는다.
목표는 다음 방법 개발에 투자할 만한 추가 이득을 찾는 것이다. PILOT_PASS는 논문 통과나 독창성 입증이 아니다.

## 후보와 강한 대조군

| 방향 | 실제 학습되는 후보 | 비교군 | 확인하려는 추가 가치 |
| --- | --- | --- | --- |
| 예측 보존 정규화 | 원시 스케일 pinball LoRA + 동결 원래 예측에 대한 L1 anchor | 원래 native-loss LoRA, raw-loss LoRA, L2 parameter anchor | 단순 손실 정렬이나 가중치 정규화 이후에도 과적합을 줄이는가 |
| 최근 변화 조건부 보정 | 과거 구간의 평균 차이·표준편차 비율로 각 LoRA rank를 조절 | native LoRA, raw LoRA, 동일 크기의 일반 moment conditioner | 같은 정보에서 변화량을 명시하는 구조가 도움이 되는가 |
| 다중 문맥 분포 압축 | 문맥 길이별 불일치가 적은 예측 패치에 증류 가중치를 두는 LoRA | raw LoRA, 균일 가중 증류, 시간 순서를 뒤집은 가중 증류 | 같은 teacher 정보와 총 증류 가중치에서 시간별 배치가 도움이 되는가 |

마지막 후보는 256/512/1024 길이의 동결 Chronos-2 예측을 혼합한다. 이는 미래 공변량 연구의 대체 실증이 아니다.
발행 시각이 있는 미래 공변량은 아직 감사하지 않았으므로 해당 실험을 했다고 주장하지 않는다.
결측 gate는 이전 Freshness 계열과 겹쳐 이번 큐에 다시 넣지 않았다.

분포 teacher는 각 성분의 단조 quantile 함수를 구간별 선형 보간하고 끝부분을 상수로 연장한다.
각 성분의 256개 고정 midpoint 샘플을 합쳐 quantile을 재계산한다. Quantile 산술평균이 아니다.
각 horizon의 주변분포만 다루며, 공동 시계열 경로 분포를 보존한다는 주장은 하지 않는다.
teacher 가중치는 성분 median의 표준편차를 train 표준편차로 나누고 16-step 패치별 평균을 취한 뒤
1/(1+disagreement)로 만든다. 각 시리즈의 horizon 평균을 1로 맞춘다.
반전 대조군은 동일 가중치를 역순으로 배치한다.

## 문헌과 독창성의 경계

- [Time-LlaMA, D-LoRA](https://arxiv.org/abs/2502.13725): 입력 조건부 LoRA 선택은 이미 존재한다.
- [MixFT](https://arxiv.org/html/2603.02840v1): TSFM 하위 분포별 LoRA 전문화가 이미 연구됐다.
- [Guard](https://arxiv.org/abs/2606.19363): TSFM의 문맥 기반 teacher 선택과 불확실도 기반 증류가 이미 연구됐다.
- [Ensemble distribution distillation](https://arxiv.org/abs/2002.11531): 앙상블 분포를 작은 모델로 압축한다는 상위 아이디어는 기존 기술이다.
- [delta-Adapter](https://arxiv.org/abs/2601.20280): 동결 예측의 보정 자체도 새로운 주장으로 삼을 수 없다.

따라서 세 후보는 알려진 구성요소를 사용하는 mechanism prototype이며 신규 방법으로 확정하지 않는다.
양성 결과가 있어도 기존 연구와 구별되는 기여, 별도 도메인 재현, 추가 ablation이 필요하다.
직접 구현이 공식 논문 재현이라고도 주장하지 않는다.

## 데이터와 선택

공식 저장소의 고정 커밋에서 ETTh1 및 Traffic를 다운로드했다. 사용 채널은 순서상 첫 네 개로 미리 고정했다.
ETTm1도 내려받았지만 ETTh1의 다른 해상도이므로 독립 도메인으로 세지 않고 사용하지 않는다.
원본 SHA와 URL은 research/overnight_20260913/data_receipt.json에 있다.
이 저장소에서 종전 평가에 사용했던 ETTm2/Electricity 대신 사용하는 새 개발 데이터이다.
사전학습 데이터와의 비중복은 보장하지 않는다.

두 데이터 모두 context 1024, horizon 48, batch 2 origins x 4 channels.
학습 origin 2048..9472 stride 32 (233), V 10752..12192 stride 96 (16),
E 13312..16288 stride 96 (32). 구간은 0-index이고 target은 origin부터 시작한다.
정규화 표준편차는 train target 합집합의 구간 [2048,9520)에서만 계산한다.
E 배열의 사전 저장은 기계적 작업이며 E 통계나 예측 점수는 사용하지 않는다.

Seed 31000/31001, LR 3e-5/1e-4, AdamW(weight_decay=0), gradient clip 1.
BF16 autocast, FP32 파라미터 및 optimizer. 원래 native loss는 quantile 수 21로 나눠 평균 단위를 맞춘다.
다른 팔은 보고 지표와 같은 sorted raw-scale 2-pinball을 최적화한다.
L2 anchor 계수 .01은 초기 파라미터와의 차이 제곱합에 적용한다.
Prediction anchor 계수 .1, distillation 계수 .2를 각각 train-std 정규화 L1 출력 차이에 적용한다.
계수 검색은 하지 않는다. 조건부 LoRA는 1,185,024개, 나머지 LoRA는 1,179,648개 trainable parameter이다.
동일 크기의 moment gate를 강한 대조로 둔다. 모든 팔의 파라미터 수가 같다는 주장은 하지 않는다.

각 fit은 같은 seed에서 같은 900개 minibatch 순서로 900 updates를 실행한다.
모델 선택은 2 LR x checkpoint {0,150,450,900} 중 V가 최소인 것, 동률이면 빠른 step/낮은 LR.
Step 0도 가능하다. 고정 update 비교이며 같은 시간 비교가 아니다.
teacher cache 작성 시간, 각 step의 실제 시간, peak memory, 전체 wall time을 별도 기록한다.
모든 후보의 선택 또는 학습 실패·teacher 중단 판정을 먼저 봉인한 뒤 E를 연다.

증류 후보는 raw LoRA의 8 fits를 먼저 수행한다. 두 데이터 모두 V에서 혼합 teacher가
가장 좋은 개별 문맥 F0와 선택된 raw LoRA seed 평균 중 더 좋은 값보다 1% 이상 좋아야 나머지 24 fits를 실행한다.
충족하지 않으면 STOP_NO_TEACHER_HEADROOM, 해당 후보 E는 열지 않는다.

## 고정 판단

각 데이터에서 제안법의 두 seed 평균 손실이 F0 및 모든 비교군보다 각각 1% 이상 낮아야 한다.
각 seed도 가장 좋은 비교군보다 1% 넘게 나빠지면 안 된다.
두 seed 모두 양수 step을 선택하고 자신의 F0보다 좋아야 한다. 두 데이터에서 모두 충족해야 PILOT_PASS.
증류의 E까지 실행된 경우에는 multi-pass teacher와 짧은 문맥 F0도 보수적으로 비교군에 포함한다.
따라서 압축 품질을 단순히 유지하는 것보다 강한 품질 기준이며, 여기서 STOP했다고 압축 연구 전체가 불가능하다는 뜻은 아니다.
4개의 연속 origin 블록 bootstrap 구간은 설명용이며 통계적 확증 조건으로 바꾸지 않는다.
E를 본 뒤 LR·계수·seed·후보를 추가하거나 기준을 바꾸지 않는다.

## 실행과 자원

프로젝트 폴더에서 다음 한 줄로 터미널과 분리된 큐를 시작한다.

    scripts/with_cuda.sh .venv/bin/python scripts/overnight_queue.py start

상태 확인:

    scripts/with_cuda.sh .venv/bin/python scripts/overnight_queue.py status

최대 96 fits / 86,400 본학습 updates. GPU smoke 24 updates는 별도이다.
teacher 중단 시 72 fits / 64,800 updates로 끝난다.
큐는 최대 8시간, 한 train/evaluate 작업은 최대 2시간, 각 fit은 최대 20분이다.
timeout은 성공이나 과학적 실패로 포장하지 않고 INCONCLUSIVE_EXECUTION으로 남긴다.
중단된 작업을 자동 재시도하거나 봉인된 결과를 덮어쓰지 않는다.

프로젝트 GPU lock을 한 큐가 소유한다. 다른 compute PID가 없고 여유 4 GiB 이상,
사용률 90% 미만인 상태가 30초 유지돼야 작업을 시작한다.
실행 중 약 5초마다 확인하고 다른 compute PID 또는 여유 1 GiB 미만이면 update 경계에서 기다린다.
외부 프로세스를 종료하지 않는다. 8 GiB GPU allocation/6 GiB RSS/2 GiB RAM 여유 guard도 적용한다.
시간 상한에는 GPU 대기 시간이 포함된다. 한 후보의 실행 오류가 다음 후보 실행을 막지 않는다.

결과는 results/overnight_20260913/REPORT.md 및 ranking.csv,
실시간 진행은 queue_status.json과 후보별 train_status.json에 남는다.
캐시·체크포인트는 .cache/overnight_20260913 아래에 있으며 Git에 올리지 않는다.
완료 후 finalize_overnight.py가 예측 지표, 샘플링 순서, V 선택, 체크포인트 해시,
teacher 혼합, 전체 후보의 봉인, 과거 결과 및 실행 소스 해시를 독립 검증한다.
수동 재검증은 scripts/with_cuda.sh .venv/bin/python scripts/finalize_overnight.py --verify-only 이다.
