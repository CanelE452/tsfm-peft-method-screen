# 교정 상태별 보존 LoRA: 고정 후속 개발 프로토콜

부모 야간 실험이 모두 STOP이면 분석 후 다시 실행하라는 사용자 승인에 따른 1회 후속 배치.
가설과 반증한 중간안은 research/calibration_anchor_20260914/FAILURE_ANALYSIS.md에 있다.
이 실행은 후보 1개와 강한 대조 6개이다. 새 방법의 독창성이나 논문 PASS를 보장하지 않는다.

## 메커니즘

train의 F0 quantile 예측에서 매 두 번째 origin을 사용한다.
origin stride가 32, target 길이가 48이므로 선택된 117개 target window는 서로 겹치지 않는다.
시간적 독립성을 주장하지 않는다. 각 channel/quantile/16-step patch의 경험적 coverage c를 구한다.

    error = abs(c - q) / sqrt(q*(1-q))
    reliability = max(exp(-4*error), 0.1)
    weight = reliability / mean(reliability over quantiles and patches, within each channel)
    objective = raw train-std-normalized 2-pinball + 0.1 * mean(weight * abs(prediction-F0) / train_std)

가중치는 train에서 한 번 고정한다. 학습 중 target 오차에 맞춰 갱신하지 않는다.
경험적 coverage는 신뢰구간이나 conformal guarantee가 아니다.
전체 channel 평균 가중치가 1이므로 가중치를 섞은 대조와 평균 보존량 예산이 같다.
가중치와 실제 residual 간 상관 때문에 최종 penalty 크기가 같다는 주장은 하지 않는다.

## 팔과 예산

1. native LoRA: native asinh-space 2-pinball / 21.
2. raw LoRA: 보고 지표와 같은 raw-scale loss.
3. L2 anchor: 초기 LoRA 파라미터와 제곱 차이 합, 계수 .01.
4. full anchor: F0 raw quantile에 균일 L1, 계수 .1.
5. weighted loss: 동일 교정 가중치를 pinball에 적용; anchor 없음.
6. shuffled anchor: 동일 가중치를 quantile 축 7칸/horizon 축 16칸 회전, 계수 .1.
7. calibration anchor: 제안법.

모두 동일 rank8, attention q/k/v/o 96개 projection, 1,179,648 trainable parameters.
같은 F0 예측과 train labels를 사용할 수 있다. 가중치 계산은 추가 train label 정보가 아니다.
2 datasets x 2 seeds(32000/32001) x 7 arms x 2 LR(3e-5/1e-4) = 최대 56 fits.
각 fit 900 updates, 동일 seed의 minibatch 순서, 합계 최대 50,400 updates.
GFLOP/시간이 같은 실험으로 주장하지 않는다. teacher와 가중치 준비 시간도 따로 기록한다.
전체 queue 최대 8시간. train/evaluate phase 최대 7시간, fit wall cap 20분.
GPU 초기 유휴 30초와 기존 memory/process guards, 단일 GPU lock을 유지한다.

최종 소스 GPU smoke 14 updates, 저장 형식 보완 전의 통과한 초기 smoke 14 updates도 별도 보존한다.
초기 스크립트 생성 시 줄바꿈 구문 오류 1회는 모델 실행 전에 발생했고 optimizer updates=0.
본학습 fit attempt로 세지 않지만 이 사실을 감추지 않는다. 전체 실제 smoke updates=28.

## 데이터와 평가

ETTh1/Traffic 첫 네 채널 및 기존 원본 해시를 유지한다.
train 2048..9472 stride32, V 10752..12192 stride96, context1024/horizon48/batch8 series.
train/V를 사후 연구에서 재사용하므로 적응적인 개발 실험이다.
부모 E의 마지막 target은 [16288,16336)이다.
이번 E는 16512..17184 stride96, 데이터당 8 origins로 그 뒤에 둔다.
raw 전체를 기계적으로 읽었던 사실과 target을 점수화했던 사실을 구분한다.

이 작은 인접 시간 tail은 독립 원천 재현이 아니다. 기존 E를 새 E라고 부르지 않는다.
모든 28 dataset/seed/arm 선택을 2 LR x {0,150,450,900} V score로 봉인하고 E를 한 번 평가한다.
동률이면 빠른 step, 낮은 LR. step0도 선택 가능.
기존과 같은 두 데이터 각각 최강 대조 대비 1% 평균 개선, 각 seed 최강 대조 대비 손실비<=1.01,
제안법 양 seed 모두 양수 step 및 F0 개선을 요구한다.
4개의 연속 블록(각 2 origins) bootstrap은 설명용이며 작은 표본의 확증으로 삼지 않는다.
기준·seed·LR·데이터 선택을 E 결과에 맞춰 변경하지 않는다.

## 실행

    scripts/with_cuda.sh .venv/bin/python scripts/calibration_anchor_queue.py start
    scripts/with_cuda.sh .venv/bin/python scripts/calibration_anchor_queue.py status

결과: results/calibration_anchor_20260914/REPORT.md
실시간: results/calibration_anchor_20260914/queue_status.json
캐시: .cache/calibration_anchor_20260914/
재검증:

    scripts/with_cuda.sh .venv/bin/python scripts/finalize_calibration_anchor.py --verify-only

부모 실행 소스는 수정하지 않고 검증된 실행기의 별도 사본을 만들었다.
새 config, objective, teacher calibration weights, manifest 분할, 작은 E 표본을 검증기에 반영했다.
기존 모든 원본 결과 파일의 해시와 새 실행 소스 커밋을 보존한다.
추가 자동 재개/CLI 버전 변경/모델 변경은 이번 실행에 포함하지 않는다.
