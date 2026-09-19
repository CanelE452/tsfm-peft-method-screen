# 고정 MAG의 독립 구현 및 CPU 복원 검증

기존 방법과 학습 결과를 바꾸지 않고, 실험 디렉터리에 묶여 있던 MAG를 `src/tsfm_peft_screen/mag`로 분리했다. core는 PyTorch만 사용하고, bundle loader는 설치된 Chronos 및 별도의 local snapshot을 이용한다. 원 실험의 TRAIN/E loader나 학습 runner를 import하지 않는다.

## 확인한 결과

| 원천 | 기존 seed | selected step | 기존 구현과 일치 | bundle 복원 일치 |
| --- | --- | --- | --- | --- |
| Electricity | 81551 | 768 | exact | exact |
| Electricity | 81552 | 768 | exact | exact |
| ETTm1 | 81551 | 0 | exact | exact |
| ETTm1 | 81552 | 0 | exact | exact |

각 모델에서 기존 TRAIN 관측8개를 사용해 원 구현·독립 구현·adapter-off·새 bundle 복원을 비교했다. B0 및 입력의 hash 불변을 확인했다. ETTm1의 selected step0도 그대로 보존했고 학습된 checkpoint로 바꿔 끼우지 않았다.

총 CPU forward21회, optimizer update0, backward0, GPU 초기화0, E prediction/정답 채점0. 검증 프로세스에서는 optimizer 생성 및 backward 호출을 명시적으로 차단했다. `/tmp`에서 독립 CLI를 실행한 예측도 원 구현과 정확히 일치했다. 변경된 threshold가 포함된 bundle config는 거부했다.

CPU 단위 검사10개도 통과했다. 일정한 입력과 extreme patch, threshold 경계, 해당 합성 사례의 affine 변환, 잘못된 입력 거부, RNG 보존과 초기 identity, 국소 residual bound를 확인했다. 이 합성 검사로 실제 forecasting 성능을 증명하지 않는다.

## 재현 범위

[사용법](../../src/tsfm_peft_screen/mag/README.md), [모델별 hash](MODELS.json), [검산](AUDIT.json), [환경](ENVIRONMENT.json), [CLI 검사](CLI_CHECK.json).

공개 코드에는 새 학습 명령이 없다. bundle은 B0 LoRA 294,912개와 MAG 8,712개 파라미터를 저장하며 원 pretrained weight는 포함하지 않는다. snapshot config/weight hash, 복원된 B0 state hash, 방법 config 및 state key를 확인한다. bundle 파일 자체도 MODELS.json의 SHA-256과 대조한다.

원 가중치·bundle·raw 입력은 로컬 `.cache`에 있으므로 GitHub만으로 전체 수치 재생이 가능하다고 주장하지 않는다. 현재 CPU와 설치 버전의 예측 동등성을 확인했으며 다른 GPU/버전에서의 동등성이나 학습 경로 동등성은 검사하지 않았다. 기존 원고의 예측 이득·손해·자원 측정과 신규성 한계를 유지한다.

## 목표와 현재 제한

이번에 보완한 것은 제안 방법의 실행 가능한 독립 구현과 복원 가능성이다. 새로운 방법론 근거나 독립 source 검증을 추가한 것은 아니다. 사용자 지시대로 새 학습을 시작하지 않으며, 기존 결과를 보존해 scoped commit/push한다.
