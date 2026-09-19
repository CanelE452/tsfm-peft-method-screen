# Fixed MAG: 독립 사용 가능한 구현

논문에서 사용한 `MAG_ONLY`의 고정 수식과 저장 가중치를 복원하는 모듈이다. 새 후보나 수정된 gate가 아니다. `experiments/`, `scripts/` 또는 원 실험 경로를 core 모듈에서 import하지 않는다. 현재 사용자의 학습 금지 지시를 따른다. 학습 runner나 optimizer 실행 명령을 포함하지 않는다.

## 지원하는 입력과 모델

- 이미 LoRA로 학습한 **Chronos-Bolt-small B0** 위의 추가 잔차다. B0 자체를 처음부터 새로 만드는 방법이 아니다.
- 관측은 `[batch,512]` FP32, 미래 출력은 `[batch,9,64]`이다. 이 길이·backbone 외의 일반적인 사용은 검증하지 않았다.
- `sigma`는 `[batch]`의 양수 유한 FP32이며 해당 계열의 고정 TRAIN 표준편차다. 새 평가 정답으로 추정하지 않는다.
- missing/NaN/Inf 입력, 빈 batch, 다른 dtype/길이는 거부한다. 오류 위치·깨끗한 원본·generator state·미래 정답을 입력으로 받지 않는다.
- 관측은 clipping하지 않는다. robust 진폭 gate로 attention 이전의 8,712개 파라미터 잔차만 제한한다. threshold3, MAD배수1.4826, TRAIN sigma floor0.1, rank8은 고정이다.

## Python API

```python
from tsfm_peft_screen.mag import load_bundle
import torch

model = load_bundle(local_chronos_snapshot, exported_bundle, device="cpu")
with torch.inference_mode():
    quantiles = model(observed, train_sigma)
    b0_quantiles = model(observed, train_sigma, adapter_off=True)
```

기존 Chronos B0 객체를 직접 전달할 때는 `MAGForecast(base, seed=...)`를 사용한다. 이때 새로 생성되는 잔차는 출력이 0인 초기 상태이므로, 논문 결과를 재현하려면 반드시 해당 저장 adapter state를 복원해야 한다. 임의 초기 모델을 완료 모델로 간주하지 않는다.

## CPU CLI

아래 명령은 **추론만** 수행한다. NPZ에는 FP32 `observed`와 `sigma` 두 배열만 넣는다. 이미 존재하는 출력 파일은 덮어쓰지 않는다.

```bash
PYTHONPATH=/path/to/repo/src python -m tsfm_peft_screen.mag \
  --snapshot /path/to/chronos-bolt-small/snapshot \
  --bundle /path/to/exported_mag.pt \
  --input /path/to/observed_and_sigma.npz \
  --output /path/to/new_quantiles.npy
```

`save_bundle(model, destination, snapshot, provenance)`는 B0의 기존 LoRA 294,912개와 MAG 8,712개 파라미터를 함께 저장한다. pretrained backbone 전체를 bundle에 복제하지 않는다. 복원할 때 local snapshot의 config/weight hash와 B0 전체 state hash, 고정 방법 config, tensor key를 검사한다. 공개된 bundle manifest와 **bundle 파일 SHA-256도 호출자가 대조**해야 한다. `save_bundle`에는 새 경로를 지정한다.

## 검증과 한계

Electricity/ETTm1 × 기존 seed81551/81552의 네 selected checkpoint에서 기존 구현·독립 구현·bundle 복원·adapter-off가 CPU에서 정확히 일치했다. 각 모델의 기존 TRAIN 관측8개를 사용했다. E 입력/정답은 읽지 않았으며 성능을 새로 채점하지 않았다. 저장소 밖 `/tmp`에서 CLI로 실행한 예측도 일치했다. 총 CPU forward21회, optimizer/backward0, GPU 초기화0이다.

검증 환경과 checkpoint/bundle/source hash는 [검산 기록](../../../results/mag_portable_20260919/AUDIT.json), [모델 manifest](../../../results/mag_portable_20260919/MODELS.json), [환경](../../../results/mag_portable_20260919/ENVIRONMENT.json)을 따른다. 이 파일들은 저장소 root 기준 링크를 포함한 원 실험 보고서에서도 확인할 수 있다.

bundle, 원 가중치, snapshot과 raw 입력은 로컬 캐시에 있다. 공개 저장소의 코드만으로 가중치를 자동 복원하거나 과거 전체 실험을 즉시 재생한다고 주장하지 않는다. 현재 PyTorch/Chronos 버전의 CPU 동일성을 확인했으며 다른 버전·GPU·배치 형식의 bitwise 동일성을 요구하거나 검증한 것은 아니다. 이 구현 정리는 신규성, 범용 PEFT 우위 또는 논문 PASS의 새 증거가 아니다.
