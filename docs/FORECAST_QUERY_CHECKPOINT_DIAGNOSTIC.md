# Forecast-query 저장·재계산 통제 진단

사용자가 2026-09-13 승인한 범위: 기존 forecast-query를 동일 수준으로
checkpointing한 뒤 개발 지속 여부를 판단한다. 기존 FAIL 및 모든 과거 결과는 보존한다.
새 모듈, 새 손실, 데이터 분할 변경, HPO, E 재선택은 하지 않는다.

## 고정 설계

- Standard / Query × checkpoint off / on, 4설정.
- ETTm2와 Electricity에서 각각 train origins [4352, 4608], 첫 4채널.
  context 4096, horizon 48, 실제 배치 8 시계열. 각 dataset의 단일 고정 batch이며
  서로 독립된 2개 연구 반복을 의미하지 않는다.
- 각 batch에서 3회 측정: 총 24회 BF16 autocast optimizer updates.
  각 측정은 같은 상태에서 출발하는 단일 업데이트이며 연속 학습 fit이 아니다.
- 먼저 Standard off를 ETTm2 고정 batch에서 2회 업데이트하여 공통 nonzero
  LoRA 및 Adam 상태를 만든다. 이를 CPU에 저장하고 모든 arm/설정/반복에 동일 복원.
  AdamW lr=3e-5, weight_decay=0, clipping=1. FP32 파라미터/optimizer.
- 각 4설정에 별도 kernel warmup 1 update 후 다시 공통 상태 복원.
  warmup 총 6 updates; 자원 통계에서 제외.
- FP32 동등성은 4설정 × 2batch = 8 updates로 별도 확인.
- 추가 4개 BF16 계측 backward는 저장 storage/단계별 peak 진단만 수행.
  optimizer update 0. 총 상한 38 optimizer updates + 4 backward-only.
- 측정마다 trainable parameters, Adam state, Python/NumPy/CPU/CUDA RNG 복원.
  gradients=None, gc 및 empty_cache 후 동일 입력 resident 상태에서 측정.
  모델/입력/optimizer CPU→GPU 복원 시간은 제외; frozen encode/cache 생성부터
  backward, clipping, Adam step까지 포함한다. CUDA 동기화로 전체 step wall time 측정.
  반복 순서는 off/on, on/off, off/on으로 교차; arm 순서는 Standard, Query.
- raw prediction, native z, loss, clipped gradients, 실제 parameter delta,
  Adam moments/step 및 RNG 사후 상태를 CPU cache에 저장한다.
  별도 전체 unclipped gradient는 반환 clip norm으로 상대 규모를 검증하고,
  gradient 동등성은 동일 clipping을 거친 full gradient vector로 평가한다.
- on/off만 같은 precision에서 비교한다. FP32 vs BF16 자체의 동일성은 요구하지 않는다.
  FP32: relative L2<=1e-5 AND max absolute<=1e-5.
  BF16: relative L2<=1e-4 AND max absolute<=1e-4.
  실제 업데이트 delta와 Adam 상태도 비교하여 큰 원래 가중치가 차이를 가리지 않게 한다.
  같은 arm의 역사적 forward 함수와 새 off 함수의 출력도 각 precision에서 비교한다.
- 같은 RNG/optimizer/parameter 상태 해시와 정확한 소스/모델/데이터 해시 기록.
  구현 검증이 실패하면 추가 측정을 중단하고 원인을 기록한다.

## 구현 제약

Query의 레이어 함수는 해당 block을 partial로 고정하고 h, 과거 K/V,
position, time mask, group mask를 명시적 tensor 인자로 받는다.
closure가 변경되는 cache/loop 변수 또는 disabled LoRA 상태를 참조하지 않게 한다.
non-reentrant checkpoint를 사용한다. frozen encoder는 매 forward마다 다시 실행한다.
혼합정밀 설정과 autocast cache 정책은 네 설정에서 동일하다.
과거 model.py와 compression.py는 변경하지 않는다.

## 저장값과 peak 해석

24회 timing에는 saved-tensor hooks나 중간 동기화를 넣지 않는다.
별도 4회 계측에서는 frozen encode+cache / query branch 또는 standard encoder /
output head / loss / backward / clip의 allocator peak를 기록한다.
storage는 StorageWeakRef로 identity를 유지해 alias 중복과 주소 재사용을 피한다.
보고하는 저장값은 outer saved-tensor hook이 관찰한 storage union이다.
동시 생존 peak나 모든 연산 임시값이 아니며 checkpoint 내부 hook이 감춘 값이 있다.
BF16 과거 K/V 24개 payload는 72MiB라는 사전 예상이며 실제 cache에서 확인한다.
이 값만으로 전체 peak 차이를 설명하지 않는다.

## 판정과 한계

먼저 같은 함수의 수치 동등성에 통과해야 한다.
자원은 memory/time 별도로 보고한다. query-on / standard-on을 주 비교로,
query-off / standard-off와 각 checkpoint 효과를 함께 공개한다.
기존 20% 메모리 감소 목표 충족 여부는 별도 칸이며 기존 FAIL을 덮어쓰지 않는다.
메모리가 줄고 시간 overhead가 5% 이하라면 후속 개발 검토 근거로만 삼는다.
시간만 좋아지면 그 사실만 보고하며 연구 목적 변경이나 대규모 fit을 자동 실행하지 않는다.
side보다 우수하다는 결론은 이 2×2에서 낼 수 없다.
고정 batch 3회는 시스템 반복이며 통계적 품질 검증이나 time-to-quality 측정이 아니다.
새 GPU fit, V/E evaluation은 0.

## 안전성과 재현

30초 연속 외부 compute PID 없음, free>=4GiB, utilization<90%를 확인한 뒤 실행.
각 step 전후 GPU 점검, 외부 학습 등장/free<1GiB이면 경계에서 대기.
startup 최대 600초, 실행 timeout 1800초 및 기존 RAM/GPU guard 적용.
다른 프로세스를 종료하지 않는다. 중단/실패도 별도 결과로 보존한다.
실행 전 이 문서/설정/코드를 main에 commit/push한다.

참조: [PyTorch 2.8 checkpoint](https://docs.pytorch.org/docs/2.8/checkpoint.html).
재계산과 backward에서 함수가 달라지면 조용히 잘못된 gradient가 생길 수 있으므로
determinism metadata check만 믿지 않고 실제 gradient/update를 비교한다.
