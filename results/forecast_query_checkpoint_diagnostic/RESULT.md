# Forecast-query checkpoint 통제 진단 결과

판정: **TRADEOFF_ONLY**. 기존 forecast-query의 **FAIL은 그대로 유지**한다.

## 실행 범위

BF16 자원 측정 24회, FP32 동등성 업데이트 8회, 워밍업 6회: 총 38 optimizer updates.
별도 저장값 계측 backward 4회는 optimizer update 없이 수행했다.
새 학습 fit, V 선택, E 접근은 모두 0이다. 모든 비교는 동일 LoRA/Adam/RNG 상태에서 시작했다.
실행 commit: 7d11f08f345ab01368af8dec24759e23baa58324. 두 데이터셋 각각 train batch 하나를 3회 반복했다.

## 실제 자원 비교

| 데이터 | 방식 | checkpoint | peak allocated MiB | peak reserved MiB | step 중앙값 s |
|---|---|---|---:|---:|---:|
| ettm2 | standard | off | 1831.18 | 1886.00 | 0.1205 |
| ettm2 | standard | on | 732.88 | 784.00 | 0.1778 |
| ettm2 | query | off | 950.10 | 960.00 | 0.1367 |
| ettm2 | query | on | 683.82 | 730.00 | 0.2060 |
| electricity | standard | off | 1831.18 | 1886.00 | 0.1207 |
| electricity | standard | on | 732.88 | 784.00 | 0.1709 |
| electricity | query | off | 947.97 | 960.00 | 0.1330 |
| electricity | query | on | 683.82 | 730.00 | 0.2027 |

peak는 3회 중 최대값, 시간은 3회 중앙값이다. frozen pass, cache 생성, backward, clipping, Adam step을 포함한다.
복원/CPU export/모니터 호출은 timing 밖에 있다. 계측 hook이 없는 24회만 자원 표에 사용했다.
데스크톱 graphics가 공유하는 GPU이므로 작은 시간 차이와 3회 반복을 통계적 우위로 해석하지 않는다.

## 사전 기준에 따른 분리 판정

- ettm2: query-on / standard-on peak=0.933065, time=1.158769. 기존 20% memory target 충족=False; memory 감소 및 시간 overhead≤5%=False. query 자체 checkpoint on/off memory=0.719739, time=1.507551.
- electricity: query-on / standard-on peak=0.933065, time=1.185885. 기존 20% memory target 충족=False; memory 감소 및 시간 overhead≤5%=False. query 자체 checkpoint on/off memory=0.721352, time=1.523934.

## 수치 및 보존 검증

독립 CPU 재계산: 16 on/off 쌍, raw cache 32개. 출력, loss, 전체 clipped gradient, 실제 parameter delta, Adam moments/step 및 RNG 사후 상태를 비교했다.
최대 absolute=0, 최대 relative L2=0.
기존 결과 230개 파일 SHA256 불변. frozen backbone 파라미터도 실행 전후 불변이다.
역사적 forward와 새 구현의 precision별 출력 비교는 historical_forward_parity.json에 기록했다.

## 저장값과 단계별 peak

storage_profiles.json은 별도 계측 4회를 보존한다. storage union과 실제 allocator peak를 구분한다.
- standard checkpoint=False: outer saved CUDA nonparameter storage union 1223.80MiB; 계측 중 가장 높은 phase=backward (1831.18MiB).
- standard checkpoint=True: outer saved CUDA nonparameter storage union 86.45MiB; 계측 중 가장 높은 phase=backward (732.88MiB).
- query checkpoint=False: outer saved CUDA nonparameter storage union 310.40MiB; 계측 중 가장 높은 phase=output_head (947.97MiB). 과거 K/V unique=72.00MiB, 전체 cache unique=78.23MiB.
- query checkpoint=True: outer saved CUDA nonparameter storage union 80.49MiB; 계측 중 가장 높은 phase=frozen_encoder_and_cache (683.82MiB). 과거 K/V unique=72.00MiB, 전체 cache unique=78.23MiB.

이 계측은 nested checkpoint hook 내부를 모두 볼 수 없고 연산별 전체 임시 메모리를 분해한 profiler가 아니다.
따라서 saved-storage union을 peak와 동일시하거나, 전체 peak 차이를 K/V만으로 설명하지 않는다.

## GPU 감시

총 92회 점검, 실행 단계 85회. 실행 단계 관측 최소 free=6899MiB. 외부 compute PID와 OOM은 없었다.
점검은 step 경계에서 수행하므로 모든 순간의 다른 프로세스 상태를 보장하는 연속 profiler는 아니다.

## 해석과 다음 범위

이번 결과는 두 데이터셋 모두 TRADEOFF_ONLY다. 같은 checkpoint-on 비교에서 메모리 약 6.69% 감소와 step 시간 약 15.88~18.59% 증가가 관찰됐다.
query 자체 checkpointing은 memory를 약 28% 줄였지만 step 시간은 약 51~52% 늘렸다.
새로운 관찰은 query-on의 peak가 학습 branch가 아니라 frozen encoder+cache 단계에서 발생한다는 점이다. 현재 측정의 그 단계가 그대로 유지된다면 query branch 저장값만 더 줄여도 standard-on 대비 20% 감소 목표인 약 586.30MiB에 도달할 수 없다. 이는 모든 구현의 수학적 하한이 아니라 이번 고정 상태의 단계별 측정에 근거한 조건부 판단이다.
현재 증거로 장기 fit이나 논문 방법 확장을 진행하지 않는다. side 대조와 품질 우위는 여전히 미검증이며, 기존 FAIL을 뒤집을 근거가 부족하다.
checkpointing은 기존 기술이며 이번 진단은 새 PEFT 방법의 성공 증거가 아니다.
RESOURCE_SIGNAL_ONLY이면 동등한 저장 전략에서 자원 이점이 남는다는 개발 근거만 얻은 것이다.
TRADEOFF_ONLY이면 메모리/시간 교환을 확인한 것이며, 목표를 바꿔 과거 FAIL을 PASS로 바꾸지 않는다.
STOP_CURRENT_QUERY_STORAGE이면 현재 구조·저장 방식의 추가 확장은 중단한다.
어느 경우에도 side/head 대비 우위, 새 데이터 예측 정확도, time-to-quality 또는 논문 신규성을 입증하지 않았다.
후속 학습이 필요하다면 강한 side 대조와 별도 미노출 평가 계획을 먼저 고정해야 한다.

## 재현

CPU: PYTHONPATH=src .venv/bin/python scripts/finalize_forecast_query_checkpoint_diagnostic.py --verify-only
GPU: scripts/with_cuda.sh .venv/bin/python scripts/run_forecast_query_checkpoint_diagnostic.py
GPU 실행기는 기존 결과 디렉터리가 존재하면 덮어쓰기를 거부한다. 원시 상태/gradient cache는 .cache 아래 ignored 파일이다.
