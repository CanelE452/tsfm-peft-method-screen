# 구현·검토 근거

읽은 사용자 저장소 커밋: `7cbe663528602136badd6be3779f0d5e7dd64b14`.

- v4 구현 및 검사: `experiments/transient_peft_readiness_v4_20260923/`.
  외부 task별 state generator와 wrapper 내부 state generator를 분리한 이전 경로는 재사용하지 않는다.
- 기존 BOPTEST 자료: `results/boptest_data_contract_20260924/{fast_probe.csv,transient_response.csv,TAU_FAST.json}`.
- 기존 그림 생성: `experiments/boptest_data_contract_20260924/make_figures.py`.

확인한 공식 구현/문서:

1. Chronos-2 v2.3.2 `model.py`: 패치, REG, 미래 공변량, `forward`, `_compute_loss`.
   https://github.com/amazon-science/chronos-forecasting/blob/v2.3.2/src/chronos/chronos2/model.py
2. Chronos-2 v2.3.2 `layers.py`: TimeSelfAttention / GroupSelfAttention의 축 차이와 q/k/v/o.
   https://github.com/amazon-science/chronos-forecasting/blob/v2.3.2/src/chronos/chronos2/layers.py
3. PyTorch state_dict 저장·복원 권고.
   https://docs.pytorch.org/tutorials/beginner/saving_loading_models
4. PyTorch Module의 파라미터·buffer 등록.
   https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html
5. PEFT checkpoint 형식과 adapter-only 저장 범위.
   https://huggingface.co/docs/peft/main/developer_guides/checkpoint
6. PEFT 모델/LoRA state 관리 API.
   https://huggingface.co/docs/peft/main/package_reference/peft_model

패키지의 변경점: 새 PEFT 이론/손실을 제안한 것이 아니라, 공유 모델 소유권·표준 LoRA 연산·정확한 key 저장/복원·공통 target 행 평균을 구현한 고정 readiness 검사다. 자체 compact checkpoint는 PEFT 표준 배포 형식이라고 부르지 않는다. native 파라미터를 병합하거나 재배포하지 않는다.

소형 테스트 모델(TinyBase)은 이 패키지 인터페이스 검사용이며 실제 Chronos 구현/사전학습 성능을 대체하지 않는다. 라이브러리와 모델이 없는 환경에서 native 결과를 만들어내지 않는다.
