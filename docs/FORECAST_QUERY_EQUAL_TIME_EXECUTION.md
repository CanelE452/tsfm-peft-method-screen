# 동일 시간 비교 실행 구현

2026-09-13: 사용자가 설계만 완료한 상태를 지적하여, 같은 고정 계획을 실제 실행하는
runner, Head/Side checkpoint wrapper, CPU tests, independent finalizer를 구현했다.

설계 기준 a33f245의 numeric settings와 fit order는 변경하지 않는다.
configs/forecast_query_equal_time_plan.json의 DESIGN_FIXED_NOT_EXECUTED는
그 시점의 설계 상태로 보존한다. 현재 실행 상태는 별도 results/forecast_query_equal_time/status.json,
최종 판정은 summary.json 및 RESULT.md에 기록한다.

실행: scripts/with_cuda.sh .venv/bin/python scripts/run_forecast_query_equal_time.py
검증: scripts/with_cuda.sh .venv/bin/python scripts/finalize_forecast_query_equal_time.py --verify-only

실행기는 기존 결과가 있으면 거부하고, clean commit 및 GPU lock을 확인한다.
고정80 preflight updates 뒤 train-only storage choice를 봉인한다.
각 fit은 새 초기 parameter/Adam state로 시작하며 모든4 time checkpoints를 저장한다.
매 epoch 대신 매 update의 host batch/transfer/forward/backward/finite gradient
check/clipping/optimizer/synchronize 시간을 누적한다.
16개 V winner를 봉인한 뒤, 재로딩한 모델의 V 예측이 완전히 같은지 확인하고 E를 연다.
원본 compressed arrays의 기계적 staging은 contract에 기록하며 E scoring과 구분한다.

사전 테스트는 학습된 Head/Side의 checkpoint on/off 출력/gradient/Adam update,
시간 경계 초과/최종 경계 후 업데이트 금지, feasible fast-off baseline 선택을 검사한다.
실제 GPU parity와 fit 상태는 결과 파일에 별도로 기록한다.
GPU가 다른 compute 작업에 사용 중이면 시작하지 않고 최대1시간 대기한다.
전체 실행은2시간 timeout이며 중단된 시도도 status와 fit attempt에 남긴다.
