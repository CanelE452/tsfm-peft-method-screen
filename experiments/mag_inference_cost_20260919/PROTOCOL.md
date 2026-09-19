# 기존 가중치의 동일 조건 추론 비용 검사

2026-09-19. 사용자의 목표 지속 실행 지시에 따른 논문 자원 근거 보완이다. 새로운 학습·후보 선택·성능 기준 수정은 하지 않는다. 과거 타이머의 범위 차이를 추정으로 보정하지 않고 현재 구현의 추론만 직접 측정한다.

- 대상: B0/PLAIN/MAG_ONLY/TOKEN_GATE/TOKEN_GATE_ENTROPY/PETSA_XY_OFFLINE. 기존 Electricity/ETTm1, seed81551/81552의 selected checkpoint 24개를 hash로 고정한다. step0도 현재 구현 그대로 실행하며 자동 생략하지 않는다.
- 입력: 기존 TRAIN epoch0의 index `33*arange(32)` (0…1023). observed x와 TRAIN sigma만 GPU로 보낸다. 라벨 파일이나 E 입력/정답을 열지 않는다. 생성 state는 표본 구성 기록에만 사용하며 모델에 전달하지 않는다.
- 계산: Chronos-Bolt-small, FP32, TF32 off, eval + inference_mode, PyTorch CPU threads4. 기존 함수의 assert/정규화/gate/분위수 계산을 그대로 포함한다. compilation/quantization/새 수식 없음.
- batch1과32. batch1은 같은32개 입력을 순서대로 한 번씩, batch32는 같은32개를32회 반복한다. 원자료의 새로운32회 반복으로 세지 않는다.
- 각 source×seed에서 여섯 모델을 CPU에 준비한다. 한 번에 모델 하나만 GPU에 올린다. 6round에서 순서를 cyclic rotate해 모든 군이 각 순서 위치에 한 번씩 온다. batch 순서는 round parity에 따라1→32/32→1로 교대한다.
- 각 모델/batch/round: warmup3회, timed32회. timed forward 전 CUDA synchronize; perf_counter 및 CUDA events로 forward 호출→완료까지 측정한다. 모델 로딩·CPU/GPU 전송·hash·GPU 점검·파일 기록은 타이머 밖이다. 출력 finite 검사는 타이머 뒤. timed call마다 allocated baseline과 peak를 기록한다. CUDA event값도 kernel-only 지표로 부르지 않는다.
- 총 timed calls9216, warmup864, 초기/최종 동일 출력 확인48회, 합계10128forward 상한. optimizer 생성/step/backward0, E prediction0, accuracy score0. 전체 wall cap1h. GPU 공통 lock·4GiB 시작여유·1GiB 도중여유·RustDesk만 예외를 유지한다. 새로운 외부 compute가 겹친 측정은 유효 결과로 만들지 않으며 중단·기록한다.
- 전/후 model state hash, 같은32batch의 초기/최종 예측 bitwise 동일성, 원 checkpoint/input/source-code hash 불변을 확인한다. 첫 입력 자체도 hash 고정한다. 기존 결과를 덮어쓰지 않는다.
- 모든24×2×6=288 측정 block을 보고한다. 군/source/seed/batch별 latency median/p10/p90, median round-ratio(MAG/대조), memory를 공개한다. 총 순위/논문PASS/accuracy-resource 합성점수 없음. 한 장치·한 실행·기존 두seed·공통 TRAIN sample의 비용이며 seed 모집단 추론이나 학습 속도 우위를 주장하지 않는다.
- 완료 후 한국어 REPORT와 논문 보충, 원시 timing CSV·검산·그림을 scoped commit/push한다. 추가 학습/새 구조/새 E 자동 실행 없음. timing 결과에 따라 측정 횟수나 구현을 바꾸지 않는다.
