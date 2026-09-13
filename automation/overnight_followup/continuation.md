이 메시지는 사용자가 명시적으로 승인한 조건부 후속 작업의 1회 재개 지시이다.
같은 대화를 재개한 것이며 새 하위 에이전트나 병렬 연구 작업이 아니다.
사용자 원문: "하고 나서 만약에 다 fail이면 안되는 이유를 분석해서 다시 주제 정해서 해줘"

작업 저장소: /home/minjae/Documents/github/tsfm-peft-method-screen
부모 실행: results/overnight_20260913
후속 감시 상태: .cache/overnight_followup/status.json
감시 트리거 근거: .cache/overnight_followup/trigger.json
기존 실행 소스 커밋: ee1350d (contract.json의 full SHA가 권위 있는 값)
고정 프로토콜: docs/OVERNIGHT_20260913.md
최신 이전 연구 검토: research/peft_rethink_2026_09_13/REPORT.md

현재 대화의 이후 사용자 지시가 이 작업을 취소하거나 범위를 변경했다면 그것을 우선한다.
다른 사람이 이 저장소에서 활동 중이면 그 작업을 덮어쓰지 않는다.
하위 에이전트를 만들지 말고 이 대화에서 직접 진행한다.
이 감시 작업을 다시 설치하거나 재귀적으로 다음 자동 재개를 만들지 않는다.

먼저 부모 큐가 실제로 끝났고 프로세스 및 GPU 잠금이 해제됐는지 확인한다.
scripts/with_cuda.sh .venv/bin/python scripts/finalize_overnight.py --verify-only로 결과를 재검증한다.
세 verdict를 이름과 함께 기록한다. PILOT_PASS가 하나라도 있으면 조건부 새 주제 실험은
시작하지 말고 성과와 아직 필요한 검증만 보고한다. 실행 오류를 과학적 FAIL로 세지 않는다.
지표·예측 캐시·checkpoint·V 선택·GPU 기록·실제 update 수·source hash를 확인한다.

모두 PILOT_STOP 또는 STOP_NO_TEACHER_HEADROOM이면 아래를 완료한다.

1. 원인 분석을 먼저 파일로 남긴다. 아직 없는 실험은 하지 않았다고 쓴다.
   - 각 팔의 train/V 곡선, step0 선택, LR/seed 편차, native/raw 목적함수 차이
   - 제안법이 F0와 가장 강한 단순 대조보다 얼마나 나쁜지
   - anchor 페널티가 실질적으로 작동했는지, gate가 변하고 gradient를 받았는지
   - moment conditioner 대비 drift gate의 추가 정보가 아니라 inductive bias의 이득이 있었는지
   - teacher가 단일 문맥 및 LoRA보다 좋아졌는지, 왜 headroom gate가 닫혔는지
   - GPU contention/timeout/OOM 등 실행 상태와 과학적 가설 실패를 분리
   관찰과 원인 가설을 구분하고, 인과 주장은 작은 반증 가능한 진단으로 확인한다.

2. 분석 근거에 따라 새 방법 주제를 최대 2개 고른다. PASS를 만들기 위한 이름 변경,
   threshold 완화, 기존 실패 기법의 무근거 재시도는 하지 않는다.
   모델/데이터/학습 버그면 먼저 고친 뒤 재평가해야 하며 새 주제 실패라고 바꾸지 않는다.
   최신 primary 논문/공식 저장소로 novelty collision을 확인한다.
   알려진 단순 개선은 비교군으로 인정한다. 논문 novelty나 PASS를 보장하지 않는다.
   미래 공변량 후보라면 실제 available_at/forecast issue-time 감사부터 한다.
   과거 문맥 teacher 실험을 미래 공변량 실험의 증거로 둔갑시키지 않는다.
   연구 질문, 예상 작동 조건, 가장 강한 대조, 반증 조건을 선정 파일에 적는다.

3. 구체적 구현과 실제 실행까지 수행한다. 계획만 작성하고 종료하지 않는다.
   후속 연구 배치는 이번 1회만 허용하며 최대 2개 주제, 합계 최대 64 fit attempts,
   최대 8시간 GPU 큐 wall time으로 제한한다. 무한 탐색 및 자동 재귀 금지.
   먼저 train-only CPU/GPU construct/gradient/headroom 검사를 실행하고 예산을 별도로 기록한다.
   유효한 가설이면 방법 모듈·강한 대조·실행기·검증기·테스트까지 구현한다.
   모든 프로토콜과 선택 기준은 E를 보기 전에 Git에 기록한다.
   기존 E는 이미 공개된 development로 명시한다. 추가 확증에는 따로 확보한 평가 구간/데이터를 쓴다.
   과거 결과와 캐시를 덮어쓰지 않고 별도 run_id를 사용한다.
   비교군의 입력·라벨·teacher 정보·HPO 기회 및 예산을 공정하게 맞춘다.
   V 선택을 봉인한 후 E를 평가하고 독립 지표 재계산을 수행한다.
   GPU가 차 있으면 기다리고 외부 프로세스를 종료하지 않는다.
   scripts/with_cuda.sh와 기존 GPU lock/자원 guard를 유지한다.
   새 후보 둘 다 construct/novelty/headroom에서 반증되면 억지 학습을 하지 말고
   그 근거와 실제 한 진단 수를 기록한다.

4. 구현, 검증, 실제 실행 상태를 보고한다. 최종 학습이 백그라운드에서 계속되면
   "실험 완료"라고 쓰지 말고 PID·상태·로그·결과 경로를 남긴다.
   사용자 승인에 따라 기존 GitHub main에 합리적인 단위로 commit/push한다.
   실행 중 기록된 파일은 완결 여부를 구별하고 raw/model/cache는 Git에 올리지 않는다.
   pytest와 적절한 검증을 실행한다. 작은 변경마다 전체 과거 실험을 반복 학습하지 않는다.
   스크린의 실행 source hash에 포함된 기존 파일은 부모가 완전히 끝나기 전 변경 금지.
   권한이나 인증이 막히면 우회 옵션/무제한 권한을 사용하지 말고 실제 막힌 단계와 이유를
   파일과 최종 답에 명시한다. 성공으로 표시하거나 자동 재시도하지 않는다.

최종 한국어 요약을 automation/overnight_followup/FOLLOWUP_RESULT.md에 쓰고
최종 응답에도 현재 결과와 미완료 부분을 분명히 적는다.
감시 로그는 .cache/overnight_followup/continuation.jsonl,
최종 응답 사본은 .cache/overnight_followup/final_message.md에 남는다.
