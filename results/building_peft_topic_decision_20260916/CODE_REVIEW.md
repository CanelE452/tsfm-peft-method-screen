# 구현 검토

[확인] runtime.py와 runner.py의 STD 소스는 std_seal 이후 변경하지 않았다. Worker는 H와 whitelist job만 받아 native loss로 학습한다. .5 index는 실제 config, F0/step0·target poison·새 모델 복원은 STD smoke에서 exact 일치했다. STD CPU5검사 통과. 미사용 평가 건물은 아직 target decode0이다.

[확인] candidate.py는 기존 native loss에 penalty 하나만 추가한다. F0 no-grad/disabled, 입력 sigma와 H내 과거 target만으로 계수를 고정한다. Model parameter 추가0, 추론 transformNone. 행렬 projector 대비 FP64 output/gradient, trace, 동치, 합법 입력 차이, zero penalty/gradient5검사 통과. 이 검사는 실모델 candidate smoke를 대체하지 않는다.

[설계] 후속 compare.py는 기존 STD를 다시 학습하지 않고 SIMPLE/CANDIDATE32를 실행한 뒤 모두 같은 규칙으로 선택한다. 후보 개발점수 부호로 LOCKED를 취소하지 않는다. 선택 seal 이후만 LOCKED 채점하며 최강 고정 대조에서 각방법 fixed120/F0/AFFINE/SEASONAL24를 누락하지 않는다.
