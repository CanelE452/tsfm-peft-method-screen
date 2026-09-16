# 감사·유지보수 기록 — 학습 결과와 구분

본학습48fits, smoke24updates 및 평가·151,200개 scalar 검산은 실행 오류 없이 완료됐다. 아래는 완료 후 추가 감사와 미사용 재개 경계에 관한 수정이다. 데이터·학습률·seed·가중치·점수·성능 기준·scalar rtol/atol1e-10을 바꾸거나 추가 학습하지 않았다.

1. 추가 TRAIN 통계 감사기가 원자료 기상 열/실행 Pandas reduction이 float32였다는 사실을 놓치고 새 float64 통계와1e-12 일치를 요구했다. 첫 감사에서 평균 차이3.21e-6이 검출됐다. 코드의 dtype 가정을 바로잡아 NumPy의 float32 평균과 float64 분산→float32 분산→float32 제곱근으로 실제 계산을 독립 재현했고, 저장된 평균·표준편차와 **exact** 일치를 확인했다. 별도 float64 대안과의 차이는 completion_audit.json에 기술적으로 기록했다. 실제 통계는 변경하지 않았다. 첫 실패 traceback은 로컬 `.cache/forecast_path_structure_v1_20260916/completion_audit_first_attempt.log`에 보존했다. 모델 점수의 float64 검산은 처음부터 통과했다.
2. 기존 guard의 `external_compute_samples`는 허용된 RustDesk도 포함한다. 원기록을 바꾸지 않고 허용 여부를 구분한 실제 미승인 compute 표본0을 보고했다.
3. runtime의 일부 smoke 추론이 이전 train phase 태그를 유지해 원시 phase counter에33회가 잘못 분류됐다. 전체 forward 수에는 누락이 없다. 별도 완료 감사의 호출 구성 대조로 optimizer24,600(본학습24,576+smoke24), 저장 예측19,892, 검증72, 전체44,564로 일치시켰다. 원시 태그와 교정된 집계를 모두 보존했다. optimizer 로그와 실제 학습 횟수는 변경되지 않았다.
4. 실제 실행은 재개 없이 완료됐다. 이후 코드 검토에서 epoch resume state는 저장됐지만 V256/V512 checkpoint 기록 직전에 중단되는 미사용 경계가 드러났다. 재개 시 step0뿐 아니라256/512도 checkpoint 함수를 통과하도록 한 줄 수정했다. 이미 있는 checkpoint는 그대로 참조한다. CPU 모의 중단 회귀 검사에서 resume512로 누락 V512를 기록하고 optimizer/GPU 호출0으로 완료함을 확인했다. 첫 모의 검사 하네스에 `data` 속성을 빠뜨린 오류도 보완했다. 본실험 재실행은 없다.
5. `evaluate.py`의 빈 줄 공백만 제거했다. 점수·선택·보정·bootstrap 계산은 변경하지 않았다.

실제 학습·채점에 사용한 실행기 버전은 **622d79c8cc80e4b2342754d1b0624af64be2ba79**이며 implementation_seal/evaluation_seal의 원래 hash를 유지한다. 현재 runtime의 미사용 재개 경계 수정과 evaluate의 공백 정리는 그 실행 이후다. 실제 실행 소스는 해당 Git commit에서 hash로 검증할 수 있다. 원시 예측과 전체 재개 상태는 로컬 ignored cache가 필요하다.
