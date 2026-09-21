# 실행 계약과 중단 범위

유일한 실행 계약은 [CONTRACT.txt](../../experiments/mag_real_levelshift_confirmation_v1_20260921/CONTRACT.txt)다. 고정 MAG를 수정하지 않았으며 과거 지시문을 병합하지 않았다.

TRAIN/V/E 경계, 실제 raw E 512→64, past-only S>=3, 14일/50원점 coverage, 두 repeat seed, 7일 UTC calendar block 2000회 및 방법별 비교는 계약 그대로다. 기존 학습은 batch32/FP32/AdamW, 1024updates, 두 LR, checkpoint0/256/512/768/1024, 다섯 V 조건의 동일 가중 nMAE다. B0 shuffle key83100과 second-stage84100을 구분한다. MAG·PLAIN8712개, LoRA294912개, OUTPUT_CONTEXT4673개는 기존 코드의 값이며 이번 source의 실제 모델 검산을 했다는 뜻은 아니다.

NYISO 전체 243 CSV에 canonical system-total이 없어 source 계약이 성립하지 않는다. 임의 zone 합산은 금지되어 생성하지 않았다. 예측과 채점0회인 상태에서 승인된 ISO-NE fallback을 확인했으나 공식 CSV 접근이403이었다. 따라서 학습·모델 smoke·평가 runner는 실행하지 않았다. 데이터가 유효하지 않은 상태에서 예산을 채우기 위한 실행을 하지 않는다.

재개 조건: 계약의 ISO-NE 공식 Hourly Real-Time System Demand 2026년1~8월 원본을 정당한 공식 다운로드 절차로 확보하고 원본 receipt와 시간·schema·품질·노출 감사를 통과해야 한다. 다른 원천/threshold/기간 변경은 이 실행 범위가 아니다. 유효 입력 확보 뒤 실제 runner 구현·검사와 실행 봉인이 아직 필요하다. 자동 재개/추가 학습은 없다.
