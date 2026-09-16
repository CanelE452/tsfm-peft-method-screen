# 실행 전 고정한 데이터 정의의 구체화

공식 메타데이터의 group/name을 NFC로 변환하고 U+001F 구분자로 연결한 UTF-8 SHA256을 사용했다. 성능과 무관한 순서는 candidate_order.json에 있다. T0를 제외한 첫 두 후보가 TRAIN 조건을 통과해 추가 탈락/교체는 없었다. 파일 SHA256과 위치·설명 identity를 대조했고 세 load 파일 hash는 서로 다르다.

시간 평균은 원자료의 연속15분 네 행을 float64로 평균한 뒤 모델 입력만 float32로 저장한다. 부하 scale은 원자료에서 계산한 unique hourly TRAIN context+label 평균의 표준편차다. 외부 기상 통계는 TRAIN 과거 입력을 구성한 원자료의 `(valid timestamp,available_at,feature)`를 중복 제거해 계산했다. 즉 중복 context의 반복 노출로 통계에 가중하지 않았으며, 미래 vintage는 외부 정규화 통계에 넣지 않았다. 모형의 native context normalization은 그대로다.

역할별 끝 후보일(6/29,7/30,8/30,12/30)의 08시 원점은 정답24시간이 그 역할의 달력 종료 경계를 넘어 제외했다. 이 규칙은 준비 코드 실행 전에 정했고 성능에 따라 바꾸지 않았다. [경계 제외](calendar_boundary_exclusions.csv). TEST의 달력 후보91개에서 경계1개와 부하 context 결측14개를 제외한 입력76개/타깃을 준비했다. 정답 결측은 봉인된 예측 이후 별도로 확인해 네 군·모든 case에 공통 적용한다. context 누락 날짜는 원점 manifest에 있다.

TEST source parquet는 합법적 context 구성에 필요하므로 파일 자체가 미열람이었다고 주장하지 않는다. 입력 함수는 origin 이전 load 행만 선택하며 미래 정답은 별도 파일로 준비하지 않았다. model/selection/calibration 단계는 TEST label 파일을 받지 않는다. 평가 예측 해시를 저장한 뒤 scorer만 TEST horizon 정답을 추출한다. 뒤 평가일의 합법적 과거에 앞 평가일의 부하가 포함되는 것은 허용하지만 그 값으로 가중치를 업데이트하지 않는다.
