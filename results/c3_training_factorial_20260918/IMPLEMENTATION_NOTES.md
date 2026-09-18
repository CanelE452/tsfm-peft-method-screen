# 고정 코드의 세부 표기

명세의 bootstrap seed89418은 봉인된 score.py에서 사용하는 기본값이다. 실제 패널별 seed는89418+패널 index, 즉 Electricity89418, Electricity전이89419, ETTm189420이다. 같은 패널의 조건·checkpoint 유형에는 동일한 block 재추출을 공유한다. 이는 결과 확인 뒤 변경한 설정이 아니라 봉인 당시 코드의 명시적 전개다.

추가된 final_audit.py와 report.py는 학습·선택·metric을 수정하지 않는 검산·문서 생성기다. 고정 학습/evaluate/score의 원본은 SEAL.json과 사전 commit에 보존되어 있다.
