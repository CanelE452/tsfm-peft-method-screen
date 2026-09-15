# 구현·연구 검토

구현: 봉인한 source와 입력/예측 hash가 일치한다. Python 문법검사, 기존 원점수 재생, scalar primary, V grid, 변환 순서·폭, D target poison, 보조지표·MSE 분해와 Markdown 링크 검사를 통과했다. main의보정7회와 검산재계산7회를 구분했다. 새 neural fit/forward는 없다.

공개 형식 검사: git diff --cached --check는 실행 소스 run.py139행 문자열 리스트의 쉼표 뒤 trailing whitespace 한 건을 알렸다. 코드 의미나 계산에는 영향이 없는 공백이다. 이미 실행·봉인된 소스의 byte hash를 유지하기 위해 그대로 보존했고, 이 형식 검사를 무경고 통과했다고 기록하지 않는다. 다른 whitespace 경고는 없다.

연구: V/D는 노출된 개발자료이고 raw/control조건을유지했다. 모든 군에 같은 보정을 적용하고 원본도 공개했다. 중심화 MSE는 설명 통계이며 배포정책이 아니다. 양의 수치와 신규성은 구분했다. 원래 selected정책은 변경하지 않았다. 일관성이라는 일반 원리는 기존 FR/PACE와 겹치므로 새 방법으로 재명명하지 않았다. 한타깃 및 V4에서의 추가선택·보정 불확실성은 남는다.
