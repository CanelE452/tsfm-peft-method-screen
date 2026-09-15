# 버전·노출·중복 감사

[확인] HEAD `6ceec43d11747505bc1d5ddaa70de435c65dd370`; 기준 commit과 tracked diff 없음: True. 원래 결과 1810개 hash를 보존했다. 현재 신규 run의 파일만 추가 중이다.

[확인] 기존 heldout6은 이전 fit/예측 원장과 이후 source/tune/dev에서 사용되지 않았고, runtime 결과 및 cache의 예측/weight 파일명에도 노출 증거가 없다. Manifest·eligibility·해시에서의 등장과 성능 노출을 구분했다. 이 감사에서 LOCKED target 배열을 decode하지 않았다.

[확인] 현재 동일 실험 worker 없음, GPU RTX3080 free9031MiB, compute는 기존 승인 RustDesk272MiB만 있었다.

[설계] 이 파일의 유한한120-fit 계약을 먼저 이행한다. 사용자 상위 목표의 이후 후보 탐색은 이번 판정·heldout을 되돌려 PASS로 바꾸는 권한이 아니다. 이번 run 종료 후 별도 근거와 새 평가 보호를 갖춘 작업으로만 검토할 수 있다.

[확인] 상세 분할/감사: [audit.json](audit.json), [episodes](episodes.json).
