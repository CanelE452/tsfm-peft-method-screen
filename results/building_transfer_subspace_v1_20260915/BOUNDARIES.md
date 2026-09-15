# 실행 해석의 추가 경계

개발 점수를 열기 전 데이터 manifest에서 확인했다. 이 실험은 **건물 분리 offline transfer**이며 모든 source 관측 시점이 모든 target origin보다 앞서는 전역 시간 분리는 아니다. 예를 들어 source에는2017년 구간, dev에는2016년 origin이 있다. Target 건물의 미래를 gradient/selection에 사용하는 것은 아니지만, 과거 그 시점에 실제 배포 가능한 온라인 예측 성능이라고 해석할 수 없다. 점수가 좋아도 동시기 정보 효과를 완전히 배제한 causal deployment 실험은 별도로 남는다.

Chronos-2 사전학습 corpus와 BDG-2 건물의 중복 부재도 이번에 입증하지 않았다. 보고서의 새 건물은 본 실험의 source/tune과 physical building이 다르다는 의미이며 foundation model이 역사상 처음 보는 시계열이라는 뜻은 아니다.

데이터 범위·학습 설정·허용오차·continuation gate를 변경하지 않았다. 이것은 효과를 통과시키기 위한 수정이 아니라 결과 해석의 제한이다. 추가 학습은 이 확인으로 자동 실행하지 않는다.
