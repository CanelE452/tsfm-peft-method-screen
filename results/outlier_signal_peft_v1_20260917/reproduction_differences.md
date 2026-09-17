# 제한 재현의 실제 차이와 실행

공개 Bolt-small/T5-small 각각49 series, 총98 series를 실행했다. optimizer0이다. T5는20 samples의 중앙 두 값 평균 median이며 내부 model forward3,136회, Bolt는49회였다. 결과는 reproduction_scores.csv와 reproduction.json에 있다.

원 notebook의 magnitude sweep은 네 고정 위치와 계수, 100 amplitudes 및 일부 미공개 patch-size-1 경로를 사용한다. count sweep은 앞256위치 permutation과50×Gaussian noise를 사용하며 seed가 고정되어 있지 않다. 이번 계약은 count1/16 × amplitude1/10/100 ×8 seeds + reference1이다. x의512위치 중 seed로 고른 점에 amplitude×Gaussian을 더했고 미래 cos는 유지했다. 따라서 원문 난수를 동일하게 재생한 것이 아니라 reduced replication이다. T5 samples20도 일부 원문 호출의1과 다르다.

cos(t+0.3), 주기160, 과거512, 미래64 및 공개 모델 연결을 확인했다. 이는 Figure8 전체나 학습된 patch-size1 재현이 아니다. patch-size1은 SKIPPED_UNAVAILABLE_REFERENCE로 보존한다. 약한 민감성이나 불리한 재현 결과로 여섯 군의 본 비교를 막지 않았다.
