# 제한 재현의 경계 — 아직 미실행

원문 notebook의 magnitude sweep은 고정 네 위치와 계수, 100 amplitudes를 사용하며 일부 경로는 미공개 patch-size-1 모델을 요구한다. count sweep의 난수 seed도 고정되어 있지 않다. 계약은 공개 Bolt/T5 두 모델 각각 reference 1 + 2 counts × 3 amplitudes × 8 seeds = 49 series, 합계 최대 98 series로 제한한다. T5는 원문 일부 호출의 1 sample 대신 20 samples의 median이다.

이는 reduced replication이며 Figure 8 전체 재현이 아니다. patch-size-1은 SKIPPED_UNAVAILABLE_REFERENCE로 남긴다. 모델 다운로드 및 실제 코드 확인만 완료했고, 제한 재현 forward 0/98이다. 실제 모델 CPU 연결 검사에서 사용한 Electricity 8개 입력의 forward는 이 98 series에 포함하지 않는다.
