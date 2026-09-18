# 자료·노출·재사용

기존 Electricity/ETTm1의 matched TRAIN256일×4채널, V64일, E128일을 정확한 cache/hash로 재사용한다. 각 source의 TRAIN sigma, 원점, draw, 상태/크기/기간 schedule과 seed별 순서 함수는 그대로다. 전력 transfer16개 ID도 기존 봉인 목록 그대로이며 새값/성능선정은 없다.

Electricity/ETTm1/transfer E는 이미 여러 번 성능을 본 개발자료다. 동일 원점과 모든 불리한 채널·seed·상태를 보존한다. 전력transfer16은 기존B0 미학습계열이지만 과거노출4/불명12이며 독립자료가아니다. 메타데이터사전학습중복은불명이다.

source별 selected B0는 부모 baseline manifest로고정하고C3/RECENCY 및전체기준선은부모가중치·저장예측을검증후재사용한다. 미래y는loss/V선택/봉인후E채점에만사용;gate·feature·정규화·원점선정에는주지않는다. raw observed입력보존, 실제사건레이블없음.

NESO2026재채점으로새확증을만들지않는다. ETTm2의빠진기전대조를채우는학습과LCL대량다운로드/연결/학습은이번범위가아니다.
