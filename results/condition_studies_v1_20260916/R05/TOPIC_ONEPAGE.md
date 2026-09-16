# R05 VINTAGE


9. R05 VINTAGE — 저장 예측으로 최신/지연 손익 정책부터 비교
======================================================================

neural fits=0. 기존 forecast_path_structure_v1_20260916의 선택된 LATEST/PATH를 고정.
원래 LR/step을 새 목적에 유리하게 다시 선택하지 않는다.
현재 결과에서 'LATEST'는 이전 네 상태평균의 V 선택모델이었다는 한계도 표시.

타깃: 기존T0/T1/T2, seed61730/61731, S0..S3 원래 입력case.
V_SELECT7월로 아래 정책만 고정하고 E10~12월을 재분석.
E는 이미 점수가 노출됐다. 명칭은 REANALYSIS_REUSED_E; 새독립평가가 아니다.
큰 예측파일이 없으면 기존 해시가 맞는 checkpoint+입력으로 필요한 예측만 재생 가능.
추론을 재생했다고 LoRA fit으로 세지 않는다. 가중치를 새로 학습하지 않는다.

q_alpha(k)=(1-alpha_k)*q_LATEST(k)+alpha_k*q_PATH(k).
quantile vector의 convex mixture이며 확률분포 mixture의 quantile이라고 부르지 않는다.
원래 정렬된 quantile을 사용. 같은tau끼리 조합하고 교차 여부 검사.

5정책:
E0 LATEST: alpha=[0,0,0,0].
E1 PATH: alpha=[1,1,1,1].
E2 GLOBAL: alpha공통, {0,.25,.5,.75,1} 중 선택.
E3 BINARY: alpha=[0,1,1,1] 고정.
E4 MONOTONE: 같은5값에서 0<=a0<=a1<=a2<=a3<=1인70개 조합.
모든 점수는 V의 same origin/case/target 사용. 타깃별계수는 두 seed평균V로 하나.
E2/E4 선택: V S0 loss <= E0 S0 loss*1.01인 조합 중 평균S1..S3 loss 최소.
동률: 평균alpha작은 것, 사전lexicographic순서.
alpha0=0인E4는S0에서LATEST와정확히같다. 그조건에서LATEST보다개선한다고 기대/요구하지 않음.
E4의 직접상대는E2뿐아니라 E3 BINARY. 동일행동이면중복실험이아니라같음으로기록.

보정은 추가하지 않는다. 원래 raw와 원래 각모델보정결과가 모두 있으면 보정판은 민감도표만.
주목표: 최신S0 손해1% 이내에서 S1..S3 평균/최악손해 감소.
실제 갱신지연빈도나 rank별확률을 추정했다고 쓰지 않는다.
S0/S3 개별표, 전체타깃/추가타깃만, seed별표 공개.
두모델 추론비용 포함. alpha끝점은한모델, 중간이면두모델이며 단일PEFT 실행아님.
모사 available_at, rank != 실제예보나이/정확도/ensemble member 한계를 유지.

정책유용성은 확인 가능하나 학습방법신규성은 주장불가.
E4가E3보다특별히좋지않으면복잡한age PEFT를만들근거는약하다.
이것이기상예보학습전체가불가능하다는판정은아니다.


[설계] 전체 계약의 공통 조항도 적용. 실행/효과/단순 대안/신규성을 분리한다.
