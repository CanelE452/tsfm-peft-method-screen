# 긴 이력 압축 PEFT — 봉인 개발 파일럿

사용자의 2026-09-17 “그렇게 해서 실험해줘” 요청으로 실행한다. 앞서 종료한 트랙은 재개하지 않고, 교정된 N02의 실제 입력·원점·기준선만 읽기 전용 재사용한다.

SHORT/LONG 완료8fits는 검산 후 참조한다. 새5군 × (선택seed 두LR + 반복2seed 고정LR) =20fits,512updates씩, 총10,240본업데이트. 군별2smoke=10폐기업데이트. 다른 데이터·후속학습 자동 실행 없음. 모든 군은 성능 gate 없이 정해진 비교를 마친다. 공통 GPU 위험은 정지/재개 상태로 보존한다.

최근336시간21patch 유지, 오래된1008시간63patch를 인접3개씩21token으로 요약한다. 전체 context token84→42, REG/미래 포함88→46. 과거 원시값을 가짜 동일간격 시계열로 재해석하지 않고 native patch embedding을 풀링하며, native 시간 feature와 원래 patch 중심 position을 보존한다. STATS_SHORT는 긴 이력 native normalization만 계산하고 최근21patch만 encoder에 넣는다. 각 압축군은 같은 full-context normalization을 쓴다.

POOL은균등convex평균, LEARN은공유rank8비선형score로같은3개 내convex가중. 초기u=0으로POOL과동일. POOL_KD/LEARN_KD는같은label MSE에0.25×교사예측MSE 추가. 교사는E와무관하게기존B1의선택seed73100,V선택체크포인트로고정, TRAIN예측만생성한다. 학생도동일1344관측을사용한다. TRAIN포함교사fit비용과cache생성비용을공개한다.

주 비교는LEARN_KD대POOL_KD(학습압축),LEARN_KD대LEARN(증류),LEARN대POOL,POOL_KD대POOL. SHORT/LONG/STATS_SHORT도모두공개한다. 낮은NRMSE가좋고,두반복평균과시간block bootstrap을보고한다. V에서만LR/checkpoint선택, 모든선택봉인후E채점, 출력보정없음. E는재사용개발기간이다.

추가가치의탐색근거는두핵심비교의평균양수및95%CI하한>0일때만기록한다. 자원절충신호는LONG대비측정peak또는latency10%이상감소와상대오차95%상한1%이내를같이요구한다. 현재결과에맞춰문턱을수정하지않는다. 이는논문PASS기준이아니다. 한쪽seed악화도보존한다.

선행의압축메모리와TS-Memory증류와겹치므로풀링/증류조합의신규성을주장하지않는다. https://arxiv.org/abs/2409.13530 , https://arxiv.org/abs/2602.11550 . 정식재현비교와독립원천은범위밖이며다음연구가자동승인된것이아니다.
