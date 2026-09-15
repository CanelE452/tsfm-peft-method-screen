# 실행 전 코드·연구 검토

- [코드] 실제 Chronos-2 설치본 pipeline.predict의 dictionary schema와 batch_size의 variate 기준을 읽었다. F0와 PAST/MEAN/PATHS는 서로 다른 schema이므로 별도 호출한다. PATHS 네 개는 각 target1+covariate3으로 batch16에 들어가며 cross_learning=False로 서로 다른 시나리오 간 group attention을 막는다.
- [코드] 부하 context [o-336h,o)와 target [o,o+24h)는 15분 grid에서 겹치지 않는다. 모든 시간 평균은 네 실제 값으로 계산한다. 입력 worker가 여는 npz에는 target context/past weather/future paths만 있고 미래 target은 별도 파일이다. 실제 target은 prepare의 completeness 검사와 report의 점수 계산에서만 읽는다.
- [코드] 28개 모두의 미래 부하·기상 poison 검사, 알려진 uniform/원자/동일분포 혼합 반례, scalar pinball 검사를 실행했다. C-only 보정 파일 작성 이후에 D target 점수 함수를 호출한다. 저장한 분위수와 비용을 개별 원점에서 보존한다.
- [연구] 나이 다른 vintage의 동일 가중은 참조 가설이며 보정된 확률 예보가 아니다. 이 비교가 부진해도 모든 미래 입력 불확실성 방법의 부재를 입증하지 않는다. 과거 날씨 실제 공개시각 문제를 피하려고 관측값 대신 명시된 simulated available_at의 예보만 쓴다. 실제 운영 가용성 보장으로 표현하지 않는다.
- [연구] C에서 동일한 편향·폭 보정을 모든 주요 방법에 주고, 계절 반복의 확률 대조도 둔다. F0보다 이겨도 보정된 단순 대조에 지면 학생 증류 근거로 삼지 않는다. 일반 LoRA/새 PEFT의 추가 가치는 이번에 미평가다.
- [연구] 본학습 0회인 참조 가치 검증이다. 신규 방법 이름을 붙이지 않는다. 범위는 한 타깃과 고정 기간으로 제한하고 결과 후 교체·추가 탐색을 하지 않는다.
