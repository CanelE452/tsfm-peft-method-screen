# 연구 비판 검토 — 메인 CLI가 코드 검토와 별도로 수행

[확인] 한 후보의 핵심 차이는 target residual의 두 성분으로 **동일 trace의 output penalty 기하**를 결정하는 것이다. uniform 보존을 SIMPLE로 직접 두고 STD/OLS/SEASONAL24/fixed120보다 나은지 본다. Native supervised task loss와 trainable 수는 유지한다. 아예 source transfer가 성립해야 한다는 진입 gate를 사용하지 않는다.

[추정] 가장 강한 반례는 과거 두 창의 잔차가 일치해도 다음날 운영 패턴이 바뀌는 경우다. 평균 분산식의 독립 가정은 현실에서 보장되지 않는다. 또24시간 형상은 여러 성분인데 scalar1개로 묶어 필요한 일부 시간대 보정을 막을 수 있다. 큰 signal이 bias/shape 어디에 있는지만으로 미래 이득을 보장하지 않는다.

[확인] 비등방 regularization, shrinkage, level/shape 분리는 알려진 원리다. 여섯 논문과 식의 직접 동일성은 확인되지 않았으나 넓은 선행 부재를 입증하지 못했다. **NOVELTY=UNRESOLVED**를 유지한다. 결과가 좋아 보인다는 이유로 신규성을 승격하지 않는다. 지금은 이 구체 가설을 한 번 직접 비교할 연구 근거가 있다는 판단이며, 새 논문 주제가 확보됐다는 판단이 아니다.

[설계] 초기 SIMPLE/CANDIDATE CPU prototype은 STD 종료 직전에 식 검토용으로 작성·검사했다. TOPIC 봉인 전 후보 GPU fit/score는0이다. 정식 후보 개발·잠금 평가에는 봉인한 이 버전 하나만 사용한다. 이 준비 순서의 차이는 기록하고 숨기지 않는다. 후보교체0, 명세의 결과 기반 변경0. 최초 공개 버전의 수식/대조를 그대로 봉인하며 추가 수정 필요성은 현재 없다.
