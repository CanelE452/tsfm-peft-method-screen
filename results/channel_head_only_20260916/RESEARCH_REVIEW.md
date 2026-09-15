# 코드·연구 비판 사전 검토

[확인] 기존 LH factory는 새6144→96 linear head와 encoder q/k/v LoRA를 동시에 학습한다. head의589920개와 LoRA172032개를 분리하지 않고 최근 이득 전체를 LoRA 성과로 말할 수 없다. 원래 factory를 사용한 뒤 encoder만 동결하면 초기 head 난수와 forward를 유지하면서 optimizer 대상만 바꿀 수 있다. 기존 LoRA dropout/encoder train 모드는 유지하므로 완전히 결정론적인 feature-cache LP와는 구분한다. 기존 모델 복원은 trainable state만 다루므로 이번 checkpoint에는 head만 들어가며 fresh factory의 동결 초기 encoder에 로드한다.

[확인] [MOMENT (ICML 2024)](https://proceedings.mlr.press/v235/goswami24a.html)의 공식 논문 페이지와 [TRACE v1](https://arxiv.org/html/2503.16991v1)의 초록·서론을 재확인했다. TRACE는 linear probing과 큰 forecasting head 문제 및 head 축소와 LoRA 선택을 이미 다룬다. 이번 head-only 비교나 단순 저랭크 head를 신규 PEFT 방법으로 명명하지 않는다. 본 실행은 두 논문의 전체 재현이 아니다.

[추정] 위상 균형 표본은 큰 head의 일반화를 회복시킨 것만으로 충분할 수 있다. 반대로 encoder 표현 적응이 추가적으로 필요할 수 있다. 두 설명을 같은 창·난수·학습률·평가에서 한 번 비교한다. head-only만 더 튜닝하거나 LoRA의 기존 좋은 결과만 선별하지 않는다. epoch1/3 비교는 과거 원래 LH의 V 선택에서 이미 정해졌고 새 LP 결과와 무관하다. 각 방법의 V 선택 비교도 함께 공개한다.

[한계] LoRA 동결에 따라 backward와 optimizer 비용이 바뀌는 것은 의도된 구성요소 제거다. 같은 parameter budget 비교는 아니며 실제 절감을 보고한다. 기존 실행과 현재 실행의 wall 차이는 시간대와 자원 상태 영향을 받을 수 있으므로 절대적 속도 벤치마크로 과장하지 않는다. 노출 E에 대해 새로운 유의성 PASS를 주장하지 않는다.
