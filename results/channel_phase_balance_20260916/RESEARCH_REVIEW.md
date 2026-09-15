# 코드 검토·해석 경계와 선행 확인

## 코드 검토

[확인] 이 실행은 이전 identity 진단의 검산된 제어 흐름을 별도 새 파일로 작성했다. 학습 모델은 기존 rank2_model.make('LH') 그대로이며 residual gate/주파수/채널 모듈은 없다. 기존 실행 모듈의 OUT/CACHE 같은 전역값을 바꾸지 않았고 이전 main/prepare/train entry point를 호출하지 않는다.

[확인] 실제 초기 파라미터 hash를 과거 LH의 각seed epoch0 checkpoint와 대조했고, 실모델 smoke에서 같은 batch의 과거 BF16 INIT 예측을 exact 재생했다. train 원점 외의 manifest 값, V/E 원점·정규화·채널이 동일하다.80개 epoch schedule의 표본 index 순열도 과거와 같으며, 그 index가 가리키는 원점만 바뀌었다. 미래값을 바꿔도 모든 train batch는 같았다.

[설계] 선택 checkpoint와 기존 선택epoch에 맞춘 checkpoint를 모두 사전에 정했다. E 결과를 보고 그중 하나만 발표하지 않는다. 같은 checkpoint일 때만 cache를 공유하며 fits 수는4 그대로다. 총학습은20epoch 상한·같은patience 규칙이지만 실제 종료epoch는 달라질 수 있다. 같은updates 비교가 이 차이를 보완한다.

## Time-PEFT 공개 loader와 로컬 파일럿의 차이

[확인] 보관한 공개 source의 [Dataset_Custom.__getitem__](../../sources/timepeft_ea4e7e1/data_provider/data_loader.py)은 s_begin=index이고 __len__은 전체 가능한 sliding-window 수다. [data_factory](../../sources/timepeft_ea4e7e1/data_provider/data_factory.py)는 train에서 shuffle을 사용한다. 해당 기본 경로는 로컬 파일럿의 stride24·최대512원점 제한과 다르다.

[한계] 이 코드 차이는 공개 논문의 모든 실제실행 옵션이나 원점 구성을 완전히 재현했다는 뜻이 아니다. 기존 R2도 전체 Time-PEFT 재현으로 기록하지 않았다. 이번 balanced512/504 역시 모든 가능한 window를 학습하는 전체 upstream recipe와 같지 않다. Time-PEFT의 성능 표를 재현했다고 주장하지 않는다.

## 새 방법을 고를 때 유지할 강한 대조

[추정] sampling 대조에서 큰 회복이 나타나더라도, 그 결과를 새로운 모듈의 기여로 바꿀 수 없다. 반대로 기존 편중된 원점에서 추가 모듈이 효과 없었다는 사실만으로 올바르게 분산된 원점에서도 모듈이 무용하다고 단정할 수 없다. 모듈 간 기존 비교는 그 지정 조건 안의 결과로 보존한다.

[확인] [CycleNet의 arXiv v1 본문 §3](https://arxiv.org/html/2409.18479v1)을 이번에 읽었다. 채널별 학습 주기 Q를 입력·예측 시점에 맞춰 정렬하고, 입력에서 주기를 뺀 잔차를 예측한 다음 미래 주기를 더한다. 지난 검토에서 접근되지 않았던 것은 v3 HTML이었다. 주기 표의 학습·정렬만 붙이는 후보는 이 가까운 대조를 무시할 수 없다. 이번 sampling 대조는 CycleNet을 구현한 것이 아니다.

[확인] [TRACE arXiv v1 본문 §4.1](https://arxiv.org/html/2503.16991v1)에서는 MOMENT의 flatten forecast head를 차원 축소·분해·pooling·convolution 등으로 재구성한다. head 경량화나 선형 두 층 분해만으로 새로운 PEFT라고 주장하지 않는다. 이 검토는 TRACE 전체 학습·논문 결과 재현이 아니다.

다음 PEFT 후보는 먼저 이 sampling 대조로 확인된 baseline 위에 추가 가치를 보여야 한다. 좋은 점수를 얻은 통계 보정은 E에서 추가 정답을 사용했으므로, 같은 정보량의 neural PEFT 비교와 섞지 않는다. 새로운 미사용 조건에서의 검증과 신규성·기여 구분은 여전히 필요하다. 이번 실행에서 sampling이 좋다는 결과와 방법론 주제를 확보했다는 결과는 다르다.
