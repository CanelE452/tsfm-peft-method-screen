# 선행·연구 비판·구현 검토 — 학습 전

## 직접 읽은 선행과 남은 차이

- [LST, NeurIPS2022, §3.2–3.3](https://proceedings.neurips.cc/paper_files/paper/2022/file/54801e196796134a2b0ae5e8adef502f-Paper-Conference.pdf): 동결 backbone의 중간표현을 작은 side network로 전달하고 그 경로만 역전파한다. 따라서 no-backbone-backward나 층별표현 전달 자체는 새롭지 않다.
- [LAST, arXiv v1(2024), §4.1–4.2와 Algorithm1](https://arxiv.org/html/2402.04009v1): LN 이후저차원Q/K/V·attention·up·residual, 층별feature합산과bias보정이다. 이번SIDE는이원리에착안했지만zero-up,잔차stream의주입규칙,연결8개·단일head/폭10은이번MOMENT통제설정이다. 원논문성능재현이라고하지않는다. PRIOR의직접변경은원래attention분포를side의logit기준으로추가하는것이다.
- [Ladder Up, Memory Down, arXiv v1(2025), §3](https://arxiv.org/html/2512.14237v1): ladder의연결위치·깊이와계산/메모리절충을다룬다. 단순히중간층을추가로연결한다는차이만으로신규성을주장하지않는다.
- [Parameter-Efficient Fine-Tuning ... Instance Segmentation, MAKE2024, §4.3](https://www.handmann.net/pdf/MAKE-BakRohHan2024.pdf): attention출력뒤의MLPadapter와key/attention출력residual을다룬다. 논문에서attention prior라고부르는부분이이번확률분포의logit추가와같은수식이라고확인되지는않았다.
- [Integrating Task Specific Information ..., Findings EMNLP2020, §3.2](https://aclanthology.org/2020.findings-emnlp.285.pdf): label임베딩과토큰의attention점수로CLS행을수정한다. target label임베딩을쓰는점과내부attention을수정하는점이다르다. prior를attention에넣는발상전체가처음이라는주장은불가하다.

원문방법부분을확인했지만위방법들을모두이환경에서재현하지않았다. LAST v2 HTML접근은실패해v1본문과PDF를확인했다. venue를확인하지않은arXiv자료는학회채택논문으로표기하지않는다.

## 수학적 반례와 비판

P=softmax(logP0+S)는P0×exp(S)의행별정규화다. 새로운확률계산원리가아니다. 특히단일frozenhead에서logP0는기존Q0K0^T와행상수만다르므로,추가QKscore와의합은고정feature와학습feature를연결한attention으로표현가능하다. 여러head분포의평균을쓰는이번구조도이런관련구조들과넓게비교해야한다. 논문최초성은UNRESOLVED다.

관찰된QK/V제거손실은공동적응효과를깨뜨린결과다. 이것만으로이번side구조가LoRA를근사한다는주장은성립하지않는다. frozen층의attention은side표현의변화를다음backbone층에반영하지않고,head평균은head별역할을버린다. 원래prior가나쁘면SIDE보다학습을방해할수있다. PRIOR와SIDE차이가작거나없다면이추가prior가필요하다는가설을지지하지않는다.

## 구현 검토

기존학습runner의data/순열/optimizer/checkpoint/선택·평가정책을사용하고독립run경로로구현한다. 초기모델은동일LHfactory,encoder LoRA는B0로동결,head초기값동일. 관측용q/k/SelfAttention hooks는출력을대체하지않으며dropout전분포를추출한다. hooks에서gradient를끊고같은batch의8층기록만소비한다. side결과는backbone에주입하지않는다. 두군의실제총trainables가LoRA와같은761,952인지검사한다.

FP64검사에서초기함수보존·동일가중치의prior반례·중립prior환원을확인하고,실제BF16모델smoke로정상gradient/초기예측/저장재생을확인한다. 검사통과시고정8fits를실행하는개발비교이며,개발결과를독립PASS나최초성으로승격하지않는다.
