# 사전 코드·연구 비판

## 관찰과 반대 설명

최근LP 대조는LoRA표현적응이실제추가이득을보인다는근거다. 하루잔차분해는본후보의인과근거가아니며, 추가비선형성이나uniform smoothing만으로충분할수있다. 따라서동일606304파라미터의POINTWISE/UNIFORM을둘다실행한다. 기존LP와균형LH도동일평가자료의외부대조로보존한다. candidate만좋은seed/채널을고르지않는다.

## 가까운 선행과 확인 범위

- [Houlsby et al., Parameter-Efficient Transfer Learning for NLP, ICML2019](https://proceedings.mlr.press/v97/houlsby19a.html): 공식페이지의adapter/parameter-efficient transfer설명을확인. bottleneck/residual PEFT 자체는알려진방법이며POINTWISE를신규방법으로보지않는다. 이페이지열람으로전체원문을읽었다고하지않는다.
- [Autoformer, NeurIPS2021 arXiv](https://arxiv.org/abs/2106.13008) 및 [저자AutoCorrelation.py](https://github.com/thuml/Autoformer/blob/main/layers/AutoCorrelation.py): primaryabstract와코드에서Q/K FFT correlation,lag선택/softmax,roll/gather를통한V집계확인. 주기유사성에따라표현을집계한다는큰원리는중복이다. 후보는학습Q/K·전역lag top-k대신원시8시간patch거리·고정동일위상mask·query별kernel을frozenencoder뒤adapter branch에쓴다. 기존Autoformer전체재현이아니다.
- [CycleNet v1 §3.1](https://arxiv.org/html/2409.18479v1): channel별학습cycle Q를입력에서빼고미래에더하는수식직접확인. 후보는학습calendar table/입력제거·출력추가가없고context내표현집계다. 주기활용자체의최초성은없다.
- [CoRA v1 §4](https://arxiv.org/html/2603.21828v1): 채널Pearson+시간가변저랭크상관,positive/negativeprojection,contrastive objective를확인. 후보는채널간상관을모델링하지않고동일채널의시간patch를다룬다. 따라서'채널상관adapter최초'라는주장은부적절하다.
- pinned [TimePEFT 공개구조](https://github.com/kaist-dmlab/TimePEFT): 이전직접코드검토의top-k FFT frequency branch와channelaffine adapter를기준으로한다. 후보는입력유사도conditioned행렬과identity residual를쓰지만TimePEFT전체의공정한재현·우위는이실험이검증하지않는다.

## 구현 검토와 신규성 한계

A/B가모든arm에서같은난수로초기화되고B=0이어서head초기화차이로이득을만들지않는다. rawpatch와hidden은기존동일forward에서얻는다. tokenizer/patchembedding/normalizer/encoder는동결되며학습head와adapter만optimizer에포함한다. 새trainableparameter가loss에실제영향을미치는지2updates로검사한다. mask는j-i mod3=0이며시각ID없이상대24h를나타낸다. 여러미래horizon의정답이나평가잔차는입력에없다.

gaussian kernel집계는attention/nonlocal averaging의알려진수학이다. 신규성상태는'구체적배치가설은있으나방법론신규성미확정'이다. prototype성능이좋다는이유만으로학회최초성이나새방법논문가능성을확정하지않는다. 하이퍼파라미터rank16/온도1/24h는이제고정하고결과에따른재튜닝을하지않는다. 새로운가설검증에기존노출E를사용한다는선택편향도그대로공개한다.
