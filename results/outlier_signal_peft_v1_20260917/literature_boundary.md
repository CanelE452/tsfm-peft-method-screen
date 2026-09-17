# 실제 선행 코드 경계

- [TSFM-Biases](https://github.com/amazon-science/TSFM-Biases), commit `3d526e0aaeb513515928c825e6f054b1228f2a5a`의 `notebooks/outlier-bias.ipynb`를 읽었다. blob `723583c1be813c54d90886e73837a543d1772580`은 계약과 일치한다. 주기 160의 cos(t+0.3), 문맥 512·미래 64, 공개 Chronos-T5-small/Bolt-small 연결과 patch-size-1 자리표시자를 확인했다. 입력 표현과 오류 민감성에 대한 직접 선행이며 새 PEFT 효과의 증거는 아니다.
- [TATO](https://github.com/thulab/TATO), commit `402bbc8998c49e2f33d9afbcc42140347a6b8c36`의 `transformation/library/inputer.py`, `model/model_factory.py`, `experiment/run.py`를 읽었다. Inputer는 none/3_sigma/1.5_iqr 검출과 선형 보간을 사용한다. Chronos 연결은 ChronosPipeline, FP16, 3 samples이며 Chronos2 연결도 별도로 있다. 검색 뒤 validation이라고 적힌 구간이 해당 코드에서 train split을 다시 만드는 점도 확인했다. 이번 고정 clip/Hampel 대조는 공식 검색이나 전체 TATO 재현이 아니다. TATO보다 우수하다는 결론은 낼 수 없다.
- [Chronos-Bolt 모델](https://huggingface.co/amazon/chronos-bolt-small)의 고정 snapshot과 설치된 chronos-forecasting 2.3.2 실제 encode/forward를 읽고 CPU에 연결했다. 설치 소스 SHA256은 MODEL_RECEIPT.json에 기록했다. native loss는 정규화된 target을 사용하므로 이번 TRAIN 고정 scale loss와 같다고 가정하지 않았다.

새 경로는 강건 전처리·residual adapter·bounded embedding과 가깝다. 신규성은 확보하지 않았다. 실제 품질 레이블, 정식 선행 비교, 독립 원천 검증은 본 파일럿 이후에도 남는 별도 과제다. 이를 자동 실행하지 않는다.
