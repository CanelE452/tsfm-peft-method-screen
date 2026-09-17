# 선행 대조와 논문 주장 범위

검토일: 2026-09-18. 논문 준비를 위한 공식 원문·저장소 확인이며 아래 논문의 수치를 이 실험의 직접 대조 수치로 가져오지 않는다. 논문 간 데이터·정보 권한·기반 모델이 다르면 보고된 개선율을 비교할 수 없다.

| 선행 | 확인한 원리 | 현재 실험의 대응 | 실제 미완료 범위 |
|---|---|---|---|
| [Houlsby et al., ICML 2019](https://proceedings.mlr.press/v97/houlsby19a.html) | 기반 모델 동결과 작은 bottleneck adapter | C2는 일반적인 추가 residual 용량 대조. 원 논문의 BERT 배치 전체 복제가 아님 | 일반 adapter 자체는 신규성으로 주장하지 않음 |
| [Hu et al., LoRA](https://arxiv.org/abs/2106.09685) | 동결 weight 위 저랭크 적응 | B0에 rank8,alpha16,q/v LoRA; C1은 같은 LoRA의 추가 학습 | 모든 LoRA 변형·rank를 비교한 연구가 아님 |
| [Bachlechner et al., UAI 2021](https://proceedings.mlr.press/v161/bachlechner21a.html) | 0으로 시작하는 residual 계수 | 현재 adapter 출력층0 초기화의 일반 원리와 관련 | identity 시작 자체를 새 기여로 세지 않음 |
| [Haugsdal et al., 2022](https://arxiv.org/abs/2208.14236) | naive persistence 예측으로 시작하는 gating/residual 초기화 | 현재의 ‘관측 extreme의 같은 부호 연속성’과 다른 persistence 정의 | 이름 유사성으로 동일 방법 또는 새로운 방법이라고 결론내리지 않음 |
| [Im and Kwon, COSA, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2a8ce71baac4c89bf9ff479d8240c7d9-Abstract-Conference.html) | context를 쓰는 출력 residual 및 도착 정답의 온라인 적응 | C3는 입력 patch residual을 source TRAIN에서 학습하고 E에서는 동결 | 정식 온라인 정보 공개·업데이트 예산 비교 미실행. BIAS는 COSA가 아님 |
| [Qiu et al., TATO, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48c5226582f41254026748c7e35d4ac2-Abstract-Conference.html) | 변환 pipeline과 configuration 탐색 | 현재 원래 입력을 보존하는 추가 patch residual과 다른 개입 위치 | 공식 전체 탐색·ranking benchmark 미실행. 일부 clipping을 TATO라고 부르지 않음 |
| [Chen et al., SOLID, KDD 2024](https://arxiv.org/abs/2310.14838) | 비슷한 context의 표본을 골라 예측층 적응 | source 고정학습과 다른 적응 시점·참조 표본 권한 | 동일 권한으로 정식 재현한 비교 없음 |
| [Na et al., Time-PEFT 공식 저장소](https://github.com/kaist-dmlab/TimePEFT) | MOMENT 기반 주파수 및 multichannel 적응 | 현재 Chronos-Bolt의 channel-independent raw-input 추가 적응과 구성이 다름 | 다른 backbone 결과를 직접 성능 우위로 비교하지 않음. 공식 README의 발표 표기를 확인했으며 이번 실행은 재현하지 않음 |
| [Gupta et al., Beyond LoRA, 2024](https://arxiv.org/abs/2409.11302) | Chronos의 BitFit,LayerNorm,VeRA,FourierFT를 의료 예측에 비교 | PEFT의 우위를 LoRA 한 가지와의 차이로 일반화할 수 없다는 관련성 | 해당 의료 자료·다른 모델에 대한 우위는 주장하지 않음 |
| [Cawley and Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html) | 모델 선택과 반복 평가의 편향 | 기존 Electricity/ETT E는 반복 노출 개발 자료 | seed 추가와 bootstrap만으로 개발 선택 편향이 제거되지 않음 |

Chronos 연구의 배경은 [Ansari et al.](https://arxiv.org/abs/2403.07815)를, 실제 사용한 Bolt 구조와 가중치는 [공식 Bolt-small 모델 카드](https://huggingface.co/amazon/chronos-bolt-small) 및 로컬 pinned hash를 인용한다. 원래 Chronos의 이산 token 학습을 이번 Bolt의 연속 patch/quantile 경로라고 설명하지 않는다.

데이터는 기존 원점과 파일 hash를 유지했다. ETT 출처는 [공식 ETDataset](https://github.com/zhouhaoyi/ETDataset), NESO는 [2025 공식 수요 자료](https://www.neso.energy/data-portal/historic-demand-data/historic_demand_data_2025)와 [수정·품질 안내](https://www.neso.energy/data-portal/historic-demand-data)를 인용한다. 국가 집계 한 계열과 개별 전력 계열을 독립 표본처럼 합산하지 않는다.

## 논문 유형별 충분성과 부족함

현재 수행한 C0/C1/C2/C3,MEAN/ROTATE16/RECENCY,SHRINK/BIAS,F0 및 단순 점예측 비교는 **추가 적응의 조건부 효과를 조사하는 통제 실증 연구**의 근거다. 이미 잘 알려진 residual/gate 원리를 최초 기여로 주장하지 않고, 구체적 관측 규칙과 그 이득·손해를 검증 대상으로 제시할 수 있다. 정확한 투고 적합성은 원고와 투고처의 심사 기준에 달렸다.

**범용 새 PEFT의 우수성**을 중심으로 쓰려면 공식 선행들과 동일 권한·예산을 정한 직접 비교가 추가로 필요하다. 이번 직접 기전 보강을 그 비교의 대체물로 부르지 않는다. 특히 RECENCY 대비 결과가 약하면 선행 수를 늘리는 것만으로 고유 기여가 생기지 않는다. 이번 작업은 모델을 바꾸거나 성공할 때까지 튜닝하는 실험을 포함하지 않는다.
