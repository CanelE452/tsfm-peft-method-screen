# 추가 선행·자료 경계

기존 선행 표를 보존하며 아래 원문과 이미 내려받아 pin한 공식 코드를 다시 확인했다. 이번111-view 검증은 정식 선행 학습 벤치마크가 아니다.

| 원문 | 확인한 내용과 C3의 차이 | 남은 일 |
|---|---|---|
| [Persistence Initialization, 2022](https://arxiv.org/html/2208.14236v1) | naive persistence 예측으로 시작하는 residual/gating 초기화. C3의 관측된 extreme 지속 구간에 대한 fixed patch gate와 같지 않지만 persistence+identity/residual이라는 표현만으로 신규성을 주장하면 안 됨 | identity 시작은 기여에서 제외 |
| [SOLID, KDD 2024](https://arxiv.org/html/2310.14838v2) | 문맥과 비슷한 예제를 골라 예측층을 sample별 적응하는 calibration. 고정 source 학습 후 정답 update 없는 C3와 적응 시점·정보 권한이 다름 | 같은 정보 권한에서 공식 절차를 재현하는 비교 미완료 |
| [COSA, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2a8ce71baac4c89bf9ff479d8240c7d9-Abstract-Conference.html) | 이전 pin527c0feb의tta/cosa.py를 다시 읽음. 원래예측+context의 출력 residual과tanh gate, 온라인 도착 정답 활용. C3는patch공간·오프라인 적응 | BIAS와동일시하지 않음; 온라인 권한과예산을정한 정식비교 필요 |
| [TATO 공식 코드](https://github.com/thulab/TATO/tree/402bbc8998c49e2f33d9afbcc42140347a6b8c36) | pipeline/base.py와 기존tuner/base.py의 전후변환/선택 경로 확인. raw-input B0위patch추가수정과 구현위치 다름 | 단순clip이나본대조를TATO전체로부르지않음 |
| [Chronos 모델 이력](https://huggingface.co/amazon/chronos-bolt-small/commits/772f3d25d38aec6d914c8949dab4462e2d46f5d8) | 최신pin은2025 README수정. 2024-11-28commit의weight LFS SHA와현재 로컬weight bytes/config가 같은지 직접감사 | 이전 영국자료 전체의중복은별도로불명 |
| [NESO2025 공식자료](https://www.neso.energy/data-portal/historic-demand-data/historic_demand_data_2025) | ND는발전측계측에기초한국가수요(MW). 반시간settlement,clock change존재 | 국가집계1계열을여러독립meter라고하지않음 |
| [NESO자료설명](https://www.neso.energy/data-portal/historic-demand-data) | 역사적자료수정과품질문제고지확인 | 실시간as-of 또는실제사건레이블검증이라고하지않음 |

추가 검색에서 OpenReview syfWdclGE1 문서는브라우저검증화면만열렸다. 전문을 읽거나그방법의공식재현을했다고주장하지않는다. 이번결과가양성이어도정식선행비교·실제사건검증은미완료다. 일반TSFM적응선행을넓게모두능가했다고쓰지않는다.
