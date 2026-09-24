# 근거와 전이 한계

확인일: 2026-09-25. 코드의 숫자 예산과 screening cutoff는 아래 논문이 제시한 합격선이 아니다. source를 그대로 재현하는 실험과 이 패키지의 신규 진단 설정을 구분한다.

## [S1] 풍력 데이터 및 신호 의미

- **SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting over a Large Turbine Array (2024/Scientific Data)**
- https://www.nature.com/articles/s41597-024-03427-5
- 공식 데이터: https://figshare.com/articles/dataset/SDWPF_dataset/24798654 (version 2)
- 공식 metadata 요청: https://api.figshare.com/v2/articles/24798654/versions/2

원문과 배포 페이지에서 134개 터빈, 10분 기록, 위치/Ndir/Wdir/발전량, KDD/full 분리, unknown·abnormal 권고를 확인했다. 위치의 x/y를 검증된 지리 방향으로 가정하지 않으며, ERA5 재분석을 미래 예보로 입력하지 않는다. 원문의 풍향 관련 관찰은 이 패키지의 경험적 관계 변화점수가 forecasting difficulty라는 증거가 아니다. Figshare 대용량 다운로드는 작성환경에서 수행하지 못했으며 자동 경로는 크기/스키마 확인 후 제한적으로 실행한다. 실패하면 원자료를 합성으로 대체하지 않는다.

## [S2] 호흡 데이터

- **Towards a Robust Estimation of Respiratory Rate from Pulse Oximeters (2017/IEEE Transactions on Biomedical Engineering; DOI online 2016)**
- DOI: https://doi.org/10.1109/TBME.2016.2613124
- **BIDMC PPG and Respiration Dataset (2018/PhysioNet, version 1.0.0)**
- https://physionet.org/content/bidmc/1.0.0/
- https://physionet.org/files/bidmc/1.0.0/bidmc_csv/

원 자료 설명과 CSV/고정 metadata 형식(Time [s],RESP 및 MIMIC II matched wdb ID)을 확인했다. 원래 목적은 호흡률 추정이다. 이 패키지의 64초→8초 파형 예측은 새 파일럿 설정이며 원 논문의 forecasting benchmark가 아니다. 수동 미래 호흡 annotation은 읽지 않는다. PhysioNet이 제시한 Open Data Commons Attribution 조건을 따르며 원자료를 GitHub 결과에 재배포하지 않는다.

## [S3] 위상/진폭의 인접 원리

- **Joint Registration and Conformal Prediction for Partially Observed Functional Data (2026/Journal of Computational and Graphical Statistics)**
- https://www.tandfonline.com/doi/abs/10.1080/10618600.2026.2634823
- https://pubmed.ncbi.nlm.nih.gov/42125249/

출판 정보와 원문 초록의 부분 관측·위상/진폭 구분을 확인했다. 그 registration 알고리즘이나 coverage 보장을 이 코드가 재현한다고 주장하지 않는다. 현재 interpeak CV는 더 단순한 후보 점수로, 위상/진폭 식별의 대체 증명이 아니다.

## [S4] 좌표 등변성과 공식 궤적 형식

- **EqMotion: Equivariant Multi-Agent Motion Prediction with Invariant Interaction Reasoning (2023/CVPR)**
- https://openaccess.thecvf.com/content/CVPR2023/html/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.html
- 저자 공식 배포: https://github.com/MediaBrain-SJTU/EqMotion
- 고정 commit: 5aec2e0b61c511fa93a24138dd90da59a089084b
- source: eth_ucy/preprocessor.py의 xind=13,zind=15, eth_ucy/data/의 17열 텍스트

저자 README의 CVPR 출판/등변성 설명과 전처리 코드를 직접 확인했다. CVF HTML은 이번 연결에서403이어서 전체 원문을 읽었다고 주장하지 않는다. 논문의 성능 수치를 현재 TSFM의 예상 성능으로 쓰지 않는다. 패키지는 공개 좌표를 파싱할 뿐 저자 코드를 실행하지 않으며 EqMotion 학습을 하지 않는다. ETH/UCY 원자료의 권리와 저장소 소스코드 license를 동일시하지 않고, 원자료는 로컬 cache에만 남긴다.

## [S5] 가까운 시계열 적응 선행

- **CoRA: Boosting Time Series Foundation Models for Multivariate Forecasting through Correlation-aware Adapter (2026/ICLR)**
- https://proceedings.iclr.cc/paper_files/paper/2026/hash/ae3e173398d6e43fea63cbcc16fbaa98-Abstract-Conference.html

원 초록의 time-varying/time-invariant 저랭크 관계 분해를 확인했다. '동적 관계를 adapter에 넣는다'는 것만으로 신규성을 주장하지 않기 위한 직접 인접 선행이다. 이번 패키지는 CoRA를 구현하거나 비교 성능을 내지 않는다. GO 이후 표준 LoRA와 함께 고려할 강한 비교 대상이다.

## [S6] 표준 수치 및 분할 원리

- Ridge objective: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- TimeSeriesSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- SciPy find_peaks: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

리지 목적식·학습/평가 시간 분리·distance/prominence의 정의를 확인했다. 본 코드는 SVD로 표준 ridge를 직접 계산하며 cutoff나 prominence의 선택은 이 문서가 보증한 값이 아니다. bootstrap은 그룹 재표집의 탐색적 불확실성 요약이고 소수 cluster의 유의성 증명이 아니다.

## [S7] Native TSFM API

- https://github.com/amazon-science/chronos-forecasting
- 로컬 실행 대상: chronos-forecasting 2.3.2
- 모델: amazon/chronos-2, snapshot 29ec3766d36d6f73f0696f85560a422f50e8498c

사용자의 앞선 실행에서 이 버전·snapshot과 target/covariate API 계약이 확인된 환경을 사용한다. 이번 코드는 quantile 반환 형상(target,H,1)을 검사하고 맞지 않으면 멈춘다. 작성환경에는 chronos가 없어 본 패키지의 실제 native 실행은 NOT_RUN이다. API 형상 mock과 제한된 단위검사는 native pretrained 결과를 대신하지 않는다.
