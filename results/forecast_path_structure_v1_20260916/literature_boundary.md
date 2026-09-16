# 선행과 신규성 경계

2026-09-16에 아래 1차 자료의 한정된 부분을 확인했다. 계약의 네 학습군·예산·지표는 문헌 탐색 뒤 바꾸지 않았다. 공개 구현을 추가 실행하거나 논문 전체 결과를 재현하지 않았다.

- **Exogenous Dropout**, arXiv2607.05452v1: §3.5 식8과 바로 뒤 train/test 설명을 읽었다. 채널별 Bernoulli keep0.7, 과거/미래 같은 mask, surviving scale1/0.7, inference dropout 없음이 이번 DROP의 대응 부분이다. 논문의 전체 비교·채택 상태는 재현/확정하지 않는다. [원문](https://arxiv.org/html/2607.05452v1#S3.SS5).
- **TFMAdapter**, arXiv2509.13906v1: §3 설정과 §4의 pseudo-forecast·instance adaptation 설명을 확인했다. 공변량을 활용하는 출력 적응은 가까운 단순 대안이다. 여기서는 CAL의21상수 보정만 사용하며 GP/pseudo-forecast 알고리즘 재현이 아니다. [원문](https://arxiv.org/html/2509.13906v1).
- **UniCA**, ICLR2026 공식 proceedings: §4.2 pre/post fusion(미래 공변량 conditional attention pooling 포함)과 기존 저장소의 한계 검토를 확인했다. 공변량 결합 자체가 새 기여라는 주장은 하지 않는다. 코드 미실행. [공식 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/0b5eb45a22ff33956c043dd271f244ea-Paper-Conference.pdf).
- **CoRA**, arXiv2510.12681v1: §3.1–3.2의 frozen embedding/selection/injection 범위를 읽었다. 표현 동결과 공변량 적응도 알려진 방향이다. 현재 계약에 새 gate를 추가하지 않으며 동치인 알고리즘이라고 단정하지 않는다. 코드 미실행. [원문](https://arxiv.org/html/2510.12681v1#S3).
- **Time-series augmentation 공개 구현/문서**: 저자 저장소의 jittering/permutation 설명 및 `utils/augmentation.py`를 확인했다. 시점별 perturbation과 시간 조각의 순서 변경은 알려진 대조 계열이다. 현재 POINT는 값을 다른 valid time으로 이동하는 permutation과 다르게 같은 valid time에서 vintage rank만 바꾸고 세 변수 묶음을 보존한다. 코드를 실행하지 않았다. [저자 문서](https://github.com/uchidalab/time_series_augmentation/blob/master/docs/AugmentationMethods.md), [코드](https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py).
- **On the Influence of Weather Forecast Errors in Short-Term Load Forecasting Models**: 저자 기관 저장소의 초록/서지 범위만 읽었다. 관측 날씨와 운영 예보 입력의 차이가 이미 연구된 문제임을 확인한다. 본문의 실험·수식까지 읽었다고 하지 않는다. [기관 원문 기록](https://mural.maynoothuniversity.ie/id/eprint/3618/).

PATH는 계약상 알려진 vintage/path augmentation 학습 규칙으로 취급한다. 가장 직접적인 로컬 선행은 20260916 vintage/order 대조이며 그 epoch 간 순서 실험과 이번 경로 내부 연결 통제는 질문이 다르다. 이번 짧은 외부 검색에서 **정확히 동일한 4-epoch, per-origin/hour/feature multiset 통제 비교**의 선행을 확인하지 못했지만, 이는 최초임의 증거가 아니다. 직접 동일 선행이 확인될 경우 KNOWN_REPLICATION으로 낮춰 읽을 수 있다.

차이는 같은 예보값의 시점별 노출을 맞추어 24시간 연결을 검사하는 평가 설계다. PATH 자체, LoRA 적용, 공변량 주입을 새로운 PEFT 아키텍처로 재명명하지 않는다. POINT는 통제용 비물리 합성 입력이다. vintage 평균/분산은 실제 기상 ensemble 확률이 아니고 rank 경로는 단일 physical issue라는 보장이 없다. 좋은 결과도 신규 구조의 필요성이나 NEW_PEFT_METHOD_ESTABLISHED를 자동 완성하지 않는다.
