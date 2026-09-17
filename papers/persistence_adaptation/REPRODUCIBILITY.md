# 재현과 자료 제공 범위

이 묶음은 기존 C3 연구와 논문 준비용 보강 실험의 근거를 연결한다. 모든 수치는 mean-error ratio를 사용하며 표의 양수 gain은 오차 감소다. 서로 다른 목표를 합쳐 우승자를 만들지 않는다.

## 근거 계보

1. `results/additive_b0_adapter_v1_20260917`: 원래 C0/C1/C2/C3 발견 단계. 과거 성능을 이미 확인한 개발 결과.
2. `results/additive_persistence_validation_v1_20260917`: 기존 후속 58 fits,59,392 main+36 smoke updates. 원래 가중치와 C3 보존,세 번째 seed·전력16계열·ETTm2·기전 대조·변화 형태 평가.
3. `results/persistence_evidence_extension_20260918`: 0 fits/111 추가 평가. NESO2025,기전 대조 확대,선택 LR/step 감사. 기준 commit `002ea63a38d731e6a2c456e40b660005ae73c64d`.
4. `results/paper_readiness_20260918`: 이번 고정1,024-update 2×2 진단,미적응 모델·단순 기준선,추가 분석. 평가 전 봉인 commit `4247075`. 원문·모델·자료는 바꾸지 않았다.

새 보강은 기존 E 자료의 후속 진단이므로 새로운 독립 test로 부르지 않는다. NESO2025의 해당 관측값은 pinned backbone 가중치의2024공개와 시간상 분리됐지만, 이번 보강 시점에는 NESO 평가 결과도 이미 보았다.

## 재생성 명령

저장소 root에서 실행한다. `.venv`와 동일 pinned 종속성,기존 cache가 필요하다.

```bash
.venv/bin/python -m experiments.paper_readiness_20260918.checks
.venv/bin/python -m experiments.paper_readiness_20260918.runner run
.venv/bin/python -m experiments.paper_readiness_20260918.mask_audit
.venv/bin/python -m experiments.paper_readiness_20260918.complete_metrics
.venv/bin/python -m experiments.paper_readiness_20260918.figures
.venv/bin/python -m experiments.paper_readiness_20260918.mask_figure
.venv/bin/python -m experiments.paper_readiness_20260918.report
```

runner는 이미 완료된 예측의 hash를 확인하고 재사용한다. 코드·입력·가중치가 봉인과 다르면 멈춘다. `score.py`는 모든 예측 manifest가 저장된 후에만 평가 정답을 읽는다. GPU 사용은 공용 lock과 자원 감시를 적용하며 RustDesk만 기존 승인 예외다. 학습/optimizer/backward 경로는 없다.

## 공개 파일과 로컬 파일

GitHub: 코드,설계와봉인,모델/자료 hash,원점별 점수,검산 기록,전체 조건의 표,PNG/PDF/SVG 그림,한국어 보고서.

로컬 cache: 원시 자료,기존 selected/fixed checkpoint,입력 packet,개별 quantile prediction 배열. 공개 저장소만 내려받으면 모델 추론 전체가 재현된다고 주장하지 않는다. 원시 자료 배포 권한과 checkpoint 공개 방식은 투고 때 별도로 확인해야 한다. `SEAL.json`은 필요한 정확한 경로와 hash를 제공한다.

## 점수와 불확실성

nMAE는 각 계열 TRAIN population sigma로 나눈 horizon64 median absolute error의 평균이다. F0는 deterministic 단일 checkpoint,단순 예측은 seed0 한 번이며,이를3seed로 복제하지 않았다. 모델 세 seed와 deterministic 기준선을 비교하는 구간은 고정된 모델과 관측기간에 조건부다.

기존 paired bootstrap과 새 보강의 seed가 다르므로 평균 점수는 같아도 CI 끝점이 소폭 다를 수 있다. 두 연구의 구간을 섞어 더 유리한 쪽을 선택하지 않는다. 이번 종합 그림은 이번 명세의 시간 block 구간을 사용하고,기존16계열의 series+time 구간은 원래 결과 파일에 보존한다. 논문 본문에서 구간 정의를 명시해야 한다.

## 실행과 논문 판단

실행 완료,숫자 검산,조건부 효과,단순 대안 대비 추가 가치,신규성은 별도의 판단이다. 운영 혼합 그래프는 가정한 합성 시나리오 비중에 대한 민감도이며 실제 현장의 발생빈도나 위험 한계가 아니다. 모델 사용 승인을 의미하지 않는다.

`RAW_SCORES_WITH_NRMSE.csv`는 사전에 지정한 채널별 nRMSE를 원점 MSE에서 집계한 보고용 파일이다. 기존 봉인된 점수 파일은 바꾸지 않았고 나머지 지표의 일치를 검산했다.

검토 문서 생성: `.venv/bin/python -m experiments.paper_readiness_20260918.build_brief`는 로컬 pandoc으로 MD/HTML/DOCX를 만든다. PDF는 로컬 Chrome headless의 `--print-to-pdf`로 같은 HTML을 출력했다. CJK 글꼴이 필요하다. 문서 생성은 모델을 호출하지 않는다. 최종 검산 명령은 `.venv/bin/python -m experiments.paper_readiness_20260918.final_audit`다.
