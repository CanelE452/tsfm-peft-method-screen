# 근거와 전이 범위

확인일: 2026-09-24. 라이브러리를 새로 설치하거나 최신 버전으로 교체하라는 뜻이 아니다.

## 기존 프로젝트에서 확인한 사실

- 원 실행 코드: https://github.com/CanelE452/tsfm-peft-method-screen/blob/948134f36a49300f6ca0c405f4757760d8f0cfdc/experiments/transient_difficulty_screen_v1_20260924/difficulty_screen.py
  - 전체 episode에서 특징을 계산해 선언된 context 밖에 접근한 부분, 계획을 F0에만 넘긴 부분, 고정 Ridge alpha와 LOEO 구조 확인.
- 실제 실행 설정: https://github.com/CanelE452/tsfm-peft-method-screen/blob/948134f36a49300f6ca0c405f4757760d8f0cfdc/results/transient_difficulty_screen_v1_20260924/run_20260924T075030Z/RUN_MANIFEST.json
  - 24시간, 8구간, 다섯 target/horizon, TRAIN 0~4 / DEV 5~6 / RESERVE 7 확인.
- 원자료 목록·해시: https://github.com/CanelE452/tsfm-peft-method-screen/blob/948134f36a49300f6ca0c405f4757760d8f0cfdc/results/transient_difficulty_screen_v1_20260924/run_20260924T075030Z/EPISODES.json
  - gzip 파일과 재구성할 일정의 SHA-256을 잠금. 이 패키지에는 episode 7의 측정 파일을 여는 경로가 없다.
- Git ignore: https://github.com/CanelE452/tsfm-peft-method-screen/blob/bdd91d2172df77bfad8ac04209dbe44bfcdf7edb/.gitignore
  - *.log / *.npz 무시를 확인. 작은 로그만 명시적 allowlist로 게시; 계수 npz는 비공개.

## 표준 방법 확인

1. scikit-learn Ridge 공식 문서
   https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
   - 표준 L2 정규화 다중출력 회귀의 목적식 및 alpha 의미.
   - 구현은 의존성 추가를 피하려고 NumPy SVD 해법을 사용하며 작은 검사에서 scikit-learn과 비교한다.
   - 새 예측 손실이나 새 방법을 제안하는 것이 아니다.
2. scikit-learn TimeSeriesSplit 공식 문서
   https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
   - 미래 자료로 과거를 평가하지 않는 시간순 검증 원리.
   - 현재는 날짜별 episode 사이 간격이 있으므로 sklearn의 등간격 표본 분할 함수를 그대로 쓰지 않고 명시적인 episode-prefix 분할을 구현한다.
3. scikit-learn Common pitfalls 공식 문서
   https://scikit-learn.org/stable/common_pitfalls.html
   - 전처리·특징 정규화·모델 선택을 학습 자료에만 맞춰야 한다는 근거.
4. BOPTEST API 공식 문서
   https://ibpsa.github.io/project1-boptest/docs-userguide/api.html
   - advance에서 명령은 control step 동안 일정하고 반환 측정은 step 끝의 값.
   - results의 제어 입력은 시뮬레이션에 사용된 값. 라이브 API 호출은 이번 범위에 없다.

## 원문이 입증하지 않는 것

이 문서들은 제안한 이력 특징의 우수성, 공급수온 15분/실내온도 6시간 과제의 필요성,
특정 최근 창이 실제 열적 과도응답 경계라는 주장, 새 PEFT의 성능·신규성을 입증하지 않는다.
이 패키지의 알파 격자·운영상 패널·원점 집계·실행 상한은 목적에 맞춘 고정 설계이며 [추정·미검증]이다.

Time-PEFT 원문을 이번 패키지의 효과 근거로 쓰지 않는다. 이번 작업은 기존 입력 계약의 정정과
표준 회귀 재평가이며, 아직 복잡도 지표나 특화 PEFT 필요성을 입증하는 연구가 아니다.
