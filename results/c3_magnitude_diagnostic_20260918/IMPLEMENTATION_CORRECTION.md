# 진단 추출 코드 수정

첫 CPU 입력 추출에서 NumPy float32 평균과 모델 PyTorch 평균이 한 패치에서 5.96e-8 달라 exact 검사가 중단됐다. 원래 코드·봉인·오류를 INITIAL_FEATURES.py.txt / INITIAL_SEAL.json / CPU_EXTRACTION_ERROR.txt에 보존했다.

특성 추출의 두 gate 평균을 실제 모델과 같은 PyTorch 연산으로 계산하도록 수정했다. exact 허용오차를 늘리지 않았고 scalar 수학 검사의 기존 1e-7 기준도 유지했다. 모델·가중치·규칙·데이터·성능 기준은 그대로다. 이 시점의 새 GPU 평가·optimizer updates는 모두 0회다.

## 집계 축 수정

36개 교차 예측을 모두 저장한 뒤 첫 집계에서 `sc[:, ci, ..., 0]`의 NumPy advanced indexing이 condition 축을 seed보다 앞으로 이동시켜 seed 루프가 중단됐다. 명시적 `np.take(sc, ci, axis=1)[...,0]`과 shape 검사를 사용해 원래 의도한 seed/condition 순서를 보존했다. 원본 코드·오류·직전 봉인을 INITIAL_ANALYZE.py.txt / AGGREGATION_ERROR.txt / PRE_AGGREGATION_FIX_SEAL.json에 보존했다. 예측·학습·규칙·표본·metric은 바꾸지 않고 CPU 집계만 다시 수행했다.
