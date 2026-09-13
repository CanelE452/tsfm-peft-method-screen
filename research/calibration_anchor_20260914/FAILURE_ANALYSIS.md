# 야간 실험 실패 분석과 후속 후보

원자료는 results/overnight_20260913/verification.json 및 각 fit의 trajectory, resource,
선택된 parameter/prediction cache이다. 재현 명령은 scripts/analyze_overnight_failure.py.
정확한 수치는 failure_diagnostics.json에 저장했다. 이는 사후 개발 분석이다.

## 확인한 결과

72/72 fits, 64,800 optimizer updates, 실행 오류 0. 376개 예측 캐시가 독립 재계산됐다.
- Prediction anchor: 최강 비교군보다 ETTh1 0.1959%, Traffic 0.3408% 개선.
  두 seed의 추가 조건도 통과했지만 미리 고정한 양 데이터 1% 평균 이득 기준에는 미달.
  개선이 없다는 뜻이 아니라, 이 개발 기준에서 추가 가치가 작다는 뜻이다.
- Drift gate: ETTh1 0.0119% 개선, Traffic 1.2816% 악화. 일반 moment gate보다 조금 낫지만
  강한 native LoRA를 넘지 못했다. 선택된 gate weight가 실제 바뀌었으므로 미작동 모듈은 아니다.
- Distillation: V에서 teacher가 raw LoRA보다 약 4.08%/4.22% 나빴다.
  raw baseline 8 fits 이후 학생 24 fits와 E를 생략한 것이 고정 프로토콜에 맞다.

ETTh1 native/raw의 마지막 V 손실은 해당 fit의 최선 V 대비 중앙값 약 31.76%/30.45% 악화했다.
Prediction anchor는 이를 17.43%, L2 anchor는 7.08%로 줄였다.
Traffic의 같은 퇴행은 훨씬 작았다. 늦은 checkpoint의 과적합과 데이터 간 차이는 관찰되지만,
이 수치만으로 샘플 부족, 분포 변화 또는 특정 층을 단일 인과 원인으로 확정하지 않는다.
V 선택에 early checkpoint와 step0가 포함돼 이 퇴행 전체가 E에 그대로 반영된 것은 아니다.

Prediction anchor의 마지막 100 step 정규화 항/예측 loss 비율은 약 11.15%/5.88%였다.
정규화가 사실상 0인 구현은 아니다. 정규화가 작동해도 일반 LoRA 이후 추가 이득은 작았다.

## 반증한 쉬운 후속안

선택된 V 예측의 median만 적응값으로 두고 나머지 quantile 간격을 F0로 되돌리는 사후 진단:
ETTh1은 약 0.12~0.19% 좋아졌지만 Traffic은 약 2.03~2.77% 나빠졌다.
적응 median과 80% 폭을 모두 유지하고 표준화한 quantile 형태만 F0로 되돌려도
Traffic은 약 0.30~0.50% 나빠졌다.

이는 새 학습 결과나 배포 가능한 학습법의 성과가 아니다. 저장된 V 예측을 조합한 반사실적 진단이다.
분포 모양을 무조건 보존하면 두 데이터에서 이득을 얻는다는 가정은 지지되지 않아 그대로 실행하지 않는다.
선택된 예측에 대한 대수적 조작만으로 soft regularizer의 최종 학습 효과를 완전히 반증할 수는 없다.

## 실행하는 후속 가설

교정 상태별 보존 LoRA:
학습 데이터에서 F0 quantile의 경험적 포함 비율이 nominal quantile에 가까운 부분은 더 보존하고,
교정 오차가 큰 부분은 덜 보존한다. 표본 평균 예측이나 전체 분포 형태를 일괄 동결하지 않는다.

이는 알려진 function anchoring, empirical calibration, teacher weighting을 결합한 검증용 후보이다.
독창성은 미확정이다. calibration-aware라는 이름이나 새 조합만으로 새로운 논문이라고 주장하지 않는다.
이 가설은 반드시 uniform anchor, 동일 가중치 분포를 섞은 anchor, 같은 정보로 loss만
reweight하는 대조, L2 anchor 및 native/raw LoRA를 넘어야 한다.

문헌의 경계:
- [Guard](https://arxiv.org/abs/2606.19363): TSFM의 신뢰도에 따른 teacher 선택·증류 강도는 기존 연구.
- [delta-Adapter](https://arxiv.org/abs/2601.20280): 예측 보정과 quantile calibration도 기존 연구.
- [Ensemble distribution distillation](https://arxiv.org/abs/2002.11531): 분포 보존·압축의 상위 개념도 기존 연구.
- [MixFT](https://arxiv.org/html/2603.02840v1): 하위 분포별 LoRA 전문화는 기존 연구.

## 자동화 장애와 회복

완료 감시와 전체 STOP 판정, 독립 재검증은 정상 동작했다.
2026-09-13 23:30 KST에 같은 세션 재개를 시도했으나 설치 CLI가 gpt-6-astra를 지원하지 않아
API 400 "requires a newer version of Codex"로 실패했다. 새 후속 fit은 자동 시작되지 않았다.
사용자가 돌아온 뒤 현재 대화에서 분석·구현을 직접 이어갔다.
실패 trigger/status/log를 지우거나 재시도 성공으로 바꾸지 않았다.
현재 한 차례 후속 실행을 만들며, 또 다음 연구를 자동으로 재귀 예약하지 않는다.
