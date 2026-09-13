# 조건부 후속 연구 완료

부모 야간 실험은 72 fits / 64,800 updates를 완료하고 세 후보 모두 STOP이었다.
자동 감지는 성공했으나 설치된 CLI의 모델 호환성 오류로 대화 재개가 실패했다.
이후 사용자가 돌아온 현재 대화에서 분석·구현·후속 실행을 직접 완료했다.

후속 calibration_anchor_20260914는 2026-09-14 02:08 KST에 종료됐다.
56/56 fits, 50,400 본학습 updates, 실행 오류 0. 최종 판정은 PILOT_STOP.
GPU smoke는 최종 소스 14 updates와 저장 형식 보완 전 14 updates, 합계 28로 별도 기록했다.
예측 캐시 286개가 독립 재계산됐고, 과거 결과 파일 514개가 모두 보존됐다.

| Dataset | F0 | Native LoRA | Uniform anchor | Shuffled anchor | Proposed | Proposed vs Native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| etth1 | 0.297980981 | 0.289287525 | 0.292596116 | 0.292535216 | 0.292622005 | -1.1527% |
| traffic | 0.167272009 | 0.159467021 | 0.160295655 | 0.160401464 | 0.160476963 | -0.6333% |

제안법은 F0보다 좋아졌지만 native LoRA보다 두 데이터 모두 나빴다.
Uniform/shuffled anchor보다도 평균 손실이 조금 높았다. 이번 비교에서는 교정 상태에 따라
보존량을 배분하는 추가 가치가 확인되지 않았다. 단순히 1% 문턱이 높아서만 실패한 사례는 아니다.
새 E는 데이터당 8 origins이고 bootstrap 구간은 넓다. 이 결과로 모든 교정 기반 PEFT의 불가능성을 주장하지 않는다.

상세 결과: results/calibration_anchor_20260914/REPORT.md
검증: results/calibration_anchor_20260914/verification.json
실패 진단: research/calibration_anchor_20260914/FAILURE_ANALYSIS.md
프로토콜: docs/CALIBRATION_ANCHOR_FOLLOWUP.md

사용자 승인에 따른 한 차례 후속 배치를 완료했다. 현재 이 연구의 학습 프로세스나 추가 자동 재개 예약은 없다.
과거 실패한 자동 재개 trigger/status/log는 수정하지 않았다.
