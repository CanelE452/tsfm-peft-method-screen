# 새 미래 구간 선택 전달 비교

완료: 새 학습 0, backward 0, forward 1064회(S 600, D 464). 실행 시간 2.81분. 검증에서 12개 셀의 모든 선택을 봉인한 뒤 미래 구간을 평가했다.
주 비교의 source-balanced 차이는 +0.1517 pp (%F0 gain 기준, 양수는 SPREAD4 우세)다. RECENT4/SPREAD4는 모두 4개 검증 origin을 사용했다. ALL16은 정보량이 다른 참조다.

| 원천 | RECENT4 %F0 gain | SPREAD4 %F0 gain | SPREAD−RECENT pp | 같은 선택 /4 |
|---|---:|---:|---:|---:|
| beijing | +0.7010% | +1.2757% | +0.5747 | 0 |
| electricity_new | -0.1145% | +0.0000% | +0.1145 | 0 |
| ettm2_later | +1.2499% | +1.0158% | -0.2341 | 1 |

양수 %F0 gain은 frozen 대비 손실 개선, 음수는 악화다. 원천별 평균은 두 seed와 두 arm을 포함하되 이들을 독립 데이터셋으로 세지 않았다. 각 셀·참조 결과는 paired_cells.csv 및 analysis_summary.json에 있다.
모든 미래 D 후보를 추론하는 oracle은 만들지 않았다. 평가한 것은 각 규칙에서 선택된 모델, 사전 고정 step150/LR3e-5, F0의 합집합뿐이다. 따라서 D에서 다른 미평가 후보가 더 좋았는지는 알 수 없다.
노출 감사는 이 체크아웃의 origin 메타데이터와 실행/데이터 기록 범위다. Electricity는 과거 실험과 겹치는 날짜의 다른 채널(4..7)을 사용한다. 외부 실험이나 foundation model 사전학습에 대한 미노출을 보장하지 않는다. 원시 데이터 형식/결측 검사는 과거에 이루어졌으므로 한 번도 읽지 않은 원시 데이터라는 뜻도 아니다.
새 미래 구간의 결과를 본 뒤 검증 규칙·학습률·후보 집합을 조정하지 않았다. 이번 결과는 표준 선택 설계의 검증이며 새로운 PEFT 방법의 성능 입증이 아니다.

[실행 계획](plan.json) · [선택 봉인](selection_seal.json) · [노출 감사](exposure_audit.json) · [검산](artifact_verification.json) · [이전 A–D](../temporal_transfer_diagnostic_v1_D/REPORT.md)
