# 최종 결정 — 다음 투자 후보 0개

2026-09-17 KST. 단일 condition_sampling_repair_cli_20260917 계약을 적용했다. **이번 repair에서도 새 PEFT 구성요소 미확보**이며, 정식 선행 비교·독립 source 검증에 추가 투자할 후보는 **0개**로 남긴다. 추가 seed/LR/방법/dataset/학습은 시작하지 않는다.

## 실제 실행과 미실행

N02/N03는 적격32/32fits·16,384 본업데이트, 폐기smoke16updates, 주 평가와 양방향 교차 평가를 완료했다. 교차 평가의 추가 optimizer는0회다. 기존 selected checkpoint48개는 모두 hash-valid했고 재학습 복원은 없었다.

N01/R04/N07/R08은 TRAIN64일을 확보했으나 계약의 정확한 circular phase 선택으로 phase 개수 최소1/최대4, 차이3이 되어 한도2를 넘었다. **BLOCKED_DIVERSITY**로 GPU smoke·본학습·교차 평가를 하지 않았다. 해당64fits·32,768 본업데이트·smoke32회는 미실행이다. 숫자를 맞추려고 날짜·phase를 재배치하지 않았다. 이 네 트랙은 repaired 성능 실패도 ROBUST_NEGATIVE도 아니다.

R05/N06/R09는 UNAFFECTED_REFERENCE로만 보존했고 재학습하지 않았다.

## 방법별 최종 판정

| 트랙 | 판정 | 다음 투자 기준을 충족하지 못한 이유 |
| --- | --- | --- |
| N01 | BLOCKED_DIVERSITY / NOT_MEASURED | 사전 phase 분산 조건 미충족. 기존 부정적 결과의 강도를 높일 수 없음 |
| N02 | POSITIVE_UNCERTAIN + SAMPLING_SENSITIVE | B3 대 B2 +0.013565%, 평균 CI에0 포함. LONG보다6.675% 나쁨 |
| N03 | POSITIVE_UNCERTAIN + SAMPLING_SENSITIVE | C3 대 C2 +0.001136%, CI에0 포함·seed 부호 불일치. 학습 τ의 추가 가치 미확보 |
| R04 | BLOCKED_DIVERSITY / NOT_MEASURED | 정확도–수정량 목표의 repaired 비교를 실행하지 않음 |
| N07 | BLOCKED_DIVERSITY / NOT_MEASURED | 사전 phase 분산 조건 미충족 |
| R08 | BLOCKED_DIVERSITY / NOT_MEASURED | 사전 phase 분산 조건 미충족 |

repaired 주 비교에서 ROBUST_NEGATIVE로 확인된 트랙은0개다. REOPEN_CANDIDATE도0개다. 이것은 이전의 부정적 결론이 모두 옳다는 판단과 다르다. 측정한 두 트랙의 작은 방법 이득은 불확실하며, 미측정 네 트랙은 판단을 유보한다.

## 기존 결론의 강도는 낮춘다

N02는 옛 가중치만 새 E로 옮겨도 가까운 대조 대비+3.391%가 나타났다. 따라서 이전 좁은 E의 음의 점추정을 일반적인 방법 실패로 읽으면 안 된다. 하지만 이 옛 가중치도 LONG을 넘지 못했고, 사전 주 비교인 새 TRAIN/V/E의 추가 이득은 불확실하다. 교차 진단을 유리한 새 주 결과로 대체하지 않는다.

N03는 평가 원점만 바꾸면 음의 효과가 약해졌고, 새 가중치와 새 E를 함께 사용했을 때 약0인 양의 평균이 나왔다. 옛 E에서는 새 가중치도 학습 커널이 고정 커널보다 나빴다. 평가 조건과 적응/선택에 민감하므로 기존 강한 음의 결론을 한정하고, 양의 점추정도 성공으로 확대하지 않는다.

## 보존과 종료

날짜 우선 selector, 다양성 검사, 같은 방법의 실행기, old/new 교차 평가, 모든 원점수·가중치·해시·검산은 보존한다. LONG, HOLD, 고정 커널 등 단순 대조도 보존한다. 단순 대조를 넘는 새 구성요소의 확정적 필요성은 확보하지 못했다. CI에0이 포함됐다는 사실을 동등성 증거로 쓰지 않는다.

날짜를 넓혔어도 같은 원천의 재사용 개발 평가이며 phase가 날짜 순서와 결정적으로 연결되는 한계도 남는다. 정식 선행 비교·독립 source 검증을 완료했다고 주장하지 않는다. 부정적 결과를 숨기거나 빈 후보 수를 채우기 위해 새 구조를 만들지 않는다. 이번 고정 범위에서 종료한다.
