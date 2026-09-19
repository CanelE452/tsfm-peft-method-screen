# 시간 반응 보존 PEFT 개발 파일럿

실행 완료와 방법론 근거를 구분한다. 신규 16/16 fits, 16384/16,384 main updates, smoke16updates, 기존 PLAIN4fits 재사용. E prediction192views 저장 후 채점·scalar 14592개 검산. 이 보고서는 학습 성공을 논문 PASS로 바꾸지 않는다.

## 사전 주 비교

전력16계열 전이, selected SHIFT8. gain 양수는 TRP가 좋음. 네 비교의 Bonferroni4는 이 family에만 적용한다.

| baseline | TRP_nmae | baseline_nmae | gain_pct | bonferroni4_low | bonferroni4_high | both_seeds_positive |
| --- | --- | --- | --- | --- | --- | --- |
| PLAIN | 0.399733 | 0.377355 | -5.9304 | -7.14091 | -4.8058 | False |
| ANCHOR | 0.399733 | 0.399759 | 0.0063426 | 0.00463543 | 0.00812557 | True |
| SHUFFLE | 0.399733 | 0.398086 | -0.413852 | -0.460938 | -0.368017 | False |
| IDEAL | 0.399733 | 0.358506 | -11.4998 | -14.2058 | -9.1527 | False |

고정 투자 신호 충족: **False**. 신호가 있어도 독립 확인·정식 선행 비교·신규성 검토 전까지 방법론 논문 완성으로 판단하지 않는다. 미충족이면 이 설정에서 새 방법의 근거를 확보하지 못한 것이며 후속 튜닝은 자동 실행하지 않는다.

## 원자료·오류·지속 변화와 반대 결과

아래는 일반 어댑터 PLAIN 대비다. C3/MAG/B0 대조, fixed1024, 모든 변화 형태와 seed 원점수는 RAW_SCORES.csv, SEED_EFFECTS.csv, EFFECTS.csv에 보존한다.

| panel | condition | TRP_nmae | baseline_nmae | gain_pct | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| electricity | FAULT | 0.171458 | 0.171823 | 0.211927 | 0.0350398 | 0.42407 |
| electricity | REFERENCE | 0.166522 | 0.166614 | 0.0551027 | -0.119895 | 0.260723 |
| electricity | SHIFT4 | 0.191301 | 0.189093 | -1.16746 | -1.57663 | -0.773535 |
| electricity | SHIFT8 | 0.210023 | 0.204404 | -2.74893 | -3.49107 | -2.09152 |
| electricity | SHIFT_POINT | 0.196211 | 0.193546 | -1.37726 | -1.69876 | -1.07624 |
| electricity_transfer | FAULT | 0.255206 | 0.255428 | 0.0865921 | -0.021325 | 0.194353 |
| electricity_transfer | REFERENCE | 0.239852 | 0.23993 | 0.0324304 | -0.0731643 | 0.149233 |
| electricity_transfer | SHIFT4 | 0.326433 | 0.318945 | -2.34775 | -2.76813 | -1.93782 |
| electricity_transfer | SHIFT8 | 0.399733 | 0.377355 | -5.9304 | -6.88617 | -5.02146 |
| electricity_transfer | SHIFT_POINT | 0.342788 | 0.334722 | -2.40994 | -2.75966 | -2.06505 |
| ettm1 | FAULT | 0.42319 | 0.423652 | 0.108971 | -0.108287 | 0.328191 |
| ettm1 | REFERENCE | 0.407324 | 0.407596 | 0.066747 | -0.14349 | 0.281731 |
| ettm1 | SHIFT4 | 0.551289 | 0.54821 | -0.561644 | -1.0409 | -0.106483 |
| ettm1 | SHIFT8 | 0.545283 | 0.540239 | -0.93369 | -1.42038 | -0.48196 |
| ettm1 | SHIFT_POINT | 0.589722 | 0.586404 | -0.56574 | -0.921849 | -0.180944 |

## 구성요소의 추가 가치

ANCHOR와의 비교는 보정 크기를 줄이는 단순 정규화 이상의 이득인지, SHUFFLE과의 비교는 같은 perturbation 값들의 시간 연결이 필요한지, IDEAL과의 비교는 B0의 실제 반응을 보존하는 것이 이상적 수준 이동 강제보다 나은지를 검토한다. 이 세 질문은 평균 손실 감소만으로 자동 확인되지 않는다. B0는 잘 학습된 teacher이지만 모든 변화에서 정확하다는 보장은 없으므로 teacher 오류도 보존할 수 있다. bootstrap은 고정된 두 seed·이미 본 데이터에 조건부인 주 단위 불확실성이다.

## 비용

신규 본학습 optimizer 구간 합계 0.388시간. 후보/신규 대조군은 update당 adapted forward2회 + teacher forward2회로 PLAIN보다 훈련비가 높다. 동일 updates를 동일 FLOPs라고 쓰지 않는다. 학습 가능8,712개 외에 동결 B0 LoRA294,912개와 backbone을 유지한다. 추론은 일반 C2와 같은 한 번의 forward다. 세부 시간·VRAM은 RESOURCES.csv, 평가 비용은 PREDICTIONS.json이다.

## 범위·신규성·미실행

본 연구의 새 구현은 temporal finite-response loss와 그 통제 비교다. Jacobian matching/교사 반응 증류 원리는 기존 연구에 있다. 아직 Time-PEFT·정식 robust PEFT 대비 우위 또는 새로운 일반 이론을 검증하지 않았다. Electricity·ETTm1 E는 반복 사용된 개발자료이며 다른 전력 계열도 독립 source가 아니다. 독립 새 source, 추가 seed, 새 모델·LR 탐색은 이번 범위 밖이며 실행하지 않았다. 실제 센서 사건 label도 없다. 본 결과로 현장 오류/실제 regime change 해결이나 모든 robust PEFT 불가능을 주장하지 않는다.

코드·데이터·기존 checkpoint hash는 SEAL.json, 수치 검산은 VERIFICATION.json, 프로토콜은 ../../experiments/temporal_response_peft_20260919/PROTOCOL.md에 있다. 원본 데이터·checkpoint·예측 cache는 로컬이며 GitHub 파일만으로 전체 수치 재현 가능하다고 쓰지 않는다.

<!-- FINAL_INTERPRETATION -->
## 실제로 확인된 구성요소의 역할

**TRP의 현재 방법론 근거는 확보하지 못했다.** 전력 전이 SHIFT8 nMAE는 TRP 0.399733396, PLAIN 0.377354736, B0 0.399761905다. 일부 조건의 작은 이득이 있는 것과 핵심 직접 대조를 통과한 것은 다르다.

같은 조건의 B0 대비 절대 보정 nMAE는 TRP 0.000195726, PLAIN 0.084777151, IDEAL 0.211232188다. TRP의 B0 반응 왜곡은 0.000150933로 작았다. 즉 반응을 유지하는 방향으로는 작동했지만, 이 고정 설정에서는 유용한 추가 적응도 거의 하지 않은 결과와 일치한다. 학습 불능·NaN·checkpoint 복원 실패가 원인은 아니다.

합성 SHIFT8의 이상적 반응 오차는 B0 0.283160608, TRP 0.283139761, PLAIN 0.259293013, IDEAL 0.222381696다. B0 반응 보존이 필요한 반응의 개선을 자동 의미하지 않는 사례다. λ=1의 이 실행에서 보인 함수 수준 결과이며, 모든 λ에서 필연적으로 실패한다는 증명이나 gradient 원인 기여율은 아니다. 보정량은 예측 MAE 개선율과 다른 지표다.

## IDEAL의 좁은 이득과 손해를 따로 남긴다

IDEAL은 전력 전이 SHIFT8에서 PLAIN 대비 4.9950%, MAG_ONLY 대비 1.2488% 낮은 오차였다. 하지만 MAG 대비 seed별 이득은 81551에서 -0.5844%, 81552에서 3.0586%로 방향이 다르다. 같은 전력 전이에서 REFERENCE 이득은 -1.7000%, FAULT 이득은 -3.8102%로 둘 다 손해다.

이것은 이미 정해둔 단순 대조군의 사후 기술 통계다. 새로운 주 성공 기준이나 우승자로 바꾸지 않는다. 일부 오류 악화만으로 모든 가능성을 폐기한 것이 아니라, TRP 자체가 핵심 직접 대조에서 뒤졌고 IDEAL의 우위도 조건·seed에 제한된다는 구분이다. IDEAL과 기존 MAG의 수치는 새로운 다중비교 유의성 결론으로 제시하지 않는다.

## 방법론 논문 목표에 대한 현재 판단

구현·학습·평가·검산은 완료했지만, 목표였던 '새 방법의 추가 가치를 뒷받침하는 방법론 논문 근거'는 아직 충족하지 못했다. 이 결과를 기존 분석 논문의 다른 제목으로 대체하지 않는다. 반응 차이를 맞추는 일반 수식은 이미 선행에 있으므로 TRP라는 이름이나 L1 변경만으로 신규성을 주장할 수 없다. [선행 수식 검토](../../research/temporal_response_method_20260919/NOVELTY_AUDIT_KO.md).

독립 자료 확인, 더 많은 seed 및 정식 PEFT 비교가 남았다는 사실은 이번 불리한 결과를 뒤집는 증거가 아니다. 현재 고정 TRP를 자동 재튜닝하거나 IDEAL을 사후 새 제안법으로 이름만 바꾸지 않는다. 새 학습은 추가하지 않았다.

[효과 그림 PDF](../../research/temporal_response_method_20260919/result_evidence/effects.pdf), [seed·자원·학습 손실 추가 근거](../../research/temporal_response_method_20260919/result_evidence/EVIDENCE_REVIEW_KO.md), [저장 예측 반응 진단](../../research/temporal_response_method_20260919/result_evidence/RESPONSE_DIAGNOSTIC_KO.md), [학습 장부 독립 검산](TRAINING_AUDIT.json), [평가 전 재사용 검사 수정](AMENDMENT_01_KO.md).
