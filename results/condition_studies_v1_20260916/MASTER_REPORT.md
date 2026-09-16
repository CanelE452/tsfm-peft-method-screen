# 아홉 조건 PEFT 비교 — 최종 결과

실제 실행·검산: 2026-09-16~17 KST. 버전 식별자는20260916으로 유지한다.

**아홉 비교의 실행·평가·독립 검산을 완료했다.** 새 본학습116/116경로·59,392updates, 폐기smoke58updates를 수행했다. smoke 상한96 중 남은38updates는 필요한 검사 완료 후 사용하지 않은 여유 예산이며 필수 본학습 누락이 아니다. R05/N06는 기존 저장 예측을 사용했고 새 LoRA 학습·신경망 추론은0회다. 실행 완료는 새 방법의 성능·논문 통과를 뜻하지 않는다. 서로 다른 목표의 점수를 합산한 우승자는 만들지 않았다.

## 계약·기존 결과·실제 실행

단일 [MASTER_PROTOCOL](MASTER_PROTOCOL.md)을 사용했고 N01 → N02 → R05 → N06 → N03 → R04 → N07 → R08 → R09 순서를 지켰다. 전체 명세·정보 권한·원점·비교군·예산을 처음에 봉인했다. 기준485b15b의 기존48forecast경로는 정확히 같은 가중치·예측 해시를 검증해 R05/N06에서 참조했고 재학습하지 않았다. 이전 실험을 함께 재개하지 않았다.

| 후보 | 실행 | 본학습 | 업데이트 | 근거 | 신규성 |
| --- | --- | --- | --- | --- | --- |
| N01 | COMPLETE | 16 | 8192 | NEGATIVE_WITHIN_SCOPE | UNVERIFIED_VARIANT |
| N02 | COMPLETE | 16 | 8192 | POSITIVE_UNCERTAIN | UNVERIFIED_VARIANT |
| R05 | COMPLETE | 0 | 0 | PROMISING_WITHIN_SCOPE | KNOWN_CONTROL |
| N06 | COMPLETE | 0 | 0 | SIMPLE_METHOD_SUFFICIENT | KNOWN_CONTROL |
| N03 | COMPLETE | 16 | 8192 | NEGATIVE_WITHIN_SCOPE | UNVERIFIED_VARIANT |
| R04 | COMPLETE | 16 | 8192 | NO_SELECTED_ADAPTATION | UNVERIFIED_VARIANT |
| N07 | COMPLETE | 16 | 8192 | NEGATIVE_WITHIN_SCOPE | UNVERIFIED_VARIANT |
| R08 | COMPLETE | 16 | 8192 | POSITIVE_UNCERTAIN | UNVERIFIED_VARIANT |
| R09 | COMPLETE | 20 | 10240 | NEGATIVE_WITHIN_SCOPE | UNVERIFIED_VARIANT |

후속 집중 문제는 **0개**로 결정했다. 알려진 단순 대안의 유용성과 새로운 PEFT 구성요소의 필요성을 구분한 판단이며 분야 전체의 불가능 결론은 아니다.

필수 비교의 미실행 경로는0개이며 추가 후보·추가 seed·추가 학습률·추가 데이터셋은 실행하지 않았다. 정식 선행 전체 재현과 새 독립 기간 확증은 이번 계약에서 실행한 비교가 아니며 여전히 남은 한계다. CPU 정책 선택225설정, 의존구조 fitting9건을 신경망 fit과 구분한다. 학습 전 TRAIN 통계/간단한 ridge fitting·검색은 각 후보 정보 장부에 별도 기록했다.

## 실제 효과·단순 대안·신규성

| 후보 | 지표 | 원점수_반복평균 | 직접대비_gain | 단순대안과추가가치 | 신규성한계 |
| --- | --- | --- | --- | --- | --- |
| N01 | 시간별 NRMSE | AGE_RESID 0.396940 / RESID 0.396248 | −0.175%; CI [−0.204, −0.054] | 기본 LoRA 0.273648이 더 낮음; age항 추가 가치 없음 | 나이·잔차 입력의 미확인 변형 |
| N02 | 시간별 NRMSE | DELTA 0.387540 / RETRIEVE 0.379895 | −2.012%; CI [−5.169, 0.761] | 짧은 이력 기본 0.371624; 검색·보정 비용의 근거 미확보 | RAFT 정식 비교 미실행 |
| R05 | 지연 normalized 2-pinball | MONOTONE 0.111180 / BINARY 0.111746 | +0.506%; CI [0.169, 0.807] | LATEST 대비 +0.303%는 CI에0 포함; 중간 혼합은 두 모델 필요 | 알려진 정책; 기존 E 재분석; 신규 LoRA0 |
| N06 | 24h 합계 normalized CRPS | AR1 0.058248 / IID 0.064730 | +10.013%; CI [8.979, 11.125] | 복잡한 SHRUNK/EMPIRICAL은 AR1보다 나쁨; Brier는 개선 안 됨 | 알려진 단순 결합; 주변분포 개선 아님; 신규 LoRA0 |
| N03 | 시간별 NRMSE | LEARN_KERNEL 0.648056 / KERNEL 0.641412 | −1.036%; CI [−1.184, −0.274] | 고정 커널은 GRID보다 낮지만 학습 τ는 불필요 | 재구성은 알려진 처리; E 원점23.75시간 제한 |
| R04 | 정확도 NRMSE / 수정 RMS | 선택 INIT: NRMSE0.370034 / 수정RMS0.125241 | 모든 학습군 INIT 선택; 추가 적응0 | 단순 평활화: 정확도0.178% 손해로 수정RMS25% 감소 | 알려진 안정화; N-BEATS-S/TARW 정식 비교 미실행 |
| N07 | 시간별 NRMSE | PRED 0.249438 / SHUFFLE 0.247272 | −0.876%; CI [−1.314, −0.542] | 추정 predictability에 따른 주파수 가중의 추가 가치 없음 | 알려진 가중 FFT; 정식 선행 미재현 |
| R08 | 시간별 NRMSE | LEAD_H 0.380829 / LINEAR_H 0.381053 | +0.059%; CI [−0.024, 0.242] | FROZEN 0.375896이 더 낮음; 작은 보정 효과 불확실 | 고정 시차·가용성; 동적 탐지·LIFT 정식 재현 아님 |
| R09 | 시간별 NRMSE | NULL 0.606694 / MIXED 0.538191 | −12.728%; CI [−18.604, −8.311] | NULL 대 UNIFORM +0.171%는 CI에0 포함; 단순 MIXED가 더 낮음 | IMPUTE=계수1 NULL; 알려진 투영·보존 강도 대조 |

gain은 양수일수록 비교군보다 낮은 손실이다. 표의 서로 다른 지표를 크기순으로 정렬하지 않는다. 모든 원점수·seed·조건·고정512체크포인트 결과는 각 보고서와 CSV에 남겼다. 불리한 타깃이나 seed를 제외하지 않았다. [보고 태그의 의미](REPORTING_SCOPE.md), [최종 문제 선택](FINAL_DECISION.md).

## 실측 자원과 계산 한도

| 후보 | 본학습_optimizer_분 | 본학습_선택포함_분 | E_실측예측_분 | 최대_allocated_MiB | 오염updates |
| --- | --- | --- | --- | --- | --- |
| N01 | 13.391951 | 23.013674 | 8.004841 | 748.022461 | 0 |
| N02 | 13.458249 | 16.684765 | 0.518613 | 783.712402 | 0 |
| N03 | 13.567822 | 17.665790 | 1.123110 | 748.025879 | 0 |
| R04 | 24.570441 | 28.889944 | 1.084037 | 672.631348 | 0 |
| N07 | 13.328145 | 16.685858 | 0.544458 | 604.877930 | 0 |
| R08 | 13.497408 | 16.925415 | 0.281886 | 604.877930 | 0 |
| R09 | 16.587233 | 20.906009 | 0.581132 | 618.396484 | 0 |

본학습+smoke optimizer는 59,450/59,488회, nativeforward는 115,034/200,000회다. controller 분류별 계수는 {'eval': 46304, 'train': 67650, 'verify': 1080}이다. 본학습/smoke/검증/E/동결예측/복원/gradient 검사의 독립 분리는 [FORWARD_LEDGER](FORWARD_LEDGER.csv)에 있으며 전체 native 호출 수와 일치한다. 주 controller의 실제 경과는 157.74분, 최종 로컬 cache는 3.354GiB/100GiB다. 원자료 다운로드0바이트. 외부 compute 비용 오염은 0updates다. GPU는 한 worker와 사전 허용된 RustDesk 예외만 사용했다.

저장 예측 CPU 비교의 실측 루프 시간은 R05 8.749초, N06 5.108초다. 기존 모델을 만드는 과거 학습 비용이나 추가 독립 검산 시간은 이 수치에 포함되지 않는다.

CPU 검산·보고 작업을 학습과 병행했으므로 작은 시간 차이는 엄격히 격리된 속도 벤치마크의 우위로 해석하지 않는다. optimizer 시간은 forward/backward/update의 실측이고, 선택 포함 경로 시간에는 검증·체크포인트 처리가 추가된다. E 예측 시간은 guard/Python 비용을 포함하며 공유 경로는 한 번만 합산했다. CPU 전처리·검산·보고 시간까지 합친 end-to-end 비용과 같지 않다. N02 검색 비용은 별도 독립 재측정이며 원래 전처리 전체 실측이라고 주장하지 않는다. R04는 update당2 nativeforward이고 나머지는1이다. 파라미터 기본값은 rank8 LoRA1,179,648개; 보조계수·입력 행·CPU 통계·고유 정답 수 차이는 각 정보 예산 표에 있다.

## 독립 검산과 보존

[전체 봉인·예산 검산](verification.json), [게시 감사](publication_audit.json)에서 기존 파일 2,516개 hash 보존과 고정 설치 모델·소스의 일치를 확인했다. [선택 재계산](independent_selection_verification.json), [116경로의 원점 순서·최종 resume/Adam/RNG](training_record_verification.json), [CPU 단순 대조 선택](independent_CPU_control_verification.json), [추가 gradient 분해](gradient_contribution_verification.json)를 분리했다. 추가 gradient 검사는 사전 지정 TRAIN 사례의 선택 가중치에서64 nativeforward·0optimizer로 수행했고 가중치를 바꾸지 않았다.

봉인 후 발견된 수식/권한·보고 수정은 [AUDIT_CORRECTIONS](AUDIT_CORRECTIONS.md)와 hash 연쇄에 남겼다. R09 TRAIN 경계의 미완료 하루 통계는 해당 후보의 첫 학습 전에 교정했고 원래 통계도 보존했다. N06 variogram의 원단위/정규화 표기도 해당 CPU 실행 전에 분리했다. 희소 원점 bootstrap의 빈 재표집은 버리거나 다시 추첨하지 않고 undefined로 기록했다. 학습률·원점·seed·목표·허용오차의 성능 맞춤 변경은 없다.

## 해석의 범위

E는 기존 프로그램에서 사용한 공개 원천의 개발 평가다. R05/N06는 이미 공개된 E의 재분석이다. Chronos 사전학습과의 비중복도 확인하지 못했다. 두 seed·네 채널은 새 도메인이 아니다. 원점64개가 곧 독립64일이라는 뜻도 아니다. 특히 N03의 E 원점 범위는23.75시간, V는11.5시간에 불과하다. 다른 다수 track도 평가 원점이5–6개 index날짜/3–4개 관측 주간 블록에 몰려 있다. [원점 시간 분산](origin_dispersion.csv)을 참고한다.

2,000회 paired7일 bootstrap은 관측 자료·채널·seed에 조건부이며 아홉 주제 탐색의 다중선택이나 새 원천 불확실성을 해결하지 않는다. 빈 재표집이 있는 후보는 계산 가능한 표본에 조건부인 CI다. 0을 포함하는 구간을 동등성 증거로 쓰지 않았다. 각 후보의 알려진 구성요소와 정식 선행 미재현 범위는 LITERATURE_BOUNDARY와 [추가 확인 기록](LITERATURE_RECHECK.md)에 남겼다.

## 후보별 한국어 보고서

- [N01 ASYNC](N01/REPORT.md)
- [N02 ARCHIVE](N02/REPORT.md)
- [R05 VINTAGE](R05/REPORT.md)
- [N06 JOINT](N06/REPORT.md)
- [N03 CLOCK](N03/REPORT.md)
- [R04 REVISION](R04/REPORT.md)
- [N07 SPECTRAL](N07/REPORT.md)
- [R08 LEAD](R08/REPORT.md)
- [R09 MIXED](R09/REPORT.md)

코드·보고서·원점수·해시·검산 기록은 GitHub에 게시한다. 큰 원자료·예측 배열·가중치는 ignored 로컬 cache에 있다. GitHub만으로 전체 수치 재생이 가능하다고 주장하지 않는다. [최종 결정](FINAL_DECISION.md) 이후 자동 후속 실행은 없다.
