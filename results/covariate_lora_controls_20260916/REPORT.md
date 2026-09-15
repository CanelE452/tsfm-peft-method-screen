# 실제 예보 입력에서 LoRA 학습 대조 — 완료 보고

[확인] 최대 6 fits를 모두 실행했다. 이 실행은 표준 LoRA, 공변량 드롭아웃, 과거 예보 augmentation이라는 **알려진 대조군의 유용성**을 비교한다. 새로운 PEFT 구조나 독립 방법론 PASS는 아니다.

## 연구 질문과 고정 비교

기존 최신 예보 입력은 target-only보다 평균적으로 유용했으나, 경로 혼합·단순 신뢰도 신호에서 새 방법의 추가 가치를 확보하지 못했다. 이번 질문은 같은 입력의 LoRA 적응에 학습 가능한 여지가 있는가, 그리고 드롭아웃 또는 이전 예보 학습이 그 위에 무엇을 더하는가이다.

OS Gorredijk 한 타깃, TRAIN 3–4월 8원점 / V 5월 4원점 / D 6–9월 16원점이다. 모두 이미 이전 실험에 노출된 DISCOVERY 자료다. 미래 기상 정답은 읽지 않았고 기존 simulated-as-of 예보 파일을 재사용했다. context336h, horizon24h, feature3, TRAIN 과거만으로 공통 표준화했다. 실제 배포 공개시각의 증거는 아니다.

세 방법 모두 rank1/alpha2, 96 q/k/v/o projection, 147456개 학습 파라미터, frozen backbone/native head, lr1e-4, 120updates, seed61710/61711이다. STD는 최신 예보, EXODROP은 과거·미래 전체 채널에 같은 p=.3 inverted dropout, VINTAGE는 epoch별 네 예보 경로를 순환한다. VINTAGE에는 추가 예보 정보가 있으므로 같은 계산량과 같은 정보량을 혼동하지 않는다.

모든 6 fits의 V 선택을 봉인한 다음 D를 채점했다. SELECTED는 V 최소 체크포인트, FIXED120은 사전 고정 진단이다. D에서 유리한 체크포인트나 조건으로 주 비교를 바꾸지 않았다.

## 실제 실행과 미실행 범위

- 본학습 **6/6 fits, 720/720 updates**, 별도 smoke 3회·6updates. 오류 0건.
- 모델 forward 1206회: gradient-enabled 학습 726회, no-grad 추론 480회. 저장 예측 448개, D 원점수416행, V검산96개, 새 모델 체크포인트 재생24개.
- 계획된 세 방법·두 seed·최신/과거 경로·selected/fixed120 평가의 미실행 0. 추가 후보·학생 모델·다른 타깃·10–12월 평가·학습률 탐색은 실행하지 않았다.

## 원점수

Primary는 정렬된 21분위수의 mean 2-pinball / 부하 context 표준편차이며 낮을수록 좋다. 아래는 두 seed 평균이다. 최신/과거 경로를 합쳐 새 지표를 만들지 않았다.

| 평가 | 입력 | INIT0 | STD | EXODROP | VINTAGE |
| --- | --- | ---: | ---: | ---: | ---: |
| SELECTED | 최신 k0 | 0.149478673 | 0.149478673 | 0.149478673 | 0.142617770 |
| SELECTED | 과거 k3 | 0.177160631 | 0.177160631 | 0.177160631 | 0.178240239 |
| FIXED120 | 최신 k0 | 0.149478673 | 0.151129372 | 0.148191278 | 0.142617770 |
| FIXED120 | 과거 k3 | 0.177160631 | 0.197165746 | 0.186411384 | 0.178240239 |

보조 지표(raw pinball·RMSE·MAE·80% 포함률/폭), 각 seed와 월 원점수는 [macro](macro_scores.csv), [월별](monthly_scores.csv), [전체](scores.csv)에 보존했다. 원본 무학습 예측과 이번 표준화 INIT0의 차이는 [직접 비교](initial_reference_comparison.csv)에 남겼다. 직접 이득의 기준은 이번 INIT0이다.

## 각 구성요소의 추가 가치

| 선택 | 입력 | 비교 | 개선율 % | 개선 월/4 | 월 block bootstrap95% % |
| --- | --- | --- | ---: | ---: | --- |
| SELECTED | k0 | STD vs INIT0 | 0.0000 | 0 | [0.0000, 0.0000] |
| SELECTED | k0 | EXODROP vs STD | 0.0000 | 0 | [0.0000, 0.0000] |
| SELECTED | k0 | VINTAGE vs STD | 4.5899 | 2 | [-1.0662, 10.7753] |
| SELECTED | k3 | STD vs INIT0 | 0.0000 | 0 | [0.0000, 0.0000] |
| SELECTED | k3 | EXODROP vs STD | 0.0000 | 0 | [0.0000, 0.0000] |
| SELECTED | k3 | VINTAGE vs STD | -0.6094 | 1 | [-7.2123, 7.5256] |
| FIXED120 | k0 | STD vs INIT0 | -1.1043 | 1 | [-9.0079, 9.0076] |
| FIXED120 | k0 | EXODROP vs STD | 1.9441 | 3 | [0.3172, 3.4560] |
| FIXED120 | k0 | VINTAGE vs STD | 5.6320 | 4 | [1.9844, 7.8893] |
| FIXED120 | k3 | STD vs INIT0 | -11.2921 | 1 | [-19.9947, 2.6471] |
| FIXED120 | k3 | EXODROP vs STD | 5.4545 | 4 | [3.2703, 7.2516] |
| FIXED120 | k3 | VINTAGE vs STD | 9.5988 | 4 | [5.1778, 13.4796] |

양의 개선율은 오차 감소다. 월 단위 2000회(seed61712) 재표집이며 seed·원점을 독립 타깃처럼 세지 않았다. 4개월·한 타깃·재사용 D에 대한 기술적 불확실성이지 독립 일반화 검정이 아니다. 두 seed 방향은 [효과 전체](effects.csv)에 별도로 공개한다.

## 자원과 재현 범위

전체 controller 202.92초(안전 대기 30.27초 포함), 최소 GPU free 8471MiB. 본학습 peak allocated 최대 588.42MiB. 비승인 compute 표본 0, 오염 기록 update 0.
모든 군의 파라미터 예산은 동일하다. 메모리·시간의 [실측](resources.csv)을 공개하며 구조적 자원 절감은 주장하지 않는다. full trajectory 시간에는 중간 검증·저장 비용이 들어가고 optimizer 시간은 그 합을 별도로 표시했다. 선택 지점 elapsed 역시 연구 중간 검증을 포함하여 순수 배포 시간을 뜻하지 않는다.

[독립 검산](verification.json): D 2496개 지표, V 96개 primary, checkpoint24개와 저장 예측 해시, TRAIN 전용 표준화, 동일 초기화·순서, 과거 결과 2257개 보존. D 지표 최대 절대오차 5.82e-11; 재생24개 중 exact 24개.
초기 native pipeline normalized max error 0, label poison 예측 불변, native loss mask/reduction 검사 통과. native loss는 target 한 행의 24시점만 관측하고 32시점·4변수의 native reduction을 유지한다. 초기 raw/표준화 입력 예측 차이는 1.34e-06(context std 단위)로 별도 측정했다.
원자료·예측·weights는 로컬 ignored cache에 있다. GitHub에는 코드·해시·원점수·보고서를 올리므로 원격 파일만으로 수치 재생이 완결되지는 않는다.

## 신규성과 남은 한계

LoRA 자체와 dropout/vintage augmentation은 알려진 기법이다. Exogenous Dropout의 채널 masking 규칙을 적용했지만 원 논문의 backbone·데이터·전체 실험을 재현한 것은 아니다. CoRA/UniCA/TFMAdapter 대비 방법 우위는 직접 비교하지 않았다. [선행 경계](../../research/covariate_reliability_diagnostic_20260916/LITERATURE_BOUNDARY.md)에 실제 읽은 범위를 기록했다.

NOVELTY=KNOWN_METHOD_CONTROLS_ONLY. 이 결과가 좋아도 새 PEFT 방법론 PASS가 아니다. 결과가 나빠도 모든 PEFT나 주제가 불가능하다는 결론은 아니다. 한 타깃, 적은 TRAIN/V, 고정 rank/lr 하나, 과거 개발 자료 재사용, simulated availability, 기상 vintage가 진짜 ensemble이 아니라는 한계가 남는다.

EXECUTION=COMPLETE. PREDICTIVE_EVIDENCE=POSITIVE_BUT_UNCERTAIN(최신 예보), 과거 예보에서는 INIT0 대비 TRADEOFF. NOVELTY=KNOWN_METHOD_CONTROLS_ONLY. 새 방법론 주제는 미확보이며, 알려진 vintage 학습에서 좁은 양성 신호를 확인했다. [구성요소·다음 연구 판단](INTERPRETATION.md)에 해석한다. 새로운 후보 학습은 이 실행에서 자동으로 추가하지 않는다.
