# 채널 압축 잔차 PEFT 주제 제안 검토

상태: **제안 검토 완료 / 실행 계약 미확정 / 신규 실험 0**. 첨부 문서는 계획 합의용 제안서이므로 이를 고정 실행 ZIP이나 14개 학습 완료로 취급하지 않는다. 원문은 [USER_PROPOSAL.md](USER_PROPOSAL.md), 확인한 파일 hash는 [EVIDENCE.json](EVIDENCE.json)에 보존했다.

## 판단

이 질문은 입력·출력 adapter와 작은 시간 모듈을 학습하는 PEFT 방법 연구로 검토할 수 있다. 다만 현재 근거로 주력 방법의 성공·신규성을 확정할 수 없으며, 준비된 Moirai 예측 API를 그대로 호출하는 학습은 제안서의 encoder 역전파 조건을 충족하지 않는다. 이 문제를 첫 실행의 실제 gradient 검사와 사전에 명시한 forward 계약으로 해결해야 한다.

AdaPTS가 잠재 특징 공간과 차원별 동결 단변량 FM을 사용하는 설명은 [공식 논문집](https://proceedings.mlr.press/v267/benechehab25a.html)과 일치한다. 전역·지역 성분 결합의 기존 연구에 대한 주의도 [Deep Factors](https://proceedings.mlr.press/v97/wang19k.html)와 부합한다. 논문 초록과 설치된 소스의 범위에서 확인했으며, 신규성을 망라한 문헌 검토나 공식 논문 수치 재현은 아니다.

## 기존 숫자와 해석 범위

고정 commit `6a04820`의 B_DECISION을 대조했다.

| 항목 | 원값 및 비교 |
|---|---:|
| R4 PILOT MSE | 0.2109640241 |
| U8 PILOT MSE | 0.2301656753 |
| R4의 U8 대비 PILOT MSE 감소 | 8.342535% |
| R4 batch2 latency | 0.015808362초 |
| U8 batch2 latency | 0.015197716초 |
| R4 latency 증가 | 4.018014% |
| R4 CAL MSE | 6.2555398941 |
| LINEAR CAL MSE | 0.5358804464 |
| R4 / LINEAR CAL MSE | 11.673387배 |

제안서의 수치는 맞다. PILOT 정확도 단서는 있지만 속도 우위와 기간 간 일관성은 입증되지 않았다. 이전 코드는 방법 이름에 따라 seed와 배치 RNG를 바꾸고 head를 무작위 초기화한다. 따라서 이전 R4/X4 차이는 같은 배치·같은 초기 출력의 잔차 입력 대조가 아니다. 원 판정 NO_GO_CURRENT_FORM은 보존한다. 새 비교가 이 공정성 조건을 맞추는 것은 타당하지만, 기존 실패를 성공으로 재명명할 근거는 아니다.

## 실제 구현의 핵심 문제

설치된 공식 Uni2TS commit `cfd46d4510ed8896f263116f32928eede05b0a75`의 `src/uni2ts/model/moirai/forecast.py:345`는 `distr.sample(...)`을 호출한다. `src/uni2ts/distribution/mixture.py:127–140`의 sample은 `torch.no_grad()` 안에서 동작하고 `has_rsample=False`다. AdaPTS의 Moirai wrapper는 이 표본의 median을 decoder에 넘긴다. 소스 확인상 이 경로로는 예측 손실의 gradient가 FM을 통과해 encoder에 전달되지 않는다. Decoder만 학습되거나 R4의 직접 잔차 경로에서만 encoder가 변하는 것을 전체 예측 경로 학습 성공으로 잘못 볼 수 있다.

이번 검토에서는 모델 로드·실제 backward를 하지 않았다. 따라서 실제 GPU smoke 검사를 통과했다고 쓰지 않는다. 향후 검사에서는 g=0 상태의 U4에서 E의 gradient·실제 update를 확인하고, R4에서 잔차 우회 경로의 gradient와 FM을 통과한 gradient를 구분해야 한다. 단순히 loss.backward가 오류 없이 끝나는 것만으로 부족하다.

가능한 미분 가능한 예측 통계나 다른 학습목적은 먼저 계약에 명시할 변경이다. 샘플 median을 분석적 mean이나 다른 손실로 조용히 교체하고 원 AdaPTS 경로의 완전 재현이라고 부르면 안 된다. 임의 straight-through gradient도 추가하지 않는다. 이 선택은 모든 비교군에 일관되게 적용하고 평가 estimator와의 관계를 기록해야 한다.

## 실행 계약에서 확정할 항목

원 제안의 후보 1개+대조 6개×2seed, 최대14fits는 유지한다. 다음은 아직 봉인되지 않은 사항이며 결과를 보고 채우면 안 된다.

1. **예측 forward·loss:** 미분 가능한 FM 경로, 학습 손실, 평가 point estimator와 sample/RNG 정책. 기존 네이티브 sample 경로와 달라지는 부분을 명시한다.
2. **일반 LoRA 대조:** 정확한 target module·rank·alpha·dropout, head 동시 학습 여부, 원채널 처리·정보 권한, 저장 상태 범위를 지정한다. 일반 LoRA의 compute가 압축 adapter와 다르면 비용 차이로 그대로 보고한다.
3. **학습 예산:** 각 seed 값, optimizer/LR/scheduler, batch·dtype, fit별 update 상한과 wall-time, 검증 간격, checkpoint 선택·early-stop 규칙을 고정한다. 두 작업일이라는 일정만으로는 optimizer 예산이 정해지지 않는다. 임의64step을 충분한 학습으로 취급하지 않는다.
4. **초기화와 공정성:** U4/R4/X4의 E/D 초기값과 원점·배치 순서 hash 일치, R4/X4의 동일 zero-head 및 bias 정책, PCA TRAIN-only, 모든 구성의 같은 채널·정규화 공간을 확인한다. 구조가 다른 U8/LoRA의 가중치 완전 동일성은 요구하지 않는다.
5. **자료·선택:** Weather의 정확한 calendar/index 경계, 합법 원점, TRAIN-only scaler, 평가 전 선택 봉인을 고정한다. V_SELECT/CAL/PILOT 개발 역할을 분리하고 TEST를 사용하지 않는다.
6. **비용·복원:** 같은 장치·dtype·batch에서 전체 21채널 출력을 생성하는 latency와 최대 메모리를 측정한다. E/D/g와 전처리를 포함하고, 예열·반복·동기화·입력전송 정책을 고정한다. 업데이트한 실제 객체의 checkpoint를 새 process에서 복원한다.
7. **판정:** 임의2%/3-of-4 기준을 새로 만들지 않는다. 두seed와 날짜별 paired 차이·비용을 보고 유지/종료/보류를 구분한다. 비열등성 주장을 원한다면 허용 손해를 실행 전에 별도로 고정한다.

## 자료 재사용 주의

Weather validation 기간은 이미 Chronos-2 주제선별에 일부 사용됐다. Moirai·L512/H96으로 바뀌어도 같은 기간이 새로운 독립 검증이 되지는 않는다. 이후 성공은 재사용 개발자료의 파일럿 신호로 보고한다. 기존 메타데이터상 21채널·52,696행, 원 MSFT 정수내림 분할 TRAIN[0,36887), VAL[36887,42156), TEST[42156,52695)와 남는1행이 기록돼 있다. 이 경계를 새 실험의 승인·봉인으로 자동 간주하지 않는다.

원자료 TRAIN의 시간 중복1건과 100분 간격1건도 이미 알려져 있다. 행 순서 benchmark를 쓸지 실제 시간 연속성을 추가 요구할지 실행 전에 적고, 결과에 따라 원점을 제외하거나 보간하지 않는다.

## 이번 검토에서 하지 않은 일

새 학습·추론·데이터 다운로드·gradient smoke·예측 채점·추가 후보 개발은 모두0이다. 원문과 이전 FAIL/NO_GO·진단 결과를 보존한다. 검토 결론은 **질문은 구체적이나, 현재 Moirai gradient 경로와 비교 계약을 해결하기 전에는 14fits를 시작할 상태가 아니다**이다.
