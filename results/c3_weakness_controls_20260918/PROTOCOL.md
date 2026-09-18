# 실행 전 고정 규약

유일한 사용자 설계: EXECUTION_CONTRACT.md (원본 Downloads/c3_weakness_resolution_plan_20260918.md). 현재 사용자의 ‘그거 해주면 좋겠는데’를 §4–6의 실행 승인으로 적용한다. §7의 LCL 준비는 별도 합의라고 명시되어 있어 범위 밖이다.

- C3/B0 재설계·재튜닝 없음. 새 대조 POS_ONLY/MAG_ONLY/OUTPUT_CONTEXT만 Electricity/ETTm1에서 학습.2sources×3controls×(2 selection LR+3 repeats)=30 fits,각1024updates. smoke6×2=12,전체30732 상한. 기존58fits 예산과 별개인 이번 승인 범위다.
- 기존 source selected B0 hash/seed를 공유. seed81550,LR .0001/.0003로 선택하고81551/52/53에 선택 LR 반복. checkpoint0/256/512/768/1024,V의REFERENCE/POINT8/BURST8/SHIFT4/SHIFT8 동일 가중 nMAE. tie는작은LR/이른step. C3와 기존 대조는 완료 가중치/예측 재사용.
- matched TRAIN cache/원점/augmentation/order 함수를 그대로 사용. source별1024examples,32epochs,32updates/epoch,effectivebatch32,micro32,FP32/TF32off/dropout0,AdamW(.9,.999),eps1e-8,wd0,clipnorm1,no scheduler. loss는 TRAIN sigma로 normalized twice-pinball. 데이터/정답/optimizer 기회 동일.
- POS_ONLY: C2와 동일 초기 어댑터+32 position logits0;gate=sigmoid(logit). 별도 gate LR .01 고정. 총8744params. MAG_ONLY: observed robust scale/threshold3,1−patch extreme fraction;8712params. OUTPUT_CONTEXT:72→64 affine+scalar gamma;W normal(0,.01),seed+300000,bias0,gamma0. 총4673params. W init의 미지정 세부는 결과 전 이 값으로 고정. 모든 feature는 실제 B0 InstanceNorm loc/scale(현재 arcsinh=False);64개 median forecast+8개 연속64관측 구간 평균. 모든 quantile에 같은 원단위 correction. 공식 COSA/TAFAS 재현이 아니다.
- 평가 panel은 electricity 기존4계열, electricity_transfer 기존16계열, ettm1 기존4계열. 표본/변형을 새로 만들지 않는다. standard10조건과기존shape9조건(원래8형태+PULSE와짝인SHIFT8reference) 모두 보존. 새 가중치18개×3개 source/transfer 조합×2종=54개 new prediction views. selection seed E 포함 금지. 새로운 ETTm2/NESO/LCL 평가·학습 없음. 원천2개와 해당 전력 전이에서 질문에 답하며 다른 완료 결과는 바꾸지 않는다.
- 주요 대조SHIFT8의 C3 대POS_ONLY/MAG_ONLY/OUTPUT_CONTEXT. C2/RECENCY/C0/C1/MEAN/ROTATE16 및 기존F0도 함께 보고. F0는fake seed복제 없음. 원자료/FAULT6종 및평균/SHIFT4/SHIFT_POINT/shapes 음수결과 모두 유지. 과거 사용한 개발E임을 명시.
- 전체 V 선택을 봉인→전체54개 E 예측 저장·hash→채점. nMAE/MAE/channel-mean nRMSE/2pinball/crossing,모든seed 원점수. gain=100(1−C3/control). 날짜7일 paired block bootstrap2000회,seed86700+1000×고정panel순서(electricity,transfer,ettm1). seed평균에조건부. 세주대조95% 및Bonferroni3 구간,transfer는계열+날짜 보조구간. 전체조건공통 fullE128의blocks로 재표집한다.
- 새구성요소의계수/gradient/변화량,동결보존,초기/off B0 정확동일,restore,source TRAIN sigma,공통draw를검사. CPU수학검사와실모델검사를구분. 시점/metadata/future 입력 없음. smoke폐기; 학습곡선 step0선택은실행실패가아님.
- GPU startup4GiB/30초·runtime1GiB/RAM2GiB/disk10GiB,승인RustDesk만예외,oneworker,4시간 cap. resume는정확한epoch/state/optimizer/RNG/ledger가일치할때만. ambiguousupdate를없던것으로재실행하지않는다. 부진한성능으로정상 비교를취소하지않는다.
- 모든 실행·선택·scoring·검산 후 한국어REPORT/FINAL_DECISION, MATCHED_CONTRASTS.csv,seed/자원/claim boundary 제공,scopedcommit/push. 후속자동학습 없음. arbitraryPASS/동등성기준 없음.
