# 예보 버전의 노출 순서 대조 — 최대4 fits

2026-09-16, 기준64532f7. 기존6-fit 및 보정 대조를 보존한다. 사용자의 후속 연구 목표 아래 예보 버전 다양성과 학습 순서의 교란을 제한적으로 검사한다. 새로운 방법론/독립 PASS를 만드는 실행이 아니다.

## 고정 데이터·모델

covariate_lora_controls의 동일 입력28원점, TRAIN3–4월8 / V5월4 / D6–9월16, 같은 TRAIN 전용 표준화와 Chronos-2 revision29ec3766d36d6f73f0696f85560a422f50e8498c. 미래 기상 정답·새 데이터·추가 날짜 없음. 모든 표본은 노출된DISCOVERY다. 기존 raw/선택/보정 결과의 판정을 바꾸지 않는다.

같은rank1alpha2,96 q/k/v/o projections,147456파라미터, frozen native backbone/head, FP32eval/dropout0/TF32off, AdamWlr1e-4betas(.9,.999)eps1e-8wd0clip1,120updates. seed61710/61711. 원점 순서는 기존seed+epoch permutation그대로. 구현 helper는 기존모듈을 읽기전용 재사용하고 OUT/global을변경하지 않는다.

## 대조군

- CYCLE: 기존VINTAGE2fits 참조, 재학습0. 각원점의15회노출 예보버전열=[0,1,2,3]×3+[0,1,2].
- REVERSE: 위15개 버전열을 뒤집음. 원점 순서는그대로, 각원점 마지막노출은k0. 신규2fits.
- SHUFFLE: 각원점의위15개버전열을 np.default_rng(seed+60000+canonical_train_origin_index).permutation으로한번섞고봉인. 원점별다른순서, 동일멀티셋. 신규2fits.

각원점각경로노출수[4,4,4,3]가 CYCLE/REVERSE/SHUFFLE모두같다. 전체120updates에서 학습 입력·label 멀티셋과계산예산동일, 순서만다르다. prefix40/80에서는정보가다를수있으므로 fixed120이순서효과의주비교다. REVERSE와SHUFFLE을새아키텍처라고하지않는다.

## 선택·평가

checkpoint0/40/80/120, 모든fit120까지학습. V4k0primary최소·동률이른step을선택하고4fit선택봉인후D평가. selected와fixed120모두저장하되순서의동일정보비교는fixed120이다. k0/k3를각각보고합치지않는다. 기존STD/EXODROP/CYCLE/INIT0도같은원점표에provenance와함께유지한다. 새로운출력보정기는fit하지않는다.

primary=정렬21분위수mean2pinball/contextstd. rawpinball/RMSE/MAE/coverage80/width도공개. 고정비교={REVERSE/SHUFFLE각각vsCYCLE,STD,INIT0},두seed및월별. paired월blockbootstrap2000(seed61712). 뒤자료에서우승방법을골라배포/독립PASS라고하지않는다.

순서변경후에도효과가남으면원래순서만의착시라는설명은약해지나일반화/신규성은미확인. 효과가크게바뀌면최적화순서와입력다양성을분리하지못했다고보고한다. 문턱추가/수정없음,knowncontrol연구판정만한다.

## 상한·검사

본학습4attempts/480updates,smoke2arms×2updates=4별도. 오류fit도차감. GPU단일worker/기존lock/Watch,사용자승인RustDesk만예외,시작30초4GiB/runtime1GiB/wait600/controller3600초. nonfinite/OOM에설정변경재시도없음. 준비API오류한번수정가능·기록필수.

기존소스/입력/modelhash,순서와멀티셋,동일초기화,학습파라미터수,finitegrad/변경/frozen/buffers,초기기존V예측재생 및 선택weights새모델재생(normalizedmaxabs<=1e-5),scalarprimary(abs/rel1e-10),원본점수재생과기존결과보존을검사한다. A첫gradient0허용. 기존nativepipeline·targetpoison·FP64LoRA수식검사는동일helper의이전검증을참조하고재실행범위를구분한다.

한국어REPORT와원점수·비용·미실행·한계·검산을push. 원자료/weights/cache는로컬. 추가후보·다른seed·학습률·데이터범위자동추가0.
