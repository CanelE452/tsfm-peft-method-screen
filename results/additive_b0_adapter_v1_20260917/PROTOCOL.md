# B0 유지 + 추가 어댑터: 고정 실행 계약

사용자 승인: “잘되는 방법에 새 요소를 더해서 얼마나 좋아지는지”를 비교하자는 네 군 제안에 “그렇게 해줘”라고 승인했다. 이번 문서는 그 범위의 구현 결정을 실행 전에 고정한다. 이전 연구를 재개하지 않으며 결과를 보고 이 문서를 바꾸지 않는다.

## 비교와 질문

C0: v2 selected B0 checkpoint/저장 예측 그대로. 새 학습0회.
C1: 같은 B0 checkpoint에서 기존 q/v rank8 LoRA만 추가 학습. 추가 계산 기회의 대조.
C2: B0의 foundation/LoRA/head/input embedding을 모두 동결하고 일반 bottleneck residual adapter만 학습.
C3: C2와 같은 어댑터/초기값/8712 parameters. 지속 변화가 보이는 patch에서 추가 보정을 줄이는 fixed observed-persistence gate를 적용한다. 유일한 제안 후보다.

C2/C3는 원래 observed context를 clip하지 않는다. B0와 같은 native normalization/inverse다. encoder 전 patch embedding e에 다음을 더한다:

- s = stopgrad(median_patch RMS(e)), floor1e-6.
- delta = s*tanh(W_up GELU(W_down e)).
- dimensions512→8→512, bias포함. down Kaiming/default Linear init, up weight/bias0.
- C2: e_eff=e+delta.
- C3: e_eff=e+(1-mean_patch p_t)*delta.
- p_t는 observed median/MAD로 abs(d)>3인 점 중 trailing8 slots에서 같은 부호인 비율. window는context 경계에서truncate. extreme이아니면p=0.
- 과거의 persistence만 추가 보정을 감쇠하며 원래 B0 경로는 지우지 않는다. 모든 특이값의 원인을 판별한다고 주장하지 않는다.
- 시작 시delta0으로 B0와 bitwise같다. 어댑터off도 B0를 복원한다. 학습 후 자동 비열등성은 보장하지 않는다.
- C3의 V에서 p=0 및 fixed permutation ablation을optimizer0으로 한다. 의존성 진단이며 우월성 증명이 아니다.

일반 residual adapter와 near-identity 초기화는 알려진 구조다([Houlsby et al.](https://proceedings.mlr.press/v97/houlsby19a.html)). 추가 persistence mask의 신규성은 검증되지 않았다. C3가 C0보다 나아도 C1/C2로 충분하면 새 방법의 근거가 아니다.

## 공정성 및 예산

Electricity/ETTm1, 이전 sealed TRAIN256/V64/E128 distinct days, 이전4채널,512 context/64 horizon, 동일 pinned Chronos-Bolt-small을 사용한다. 모든 군이 같은 v2 matched TRAIN 입력·정답을 공유한다. TRAIN/V/E/원점·severity·기간을 다시 생성하거나 바꾸지 않는다.

baseline selection seed81550은 v2 B0 V선택 LR/checkpoint를, repeat81551/81552는 각 대응 v2 B0 selected checkpoint를 재사용한다. baseline E를 보고 seed/checkpoint를 고르지 않는다. 추가 adapter initial seed=optimizer seed+200000. C2/C3초기값은동일하다.

선택81550 × LR1e-4/3e-4, 반복81551/81552는고른LR만. 두원천×세학습군×(선택2+반복2)=24 fits. 32epochs×32updates=1024/fit, 본학습24576회. smoke는원천2×군3×2updates=12회. 전체24588 cap. 기존B0학습비용은공유된기존비용으로별도기록하며 추가비용0이라고전체학습이없었다고주장하지않는다.

C1/C2/C3모두AdamW(.9,.999),eps1e-8,wd0,clipnorm1,스케줄러없음,새 optimizer moments로시작한다. fixed TRAINsigma로normalized2pinball. FP32/TF32off/dropout0. effectivebatch32; OOM시micro32/16/8/4선택은optimizer0검사후고정한다.

checkpoint0/256/512/768/1024. V의REFERENCE/POINT8/BURST8/SHIFT4/SHIFT8 동일가중nMAE만선택. tie는작은LR,이른checkpoint. stage0도정식선택후보다. 같은기존TRAIN을추가로보는것의효과는C1/C2대조로분리한다. 결과에따른새LR/seed/rank/gate/threshold추가없음.

## 평가와 해석

모든선택봉인후새모델12개의전체Eprediction저장,기존C0예측4개는hash확인후참조한다. 이후정답채점. 이미여러번노출된E는개발근거이며독립시험이아니다. 앞으로의독립검증은이번범위에없다.

FAULT6조건동가중/REFERENCE/SHIFT4/SHIFT8/SHIFT_POINT를분리한다. SHIFT평균과history subset은참고. nMAE주지표,rawMAE/channelNRMSE/2pinball/crossing보조. 전후gain과C3/C2/C1직접대비,seed별방향,2000회7day paired blockbootstrap. 채널/조건/합성draw/optimizerseed를독립날짜로세지않는다. optimizerseed2의한계를적는다. 서로다른목표를합쳐우승자를만들지않는다. 임의1%문턱없음.

## 검사·안전·보존

GPU전snapshot/origin/입력label/checkpoint hash,추가모듈0출력,같은past에서future교체가출력에영향없음,공통C2/C3초기값/parameter count를검사한다. 실제12smoke updates로gradient/학습변화/frozenB0보존/restore/batch permutation/scalarpinball을검사한다. main동안update intent+journal,epoch exact resume. ambiguous update는자동replay하지않는다.

기존Watch사용,oneGPU/oneworker,startupfree4GiB,boundary1GiB,disk10GiB,cache50GiB,24h cap. RustDesk만기존승인예외. 다른작업종료금지. 중단은성능실패와구분한다.

REPORT.md/FINAL_DECISION.md에실제횟수/미실행/원점수/추가이득/손해/자원/신규성한계를한국어로작성하고검산후scoped commit/push한다. 원자료/weights/prediction cache는로컬보존하고manifest를공개한다. 후속후보·자동추가학습없음.
