# 예보 버전 노출 순서 대조 — 완료

기존 예보 버전 학습의 이득이 노출 순서에 얼마나 민감한지 비교했다. CYCLE은기존VINTAGE를표시하는별칭이며재학습하지않았다. REVERSE/SHUFFLE은같은입력·정답멀티셋과초기값에서예보버전순서만달라진다. 노출된개발자료이며새PEFT발명/독립PASS가아니다.

## 실제 실행

신규4/4fits·480updates, smoke2회·4updates. 이전6fits는원본예측참조로유지했고추가학습으로세지않았다. 신규모델forward 758회(학습484,추론274). 계획된신규평가256행의미실행0. 새로운보정기·다른후속후보학습0.
각원점k0/k1/k2/k3를4/4/4/3회보는15epoch,각epoch같은8원점순열,총120updates. 새 학습률·기간·seed를추가하지않았다. 부하context336h/예측24h,rank1alpha2/147456개LoRA파라미터와기존native모델경로유지. 동일정보멀티셋의직접비교는FIXED120이며selected의prefix정보는서로다를수있다.

## 원점수 — 두seed 평균

| 정책 | 경로 | INIT0 | STD | EXODROP | CYCLE | REVERSE | SHUFFLE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FIXED120 | k0 | 0.149478673 | 0.151129372 | 0.148191278 | 0.142617770 | 0.142647084 | 0.142398379 |
| FIXED120 | k3 | 0.177160631 | 0.197165746 | 0.186411384 | 0.178240239 | 0.177899909 | 0.178285583 |
| SELECTED | k0 | 0.149478673 | 0.149478673 | 0.149478673 | 0.142617770 | 0.145767640 | 0.146014092 |
| SELECTED | k3 | 0.177160631 | 0.177160631 | 0.177160631 | 0.178240239 | 0.177740282 | 0.178103923 |

Primary는기존scaled2pinball,낮을수록좋다. 모든seed/보조지표와원점수를[전체표](comparison_scores.csv)에보존했다. 이전보정연구결과를이번raw표와같은조건으로합치지않았다.

| 고정120 비교 | 경로 | 개선율 % | 개선월/4 | 월block95% % |
| --- | --- | ---: | ---: | --- |
| REVERSE vs CYCLE | k0 | -0.0206 | 3 | [-0.1426, 0.0432] |
| REVERSE vs STD | k0 | 5.6126 | 4 | [2.0193, 7.9272] |
| REVERSE vs INIT0 | k0 | 4.5703 | 2 | [-1.1387, 10.7567] |
| SHUFFLE vs CYCLE | k0 | 0.1538 | 3 | [-0.1873, 0.4284] |
| SHUFFLE vs STD | k0 | 5.7772 | 4 | [2.0311, 8.0066] |
| SHUFFLE vs INIT0 | k0 | 4.7367 | 2 | [-0.6332, 10.8557] |
| REVERSE vs CYCLE | k3 | 0.1909 | 3 | [-0.0302, 0.3485] |
| REVERSE vs STD | k3 | 9.7714 | 4 | [5.3612, 13.7391] |
| REVERSE vs INIT0 | k3 | -0.4173 | 1 | [-7.1624, 7.8172] |
| SHUFFLE vs CYCLE | k3 | -0.0254 | 2 | [-0.4016, 0.4028] |
| SHUFFLE vs STD | k3 | 9.5758 | 4 | [5.2570, 13.1198] |
| SHUFFLE vs INIT0 | k3 | -0.6350 | 1 | [-7.0700, 7.8070] |

## 검산·비용·한계

독립scalar지표4032개·V primary64개,입력멀티셋/원점순서,새모델복원16개(exact16),기존결과2299개hash보존을검사했다. 기존nativepipeline/targetpoison/FP64수식검사는같은helper의이전검증을참조했다. [검산](verification.json).
GPU controller 147.52초,대기30.27초,최소free 8096MiB,peak allocated최대589.29MiB. 비승인compute표본0,오염update0. 외부compute통계의RustDesk는승인된예외다. 상세fit비용은[resources](resources.csv).
한타깃·TRAIN8/V4/D16·4개월·동일개발기간재사용이다. 월block2000회(seed61712)는서술적구간이며독립확증이아니다. 순서가달라도좋다는관측만으로추가정보의원인이나새구조필요성이증명되지는않는다. 같은정보를다른순서로학습하는알려진대조이며신규성은KNOWN_ORDER_CONTROLS_ONLY다. 원자료·예측·weights는로컬cache에남기고코드·해시·점수·보고서를공개한다.

EXECUTION=COMPLETE. 성능/연구해석은[INTERPRETATION.md](INTERPRETATION.md). 사용자최신지시에따라이기상예보축에서추가후속을시작하지않고,별도MOMENT단일계약으로이동한다.
