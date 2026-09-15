# 동결 attention prior의 추가 가치 — 8-fit 결과

**후보PRIOR와 같은 용량의SIDE 직접 비교를 완료했다. 이번 E는 반복 노출된 개발 자료이며 독립 SCREEN_PASS가 아니다.**

실제8/8fits, 9143/10160 본학습updates, smoke8updates. 기존 LH/head-only/동일용량pointwise 각4fits는 재학습하지 않았다. 미완료fit0, 추가후속학습0, 실행오류/재시도0.

## 무엇을 바꿨는가

두 군 모두8층 중간표현을 받는 저랭크side attention과 같은head를 학습한다. PRIOR만 각층 frozen attention의head평균분포를logit기준으로 사용한다. SIDE는그기준없이연결을학습한다. 둘 다 Q/K/V와출력변환을학습하고backbone으로역전파하지않는다. 각층폭10은LoRA와같은761,952 trainables를맞추는등식으로정했다. 학습률·표본·초기head·선택정책은고정했다. [사전프로토콜](PROTOCOL.md).

## 원점수

| 원천 | recipe | SIDE MSE | PRIOR MSE | LH MSE | head-only MSE | pointwise168 MSE |
|---|---|---:|---:|---:|---:|---:|
| electricity | selected | 0.282389 | 0.279546 | 0.280010 | 0.297230 | 0.290154 |
| electricity | matched_old_epoch | 0.324322 | 0.321582 | 0.312357 | 0.345150 | 0.325419 |
| traffic | selected | 0.406510 | 0.400725 | 0.386548 | 0.436876 | 0.407270 |
| traffic | matched_old_epoch | 0.520836 | 0.532565 | 0.507064 | 0.553986 | 0.524193 |

각방법의 V선택과사전에정한동일epoch/updates를함께공개했다. E점수로epoch를바꾸지않았다. [seed별MSE·MAE·rawMAE](scores.csv), [모든채널](channel_scores.csv), [비교별효과](comparisons.csv).

| 원천 | PRIOR의 대조 | MSE 개선율 | 두seed 양수 | 사전1% 개발조건 |
|---|---|---:|---|---|
| electricity | SIDE | +1.007% | False | False |
| electricity | BALANCED_LH | +0.166% | False | False |
| traffic | SIDE | +1.423% | False | False |
| traffic | BALANCED_LH | -3.668% | False | False |

[설명용95%구간](descriptive_uncertainty.csv)은 같은원점에서두seed를평균하고시간순8블록paired bootstrap2000회(seed9019)로계산했다. 독립확증이나탐색편향보정이아니다. [레벨/일주기/나머지잔차분해](residual_components.csv)는정답을사용한설명용이며배포가능한보정기가아니다.

## 자원과 실제 실행량

| fit | epochs | updates | peak allocated MiB | peak reserved MiB | LH 대비allocated절감 |
|---|---:|---:|---:|---:|---:|
| electricity/41000/SIDE | 20 | 1280 | 425.869 | 450.000 | 60.18% |
| electricity/41000/PRIOR | 20 | 1280 | 425.635 | 462.000 | 60.21% |
| electricity/41001/SIDE | 13 | 832 | 425.885 | 462.000 | 60.16% |
| electricity/41001/PRIOR | 18 | 1152 | 427.619 | 462.000 | 60.00% |
| traffic/41000/SIDE | 20 | 1260 | 426.385 | 468.000 | 60.04% |
| traffic/41000/PRIOR | 20 | 1260 | 426.385 | 468.000 | 60.04% |
| traffic/41001/SIDE | 13 | 819 | 426.385 | 468.000 | 60.15% |
| traffic/41001/PRIOR | 20 | 1260 | 426.385 | 468.000 | 60.15% |

Controller 1228.7초, 최소GPU여유8582MiB, 비승인compute0표본. 두군trainables는LH와같아학습파라미터절감0이다. 원래0-LoRA가상주한채side모듈이추가되므로총모델이작아졌다고주장하지않는다. 학습peak와전체GPU사용량은다르다. prior추출을두군모두수행해비교조건을맞췄으며SIDE를최적화한최소비용벤치마크는아니다. [상세자원](resources.csv). 선택checkpoint까지step시간과총연구step시간은scores.csv에구분했다.

## 검산과 남은 한계

고유예측200개/200개 MSE·MAE·rawMAE를독립float64 scalar로검산했다. 최대MSE차1.11e-16. 선택checkpoint8개의새모델재생exact,동일target/origin/std·순열·source/model/data hash·선택봉인·LR/조기종료정책을확인했다. 기존2155개결과파일보존. [검산](verification.json).

사전개발조건충족=False. 신규성UNRESOLVED_PRIOR_COMPONENTS_KNOWN, 독립SCREEN_PASS=NOT_EVALUATED, 새방법론주제=NOT_CONFIRMED. 양성이라도현재E에대한사후개발신호이며별도미노출평가와선행대비차별성확인이남아있다.

LST/LAST의side경로·저차원attention, prior가중정규화의수학은알려진원리다. 현재차이는고정backbone의층별attention을side기준으로활용하는선택이며최초성은확정하지않았다. 원논문LAST전체재현도아니다. [선행·반례검토](RESEARCH_REVIEW.md). 후보가실패한조건의설정재탐색·추가backbone·미노출평가·후속후보학습은미실행이다. 원자료/weights/예측배열은로컬,코드·원점수·manifest·보고서는GitHub에보존한다.

## 학습 입력에서 확인한 attention 기준분포

봉인 후 보조 관찰로 각 원천의 기존 학습 입력 8개만 CPU FP32로 읽었다. 학습과 V/E 채점은 추가하지 않았다. GPU BF16 분포와 수치적으로 같다는 검사는 아니다.

| 원천 | 층별 평균 정규화 엔트로피 범위 | 균일분포와의 평균 L1 거리 범위 |
|---|---:|---:|
| electricity | 0.716–0.946 | 0.424–1.001 |
| traffic | 0.737–0.938 | 0.454–0.963 |

이 입력들에서 기준분포는 균일하지 않았다. 따라서 추가 prior가 단순히 균일분포여서 SIDE와 같은 연산이 됐다는 설명은 맞지 않는다. 그렇다고 이 분포가 예측에 유용하거나 모든 입력에서 같은 특성을 가진다는 뜻은 아니다. [관찰 원기록](prior_probe.json).

본학습 update 상한 중 미사용 1017회는 고정한 조기 종료 정책에 따른 것이다. 여유분으로 대체 학습을 실행하지 않았다.
