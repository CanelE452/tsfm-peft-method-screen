# 논문 방법·실험 설정의 정확한 명세

## 기준선과 입력

Chronos-Bolt-small의 고정 snapshot을 사용한다. 문맥512,예측64,patch16,32개patch,embedding512,원래quantile0.1~0.9다. 원래Chronos-T5의이산token모델과혼동하지않는다. B0는원래관측입력과고정합성증강으로학습한LoRA모델이다. attention의q/v36개512×512선형층에rank8,alpha16을부착해294,912개를학습했다. C0는선택된B0를그대로고정한평가명칭이다. C1은같은LoRA를추가학습하고,C2/C3는B0의LoRA를포함한기존가중치를모두동결한다.

모델이 받는것은오류또는변화가포함된관측과거x와TRAIN구간의계열별population표준편차sigma뿐이다. generator상태,clean x0,실제합성변화량,오류위치,평가정답은모델·gate입력에들어가지않는다. C3는x를clipping하지않는다.

## 관측 기반 고정 gate

문맥중앙값m과r=max(1.4826 median(|x−m|),0.1 sigma)를계산한다. d_t=(x_t−m)/r, e_t=1[|d_t|>3]다. 각관측t에서최근최대8개관측중같은부호의extreme비율을계산하고현재e_t를곱해p_t를정한다. 시작부는가용관측수로분모를줄인다. 문맥전체중앙값을쓰지만예측원점이후의정답을쓰지않는다.

patch j의추가수정gate는 g_j=1−mean(p_t : t∈patch j)다. C2는항상g_j=1이다. MEAN은원래g의patch평균을모든patch에적용,ROTATE16은32개gate를16칸회전,RECENCY는gate를내림차순정렬해최근patch일수록작은추가수정을주도록한다. RECENCY도원래관측으로계산한gate값을사용하므로단순히고정된최근N개를0으로하는방법과같지않다.

## 추가 residual

기존patch embedding h_j에대해 delta_j=c(x)tanh(W_up GELU(W_down h_j+b_down)+b_up)를계산하고 h'_j=h_j+g_j delta_j로바꾼다. bottleneck은8,출력은512다. c(x)는patch별embedding RMS의중앙값을1e−6이상으로제한하고gradient를차단한값이다. up weight와bias를0으로초기화하므로시작시정확히B0와같고,adapter를끄면학습후에도B0로복원된다. 이것은adapter를켜두었을때원자료오차가절대악화하지않는다는보장이아니다.

추가파라미터는512×8+8+8×512+512=8,712개다. 배포시원래LoRA와합친적응파라미터는303,624개이며backbone전체를제외한수치다. 학습파라미터비율을GPU메모리절감률로표기하지않는다.

## 부모 연구의 학습·선택

TRAIN256개서로다른날짜×4채널=1,024기본문맥,32epochs,유효batch32로1,024updates. AdamW(beta=.9/.999,eps1e−8,weight_decay0),gradient norm1로제한,FP32·TF32 off·dropout0. 학습loss는TRAIN sigma로정규화한9quantile의2pinball평균이다.

0/256/512/768/1024 checkpoint를저장한다. LR grid는1e−4/3e−4이며선택seed에서V로고른LR를반복seed에고정한다. V는64날짜×4채널,REFERENCE/POINT8/BURST8/SHIFT4/SHIFT8의동등가중nMAE다. 동률은작은LR·이른checkpoint. 반복seed는Electricity/ETTm1에서81551/81552/81553,ETTm2에서85551/85552/85553이다. 모델별최종checkpoint와LR는부모MODEL_SELECTION.json이정확한명세다.

TRAIN은REFERENCE/POINT/BURST/SHIFT를균등순환한다. fault강도는4/8/16 robust scale,SHIFT는4/8,지속길이는24/48이다. standard E의SHIFT는길이32이며과거최근구간과미래전체에같은변화를더한다. fault는입력만수정한다. SHIFT_POINT는SHIFT4+POINT8이다. E의POINT는2점,BURST는연속8점이다. 별도형태는STEP6/12의길이17/63,STEP8의31/33,RAMP8_D32,PULSE8_D32다. PULSE와pairedSHIFT8은같은입력·다른미래로식별성한계를드러낸다.

## 데이터와 노출

Electricity는26,304시간 index,321열중0~3을source학습에사용하고16개별도열로전이평가한다. 경계는0/15782/21043/26304다. ETTm1/2는69,680개15분값,HUFL/HULL/MUFL/MULL,경계0/41808/55744/69680이다. TRAIN256/V64/E128개서로다른날짜를기간전체에서먼저균등선택하고일중phase를분산했다. 이전부터노출된개발자료임을명시한다.

NESO는2025 ND단일국가수요의반시간값을UTC시간별평균으로변환한8,760개다. sigma는상반기,평가는하반기128개서로다른날짜다. Electricity모델·선택값을그대로이전하고NESO에서학습·선택하지않았다. source전이16계열도자체TRAIN구간sigma를허용했다. 원점·열·날짜·hash의기계적명세는부모및외부확장의DATA_MANIFEST/ORIGIN_AUDIT에있다.

## 이번 보강의 위치

부모연구58개새fit를다시하지않고,저장된가중치로114개논리적view를검증했다. 이중90개새평가/24개재사용이다. 고정1024비교는selected주결과의기전진단용보조자료다. 추가된F0와단순예측의seed는0 하나이며3회학습한것처럼세지않는다. 전체E예측을저장한후채점했다. 실행완료와논문성공은별도판단이다.
