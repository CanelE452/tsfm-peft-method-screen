# 동결 attention을 기준으로 수정하는 side adapter — 후보1·대조1, 최대8fits

기준fff2dadd0e511408b41835a5517c8a9e47960f7a. 사용자 활성 목표의 실패 후 후보 검토 범위에서 진행하는 별도 유한 개발 비교다. 완료된 건물120fits와 이전 후보의 판정·설정은 보존한다. 현재 E는DISCOVERY_REUSED_E이며 독립 PASS를 부여할 수 없다.

## 관찰과 한 가지 가설

동일 파라미터의 마지막 adapter는 LoRA보다 두 원천에서 나빴다. 저장 LoRA에서 q/k 또는v를 제거하면 모두 악화했다. 이 관찰은 내부 표현을 이용하는 side 경로를 검토할 근거지만, 시간관계만의 수정이나 모듈 제거에 따른 학습가치를 입증하지 않는다.

가설: 동결 모델의 층별 attention을 side attention의 기준분포로 주면, 같은 용량으로 처음부터 연결을 배우는 side 경로보다 예측에 유용한 시간·내용의 공동 적응을 배울 수 있다. 반례: 원래attention이 타깃에 부적합하거나 층별표현을 더하는 경로가 충분하지 않으면 prior가 이득을 주지 않는다.

## 완전한 명세

MOMENT-small의 동결encoder8층, hidden512, 원래LoRA는 초기0인채동결. z_i=encoder가 반환하는 hidden_states[1:]의 i번째(마지막은최종norm후), s_0=0. 각층 x_i=LayerNorm(z_i+s_{i-1},eps1e-5), Q_i=x_i A_Q, K_i=x_i A_K, V_i=x_i A_V. A세개와B는각층독립,bias없음,폭10,단일head. B는0초기화, 나머지Linear기본초기화,LayerNormweight1/bias0, seed+9019의독립RNG.

SIDE: P_i=softmax(Q_i K_i^T/sqrt(10)).
PRIOR: P_i=softmax(log P0_i +Q_i K_i^T/sqrt(10)).
공통 s_i=s_{i-1}+B_i(P_i V_i), 최종hidden=z_8+s_8, 기존공통forecast head와normalizer사용.

P0_i는동결encoder해당층의dropout전attention을head평균한분포다. 원래q/k출력과position_bias로nativeBF16 score matmul을재계산하고, FP32 log_softmax와head방향logsumexp-log(heads)로 logP0를구한다. 임의epsilon/floor없음. backbone의 train/eval/dropout정책은기존과같고, 모든backbone계산은no_grad다. 두군모두P0를추출해프로파일조건을맞춘다. SIDE는P0를사용하지않는다. 모든입력mask가present인기존96h프로토콜만지원한다.

각층파라미터=4×512×10+2×512=21,504, 8층=172,032. head589,920을더해761,952로기존LoRA와같다. 폭10은이등식으로고정하고탐색하지않는다. SIDE는LAST의저랭크side attention에착안한직접대조이며, 원논문의bias보정/초기화/연결전체를그대로재현했다고하지않는다. 두새군차이는logP0추가하나뿐이다.

## 데이터·학습·예산

기존balanced origins·epoch순열·staged data·32채널·96→96·70/10/20분할·train통계·seed41000/41001을보존한다. 두원천×두seed×두군=최대8fits,최대10,160updates. smoke는각원천·군2updates=8별도. 기존balancedLH/head-only/POINTWISE168은재학습하지않고기록참조한다.

AdamW lr.001 betas(.9,.999),eps1e-8,wd0,clip1; BF16batch8;StepLR5epochs gamma.5;최대20epochs;Vpatience5,min_delta1e-4;INIT포함최소V선택. 성능에따라폭·lr·epoch·seed·source를변경하지않는다. 준비/구현의의미오류는기록후수정1회까지,불리한점수를오류로처리하지않는다. 실패attempt도상한에서차감,새후보/대체fit없음.

한GPU/한worker/기존전역lock/guard,RustDesk만승인예외. 시작안정30초/free4GiB,실행중free1GiB/외부compute경계대기,누적대기600초/controller5400초. 공통환경안전하지않으면실행중지하고성능실패와구분한다.

## 평가·판정

8개V선택을먼저봉인한뒤노출E를평가한다. 각군의V선택과기존고정epoch(e41000=1,e41001=3,t둘=1)의같은updates비교를공개한다. 해당epoch미도달이면연장하지않고미실행으로기록한다. MSE/MAE/rawMAE·채널별원점수·source/seed효과·학습peak allocated/reserved·step시간·선택비용·전체비용구분.

개발추가지지신호는PRIOR가SIDE와balancedLH 각각대비두원천평균MSE1%이상개선하고두seed모두개선방향일때다. 설명용시간8블록paired bootstrap2000회(seed9019)구간도함께보여주되독립확증으로쓰지않는다. 메모리이득은별도로기록하며예측악화를자원PASS로전환하지않는다. 후보가좋아도미노출평가·직접선행검토가남아새방법론PASS는아니다. 부호가나빠도고정8fits의나머지군실행은성능만을이유로막지않는다.

## 사전검사

FP64explicit수식·parameter/input gradient일치(<1e-10),동일가중치에서비균일prior차이/균일prior의SIDE환원(<1e-12),B0 identity. 실모델761,952trainables·공통headhash·INIT BF16 V8기존LH와exact·frozen/buffer불변·smoke후side q/k/v/up/norm과head실제변경·finite. no_grad snapshot확인,미래poison학습창불변,checkpoint새모델V재생exact,전저장예측float64 scalar MSE/MAE검산(atol/rtol1e-12),기존결과·source/model/data해시보존. 검사실패후허용오차인상없음.
