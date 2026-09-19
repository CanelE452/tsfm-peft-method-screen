from .common import *
import pandas as pd

def markdown(frame):
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))

def report():
    e=pd.read_csv(OUT/'EFFECTS.csv');raw=pd.read_csv(OUT/'RAW_SCORES.csv');v=read(OUT/'VERIFICATION.json');receipts=[read(p) for p in (OUT/'fits').glob('*/receipt.json')];new=[r for r in receipts if not r.get('reused')]
    costs=pd.DataFrame([dict(source=r['source'],arm=r['arm'],seed=r['seed'],reused=r.get('reused',False),train_seconds=r['optimizer_seconds'],validation_seconds=r['validation_seconds'],peak_allocated_GiB=r['peak_allocated']/2**30,selected_step=r['selected']['step']) for r in receipts]);costs.to_csv(OUT/'RESOURCES.csv',index=False)
    primary=e[e.primary_family];compact=e[(e.stage=='selected')&(e.condition.isin(['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT']))&(e.baseline=='PLAIN')]
    text=f'''# 시간 반응 보존 PEFT 개발 파일럿

실행 완료와 방법론 근거를 구분한다. 신규 {len(new)}/16 fits, {sum(r['updates'] for r in new)}/16,384 main updates, smoke16updates, 기존 PLAIN4fits 재사용. E prediction192views 저장 후 채점·scalar {v['scalar_metrics']}개 검산. 이 보고서는 학습 성공을 논문 PASS로 바꾸지 않는다.

## 사전 주 비교

전력16계열 전이, selected SHIFT8. gain 양수는 TRP가 좋음. 네 비교의 Bonferroni4는 이 family에만 적용한다.

{markdown(primary[['baseline','TRP_nmae','baseline_nmae','gain_pct','bonferroni4_low','bonferroni4_high','both_seeds_positive']])}

고정 투자 신호 충족: **{v['four_comparisons_meet_predeclared_signal']}**. 신호가 있어도 독립 확인·정식 선행 비교·신규성 검토 전까지 방법론 논문 완성으로 판단하지 않는다. 미충족이면 이 설정에서 새 방법의 근거를 확보하지 못한 것이며 후속 튜닝은 자동 실행하지 않는다.

## 원자료·오류·지속 변화와 반대 결과

아래는 일반 어댑터 PLAIN 대비다. C3/MAG/B0 대조, fixed1024, 모든 변화 형태와 seed 원점수는 RAW_SCORES.csv, SEED_EFFECTS.csv, EFFECTS.csv에 보존한다.

{markdown(compact[['panel','condition','TRP_nmae','baseline_nmae','gain_pct','ci_low','ci_high']])}

## 구성요소의 추가 가치

ANCHOR와의 비교는 보정 크기를 줄이는 단순 정규화 이상의 이득인지, SHUFFLE과의 비교는 같은 perturbation 값들의 시간 연결이 필요한지, IDEAL과의 비교는 B0의 실제 반응을 보존하는 것이 이상적 수준 이동 강제보다 나은지를 검토한다. 이 세 질문은 평균 손실 감소만으로 자동 확인되지 않는다. B0는 잘 학습된 teacher이지만 모든 변화에서 정확하다는 보장은 없으므로 teacher 오류도 보존할 수 있다. bootstrap은 고정된 두 seed·이미 본 데이터에 조건부인 주 단위 불확실성이다.

## 비용

신규 본학습 optimizer 구간 합계 {sum(r['optimizer_seconds'] for r in new)/3600:.3f}시간. 후보/신규 대조군은 update당 adapted forward2회 + teacher forward2회로 PLAIN보다 훈련비가 높다. 동일 updates를 동일 FLOPs라고 쓰지 않는다. 학습 가능8,712개 외에 동결 B0 LoRA294,912개와 backbone을 유지한다. 추론은 일반 C2와 같은 한 번의 forward다. 세부 시간·VRAM은 RESOURCES.csv, 평가 비용은 PREDICTIONS.json이다.

## 범위·신규성·미실행

본 연구의 새 구현은 temporal finite-response loss와 그 통제 비교다. Jacobian matching/교사 반응 증류 원리는 기존 연구에 있다. 아직 Time-PEFT·정식 robust PEFT 대비 우위 또는 새로운 일반 이론을 검증하지 않았다. Electricity·ETTm1 E는 반복 사용된 개발자료이며 다른 전력 계열도 독립 source가 아니다. 독립 새 source, 추가 seed, 새 모델·LR 탐색은 이번 범위 밖이며 실행하지 않았다. 실제 센서 사건 label도 없다. 본 결과로 현장 오류/실제 regime change 해결이나 모든 robust PEFT 불가능을 주장하지 않는다.

코드·데이터·기존 checkpoint hash는 SEAL.json, 수치 검산은 VERIFICATION.json, 프로토콜은 ../../experiments/{NAME}/PROTOCOL.md에 있다. 원본 데이터·checkpoint·예측 cache는 로컬이며 GitHub 파일만으로 전체 수치 재현 가능하다고 쓰지 않는다.
'''
    (OUT/'REPORT.md').write_text(text)
    signal=v['four_comparisons_meet_predeclared_signal']
    decision='DEVELOPMENT_SIGNAL_REQUIRES_INDEPENDENT_CONFIRMATION' if signal else 'NO_METHOD_EVIDENCE_AT_FIXED_PROTOCOL'
    (OUT/'FINAL_DECISION.md').write_text(f'''# 최종 판단\n\n상태: **{decision}**. 논문 PASS 아님.\n\n사전 봉인된 예측 이득·단순 대안·오류/원자료 보호 기준 충족={signal}. 구현·검산이 성공한 것과 방법의 추가 가치가 있는 것은 다르다. 모든 adverse source/state/seed를 보존했다.\n\n남길 구현: 재현 가능한 finite-response loss, 입력 권한/변환 검사, 해시·예산·복원 실행기와 모든 결과. 새 학습·추가 탐색은 자동 연결하지 않는다. {'다음 검토 후보는 TRP 한 개이며 독립 source와 정식 선행 비교가 남았다.' if signal else '이 고정 후보를 자동 재튜닝하지 않으며, 이번 결과에서 다음 투자 후보는 0개다.'}\n''')
