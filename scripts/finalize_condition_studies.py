"""Post-evaluation reporting/verification only. Never trains or selects a new model."""
import sys,json,time,math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from experiments.condition_studies_v1_20260916.common import *
from experiments.condition_studies_v1_20260916.report import report_track,report_all,table

def classify(t):
 st=read(OUT/t/'STATUS.json')
 if st['EXECUTION']!='COMPLETE':return
 df=pd.read_csv(OUT/t/'contrasts.csv');reason=[]
 if t in SOURCES:
  d=df[(df.policy=='selected')&(df.seed.astype(str)=='MEAN')&(df.condition=='PRIMARY')];sel=read(OUT/t/'selections.json')['selections'];prop=[s for s in sel if s['arm']==PROPOSED[t]]
  if prop and all(s['selected']['step']==0 for s in prop):e='NO_SELECTED_ADAPTATION';reason.append('제안군의 두 반복 모두 INIT 선택; 학습 적응의 추가 가치가 선택되지 않음')
  elif len(d)==len(CONTRASTS[t]) and (d.ci_low>0).all():e='PROMISING_WITHIN_SCOPE';reason.append('모든 사전 핵심 대비의 개발 평가 gain CI가 양수')
  elif len(d) and (d.ci_high<0).any():e='NEGATIVE_WITHIN_SCOPE';reason.append('적어도 하나의 필수 직접 대조에서 gain CI 전체가 음수')
  else:e='POSITIVE_UNCERTAIN';reason.append('추가 가치의 부호/크기/조건부 CI가 일관된 우위를 확정하지 못함; POSITIVE_UNCERTAIN 태그는 성능 성공이 아님')
  if t=='R04':
   f=pd.read_csv(OUT/t/'accuracy_revision_frontier.csv');p=f[(f.arm=='D3')&(f.policy=='selected')]
   rev=df[(df.policy=='selected')&(df.seed.astype(str)=='MEAN')&(df.condition=='REVISION')]
   reason=['R04의 주목적은 정확도 개선 자체가 아닌 정확도1% 보호 아래 원 발행 예측 수정량 감소']
   if prop and all(s['selected']['step']==0 for s in prop):e='NO_SELECTED_ADAPTATION';reason.append('두 반복 모두 제안군 INIT 선택')
   elif not p.E_accuracy_protected.all():e='NEGATIVE_WITHIN_SCOPE';reason.append('제안군 반복 중 E 정확도 보호1%를 충족하지 않은 경우가 있음')
   elif len(rev)==len(CONTRASTS[t]) and (rev.ci_low>0).all():e='PROMISING_WITHIN_SCOPE';reason.append('정확도 보호 조건 안에서 필수 직접 대조 대비 수정량 CI 개선')
   elif len(rev) and (rev.ci_high<0).any():e='NEGATIVE_WITHIN_SCOPE';reason.append('정확도는 보호했으나 필수 직접 대조보다 수정량 악화')
   else:e='POSITIVE_UNCERTAIN';reason.append('정확도 보호와 수정량 추가 가치의 직접 대비를 함께 읽어야 함')
 elif t=='R05':
  d=df[(df.scope=='ALL')&(df.variant=='RAW')&(df.condition=='DELAYED')];sel=read(OUT/t/'selections.json')['policies'];same=all(sel[k]['E4']==sel[k]['E3'] for k in sel)
  if same:e='SIMPLE_METHOD_SUFFICIENT';reason.append('MONOTONE 정책이 세 타깃 모두 고정 BINARY와 동일하여 복잡한 정책 추가가치 없음')
  elif len(d) and (d[d.baseline.isin(['E2','E3'])].ci_low>0).all():e='PROMISING_WITHIN_SCOPE';reason.append('단순 정책 대비 지연 손실 우위; 최신 보호는 별도 점검')
  else:e='POSITIVE_UNCERTAIN';reason.append('최신/지연 정책의 개발 손익이며 새 PEFT 근거는 아님')
 else:
  d=df[(df.scope=='ALL')&(df.metric=='sum_CRPS_normalized')&(df.variant=='RAW')&(df.baseline=='F1')]
  ref=df[(df.scope=='ALL')&(df.metric=='sum_CRPS_normalized')&(df.variant=='RAW')&(df.method=='F1')&(df.baseline=='F0')]
  if len(d)==2 and (d.ci_high<0).all() and len(ref)==1 and (ref.ci_low>0).all():e='SIMPLE_METHOD_SUFFICIENT';reason.append('단순 AR1은 독립 결합보다 개선했고 복잡한 두 결합은 AR1보다 악화; 관찰된 주목적에는 단순 대안이 충분')
  else:e='PROMISING_WITHIN_SCOPE' if len(d) and (d.ci_low>0).any() else 'POSITIVE_UNCERTAIN'
  reason.append('동일 주변분포의 알려진 coupling 비교. 개선되어도 새 PEFT 설계/신규성은 미확보')
 st['EVIDENCE']=e;st['interpretation']=reason;save(OUT/t/'STATUS.json',st)

def parameter_table(t):
 env=read(OUT/'environment_receipt.json');spec=read(OUT/t/'PROTOCOL.json');aux=read(OUT/t/'feature_or_transform_manifest.json');extra={'A2':36,'A3':40,'B3':8,'C3':4,'H1':4,'H2':8,'H3':8};rows=[]
 for arm in ARMS[t]:
  group=12 if t in ['N01','N03'] or arm in ['B2','B3'] else 5 if t=='R09' else 4;cpu=aux.get('regression_coefficients',aux.get('cpu_regression_scalar_coefficients',0));stored_cpu=cpu;cpu=0 if arm in ['A0','H0','G0','G1'] else 1000 if arm in ['G2','G3'] else cpu;rows.append(dict(arm=arm,LoRA_parameters=1179648,extra_parameters=extra.get(arm,0),total_trainable=1179648+extra.get(arm,0),frozen_parameters=env['frozen_backbone_parameters'],input_rows=group,context=1344 if arm=='B1' else 336,horizon=spec['H'],native_calls_per_update=2 if t=='R04' else 1,CPU_fitted_scalar_entries_used=cpu,shared_prepared_CPU_scalar_storage=stored_cpu,CPU_scalar_entries_in_forward=0 if t=='N07' else cpu,unique_training_origins=16 if arm=='I0' else 64,occurrences_per_unique_origin=32 if arm=='I0' else 8))
 csvwrite(OUT/t/'parameter_information_budget.csv',rows)

def cpu_artifacts(t):
 from experiments.condition_studies_v1_20260916.cached import Prior,TAU
 prior=Prior();origins=[];role='V_SELECT' if t=='R05' else 'V_CALIBRATE'
 for j in prior.jobs.values():
  if j['role'] in [role,'TEST'] and (j['role']!='TEST' or j['id'] in prior.labels):origins.append(dict(target=j['target'],role='E_REUSED' if j['role']=='TEST' else role,id=j['id'],origin=j['origin']))
 csvwrite(OUT/t/'origins.csv',origins);csvwrite(OUT/t/'optimizer_log.csv',[])
 if t=='R05':
  p=read(OUT/t/'selections.json')['policies'];fits=[dict(target=target,arm=arm,type='CPU_policy_selection',neural_fits=0,optimizer_updates=0,configurations=5 if arm=='E2' else 70,selected_alpha=json.dumps(p[target][arm])) for target in p for arm in ['E2','E4']];csvwrite(OUT/t/'fit_manifest.csv',fits)
  df=pd.read_csv(OUT/t/'scores_by_origin.csv');scores=df[(df.variant=='RAW')&(df.condition.isin(['S0','DELAYED']))].groupby(['target','seed','arm','condition']).value.mean();rows=[]
  for target in p:
   for seed in [61730,61731]:
    base=scores[(target,seed,'E0','S0')]
    for arm in ARMS[t]:rows.append(dict(target=target,seed=seed,arm=arm,S0=scores[(target,seed,arm,'S0')],latest_harm_percent=100*(scores[(target,seed,arm,'S0')]/base-1),E_latest_protected=scores[(target,seed,arm,'S0')]<=1.01*base,delayed=scores[(target,seed,arm,'DELAYED')],alpha=json.dumps(p[target][arm]),models_per_case=json.dumps([1 if v in [0,1] else 2 for v in p[target][arm]])))
  csvwrite(OUT/t/'latest_protection_and_cost.csv',rows)
 else:
  pp=read(OUT/t/'selections.json')['parameters'];csvwrite(OUT/t/'fit_manifest.csv',[dict(target=target,arm=arm,type='CPU_dependence_estimation',neural_fits=0,optimizer_updates=0,CAL_origins=16,coefficients=1 if arm=='F1' else 276 if arm=='F2' else 384) for target in pp for arm in ['F1','F2','F3']]);m=read(OUT/t/'marginal_verification.json');check=[];memo={}
  for r in m:
   if r['target'] not in memo:memo[r['target']]=prior.get(r['target'],'TEST','FROZEN_WEATHER')
   q,y,ids=memo[r['target']];i=ids.index(r['id']);err=y[i][None]-q[i,0];pb=(2*np.maximum(TAU[:,None]*err,(TAU[:,None]-1)*err)).mean(0)
   for h in range(24):check.append(dict(target=r['target'],id=r['id'],arm=r['arm'],lead=h,original_2pinball=float(pb[h]),ensemble_CRPS=r['lead_CRPS'][h]))
  df=pd.DataFrame(check)
  for _,g in df.groupby(['target','id','lead']):assert g.original_2pinball.max()==g.original_2pinball.min();assert np.allclose(g.ensemble_CRPS,g.ensemble_CRPS.iloc[0],rtol=1e-10,atol=1e-10)
  csvwrite(OUT/t/'marginal_pinball_CRPS_identity.csv',check)

def run():
 for t in ORDER:
  st=read(OUT/t/'STATUS.json')
  if st['EXECUTION']!='COMPLETE':continue
  classify(t)
  if t in SOURCES:
   parameter_table(t)
   journal=[json.loads(line) for line in (OUT/t/'optimizer_log.jsonl').read_text().splitlines() if line.strip()]
   if not (OUT/t/'optimizer_log.csv').exists():csvwrite(OUT/t/'optimizer_log.csv',journal)
  else:cpu_artifacts(t)
  report_track(t)
  if t in ['R04','R05']:
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   fig,ax=plt.subplots(figsize=(9,5))
   if t=='R04':
    f=pd.read_csv(OUT/t/'accuracy_revision_frontier.csv');f=f[f.policy=='selected']
    for (x,y),g in f.groupby(['normalized_RMSE','raw_revision_RMS']):
     ax.scatter(x,y);ax.annotate(','.join(sorted(g.arm.unique()))+'\nseeds '+','.join(map(str,sorted(g.seed.unique()))),(x,y),xytext=(5,-25),textcoords='offset points',fontsize=7)
    ax.set_xlabel('Accuracy: normalized RMSE');ax.set_ylabel('Revision RMS / TRAIN sigma')
   else:
    f=pd.read_csv(OUT/t/'latest_protection_and_cost.csv')
    for r in f.itertuples():ax.scatter(r.latest_harm_percent,r.delayed);ax.annotate(f'{r.target}/{r.arm}/{r.seed}',(r.latest_harm_percent,r.delayed),fontsize=5)
    ax.axvline(1,color='black',linestyle='--');ax.set_xlabel('S0 harm vs LATEST (%)');ax.set_ylabel('Mean S1..S3 normalized pinball')
   ax.set_title(t+' accuracy / robustness trade-offs');fig.tight_layout();fig.savefig(OUT/t/'condition_tradeoffs.png',dpi=140);plt.close(fig)
  path=OUT/t/'REPORT.md';s=path.read_text()
  if t!='R09':s=s.replace(' R09 숨긴 상세값의 권한은 절대시간 블록에 적용한다.','')
  if t=='R05':
   raw=pd.read_csv(OUT/t/'raw_scores.csv');raw=raw[(raw.variant=='RAW')&(raw.metric=='normalized_2pinball')]
   wide=raw.pivot(index=['target','seed','arm'],columns='condition',values='score').reset_index()
   start=s.index('## ⑤');end=s.index('![직접대비]',start)
   s=s[:start]+'## ⑤ 원점수·효과·seed·조건 손익\n\n'+table(wide,['target','seed','arm','S0','S1','S2','S3','DELAYED'],40)+'\n\n모든 타깃·seed·조건을 포함했다. [원단위·보정 민감도 포함 원점수](raw_scores.csv), [직접 대비와 CI](contrasts.csv), [최악 지연 조건 대비](worst_delayed_contrasts.csv).\n\n'+s[end:]
  if (OUT/t/'INTERPRETATION.md').exists():s+='\n## 실제 결과 해석\n\n'+(OUT/t/'INTERPRETATION.md').read_text().split('\n',1)[-1].lstrip()+'\n'
  s+='\n## 판정 해석과 추가 감사\n\n'+' '.join(read(OUT/t/'STATUS.json').get('interpretation',[]))+'\n\n'
  if t in SOURCES:
   s+='[정보·파라미터·노출 횟수](parameter_information_budget.csv). 보조계수가 있는 군을 완전히 동일 파라미터 예산이라고 하지 않는다. CPU fitted scalar entries는 독립 자유도나 신경망 trainable 수가 아니다.\n\n';b=read(OUT/t/'bootstrap_manifest.json');s+=f'평가 원점이 걸친 관측 주간 블록은 {b["observed_blocks"]}개다. bootstrap 2,000회 중 {b["defined"]}회가 계산 가능하고 {b["empty_resamples"]}회는 관측 없는 재표집으로 보존했다. CI는 계산 가능한 재표집에 조건부다. 중복 horizon의64원점을64개의 독립 기간으로 해석하지 않는다.\n'
  if t=='N07':s+='\n공유 CPU 배열1,920개에는 음의 주파수 mirror가 포함되고 양의 주파수·DC/Nyquist의 원래 저장 회귀계수는1,000개다. G0/G1은 그 ridge 정보를 loss에 사용하지 않고 G2/G3도 추론 때는 loss 통계를 사용하지 않는다.\n'
  if t=='R05':s+='[최신 보호·실제 모델 호출 비용](latest_protection_and_cost.csv), [CPU 선택 장부](fit_manifest.csv).\n'
  if t in SOURCES:s+='\n[실제 clipping·보조계수 gradient·GPU 오염 기록](optimization_diagnostics.csv), [저장된 실제 예측 루프 비용](prediction_costs.csv). 예측 시간은 guard/Python 비용을 포함하며 같은 prediction 경로를 여러 정책에서 참조하면 중복 합산하지 않는다. CPU 검산·보고를 병행했으므로 작은 시간 차이를 격리된 속도 우위라고 해석하지 않는다.\n'
  if t=='R04':s+='\n[원점별 정확도·수정량·혁신량](scores_by_origin.csv), [원점 점수 재집계 검산](origin_score_verification.json). frontier의 raw_revision은 adapter 보정량이 아닌 발행 예측 차이이며 TRAIN sigma로 정규화한 RMS다.\n'
  if t=='R09':s+='\n[허용 context 해상도 상태별 결과](context_state_scores.csv), [전체 원점 상태](evaluation_context_states.csv). 모든 상태와 원점을 유지한다.\n'
  if t=='N06':s+='[모든 타깃·원점·lead의 원래 pinball/ensemble CRPS 동일성](marginal_pinball_CRPS_identity.csv), [CPU fitting 장부](fit_manifest.csv). 원래 quantile의 .01/.99 밖은 공통 clamp라 극단 꼬리 정확도는 확인하지 않았다.\n'
  path.write_text(s)
 report_all();print('POSTPROCESS_COMPLETE_ONLY')
if __name__=='__main__':run()
