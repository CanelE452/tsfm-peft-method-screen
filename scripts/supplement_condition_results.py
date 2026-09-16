"""Complete-only descriptive summaries from immutable journals/predictions. No training."""
import sys,json,math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.condition_studies_v1_20260916.common import *

def execution(t):
 rows=[json.loads(x) for x in (OUT/t/'optimizer_log.jsonl').read_text().splitlines()];main=[r for r in rows if not r['smoke']];out=[]
 for arm in ARMS[t]:
  rr=[r for r in main if r['id'].split('_')[0]==arm]
  if not rr:continue
  extras=sorted({k for r in rr for k in r['extra_grad']});v=dict(arm=arm,updates=len(rr),clip_fraction=np.mean([r['clip_active'] for r in rr]),gradnorm_median=np.median([r['gradnorm'] for r in rr]),gradnorm_max=max(r['gradnorm'] for r in rr),contaminated_updates=sum(r['contaminated'] for r in rr),minimum_boundary_free_MiB=min(r['free_mib'] for r in rr),mean_update_seconds=np.mean([r['seconds'] for r in rr]))
  for k in extras:v['extra_'+k+'_nonzero_fraction']=np.mean([r['extra_grad'][k]>0 for r in rr]);v['extra_'+k+'_mean_norm']=np.mean([r['extra_grad'][k] for r in rr])
  if 'a_saturation' in rr[0]:v['mean_a_saturation']=np.mean([r['a_saturation'] for r in rr]);v['max_a_saturation']=max(r['a_saturation'] for r in rr)
  parts=sorted({k for r in rr for k in r['loss_parts']})
  for k in parts:v['mean_'+k]=np.mean([r['loss_parts'][k] for r in rr if k in r['loss_parts']])
  out.append(v)
 csvwrite(OUT/t/'optimization_diagnostics.csv',out)
 pred=[]
 for r in read(OUT/t/'predictions_manifest.json')['predictions']:
  p=ROOT/r['path'];meta=read(p.with_suffix('.json'));z=np.load(p);pred.append(dict(arm=r['arm'],seed=r['seed'],policy=r['policy'],path=r['path'],seconds=meta['seconds'],origins=len(z['origins']),conditions=len(z['conditions']),seconds_per_origin=meta['seconds']/len(z['origins']),timing_scope='actual prediction loop including guard/Python; shared path timed once, never sum alias timings'))
 csvwrite(OUT/t/'prediction_costs.csv',pred)

def r04_origins():
 t='R04';b=np.load(CACHE/t/'evaluation_bundle.npz');stats=read(OUT/t/'train_statistics.json');sigma=np.array(stats['sigma']);inp=np.load(CACHE/t/'E_DISCOVERY_inputs.npz');fr=np.load(CACHE/t/'frozen.npz');v=(((inp['innovation_observed']-fr['E_DISCOVERY'][:,:,:24])/sigma[None,:,None])**2).mean(-1);cut=read(OUT/t/'frozen_parameters.json')['innovation_upper_quartile'];rows=[]
 for key in b.files:
  if key in ['y','origins','conditions']:continue
  arm,seed,policy=key.split('__');p=b[key][:,0];y=np.stack([b['y'][:,:,:48],b['y'][:,:,24:72]],1)
  for i,o in enumerate(b['origins']):
   for c in range(4):
    e=p[i,:,c]-y[i,:,c];d=p[i,1,c,:24]-p[i,0,c,24:];rows.append(dict(arm=arm,seed=int(seed),policy=policy,origin=int(o),channel=c,normalized_MSE=float(np.mean(e*e)/sigma[c]**2),raw_MAE=float(np.mean(abs(e))),revision_normalized_MSE=float(np.mean(d*d)/sigma[c]**2),revision_raw_MSE=float(np.mean(d*d)),innovation_MSE=float(v[i,c]),high_innovation=bool(v[i,c]>=cut)))
 csvwrite(OUT/t/'scores_by_origin.csv',rows)
 df=pd.DataFrame(rows);summary=pd.read_csv(OUT/t/'scores_summary.csv')
 for (a,s,p),g in df.groupby(['arm','seed','policy']):
  expected=summary[(summary.arm==a)&(summary.seed==s)&(summary.policy==p)].iloc[0];assert math.isclose(g.groupby('channel').normalized_MSE.mean().pow(.5).mean(),expected.score,rel_tol=1e-10,abs_tol=1e-10);assert math.isclose(g.groupby('channel').revision_normalized_MSE.mean().pow(.5).mean(),expected.revision,rel_tol=1e-10,abs_tol=1e-10)
 save(OUT/t/'origin_score_verification.json',dict(passed=True,rows=len(df),new_model_calls=0,rtol=1e-10,atol=1e-10,revision_unit='TRAIN sigma normalized RMS of raw issued forecasts; raw means not adapter correction'))
 frontier=pd.read_csv(OUT/t/'accuracy_revision_frontier.csv');ref=frontier[(frontier.arm=='D0')&(frontier.policy=='selected')].set_index('seed').normalized_RMSE
 frontier['selected_D0_reference']=frontier.seed.map(ref);frontier['accuracy_harm_vs_selected_D0_percent']=100*(frontier.normalized_RMSE/frontier.selected_D0_reference-1);frontier['protected_vs_selected_D0']=frontier.normalized_RMSE<=1.01*frontier.selected_D0_reference;frontier.to_csv(OUT/t/'accuracy_protection_references.csv',index=False)

def r09_context():
 t='R09';b=np.load(CACHE/t/'evaluation_bundle.npz');inp=np.load(CACHE/t/'E_DISCOVERY_inputs.npz');sigma=np.array(read(OUT/t/'train_statistics.json')['sigma']);counts=inp['resolution'].reshape(64,14,24)[:,:,0].sum(1).astype(int);rows=[]
 csvwrite(OUT/t/'evaluation_context_states.csv',[dict(origin=int(o),detailed_blocks=int(c),aggregate_only_blocks=14-int(c),all_context_blocks=14) for o,c in zip(b['origins'],counts)])
 for key in b.files:
  if key in ['y','origins','conditions']:continue
  arm,seed,policy=key.split('__');p=b[key][:,0]
  for count in sorted(set(counts)):
   m=counts==count;err=(p[m]-b['y'][m])/sigma[None,:,None];dm=err.mean(-1);pattern=err-err.mean(-1,keepdims=True);rows.append(dict(arm=arm,seed=int(seed),policy=policy,detailed_context_blocks=int(count),origins=int(m.sum()),normalized_RMSE=float(np.sqrt((err**2).mean((0,2))).mean()),daily_mean_RMSE_normalized=float(np.sqrt((dm**2).mean(0)).mean()),pattern_RMSE_normalized=float(np.sqrt((pattern**2).mean((0,2))).mean())))
 csvwrite(OUT/t/'context_state_scores.csv',rows);save(OUT/t/'context_state_verification.json',dict(passed=True,origins=len(counts),counts_sum=int(sum((counts==k).sum() for k in set(counts))),rule='every distinct lawful number of detailed 24h context blocks, no score-dependent threshold',new_model_calls=0,all_origins_included=True))

def r05_seed_plot():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 df=pd.read_csv(OUT/'R05/raw_scores.csv');df=df[(df.variant=='RAW')&(df.metric=='normalized_2pinball')];p=df.pivot(index=['target','seed','condition'],columns='arm',values='score');fig,axs=plt.subplots(1,3,figsize=(15,14),sharey=True);labels=[f'{t}/{s}/{c}' for t,s,c in p.index]
 for ax,base in zip(axs,['E0','E2','E3']):
  gain=100*(p[base]-p.E4)/p[base];ax.barh(range(len(gain)),gain,color=np.where(gain>=0,'#26786b','#b95158'));ax.axvline(0,color='black',lw=.7);ax.set_yticks(range(len(gain)),labels,fontsize=6);ax.set_title('E4 vs '+base);ax.set_xlabel('Paired gain (%)')
 fig.suptitle('R05 all targets / repeat seeds / prespecified conditions');fig.tight_layout();fig.savefig(OUT/'R05/paired_gains.png',dpi=140);plt.close(fig)

def condition_plot(t):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 if t in ['N01','N03']:
  d=pd.read_csv(OUT/t/'scores_summary.csv');d=d[(d.policy=='selected')&(d.condition!='PRIMARY')]
  if t=='N01':d=d.assign(condition=d.condition.str.replace(r'_c[0-3]$','',regex=True)).groupby(['arm','seed','condition'],as_index=False).score.mean()
  metrics=list(d.condition.unique());fig,ax=plt.subplots(figsize=(10,5));arms=list(d.arm.unique());colors=plt.get_cmap('tab10')
  for ai,a in enumerate(arms):
   for seed,sty in [(73101,'-o'),(73102,'--x')]:
    g=d[(d.arm==a)&(d.seed==seed)].set_index('condition');ax.plot(range(len(metrics)),g.loc[metrics,'score'],sty,color=colors(ai),label=f'{a}/{seed}')
  ax.set_xticks(range(len(metrics)),metrics);ax.set_ylabel('Normalized RMSE');ax.legend(fontsize=7,ncol=2);ax.set_title(t+' all input conditions and repeat seeds')
 elif t in ['N07','R08','R09']:
  d=pd.read_csv(OUT/t/'secondary_scores.csv');d=d[d.policy=='selected'];metrics=list(d.metric.unique());fig,axs=plt.subplots(1,len(metrics),figsize=(5*len(metrics),5),squeeze=False)
  for ax,metric in zip(axs[0],metrics):
   g=d[d.metric==metric];arms=list(g.arm.unique())
   for seed in [73101,73102]:
    q=g[g.seed==seed];ax.scatter([arms.index(a) for a in q.arm],q.value,label=str(seed))
   ax.set_xticks(range(len(arms)),arms,rotation=35);ax.set_title(metric,fontsize=9);ax.legend(fontsize=7)
 else:return
 fig.tight_layout();fig.savefig(OUT/t/'condition_tradeoffs.png',dpi=140);plt.close(fig)

def run():
 for t in ORDER:
  if read(OUT/t/'STATUS.json')['EXECUTION']!='COMPLETE':continue
  if t in SOURCES:execution(t);condition_plot(t)
  if t=='R04':r04_origins()
  if t=='R09':r09_context()
  if t=='R05':r05_seed_plot()
 print('SAVED_RESULTS_SUPPLEMENTED_NO_NEW_FITS')
if __name__=='__main__':run()
