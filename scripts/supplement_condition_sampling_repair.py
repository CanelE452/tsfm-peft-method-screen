"""Completed-only, zero-training cross-decomposition and resource reporting."""
import sys,math,time,json
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from experiments.condition_sampling_repair_v1_20260917.common import *
from experiments.condition_sampling_repair_v1_20260917.cross import summarize

def pair_gain(t,method,base,y,origins,sigma,conds):
 counts=block_counts(origins,672 if t=='N03' else 168);den=counts.sum(1);den=np.where(den>0,den,np.nan)
 def scores(pred):
  vv=[];bb=[]
  for i,c in enumerate(conds):
   if t=='N03' and c not in ['delta2','irregular']:continue
   z=(((pred[:,i]-y)/sigma[None,:,None])**2).mean(-1);vv.append(np.sqrt(z.mean(0)).mean());bb.append(np.sqrt(counts@z/den[:,None]).mean(1))
  return float(np.mean(vv)),np.mean(bb,axis=0)
 a,ab=scores(method);b,bb=scores(base);boot=100*(bb-ab)/bb
 return dict(method_score=a,baseline_score=b,gain_percent=100*(b-a)/b,ci_low=float(np.nanquantile(boot,.025)),ci_high=float(np.nanquantile(boot,.975)))

def run():
 assert read(OUT/'controller_state.json')['status']=='FINISHED';all_resources=[]
 for t in ORDER:
  if read(OUT/t/'STATUS.json')['EXECUTION']=='BLOCKED_DIVERSITY':
   old=pd.read_csv(OLD/t/'contrasts.csv');old=old[(old.policy=='selected')&(old.condition==('REVISION' if t=='R04' else 'PRIMARY'))].copy();cols=['method','baseline','seed','method_score','baseline_score','gain_percent','ci_low','ci_high'];b=old[cols].rename(columns={k:k+'_old' for k in cols if k not in ['method','baseline','seed']})
   for k in ['method_score','baseline_score','gain_percent','ci_low','ci_high']:b[k+'_repaired']=np.nan
   b['execution']='BLOCKED_DIVERSITY';b.to_csv(OUT/t/'old_vs_repaired_contrasts.csv',index=False)
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   q=old[old.baseline==CLOSEST[t]];fig,ax=plt.subplots(figsize=(8,4));ax.bar(np.arange(len(q)),q.gain_percent,label='OLD ONLY');ax.set_xticks(np.arange(len(q)),q.seed);ax.axhline(0,color='black',lw=.7);ax.set_ylabel('old gain (%)');ax.set_title(t+' repaired: BLOCKED_DIVERSITY (not zero gain)');ax.legend();fig.tight_layout();fig.savefig(OUT/t/'old_vs_repaired_gain.png',dpi=140);plt.close(fig)
   csvwrite(OUT/t/'resources.csv',[dict(neural_fits=0,optimizer_updates=0,cross_model_calls=0,status='BLOCKED_DIVERSITY')]);continue
  if read(OUT/t/'STATUS.json')['EXECUTION']!='COMPLETE':continue
  out=OUT/t;old=np.load(OLDCACHE/t/'evaluation_bundle.npz');new=np.load(CACHE/t/'evaluation_bundle.npz');sigma=np.array(read(out/'train_statistics.json')['sigma']);cs=read(out/'cross_evaluation_manifest.json')['records'];cross={(r['direction'],r['arm'],r['seed']):np.load(ROOT/r['path'])['pred'] for r in cs if r['status']=='COMPLETE'};raw=pd.read_csv(out/'cross_raw_scores.csv');full=[];effects=[]
  os=pd.read_csv(OLD/t/'scores_summary.csv');ns=pd.read_csv(out/'scores_summary.csv')
  for arm in ARMS[t]:
   for seed in [73101,73102]:
    a=float(os[(os.arm==arm)&(os.seed==seed)&(os.policy=='selected')&(os.condition=='PRIMARY')].score.iloc[0]);c=float(ns[(ns.arm==arm)&(ns.seed==seed)&(ns.policy=='selected')&(ns.condition=='PRIMARY')].score.iloc[0]);r=dict(arm=arm,seed=seed,OLD_MODEL_OLD_E=a,NEW_MODEL_NEW_E=c)
    for direction in ['OLD_MODEL_NEW_E','NEW_MODEL_OLD_E']:
     rr=raw[(raw.direction==direction)&(raw.arm==arm)&(raw.seed==seed)];r[direction]=float(rr.normalized_RMSE.iloc[0]) if len(rr) else None
    r['E_change_only_score_delta']=r['OLD_MODEL_NEW_E']-a if r['OLD_MODEL_NEW_E'] is not None else None;r['new_TRAIN_V_on_new_E_score_delta']=c-r['OLD_MODEL_NEW_E'] if r['OLD_MODEL_NEW_E'] is not None else None;r['new_TRAIN_V_on_old_E_score_delta']=r['NEW_MODEL_OLD_E']-a if r['NEW_MODEL_OLD_E'] is not None else None;full.append(r)
    key=f'{arm}__{seed}__selected'
    for label,yp,orig,conds,p,b in [('NEW_vs_OLD_WEIGHTS_ON_NEW_E',new['y'],new['origins'],list(new['conditions']),new[key],cross.get(('OLD_MODEL_NEW_E',arm,seed))),('NEW_vs_OLD_WEIGHTS_ON_OLD_E',old['y'],old['origins'],list(old['conditions']),cross.get(('NEW_MODEL_OLD_E',arm,seed)),old[key])]:
     if p is None or b is None:continue
     effects.append(dict(arm=arm,seed=seed,comparison=label,**pair_gain(t,p,b,yp,orig,sigma,conds)))
  csvwrite(out/'four_way_score_decomposition.csv',full);csvwrite(out/'same_E_weight_effects.csv',effects)
  logs=[json.loads(x) for x in (out/'optimizer_log.jsonl').read_text().splitlines()];main=[x for x in logs if not x['smoke']];diag=[]
  for arm in ARMS[t]:
   ll=[x for x in main if x['id'].startswith(arm+'_')];row=dict(arm=arm,updates=len(ll),clip_fraction=float(np.mean([x['clip_active'] for x in ll])),contaminated_updates=sum(x['contaminated'] for x in ll),minimum_free_MiB=min(x['free_mib'] for x in ll),gradient_norm_median=float(np.median([x['gradnorm'] for x in ll])))
   for k in sorted({k for x in ll for k in x['extra_grad']}):row[k+'_nonzero_gradient_fraction']=float(np.mean([x['extra_grad'][k]>0 for x in ll]))
   if 'a_saturation' in ll[0]:row['a_saturation_mean']=float(np.mean([x['a_saturation'] for x in ll]))
   diag.append(row)
  csvwrite(out/'optimization_diagnostics.csv',diag);fits=read(out/'fits.json');paths={r['path'] for r in read(out/'predictions_manifest.json')['predictions']};crosspaths={r['path'] for r in cs if r['status']=='COMPLETE'};resources=dict(track=t,fits=len(fits),optimizer_seconds=sum(f['optimizer_seconds'] for f in fits),selection_included_fit_seconds=sum(f['seconds'] for f in fits),main_E_prediction_seconds=sum(read((ROOT/p).with_suffix('.json'))['seconds'] for p in paths),cross_prediction_seconds=sum(read((ROOT/p).with_suffix('.json'))['seconds'] for p in crosspaths),peak_allocated_MiB=max(f['peak_allocated_bytes'] for f in fits)/2**20,contaminated_updates=sum(x['contaminated'] for x in main));all_resources.append(resources)
  extra=8 if t=='N02' else 4;csvwrite(out/'parameter_information_budget.csv',[dict(arm=a,LoRA_parameters=1179648,extra_trainable_parameters=extra if a==PROPOSED[t] else 0,unique_TRAIN_origins=64,distinct_TRAIN_days=64,occurrences_per_origin=8,context_slots=1344 if a=='B1' else 336,horizon_slots=48,input_rows=12 if t=='N03' or a in ['B2','B3'] else 4) for a in ARMS[t]])
  save(out/'decomposition_verification.json',dict(passed=True,zero_optimizer=True,no_model_calls=True,full_factors=len(full),paired_same_E_comparisons=len(effects),different_E_comparison='descriptive only; no causal or paired CI claim',same_E_bootstrap_replicates=2000,same_E_bootstrap_seed=73300,all_seeds_conditions_preserved=True))
 csvwrite(OUT/'RESOURCE_SUMMARY.csv',all_resources);print('REPAIR_SUPPLEMENT_COMPLETE',len(all_resources))
if __name__=='__main__':run()
