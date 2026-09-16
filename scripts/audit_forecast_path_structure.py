#!/usr/bin/env python3
"""Read-only numerical/resume completion audit; never launches optimization."""
import sys,json,math,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.forecast_path_structure_v1_20260916.common import *
import pandas as pd
import torch

def audit():
 state=read(OUT/'status.json');assert state['status']=='EXECUTION_COMPLETE'
 fits=read(OUT/'fits.json');data=read(OUT/'data_seal.json');selections=read(OUT/'selection_seal.json')['selections'];resume_checks=[]
 for f in fits:
  p=CACHE/'resume'/f"{f['id']}.pt";r=torch.load(p,map_location='cpu',weights_only=False)
  assert r['id']==f['id'] and r['step']==512 and r['epoch']==8 and r['stream_position']==0
  assert r['schedule_sha256']==sha(CACHE/'augmentation_schedule.npz')
  assert r['implementation_sha256']==sha(OUT/'implementation_seal.json')
  assert set(r['rng'])=={'python','numpy','torch','cuda'} and r['rng']['cuda']
  assert len(r['parameters'])==192 and sum(x.numel() for x in r['parameters'].values())==147456
  cp=next(c for c in f['checkpoints'] if c['step']==512);c=torch.load(ROOT/cp['path'],map_location='cpu',weights_only=True)
  assert c.keys()==r['parameters'].keys() and all(torch.equal(c[k],r['parameters'][k]) for k in c)
  opt=r['optimizer'];assert len(opt['state'])==192
  for x in opt['state'].values():
   assert float(x['step'])==512 and torch.isfinite(x['exp_avg']).all() and torch.isfinite(x['exp_avg_sq']).all()
  for g in opt['param_groups']:
   assert g['lr']==f['lr'] and g['betas']==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0
  resume_checks.append(dict(id=f['id'],complete_step=512,parameter_tensors=192,adam_tensors=192,rng_complete=True,checkpoint_exact=True,sha256=sha(p)))
 # Re-estimate TRAIN scale independently from the original hourly load and unique source weather rows.
 scales=read(OUT/'train_scaling.json');scalechecks=[]
 for t in data['targets']:
  tid=t['target'];js=[j for j in data['jobs'] if j['target']==tid and j['role']=='TRAIN'];rows=[];hours=set()
  for j in js:
   rows.append(pd.read_parquet(CACHE/'rows'/f"{j['id']}.parquet").query("part=='PAST'"));o=pd.Timestamp(j['origin']);hours.update(pd.date_range(o-pd.Timedelta(hours=336),periods=360,freq='h'))
  unique=pd.concat(rows).drop_duplicates(['timestamp','available_at']);weather=unique[FEATURES].to_numpy(float);mu=weather.mean(0);sd=weather.std(0)
  # Source columns and the executed Pandas reductions were float32. Reproduce
  # that arithmetic independently; do not impose equality to a new FP64 recipe.
  mu32=np.array([np.mean(unique[c].to_numpy(),dtype=np.float32) for c in FEATURES])
  sd32=np.array([np.sqrt(np.float32(np.var(unique[c].to_numpy(dtype=np.float64),ddof=0))) for c in FEATURES])
  np.testing.assert_array_equal(mu32,np.array(scales[tid]['weather_mean']))
  np.testing.assert_array_equal(sd32,np.array(scales[tid]['weather_std']))
  load=pd.read_parquet(ROOT/t['load_path'],columns=['timestamp','load']).set_index('timestamp')['load'];vals=[math.fsum(float(load.loc[h+pd.Timedelta(minutes=m)]) for m in [0,15,30,45])/4 for h in sorted(hours)];sig=float(np.std(vals))
  assert math.isclose(sig,scales[tid]['sigma_y'],rel_tol=1e-12,abs_tol=1e-12)
  scalechecks.append(dict(target=tid,unique_weather_rows=len(unique),unique_hourly_load_values=len(vals),sigma_y=sig,source_weather_dtype='float32',executed_statistics_bitwise_reproduced=True,alternative_fp64_mean_maxabs=float(np.max(abs(mu-mu32))),alternative_fp64_std_maxabs=float(np.max(abs(sd-sd32)))))
 # Distinguish optimization forwards from diagnostic inference despite legacy phase tags.
 preds=read(OUT/'prediction_manifest.json');total=sum(state[k] for k in ['train_forwards','eval_forwards','verify_forwards'])
 optimization=state['main_updates']+state['smoke_updates'];prediction_calls=sum(r['forwards'] for r in preds)
 diagnostic_expected=len(data['targets'])*(4+4*3)+len(selections)
 assert total==optimization+prediction_calls+diagnostic_expected,(total,optimization,prediction_calls,diagnostic_expected)
 forwards=dict(primitive_total=total,optimizer_forwards=optimization,main_training_forwards=state['main_updates'],smoke_training_forwards=state['smoke_updates'],saved_prediction_forwards=prediction_calls,verification_forwards=diagnostic_expected,raw_phase_tags={k:state[k] for k in ['train_forwards','eval_forwards','verify_forwards']},note='Runtime phase tags retain train phase during some smoke inference; this call-graph reconciliation separates actual gradient updates from verification forwards.')
 # Independently aggregate origin rows, including worst max AFTER case means and seed averaging.
 df=pd.read_csv(OUT/'scores_by_origin.csv');contr=pd.read_csv(OUT/'contrasts.csv');maxerr=0.;contrastchecks=0
 for r in contr.itertuples():
  tids=sorted(df.target.unique()) if r.scope=='ALL' else [t for t in sorted(df.target.unique()) if t!='T0'] if r.scope=='ADDITIONAL' else [r.scope]
  scores=[]
  for arm in [r.method,r.baseline]:
   pertarget=[]
   for tid in tids:
    policy='SELECTED' if arm.startswith('FROZEN') else r.policy
    dd=df[(df.target==tid)&(df.arm==arm)&(df.policy==policy)&(df.variant==r.variant)]
    percase=dd.groupby(['seed','case']).primary.mean().groupby('case').mean()
    sc=percase.mean() if r.case=='MEAN' else percase.max() if r.case=='WORST' else percase.loc[r.case];pertarget.append(sc)
   scores.append(float(np.mean(pertarget)))
  gain=100*(scores[1]-scores[0])/scores[1]
  assert math.isclose(gain,r.gain_percent,rel_tol=1e-10,abs_tol=1e-10);maxerr=max(maxerr,abs(gain-r.gain_percent));contrastchecks+=1
 # Input exclusions and their calendar coverage are descriptive, never a score-based filter.
 om=pd.read_csv(OUT/'origin_manifest.csv');excluded=om[(om.role=='TEST')&(~om.selected)]
 exclusion_records=[dict(target=t,reason=reason,dates=g.date.to_list()) for (t,reason),g in excluded.groupby(['target','exclude_reason'])]
 result=dict(status='PASS',resume_states=resume_checks,train_scale_checks=scalechecks,forwards=forwards,aggregate_contrast_checks=contrastchecks,max_contrast_gain_error=maxerr,input_exclusions=exclusion_records,implementation_file_sha256=sha(Path(__file__)))
 save(OUT/'completion_audit.json',result)
 print(json.dumps({k:result[k] for k in ['status','forwards','aggregate_contrast_checks','max_contrast_gain_error']},indent=2))
if __name__=='__main__':audit()
