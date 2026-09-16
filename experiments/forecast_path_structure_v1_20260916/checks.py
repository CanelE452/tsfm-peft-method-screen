"""Independent CPU audit of intervention, masks, splits and sealed input receipts."""
import math
import pandas as pd
from .common import *
def run_checks():
 s=read(OUT/'data_seal.json');schedule=dict(np.load(CACHE/'augmentation_schedule.npz'));scales=read(OUT/'train_scaling.json');records=[];maxerr=0.
 for t in s['targets']:
  tid=t['target'];jobs=[j for j in s['jobs'] if j['target']==tid];tr=[j for j in jobs if j['role']=='TRAIN'];assert len(tr)==64
  sigma=scales[tid]['sigma_y'];assert sigma>1e-6
  allsets={}
  for role in ['TRAIN','V_SELECT','V_CALIBRATE','TEST']:
   jj=[j for j in jobs if j['role']==role];h=[h for j in jj for h in pd.date_range(j['origin'],periods=24,freq='h')];assert len(h)==len(set(h));allsets[role]=set(h)
  for r in allsets:
   for rr in allsets:
    if r!=rr:assert not allsets[r]&allsets[rr]
  for seed in SEEDS:
   pref=f'{tid}_{seed}';order=schedule[pref+'_order'];offset=schedule[pref+'_offsets'];masks=schedule[pref+'_masks'];count=0
   for e in range(8):np.testing.assert_array_equal(order[e],np.random.default_rng(seed+e).permutation(64))
   for i,j in enumerate(tr):
    x=dict(np.load(ROOT/j['input']));a=x['paths']
    for b in range(2):
     off=offset[b,i];np.testing.assert_array_equal(off,point_offsets(tid,j['origin'],seed,b))
     # Independently enumerate ranks, weather tuples, and pointwise multisets.
     seen=np.zeros((24,4),dtype=int)
     for e in range(4):
      qq=select_path(a,e,'POINT',off)
      for h in range(24):
       k=int((e+off[h])%4);seen[h,k]+=1;np.testing.assert_array_equal(qq[:,h],a[k,:,h])
     np.testing.assert_array_equal(seen,np.ones((24,4),dtype=int));count+=1
    original=payload(x,scales[tid],arm='LATEST')
    for e in range(8):
     mask=masks[e,i];assert np.all((mask==0)|(mask==np.float32(1)/np.float32(.7)))
     dropped=payload(x,scales[tid],arm='DROP',epoch=e,mask=mask)
     np.testing.assert_array_equal(dropped['target'],original['target'])
     for f,m in zip(FEATURES,mask):
      for role in ('past_covariates','future_covariates'):np.testing.assert_array_equal(dropped[role][f],original[role][f]*m)
   records.append(dict(target=tid,seed=seed,four_epoch_multiset_checks=count,shared_order=True,drop_shared_mask_load_unchanged=True))
  for j in tr[:2]:
   y=np.load(ROOT/j['label']);q=y[None]+np.linspace(-sigma,sigma,21)[:,None];v=metric(q,y,sigma)
   total=math.fsum(2*(tau*(float(y[h])-float(q[k,h])) if y[h]>=q[k,h] else (tau-1)*(float(y[h])-float(q[k,h]))) for k,tau in enumerate(TAUS) for h in range(24))/(21*24*sigma)
   err=abs(total-v['primary']);assert math.isclose(total,v['primary'],rel_tol=1e-10,abs_tol=1e-10);maxerr=max(maxerr,err)
   med=np.flatnonzero(TAUS==.5)[0];mae=math.fsum(abs(float(q[med,h])-float(y[h])) for h in range(24))/24
   assert math.isclose(mae,v['MAE'],rel_tol=1e-10,abs_tol=1e-10)
  x=np.zeros((16,4,21,24));y=np.arange(16*24).reshape(16,24);d=calibrate(x,y);assert len(d)==21 and np.all(np.diff(apply_cal(x,d),axis=-2)>=0)
 for rr in read(OUT/'selected_weather_rows.json'):
  assert sha(ROOT/rr['path'])==rr['sha256'];rows=pd.read_parquet(ROOT/rr['path']);j=next(j for j in s['jobs'] if j['id']==rr['id']);o=pd.Timestamp(j['origin']);assert (rows.available_at<=o).all();assert (rows.query("part=='PAST'").timestamp<o).all()
 save(OUT/'cpu_checks.json',dict(status='PASS',interventions=records,scalar_maxabs=maxerr,selected_weather_files_checked=len(read(OUT/'selected_weather_rows.json')),label_intervals_disjoint=True,calibration_monotone=True,test_label_files_absent=all('label' not in j for j in s['jobs'] if j['role']=='TEST')))
 print('CPU_CHECKS_PASS',len(records),'target/seed pairs',flush=True)
