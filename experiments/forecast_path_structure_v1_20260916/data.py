"""Pinned receipt, TRAIN-only target selection, label-separated as-of construction."""
import subprocess, unicodedata, shutil
from urllib.parse import quote
import pandas as pd
import yaml, requests
from .common import *
OLD=ROOT/'.cache/covariate_availability_audit_20260916'
SPLITS={'TRAIN':('2024-03-01','2024-06-29',64,'2024-06-30'), 'V_SELECT':('2024-07-01','2024-07-30',16,'2024-07-31'), 'V_CALIBRATE':('2024-08-01','2024-08-30',16,'2024-08-31'), 'TEST':('2024-10-01','2024-12-30',None,'2024-12-31')}
class InputsInvalid(ValueError):pass

def frames(lp,wp,train_only=False):
 filt=[('timestamp','<',pd.Timestamp('2024-06-30',tz='UTC'))] if train_only else None
 load=pd.read_parquet(lp,filters=filt,columns=['timestamp','available_at','load'])
 weather=pd.read_parquet(wp,filters=filt,columns=['timestamp','available_at']+FEATURES)
 for f in (load,weather):
  for c in ('timestamp','available_at'):f[c]=pd.to_datetime(f[c],utc=True)
 assert not load.duplicated('timestamp').any(),'DUPLICATE_LOAD_TIMESTAMP'
 assert not weather.duplicated(['timestamp','available_at']).any(),'DUPLICATE_WEATHER_PAIR'
 return load,weather

def build_inputs(load,weather,o):
 # Label values at/after o never enter this function's selected load frame.
 past=pd.date_range(o-pd.Timedelta(hours=336),periods=1344,freq='15min');future=pd.date_range(o,periods=96,freq='15min')
 lc=load[(load.timestamp>=past[0])&(load.timestamp<o)&(load.available_at<=o)].set_index('timestamp').reindex(past)
 w=weather[(weather.timestamp>=past[0])&(weather.timestamp<=future[-1])&(weather.available_at<=o)].sort_values(['timestamp','available_at'],ascending=[True,False]).copy()
 w['rank']=w.groupby('timestamp').cumcount()
 hist=w[(w['rank']==0)&(w.timestamp<o)].set_index('timestamp').reindex(past)
 fs=[w[(w['rank']==k)&(w.timestamp>=o)].set_index('timestamp').reindex(future) for k in range(4)]
 if not np.isfinite(lc['load'].to_numpy(float)).all():raise InputsInvalid('LOAD_CONTEXT_MISSING_OR_UNAVAILABLE')
 if not all(np.isfinite(f[FEATURES].to_numpy(float)).all() for f in [hist]+fs):raise InputsInvalid('WEATHER_INPUT_MISSING_OR_UNAVAILABLE')
 rows=pd.concat([hist.reset_index(names='timestamp').assign(part='PAST',k=0)]+[f.reset_index(names='timestamp').assign(part='FUTURE',k=k) for k,f in enumerate(fs)],ignore_index=True)
 assert (rows.available_at<=o).all() and (lc.available_at<=o).all() and (lc.index<o).all()
 x=dict(context=lc['load'].to_numpy(float).reshape(336,4).mean(1).astype('float32'),past=hist[FEATURES].to_numpy(float).reshape(336,4,3).mean(1).T.astype('float32'),paths=np.stack([f[FEATURES].to_numpy(float).reshape(24,4,3).mean(1).T for f in fs]).astype('float32'))
 return x,rows

def label(load,o):
 idx=pd.date_range(o,periods=96,freq='15min');v=load.set_index('timestamp').reindex(idx)['load'].to_numpy(float)
 if not np.isfinite(v).all():raise InputsInvalid('TARGET_MISSING')
 return v.reshape(24,4).mean(1)

def origins(role):
 a,b,_,end=SPLITS[role]
 return [o+pd.Timedelta(hours=8) for o in pd.date_range(a,b,tz='UTC') if o+pd.Timedelta(hours=32)<=pd.Timestamp(end,tz='UTC')]

def prepare():
 assert not (OUT/'data_seal.json').exists(),'Already prepared; no duplicate preparation'
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 oldfiles=subprocess.check_output(['git','ls-files','results','research'],cwd=ROOT,text=True).splitlines()
 save(OUT/'historical_hashes.json',{p:sha(ROOT/p) for p in oldfiles})
 save(OUT/'repo_audit.json',dict(head=head,base='a5cbff35572d01cfb5d5204f71ae571eab61efc1',branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),base_diff=subprocess.check_output(['git','diff','--stat','a5cbff3..HEAD'],cwd=ROOT,text=True),processes=subprocess.check_output(['ps','-eo','pid,etimes,args'],text=True),disk_free_bytes=shutil.disk_usage(ROOT).free,reused_fits=0,reason='New TRAIN64, three targets, two LRs/two seeds, 512 updates and POINT control; old 120-update paths incompatible',protocol_sha256=sha(OUT/'PROTOCOL.md')))
 assert shutil.disk_usage(ROOT).free>10*2**30,'DISK_BELOW_10GiB'
 previous={r['path']:r for r in read(ROOT/'research/covariate_availability_audit_20260916/download_manifest.json')}
 receipt=[];remote={};downloaded=0
 def acquire(path):
  nonlocal downloaded,remote
  old=OLD/path;dest=CACHE/'source'/path
  if path in previous and old.exists():
   assert sha(old)==previous[path]['sha256'];p=old;h=sha(p);mode='EXISTING_HASH_VERIFIED'
  else:
   if not remote:
    url=f'https://huggingface.co/api/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark/tree/{REV}?recursive=true&limit=1000'
    r=requests.get(url,timeout=60);r.raise_for_status();remote={x['path']:x for x in r.json()};save(OUT/'remote_file_metadata.json',remote)
   meta=remote[path];size=meta['size'];assert downloaded+size<=2*2**30,'DOWNLOAD_CAP'
   dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():
    url=f'https://huggingface.co/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark/resolve/{REV}/{quote(path)}'
    r=requests.get(url,stream=True,timeout=120);r.raise_for_status();tmp=dest.with_suffix('.partial')
    with open(tmp,'wb') as f:
     for b in r.iter_content(1048576):
      downloaded+=len(b);assert downloaded<=2*2**30;f.write(b)
    tmp.replace(dest)
   p=dest;h=sha(p);assert h==meta['lfs']['oid'];mode='PINNED_LFS_VERIFIED'
  receipt.append(dict(path=path,local=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=h,mode=mode))
  save(OUT/'data_receipt.json',dict(revision=REV,files=receipt,new_download_bytes=downloaded,limit_bytes=2*2**30,information_contract='SIMULATED_ASOF',license='CC-BY-4.0 with source-specific Liander terms; see pinned README'))
  return p
 card=acquire('README.md');meta_path=acquire('liander2024_targets.yaml')
 metadata=yaml.safe_load(meta_path.read_text());cand=[]
 for m in metadata:
  if m['group_name']=='mv_feeder':
   # Only identity/location metadata, never upper/lower_limit or held-out values.
   ident={k:m[k] for k in ['group_name','name','latitude','longitude','description'] if k in m}
   canon='\x1f'.join(unicodedata.normalize('NFC',str(ident[k])) for k in ('group_name','name'))
   ident['canonical']=canon;ident['canonical_sha256']=hashlib.sha256(canon.encode()).hexdigest();cand.append(ident)
 cand.sort(key=lambda m:m['canonical_sha256']);save(OUT/'candidate_order.json',cand)
 selected=[];tmanifest=[];seen_hash=set();seen_physical=set();traincache={}
 candidates=[next(m for m in cand if m['name']=='OS Gorredijk')]+[m for m in cand if m['name']!='OS Gorredijk'][:8]
 for rank,m in enumerate(candidates):
  if len(selected)==3:break
  print('CANDIDATE_TRAIN_ONLY',rank,m['name'],flush=True)
  lp=acquire('load_measurements/mv_feeder/'+m['name']+'.parquet');wp=acquire('weather_forecasts_versioned/mv_feeder/'+m['name']+'.parquet')
  rec=dict(candidate_order=rank,name=m['name'],group=m['group_name'],canonical_sha256=m['canonical_sha256'],load_path=str(lp.relative_to(ROOT)),weather_path=str(wp.relative_to(ROOT)),load_sha256=sha(lp),weather_sha256=sha(wp))
  try:
   physical=(m.get('latitude'),m.get('longitude'),m.get('description'))
   assert rec['load_sha256'] not in seen_hash,'DUPLICATE_LOAD_FILE'
   assert physical not in seen_physical,'DUPLICATE_METADATA_IDENTITY'
   load,weather=frames(lp,wp,train_only=True);valid=[]
   for o in origins('TRAIN'):
    try:x,rows=build_inputs(load,weather,o);y=label(load,o);valid.append((o,x,rows,y))
    except InputsInvalid:continue
   assert len(valid)>=64,'TRAIN_BELOW_64'
   idx=np.floor(np.linspace(0,len(valid)-1,64)).astype(int);assert len(np.unique(idx))==64
   chosen=[valid[i] for i in idx]
   dedup=pd.concat([z[2].query("part == 'PAST'") for z in chosen]).drop_duplicates(['timestamp','available_at'])
   sd=dedup[FEATURES].std(ddof=0).to_numpy();assert (sd>1e-6).all(),'CONSTANT_TRAIN_WEATHER'
   hours=pd.concat([pd.Series(np.r_[z[1]['context'].astype(float),z[3]],index=pd.date_range(z[0]-pd.Timedelta(hours=336),periods=360,freq='h')) for z in chosen])
   # sigma uses raw hourly means, not FP32-rounded context: recompute unique hours below.
   hourly=load.set_index('timestamp')['load'];vals=[]
   for h in hours.index.unique():vals.append(hourly.reindex(pd.date_range(h,periods=4,freq='15min')).to_numpy(float).mean())
   sigma=float(np.std(vals));assert sigma>1e-6 and np.isfinite(sigma),'CONSTANT_TRAIN_LOAD'
   tid=f'T{len(selected)}';rec.update(target=tid,status='SELECTED',valid_train=len(valid),train_sigma=sigma)
   selected.append(dict(rec,metadata=m));traincache[tid]=chosen;seen_hash.add(rec['load_sha256']);seen_physical.add(physical)
  except (AssertionError,InputsInvalid) as e:rec.update(status='PREPARATION_FAILED',reason=str(e))
  tmanifest.append(rec);csvwrite(OUT/'target_manifest.csv',tmanifest)
 save(OUT/'selected_targets.json',selected)
 jobs=[];oman=[];scales={};rowfiles=[];tests=[];aug={};intensity=[]
 for t in selected:
  tid=t['target'];load,weather=frames(ROOT/t['load_path'],ROOT/t['weather_path']);chosen=traincache[tid]
  trainrows=pd.concat([z[2].query("part == 'PAST'") for z in chosen]).drop_duplicates(['timestamp','available_at'])
  scales[tid]=dict(weather_mean=trainrows[FEATURES].mean().to_list(),weather_std=trainrows[FEATURES].std(ddof=0).to_list(),sigma_y=t['train_sigma'],unique_weather_rows=len(trainrows),statistics='TRAIN only; unique selected quarter-hour (valid timestamp, available_at, feature); sigma unique raw hourly TRAIN context+label timestamps',features=FEATURES)
  targetjobs=[];ready=True
  for role in SPLITS:
   valid=[]
   for o in origins(role):
    rec=dict(target=tid,role=role,origin=o.isoformat(),date=o.date().isoformat(),selected=False,input_valid=False,target_valid='DEFERRED_UNTIL_SEALED_PREDICTIONS' if role=='TEST' else False)
    try:
     x,rows=build_inputs(load,weather,o);rec['input_valid']=True
     y=None if role=='TEST' else label(load,o)
     if role!='TEST':rec['target_valid']=True
     valid.append((o,x,rows,y,rec))
    except InputsInvalid as e:rec['exclude_reason']=str(e)
    oman.append(rec)
   n=SPLITS[role][2]
   if n is not None and len(valid)<n:ready=False;t['preparation_failure']=f'{role}_BELOW_{n}';break
   indexes=np.arange(len(valid)) if n is None else np.floor(np.linspace(0,len(valid)-1,n)).astype(int)
   assert len(np.unique(indexes))==len(indexes)
   for i in indexes:
    o,x,rows,y,rec=valid[i];rec['selected']=True;jid=f'{tid}_{role}_{o:%m%d}';p=CACHE/'inputs'/f'{jid}.npz';p.parent.mkdir(exist_ok=True);np.savez_compressed(p,**x)
    j=dict(id=jid,target=tid,role=role,origin=o.isoformat(),date=o.date().isoformat(),input=str(p.relative_to(ROOT)),input_sha256=sha(p))
    if y is not None:
     yp=CACHE/'labels'/f'{jid}.npy';yp.parent.mkdir(exist_ok=True);np.save(yp,y);j.update(label=str(yp.relative_to(ROOT)),label_sha256=sha(yp))
    targetjobs.append(j)
    rp=CACHE/'rows'/f'{jid}.parquet';rp.parent.mkdir(exist_ok=True);rows.to_parquet(rp,index=False)
    rowfiles.append(dict(id=jid,path=str(rp.relative_to(ROOT)),sha256=sha(rp),rows=len(rows),max_available_at=rows.available_at.max().isoformat(),age_hours_quantiles=np.quantile((o-rows.available_at).dt.total_seconds()/3600,[0,.5,1]).tolist(),lead_hours_quantiles=np.quantile((rows.timestamp-rows.available_at).dt.total_seconds()/3600,[0,.5,1]).tolist()))
   # Independent poison and direct row selection on one per role; structural assertions on all origins.
   if len(indexes):
    o,x,rows,y,rec=valid[indexes[0]];lp=load.copy();lp.loc[lp.timestamp>=o,'load']=1e25;wp=weather.copy();wp.loc[wp.available_at>o,FEATURES]=1e24
    xp,_=build_inputs(lp,wp,o)
    for k in x:np.testing.assert_array_equal(x[k],xp[k])
    available=weather[(weather.timestamp==o-pd.Timedelta(hours=1))&(weather.available_at<=o)].index
    wp=weather.copy();wp.loc[available,FEATURES]=wp.loc[available,FEATURES]+1000;xr,_=build_inputs(load,wp,o);assert not np.array_equal(x['past'],xr['past'])
    for ri in [0,1000,1344,len(rows)-1]:
     rr=rows.iloc[ri];c=weather[(weather.timestamp==rr.timestamp)&(weather.available_at<=o)].sort_values('available_at',ascending=False).iloc[int(rr.k)]
     np.testing.assert_array_equal(c[FEATURES].to_numpy(float),rr[FEATURES].to_numpy(float));assert c.available_at==rr.available_at
    tests.append(dict(target=tid,role=role,future_label_poison=True,unavailable_weather_poison=True,available_past_sensitive=True,independent_row_checks=4))
  if not ready:t['status']='PREPARATION_FAILED';continue
  jobs+=targetjobs;tr=[j for j in targetjobs if j['role']=='TRAIN'];assert len(tr)==64
  for seed in SEEDS:
   orders=np.stack([np.random.default_rng(seed+e).permutation(64) for e in range(8)])
   offsets=np.stack([[point_offsets(tid,j['origin'],seed,b) for j in tr] for b in range(2)])
   masks=np.stack([[ (np.random.default_rng(stable_seed('DROP_V1',tid,j['origin'],seed,e)).random(3)<.7).astype('float32')/np.float32(.7) for j in tr] for e in range(8)])
   aug[f'{tid}_{seed}_order']=orders;aug[f'{tid}_{seed}_offsets']=offsets;aug[f'{tid}_{seed}_masks']=masks
   for i,j in enumerate(tr):
    paths=np.load(ROOT/j['input'])['paths']
    for b in range(2):
     off=offsets[b,i];pp=np.stack([select_path(paths,e,'PATH') for e in range(4)]);qq=np.stack([select_path(paths,e,'POINT',off) for e in range(4)])
     np.testing.assert_array_equal(np.sort(pp,axis=0),np.sort(qq,axis=0));assert np.array_equal(np.bincount(off,minlength=4),[6]*4)
     intensity.append(dict(target=tid,seed=seed,id=j['id'],block=b,distinct=bool(np.any(pp!=qq)),path_abs_temporal_diff=float(np.mean(abs(np.diff(pp,axis=-1)))),point_abs_temporal_diff=float(np.mean(abs(np.diff(qq,axis=-1)))),version_abs_difference=float(np.mean(abs(paths-paths[0])))))
  print('TARGET_PREPARED',tid,t['name'],len(targetjobs),flush=True)
 selected=[t for t in selected if t['status']=='SELECTED'];save(OUT/'selected_targets.json',selected)
 csvwrite(OUT/'origin_manifest.csv',oman);save(OUT/'train_scaling.json',scales);save(OUT/'selected_weather_rows.json',rowfiles);csvwrite(OUT/'train_intervention.csv',intensity)
 np.savez_compressed(CACHE/'augmentation_schedule.npz',**aug)
 save(OUT/'augmentation_schedule.json',dict(path=str((CACHE/'augmentation_schedule.npz').relative_to(ROOT)),sha256=sha(CACHE/'augmentation_schedule.npz'),seeds=SEEDS,indexing='order[epoch,position] -> TRAIN index; offsets[block,TRAIN index,hour]; masks[epoch,TRAIN index,feature]',uses='TRAIN ONLY; same schedules for both LRs and common order for all arms'))
 # All supervised hourly intervals are pairwise disjoint across split roles and origins.
 for t in selected:
  sets={}
  for role in SPLITS:
   times=[h for j in jobs if j['target']==t['target'] and j['role']==role for h in pd.date_range(j['origin'],periods=24,freq='h')]
   assert len(times)==len(set(times));sets[role]=set(times)
  for a in sets:
   for b in sets:
    if a!=b:assert not sets[a]&sets[b]
 # The reference test is on raw float64 arrays, independent from model/GPU precision.
 toy=np.arange(4*3*24,dtype=float).reshape(4,3,24);off=point_offsets('toy','origin',61730,0)
 assert not np.array_equal(select_path(toy,0,'PATH'),select_path(toy,0,'POINT',off))
 same=np.repeat(toy[:1],4,axis=0);assert np.array_equal(select_path(same,0,'PATH'),select_path(same,0,'POINT',off))
 save(OUT/'data_seal.json',dict(at=time.time(),jobs=jobs,targets=selected,tests=tests,planned_fits=16*len(selected),planned_updates=8192*len(selected),files={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'PROTOCOL.md',OUT/'train_scaling.json',OUT/'selected_targets.json',OUT/'augmentation_schedule.json',CACHE/'augmentation_schedule.npz']},information_contract='SIMULATED_ASOF',test_labels_materialized=False,partial_target_coverage=len(selected)<3))
 print('PREPARED',len(selected),'targets',len(jobs),'inputs; TEST labels deferred',flush=True)
