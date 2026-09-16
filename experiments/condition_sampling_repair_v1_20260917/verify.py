"""Final independent accounting and immutable-old-artifact audit, without training."""
import ast,time,math
import pandas as pd
from .common import *

def verify_all():
 seal=read(OUT/'MASTER_SEAL.json')
 for path,h in {**seal['files'],**seal['implementation']}.items():assert sha(ROOT/path)==h,('SEALED_CHANGE',path)
 for path,h in {**seal['installed_code'],**seal['model_files']}.items():assert sha(path)==h
 for path,h in read(OUT/'historical_hashes.json').items():assert sha(ROOT/path)==h,('OLD_CHANGED',path)
 from .audit_selection import run as selection
 from .audit_training import run as training
 from .audit_cpu import run as cpu
 selection();training();cpu()
 state=read(OUT/'controller_state.json');assert state['status']=='FINISHED';assert not (OUT/'pending_update.json').exists();main=smoke=fits_n=0;trainforward=0;rows=[];forward=[]
 for t in ORDER:
  st=read(OUT/t/'STATUS.json');div=read(OUT/t/'origin_diversity.json')
  if st['EXECUTION']=='BLOCKED_DIVERSITY':
   assert not div['passed'];assert not (OUT/t/'optimizer_log.jsonl').exists() and not (CACHE/t/'checkpoints').exists();rows.append(dict(track=t,status='BLOCKED_DIVERSITY',main_fits=0,main_updates=0,smoke_updates=0,performance_measured=False));continue
  assert div['passed'];fits=read(OUT/t/'fits.json');logs=[json.loads(x) for x in (OUT/t/'optimizer_log.jsonl').read_text().splitlines()];sm=sum(x['smoke'] for x in logs);ma=len(logs)-sm;main+=ma;smoke+=sm;fits_n+=len(fits);factor=2 if t=='R04' else 1;trainforward+=len(logs)*factor
  csvwrite(OUT/t/'optimizer_log.csv',[{k:json.dumps(v,sort_keys=True) if isinstance(v,dict) else v for k,v in r.items()} for r in logs])
  for f in fits:
   assert f['status']=='COMPLETE' and f['updates']==512 and f['frozen_unchanged'] and f['buffers_unchanged'];ll=[r for r in logs if r['id']==f['id']];assert len(ll)==512 and [r['step'] for r in ll]==list(range(1,513))
   for cp in f['checkpoints']:assert sha(ROOT/cp['path'])==cp['sha256']
  for r in read(OUT/t/'predictions_manifest.json')['predictions']:assert sha(ROOT/r['path'])==r['sha256']
  for r in read(OUT/t/'cross_evaluation_manifest.json')['records']:
   if r['status']=='COMPLETE':assert sha(ROOT/r['path'])==r['sha256'] and sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
  assert read(OUT/t/'verification.json')['passed'] and read(OUT/t/'independent_selection_verification.json')['passed'] and read(OUT/t/'training_record_verification.json')['passed']
  cross=read(OUT/t/'cross_verification.json');assert cross['passed'] and cross['optimizer_updates']==0
  rows.append(dict(track=t,status=st['EXECUTION'],main_fits=len(fits),main_updates=ma,smoke_updates=sm,scalar_metric_checks=read(OUT/t/'verification.json')['scalar_metric_checks'],cross_metric_checks=cross['scalar_metric_checks'],contaminated_updates=sum(r['contaminated'] for r in logs)))
  forward.extend([dict(track=t,phase='main_train',calls=ma*factor),dict(track=t,phase='smoke_train',calls=sm*factor),dict(track=t,phase='smoke_verify',calls=4*len(read(OUT/t/'smoke.json'))),dict(track=t,phase='restore_verify',calls=factor*len(read(OUT/t/'restore_verification.json')))])
  if (CACHE/t/'frozen.npz').exists():z=np.load(CACHE/t/'frozen.npz');forward.append(dict(track=t,phase='frozen_cache',calls=sum(len(z[k]) for k in z.files)))
  phases={}
  for path in (CACHE/t/'predictions').glob('*.npz'):
   z=np.load(path);c=len(z['conditions']);phase='cross' if '_MODEL_' in path.name else 'V' if path.name.startswith('V_SELECT') else 'E';phases[phase]=phases.get(phase,0)+len(z['origins'])*(c-3 if t=='N01' else c*factor)
  forward.extend(dict(track=t,phase=k,calls=v) for k,v in phases.items())
 assert main==state['main_updates']<=49152 and smoke==state['smoke_updates']<=48 and main+smoke<=49200;assert fits_n==state['completed_fits']<=96;assert trainforward==state['forwards']['train'];assert sum(v['calls'] for v in forward)==sum(state['forwards'].values())<=200000
 assert sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())<=100*2**30
 csvwrite(OUT/'FORWARD_LEDGER.csv',forward);save(OUT/'verification.json',dict(passed=True,at=time.time(),historical_files_preserved=len(read(OUT/'historical_hashes.json')),main_fits=fits_n,main_updates=main,smoke_updates=smoke,native_forwards=state['forwards'],independent_forward_ledger_matches=True,tracks=rows,blocked_not_performance_failures=True,no_unaffected_retraining=True))
 print('REPAIR_VERIFIED',fits_n,main,smoke,flush=True)
