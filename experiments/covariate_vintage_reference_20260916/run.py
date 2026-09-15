"""Finite forecast-only vintage reference; inference reads no evaluation labels."""
import argparse,subprocess,traceback,time,sys
import numpy as np
import pandas as pd
from common import *

def build_inputs(load,weather,o):
 past=pd.date_range(o-pd.Timedelta(hours=336),periods=336*4,freq='15min')
 future=pd.date_range(o,periods=24*4,freq='15min')
 target=load[(load.timestamp.isin(past))&(load.available_at<=o)].set_index('timestamp').reindex(past)['load'].to_numpy(dtype=float)
 assert len(target)==1344 and np.isfinite(target).all(),'INVALID_LOAD_CONTEXT'
 w=weather[(weather.timestamp.isin(past.append(future)))&(weather.available_at<=o)].sort_values(['timestamp','available_at'],ascending=[True,False]).copy()
 w['rank']=w.groupby('timestamp').cumcount()
 history=w[(w['rank']==0)&w.timestamp.isin(past)].set_index('timestamp').reindex(past)[FEATURES].to_numpy(dtype=float)
 paths=np.stack([w[(w['rank']==k)&w.timestamp.isin(future)].set_index('timestamp').reindex(future)[FEATURES].to_numpy(dtype=float) for k in range(4)])
 assert np.isfinite(history).all() and np.isfinite(paths).all(),'INVALID_WEATHER_INPUT'
 assert (w.available_at<=o).all()
 return dict(target=target.reshape(336,4).mean(1).astype('float32'),past=history.reshape(336,4,3).mean(1).T.astype('float32'),paths=paths.reshape(4,24,4,3).mean(2).transpose(0,2,1).astype('float32'))

def prepare():
 assert not (OUT/'seal.json').exists(),'No repeated preparation'
 tests=checks();manifest=read(ROOT/'research/covariate_availability_audit_20260916/download_manifest.json')
 for item in manifest[:5]:assert sha(SOURCE/item['path'])==item['sha256']
 load=pd.read_parquet(SOURCE/'load_measurements/mv_feeder/OS Gorredijk.parquet');weather=pd.read_parquet(SOURCE/'weather_forecasts_versioned/mv_feeder/OS Gorredijk.parquet')
 for frame in [load,weather]:
  for col in ['timestamp','available_at']:frame[col]=pd.to_datetime(frame[col],utc=True)
 assert not load.duplicated('timestamp').any() and not weather.duplicated(['timestamp','available_at']).any()
 jobs=[];poison=[]
 for role,months in [('C',[3,4,5]),('D',[6,7,8,9])]:
  for month in months:
   for day in [1,8,15,22]:
    o=pd.Timestamp(year=2024,month=month,day=day,hour=8,tz='UTC');jid=f'{role}_{month:02d}{day:02d}';x=build_inputs(load,weather,o)
    # For every job, poison unavailable load and weather then reconstruct inputs.
    lp=load.copy();lp.loc[lp.available_at>=o,'load']=123456789.;wp=weather.copy();wp.loc[wp.available_at>o,FEATURES]=123456.
    xp=build_inputs(lp,wp,o)
    for k in x:np.testing.assert_array_equal(x[k],xp[k])
    poison.append(dict(id=jid,future_load_and_weather_invariant=True))
    idx=pd.date_range(o,periods=96,freq='15min');label=load.set_index('timestamp').reindex(idx)['load'].to_numpy(dtype=float)
    assert np.isfinite(label).all(),'INVALID_EVALUATION_TARGET'
    y=label.reshape(24,4).mean(1);std=max(float(x['target'].astype(float).std(ddof=0)),1e-6)
    inp=CACHE/f'{jid}_input.npz';yp=CACHE/f'{jid}_target.npy';np.savez_compressed(inp,**x);np.save(yp,y)
    jobs.append(dict(id=jid,role=role,month=month,origin=str(o),std=std,input=str(inp.relative_to(ROOT)),input_sha256=sha(inp),target=str(yp.relative_to(ROOT)),target_sha256=sha(yp)))
 tracked=subprocess.check_output(['git','ls-files','results','research'],cwd=ROOT,text=True).splitlines();old={p:sha(ROOT/p) for p in tracked}
 save(OUT/'historical_hashes.json',old)
 files=list(Path(__file__).parent.glob('*.py'))+[OUT/'PROTOCOL.md',ROOT/'src/tsfm_peft_screen/backbone.py',ROOT/'experiments/peft_rank12_20260915/common.py',ROOT/'scripts/priority12/common.py']
 import chronos.chronos2.pipeline as pipeline
 files+=[Path(pipeline.__file__)]
 snapshot=Path.home()/'.cache/huggingface/hub/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c'
 weights={str(p):sha(p) for p in snapshot.iterdir() if p.is_file()};assert any(p.endswith('.safetensors') for p in weights)
 save(OUT/'seal.json',dict(baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),at=time.time(),files={str(p):sha(p) for p in files},source_manifest_sha256=sha(ROOT/'research/covariate_availability_audit_20260916/download_manifest.json'),model_files=weights,jobs=jobs,tests=tests,poison_checks=poison,planned_target_forecasts=233,planned_pipeline_calls=128,planned_neural_fits=0,planned_postprocessors=7))
 print('PREPARED',len(jobs),'origins, mathematical and poison checks passed',flush=True)

def verify_seal():
 s=read(OUT/'seal.json')
 for p,h in s['files'].items():assert sha(p)==h,('SOURCE_CHANGED',p)
 for p,h in s['model_files'].items():assert sha(p)==h,('MODEL_CHANGED',p)
 for j in s['jobs']:
  for key in ['input','target']:assert sha(ROOT/j[key])==j[key+'_sha256']
 return s

def payloads(x):
 past={f:x['past'][i] for i,f in enumerate(FEATURES)}
 def item(v):return dict(target=x['target'],past_covariates=past,future_covariates={f:v[i] for i,f in enumerate(FEATURES)})
 return [('F0',[{'target':x['target']}]),('PAST',[dict(target=x['target'],past_covariates=past)]),('MEAN_INPUT',[item(x['paths'].mean(0))]),('PATHS',[item(v) for v in x['paths']])]

def predict_all():
 s=verify_seal();assert not (OUT/'status.json').exists(),'Existing attempt: inspect; do not rerun'
 sys.path.insert(0,str(ROOT))
 import torch
 from chronos import Chronos2Pipeline
 from tsfm_peft_screen.backbone import load_base
 from experiments.peft_rank12_20260915.common import Watch,tensor_hash
 torch.set_num_threads(4);torch.manual_seed(61700);np.random.seed(61700);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 state=dict(status='RUNNING',fits=0,updates=0,target_forecasts=0,pipeline_calls=0,primitive_forwards=0,complete_origins=0,errors=[]);save(OUT/'status.json',state)
 watch=None;calls=[];preds=[]
 try:
  watch=Watch(OUT,'predict',wall_cap=3600);watch.boundary(startup=True)
  model=load_base();assert not any(p.requires_grad for p in model.parameters());pipe=Chronos2Pipeline(model)
  before=tensor_hash(dict(model.named_parameters()));buffers=tensor_hash(dict(model.named_buffers()))
  def hook(*args):state['primitive_forwards']+=1
  handle=model.register_forward_hook(hook)
  def call(inp,label):
   assert state['pipeline_calls']<128 and state['target_forecasts']+len(inp)<=233
   start=watch.before();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
   state['pipeline_calls']+=1;state['target_forecasts']+=len(inp);save(OUT/'status.json',state)
   with torch.inference_mode():out=pipe.predict(inp,prediction_length=24,context_length=336,batch_size=16,cross_learning=False,after_batch=watch.boundary)
   torch.cuda.synchronize();seconds=time.perf_counter()-t0
   raw=np.stack([q[0].double().cpu().numpy() for q in out]);assert raw.shape==(len(inp),21,24) and np.isfinite(raw).all(),'INVALID_PREDICTION'
   end,contaminated=watch.after(start);calls.append(dict(label=label,target_forecasts=len(inp),seconds=seconds,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),free_mib=end['free_mib'],contaminated=contaminated));save(OUT/'calls.json',calls);save(OUT/'status.json',state)
   return raw
  for j in s['jobs']:
   with np.load(ROOT/j['input']) as x:groups=payloads(x)
   outputs={name:call(items,j['id']+'/'+name) for name,items in groups}
   p=CACHE/f"{j['id']}_prediction.npz";np.savez_compressed(p,**outputs)
   preds.append(dict(id=j['id'],path=str(p.relative_to(ROOT)),sha256=sha(p)));save(OUT/'predictions.json',preds);state['complete_origins']+=1;save(OUT/'status.json',state)
   print('ORIGIN_COMPLETE',j['id'],state['complete_origins'],'/28',flush=True)
  first=s['jobs'][0]
  with np.load(ROOT/first['input']) as x:groups=payloads(x)
  replay=[]
  with np.load(ROOT/preds[0]['path']) as old:
   for name,items in groups:
    q=call(items,'REPLAY/'+name);err=float(np.max(abs(q-old[name])));relative=err/first['std'];assert relative<=1e-6,'REPLAY_ERROR'
    replay.append(dict(name=name,raw_max_abs=err,normalized_max_abs=relative,exact=bool(np.array_equal(q,old[name]))))
  paths=groups[-1][1]
  for name,items in [('LATEST',[paths[0]]),('PATHS',paths)]:
   for rep in range(6):call(items,f'TIMING/{name}/{rep}')
  handle.remove();assert tensor_hash(dict(model.named_parameters()))==before and tensor_hash(dict(model.named_buffers()))==buffers,'FROZEN_CHANGED'
  save(OUT/'replay.json',replay);state.update(status='COMPLETE',frozen_unchanged=True,buffers_unchanged=True)
 except BaseException as exc:
  state.update(status='PARTIAL',errors=[dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())]);print(traceback.format_exc(),flush=True)
 finally:
  if watch:watch.close()
  save(OUT/'status.json',state)
 print('PREDICTION_STATUS',state['status'],state['target_forecasts'],state['pipeline_calls'],flush=True)
 if state['status']!='COMPLETE':raise SystemExit(1)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','predict']);a=p.parse_args()
 if a.action=='prepare':prepare()
 else:predict_all()
