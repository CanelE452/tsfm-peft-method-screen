"""Four bounded order controls, with explicit attempt and inference ledgers."""
import argparse,traceback
from common import *

def run():
 s=verified();assert not (OUT/'status.json').exists(),'Existing attempt: inspect, never blindly restart'
 base.configure();jobs={j['id']:j for j in s['jobs']};vjobs=[j for j in s['jobs'] if j['split']=='V'];djobs=[j for j in s['jobs'] if j['split']=='D'];schedule=read(OUT/'schedule.json')
 state=dict(status='RUNNING',phase='PREFLIGHT',fit_attempts=0,completed_fits=0,updates=0,smoke_attempts=0,smoke_updates=0,training_forwards=0,inference_forwards=0,errors=[]);save(OUT/'status.json',state)
 fits=[];preds=[];scores=[];replays=[];smoke=[];w=None
 def make(seed):
  m=base.make(seed)
  def hook(*args):state['training_forwards' if torch.is_grad_enabled() else 'inference_forwards']+=1
  m.register_forward_hook(hook);return m
 def optimizer(m):return torch.optim.AdamW(base.parameters(m).values(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
 def optimize(m,opt,entry,is_smoke=False):
  start=w.before();torch.cuda.synchronize();t0=time.perf_counter();j=jobs[entry['id']]
  opt.zero_grad(set_to_none=True);o=base.forward(m,base.payload(j,entry['path']),np.load(ROOT/j['target']));o.loss.backward();ps=base.parameters(m)
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps.values()),'NONFINITE_OR_MISSING_GRAD'
  norm=torch.nn.utils.clip_grad_norm_(list(ps.values()),1.,error_if_nonfinite=True);opt.step();state['smoke_updates' if is_smoke else 'updates']+=1
  assert all(torch.isfinite(p).all() for p in ps.values()),'NONFINITE_PARAMETER'
  torch.cuda.synchronize();seconds=time.perf_counter()-t0;end,contaminated=w.after(start)
  r=dict(step=entry['step'],id=j['id'],path_index=entry['path'],loss=float(o.loss.detach()),gradient_norm=float(norm),seconds=seconds,free_mib=end['free_mib'],contaminated=contaminated)
  save(OUT/'status.json',state);return r
 def predict(m,j,path,label):
  q,raw=base.predict(m,base.payload(j,path),w);p=CACHE/f'{label}_{j["id"]}_k{path}.npz';assert not p.exists();np.savez_compressed(p,q=q,raw=raw)
  record=dict(label=label,id=j['id'],path_index=path,path=str(p.relative_to(ROOT)),sha256=sha(p));preds.append(record);save(OUT/'predictions.json',preds);return q,record
 try:
  w=base.Watch(OUT,'run',wall_cap=3600);w.boundary(startup=True)
  oldfits=read(OLD/'fits.json');oldpreds=read(OLD/'predictions.json')
  for arm in ARMS:
   state['smoke_attempts']+=1;save(OUT/'status.json',state);m=make(SEEDS[0]);initial=base.cpu_state(m);frozen=base.frozen_hash(m);buffers=base.tensor_hash(dict(m.named_buffers()))
   j=vjobs[0];q,_=base.predict(m,base.payload(j),w);rp=next(p for p in oldpreds if p['label']=='STD_61710_CP0' and p['id']==j['id'])
   with np.load(ROOT/rp['path']) as p:err=float(np.max(abs(q-p['q']))/j['std'])
   assert err<=1e-5,('INITIAL_REPLAY',err)
   opt=optimizer(m);entries=[r for r in schedule if r['arm']==arm and r['seed']==SEEDS[0]];curve=[optimize(m,opt,e,True) for e in entries[:2]]
   changed=sum(not torch.equal(initial[n],p.detach().cpu()) for n,p in base.parameters(m).items());assert changed>0
   assert frozen==base.frozen_hash(m) and buffers==base.tensor_hash(dict(m.named_buffers()))
   smoke.append(dict(arm=arm,initial_normalized_max_abs=err,curve=curve,changed_tensors=changed,frozen_unchanged=True,buffers_unchanged=True));save(OUT/'smoke.json',smoke)
   del m,opt,initial;base.cleanup()
  save(OUT/'preflight.json',dict(status='PASS',new_smoke_updates=4,initial_reference_replays=2,inherited_numeric_tests=str((OLD/'preflight.json').relative_to(ROOT)),native_pipeline_and_poison_repeated=False,reason='Model, payload, forward and loss helpers unchanged and hash checked. New logic is the sealed input order.'))
  state['phase']='TRAIN';save(OUT/'status.json',state)
  for seed in SEEDS:
   for arm in ARMS:
    assert state['fit_attempts']<4;state['fit_attempts']+=1;save(OUT/'status.json',state)
    fid=f'{arm}_{seed}';fit=dict(id=fid,arm=arm,seed=seed,status='RUNNING',updates=0,checkpoints=[],curve=[]);fits.append(fit);save(OUT/'fits.json',fits)
    m=make(seed);initial=base.cpu_state(m);frozen=base.frozen_hash(m);buffers=base.tensor_hash(dict(m.named_buffers()));ih=base.tensor_hash(initial)
    reference=next(f for f in oldfits if f['arm']=='VINTAGE' and f['seed']==seed);assert ih==reference['initial_hash']
    fit.update(initial_hash=ih,trainable_parameters=sum(p.numel() for p in base.parameters(m).values()),trainable_tensors=len(base.parameters(m)))
    assert fit['trainable_parameters']==147456 and fit['trainable_tensors']==192
    opt=optimizer(m);entries=[e for e in schedule if e['arm']==arm and e['seed']==seed];torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
    for step in range(121):
     if step:fit['curve'].append(optimize(m,opt,entries[step-1]));fit['updates']=step
     if step in [0,40,80,120]:
      cp=CACHE/f'{fid}_{step}.pt';assert not cp.exists();torch.save(base.cpu_state(m),cp);values=[]
      for j in vjobs:
       q,r=predict(m,j,0,f'{fid}_CP{step}');values.append(base.score(q,np.load(ROOT/j['target']),j['std'])['primary'])
      fit['checkpoints'].append(dict(step=step,path=str(cp.relative_to(ROOT)),sha256=sha(cp),validation_primary=float(np.mean(values)),at=time.time(),elapsed_seconds=time.perf_counter()-t0));save(OUT/'fits.json',fits)
      print('CHECKPOINT',fid,step,'V',np.mean(values),flush=True)
    fit.update(status='COMPLETE',seconds=time.perf_counter()-t0,optimizer_seconds=sum(r['seconds'] for r in fit['curve']),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),changed_tensors=sum(not torch.equal(initial[n],p.detach().cpu()) for n,p in base.parameters(m).items()))
    assert fit['changed_tensors']>0 and frozen==base.frozen_hash(m) and buffers==base.tensor_hash(dict(m.named_buffers()))
    fit.update(frozen_unchanged=True,buffers_unchanged=True);state['completed_fits']+=1;save(OUT/'fits.json',fits);save(OUT/'status.json',state);del m,opt,initial;base.cleanup()
  selection=[dict(id=f['id'],arm=f['arm'],seed=f['seed'],checkpoint=min(f['checkpoints'],key=lambda c:(c['validation_primary'],c['step']))) for f in fits]
  save(OUT/'selection.json',dict(at=time.time(),fits_sha256=sha(OUT/'fits.json'),selections=selection));state.update(phase='EVALUATE',selection_sha256=sha(OUT/'selection.json'));save(OUT/'status.json',state)
  for f,sel in zip(fits,selection):
   m=make(sel['seed']);frozen=base.frozen_hash(m);buffers=base.tensor_hash(dict(m.named_buffers()));seen={}
   for source,cp in [('SELECTED',sel['checkpoint']),('FIXED120',f['checkpoints'][-1])]:
    assert sha(ROOT/cp['path'])==cp['sha256'];base.restore(m,torch.load(ROOT/cp['path'],map_location='cpu',weights_only=True))
    if source=='SELECTED':
     for j in vjobs:
      q,_=base.predict(m,base.payload(j),w)
      with np.load(CACHE/f"{f['id']}_CP{cp['step']}_{j['id']}_k0.npz") as p:err=float(np.max(abs(q-p['q']))/j['std']);exact=bool(np.array_equal(q,p['q']))
      assert err<=1e-5;replays.append(dict(id=f['id'],origin=j['id'],normalized_max_abs=err,exact=exact))
     save(OUT/'replay.json',replays)
    for j in djobs:
     for path in [0,3]:
      key=(cp['step'],j['id'],path)
      if key not in seen:seen[key]=predict(m,j,path,f"{f['id']}_EVAL{cp['step']}")
      q,r=seen[key];scores.append(dict(arm=f['arm'],seed=f['seed'],source=source,step=cp['step'],id=j['id'],month=j['month'],path_index=path,prediction=r['path'],**base.score(q,np.load(ROOT/j['target']),j['std'])))
    base.csvwrite(OUT/'scores.csv',scores)
   assert frozen==base.frozen_hash(m) and buffers==base.tensor_hash(dict(m.named_buffers()));del m;base.cleanup();print('EVALUATED',f['id'],flush=True)
  assert state['updates']==480 and state['smoke_updates']==4 and len(scores)==256
  state.update(status='COMPLETE',phase='DONE',score_rows=256,saved_predictions=len(preds),replay_checks=16)
 except BaseException as exc:
  state.update(status='PARTIAL',errors=[dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())]);print(traceback.format_exc(),flush=True)
 finally:
  if w:w.close()
  save(OUT/'fits.json',fits);save(OUT/'status.json',state)
 print('STATUS',state,flush=True)
 if state['status']!='COMPLETE':raise SystemExit(1)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','status']);a=p.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='run':run()
 else:print(read(OUT/'status.json'))
