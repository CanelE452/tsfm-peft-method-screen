"""Bounded, known LoRA controls. No automatic extra fits or outcome-driven retries."""
import argparse, subprocess, traceback, resource
from common import *


def prepare():
 assert not (OUT/'seal.json').exists(), 'Preparation already exists'
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 ref=ROOT/'results/covariate_vintage_reference_20260916/seal.json';old=read(ref)
 for p,h in {**old['files'],**old['model_files']}.items():assert sha(p)==h, p
 jobs=[]
 for j in old['jobs']:
  for k in ['input','target']:assert sha(ROOT/j[k])==j[k+'_sha256']
  jobs.append(dict(j,split='TRAIN' if j['month'] in [3,4] else 'V' if j['month']==5 else 'D'))
 train=[j for j in jobs if j['split']=='TRAIN'];assert len(train)==8
 values=[]
 for j in train:
  with np.load(ROOT/j['input']) as x:values.append(x['past'].astype(float))
 values=np.concatenate(values,axis=1)
 save(OUT/'scaling.json',dict(features=FEATURES,mean=values.mean(1).tolist(),std=np.maximum(values.std(1),1e-6).tolist(),source_ids=[j['id'] for j in train],n_values_per_feature=values.shape[1],policy='TRAIN past arrays only; overlapping contexts retain their frequency'))
 # FP64 algebra and all input/parameter gradients, with nonzero B.
 torch.manual_seed(61710);m=LoRA(nn.Linear(5,4,bias=True)).double();m.b.data.normal_()
 x=torch.randn(2,3,5,dtype=torch.float64,requires_grad=True)
 y=m(x);explicit=x @ (m.base.weight+2*m.b@m.a).T+m.base.bias
 args=[x,m.a,m.b,m.base.weight,m.base.bias]
 g1=torch.autograd.grad(y.square().sum(),args,retain_graph=True)
 g2=torch.autograd.grad(explicit.square().sum(),args)
 err=max(float((y-explicit).abs().max().detach()),*[float((a-b).abs().max()) for a,b in zip(g1,g2)])
 assert err<=1e-12,err
 schedule=[]
 for seed in SEEDS:
  for epoch in range(15):
   order=np.random.default_rng(seed+epoch).permutation(8)
   for offset,i in enumerate(order):
    k=epoch*8+offset
    for arm in ARMS:
     path,mask=augmentation(arm,seed,k)
     schedule.append(dict(arm=arm,seed=seed,step=k+1,epoch=epoch,id=train[i]['id'],path=path,mask=None if mask is None else mask.tolist()))
 assert len(schedule)==720
 save(OUT/'schedule.json',schedule)
 tracked=subprocess.check_output(['git','ls-files','results','research'],cwd=ROOT,text=True).splitlines()
 save(OUT/'historical_hashes.json',{p:sha(ROOT/p) for p in tracked})
 import chronos.chronos2.model as model
 import chronos.chronos2.preprocess as prep
 files=list(Path(__file__).parent.glob('*.py'))+[OUT/'PROTOCOL.md',OUT/'scaling.json',OUT/'schedule.json',ROOT/'src/tsfm_peft_screen/backbone.py',ROOT/'src/tsfm_peft_screen/lora.py',ROOT/'experiments/peft_rank12_20260915/common.py',ROOT/'scripts/priority12/common.py',Path(model.__file__),Path(prep.__file__)]
 save(OUT/'seal.json',dict(at=time.time(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),reference=str(ref.relative_to(ROOT)),reference_sha256=sha(ref),jobs=jobs,model_files=old['model_files'],files={str(p):sha(p) for p in files},algebra_fp64_max_error=err,planned_fits=6,planned_updates=720,planned_smoke_updates=6))
 print('PREPARED: 8 TRAIN / 4 V / 16 D, 720 scheduled steps, FP64 checks passed',flush=True)


def verify_seal():
 s=read(OUT/'seal.json')
 for p,h in {**s['files'],**s['model_files']}.items():assert sha(p)==h,('SEALED_FILE_CHANGED',p)
 assert sha(ROOT/s['reference'])==s['reference_sha256']
 for j in s['jobs']:
  for k in ['input','target']:assert sha(ROOT/j[k])==j[k+'_sha256']
 return s


def run():
 s=verify_seal();assert not (OUT/'status.json').exists(),'Existing attempt: do not restart'
 configure();jobs={j['id']:j for j in s['jobs']};vjobs=[j for j in s['jobs'] if j['split']=='V'];djobs=[j for j in s['jobs'] if j['split']=='D']
 state=dict(status='RUNNING',phase='PREFLIGHT',fit_attempts=0,completed_fits=0,updates=0,smoke_attempts=0,smoke_updates=0,primitive_forwards=0,inference_forwards=0,training_forwards=0,errors=[])
 save(OUT/'status.json',state);w=None;fits=[];preds=[];rows=[];smokes=[];replays=[]
 def counted(m):
  def hook(model,args,output):
   state['primitive_forwards']+=1
   state['training_forwards' if torch.is_grad_enabled() else 'inference_forwards']+=1
  m.register_forward_hook(hook);return m
 def prediction(m,j,path,label):
  q,raw=predict(m,payload(j,path),w);p=CACHE/(label+'_'+j['id']+f'_k{path}.npz');assert not p.exists()
  np.savez_compressed(p,q=q,raw=raw)
  r=dict(label=label,id=j['id'],path_index=path,path=str(p.relative_to(ROOT)),sha256=sha(p))
  preds.append(r);save(OUT/'predictions.json',preds)
  return q,r
 def optimize(m,opt,j,arm,seed,k,smoke=False):
  start=w.before();torch.cuda.synchronize();t0=time.perf_counter();path,mask=augmentation(arm,seed,k)
  opt.zero_grad(set_to_none=True);o=forward(m,payload(j,path,mask),np.load(ROOT/j['target']));o.loss.backward()
  ps=parameters(m);assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps.values()),'NONFINITE_OR_MISSING_GRAD'
  norm=torch.nn.utils.clip_grad_norm_(list(ps.values()),1.,error_if_nonfinite=True)
  opt.step();state['smoke_updates' if smoke else 'updates']+=1
  assert all(torch.isfinite(p).all() for p in ps.values()),'NONFINITE_PARAMETER'
  torch.cuda.synchronize();seconds=time.perf_counter()-t0;end,contaminated=w.after(start)
  r=dict(step=k+1,id=j['id'],path_index=path,loss=float(o.loss.detach()),gradient_norm=float(norm),seconds=seconds,free_mib=end['free_mib'],contaminated=contaminated)
  del o;save(OUT/'status.json',state);return r
 def optimizer(m):return torch.optim.AdamW(parameters(m).values(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
 try:
  w=Watch(OUT,'run',wall_cap=3600);w.boundary(startup=True)
  j=s['jobs'][0];item=payload(j);base=counted(load_base());native,_=predict(base,item,w)
  from chronos import Chronos2Pipeline
  with torch.no_grad():p=Chronos2Pipeline(base).predict([item],prediction_length=24,context_length=336,batch_size=4,cross_learning=False)[0][0].double().cpu().numpy()
  parity=float(np.max(abs(native-np.sort(p,axis=0)))/j['std']);assert parity<=1e-5,parity
  raw,_=predict(base,payload(j,raw=True),w)
  scaling_difference=float(np.max(abs(raw-native))/j['std'])
  with torch.no_grad():
   ordinary=forward(base,item,np.load(ROOT/j['target'])).quantile_preds.detach().clone()
   poisoned=forward(base,item,np.load(ROOT/j['target'])+123456.).quantile_preds
  assert torch.equal(ordinary,poisoned),'LABEL_LEAKAGE'
  # Test native reduction using synthetic normalized forecasts and actual masks.
  a=tensors(item,np.load(ROOT/j['target']));_,loc_scale=base.instance_norm(a['context'])
  normalized,_=base.instance_norm(a['future_target'],loc_scale)
  mask=torch.zeros(4,2,16,device='cuda');mask[1:,:,:]=1
  z=torch.zeros(4,21,32,device='cuda',requires_grad=True)
  loss=base._compute_loss(z,a['future_target'],None,mask,loc_scale,2)
  yy=normalized[0].double().cpu().numpy();qs=base.quantiles.double().cpu().numpy()
  expected=sum(2*(t*y if y>=0 else (t-1)*y) for t in qs for y in yy)/(4*32)
  loss_error=abs(float(loss)-expected)/max(1.,abs(expected));assert loss_error<=1e-5
  grad=torch.autograd.grad(loss,z)[0];assert torch.count_nonzero(grad[1:])==0 and torch.count_nonzero(grad[0,:,24:])==0
  del base,ordinary,poisoned,a,z,grad,loss;cleanup()
  for arm in ARMS:
   state['smoke_attempts']+=1;save(OUT/'status.json',state)
   m=counted(make(SEEDS[0]));before=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()));init=cpu_state(m)
   initial,_=predict(m,item,w);initerr=float(np.max(abs(initial-native))/j['std']);assert initerr<=1e-5
   opt=optimizer(m);curve=[]
   for k in range(2):curve.append(optimize(m,opt,j,arm,SEEDS[0],k,True))
   changed=sum(not torch.equal(init[n],p.detach().cpu()) for n,p in parameters(m).items());assert changed>0
   assert frozen_hash(m)==before and tensor_hash(dict(m.named_buffers()))==buffers
   smokes.append(dict(arm=arm,updates=2,changed_tensors=changed,init_normalized_max_abs=initerr,frozen_unchanged=True,buffers_unchanged=True,curve=curve));save(OUT/'smoke.json',smokes)
   del m,opt,init;cleanup()
  save(OUT/'preflight.json',dict(status='PASS',pipeline_normalized_max_abs=parity,raw_vs_standardized_normalized_max_abs=scaling_difference,label_poison_invariant=True,native_loss_relative_error=loss_error,native_reduction='target only, 24 observed / 32 padded horizon, sum21 quantiles /4 variates',smoke_updates=state['smoke_updates']))
  state['phase']='TRAIN';save(OUT/'status.json',state);schedule=read(OUT/'schedule.json')
  for seed in SEEDS:
   for arm in ARMS:
    assert state['fit_attempts']<6 and state['updates']<720
    fid=f'{arm}_{seed}';state['fit_attempts']+=1;save(OUT/'status.json',state)
    fit=dict(id=fid,arm=arm,seed=seed,status='RUNNING',updates=0,checkpoints=[],curve=[]);fits.append(fit);save(OUT/'fits.json',fits)
    m=counted(make(seed));before=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()));init=cpu_state(m);fit['initial_hash']=tensor_hash(init)
    fit['trainable_parameters']=sum(p.numel() for p in parameters(m).values());fit['trainable_tensors']=len(parameters(m))
    opt=optimizer(m);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.perf_counter()
    items=[r for r in schedule if r['arm']==arm and r['seed']==seed]
    for step in range(121):
     if step:
      entry=items[step-1];fit['curve'].append(optimize(m,opt,jobs[entry['id']],arm,seed,step-1));fit['updates']=step
     if step in CHECKPOINTS:
      cp=CACHE/f'{fid}_{step}.pt';assert not cp.exists();torch.save(cpu_state(m),cp);scores=[]
      for jv in vjobs:
       q,r=prediction(m,jv,0,f'{fid}_CP{step}');scores.append(score(q,np.load(ROOT/jv['target']),jv['std'])['primary'])
      fit['checkpoints'].append(dict(step=step,path=str(cp.relative_to(ROOT)),sha256=sha(cp),validation_primary=float(np.mean(scores)),at=time.time(),elapsed_seconds=time.perf_counter()-t0))
      save(OUT/'fits.json',fits);print('CHECKPOINT',fid,step,'V',np.mean(scores),flush=True)
    torch.cuda.synchronize();fit.update(status='COMPLETE',seconds=time.perf_counter()-t0,optimizer_seconds=sum(r['seconds'] for r in fit['curve']),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),changed_tensors=sum(not torch.equal(init[n],p.detach().cpu()) for n,p in parameters(m).items()))
    assert fit['changed_tensors']>0 and frozen_hash(m)==before and tensor_hash(dict(m.named_buffers()))==buffers
    fit.update(frozen_unchanged=True,buffers_unchanged=True);state['completed_fits']+=1
    save(OUT/'fits.json',fits);save(OUT/'status.json',state);del m,opt,init;cleanup()
  selections=[dict(id=f['id'],arm=f['arm'],seed=f['seed'],checkpoint=min(f['checkpoints'],key=lambda c:(c['validation_primary'],c['step']))) for f in fits]
  save(OUT/'selection.json',dict(at=time.time(),fits_sha256=sha(OUT/'fits.json'),selections=selections,rule='minimum V primary; earlier step breaks ties'))
  state['phase']='EVALUATE';state['selection_sha256']=sha(OUT/'selection.json');save(OUT/'status.json',state)
  # Common INIT0 is recomputed through the exact normalized input path.
  m=counted(make(SEEDS[0]));before=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()))
  for j in djobs:
   for path in [0,3]:
    q,r=prediction(m,j,path,'INIT0');rows.append(dict(arm='INIT0',seed=-1,source='INIT0',step=0,id=j['id'],month=j['month'],path_index=path,prediction=r['path'],**score(q,np.load(ROOT/j['target']),j['std'])))
  assert frozen_hash(m)==before and tensor_hash(dict(m.named_buffers()))==buffers
  del m;cleanup();csvwrite(OUT/'scores.csv',rows)
  for sel,fit in zip(selections,fits):
   m=counted(make(sel['seed']));before=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()));seen={}
   for source,cp in [('SELECTED',sel['checkpoint']),('FIXED120',fit['checkpoints'][-1])]:
    assert sha(ROOT/cp['path'])==cp['sha256'];restore(m,torch.load(ROOT/cp['path'],weights_only=True,map_location='cpu'))
    if source=='SELECTED':
     for jv in vjobs:
      q,_=predict(m,payload(jv),w)
      old=np.load(CACHE/f"{sel['id']}_CP{cp['step']}_{jv['id']}_k0.npz")['q'];err=float(np.max(abs(q-old))/jv['std']);assert err<=1e-5
      replays.append(dict(id=sel['id'],origin=jv['id'],normalized_max_abs=err,exact=bool(np.array_equal(q,old))))
     save(OUT/'replay.json',replays)
    for j in djobs:
     for path in [0,3]:
      key=(cp['step'],j['id'],path)
      if key not in seen:seen[key]=prediction(m,j,path,f"{sel['id']}_EVAL{cp['step']}")
      q,r=seen[key];rows.append(dict(arm=sel['arm'],seed=sel['seed'],source=source,step=cp['step'],id=j['id'],month=j['month'],path_index=path,prediction=r['path'],**score(q,np.load(ROOT/j['target']),j['std'])))
    csvwrite(OUT/'scores.csv',rows)
   assert frozen_hash(m)==before and tensor_hash(dict(m.named_buffers()))==buffers
   del m;cleanup();print('EVALUATION_COMPLETE',sel['id'],flush=True)
  assert state['updates']==720 and state['smoke_updates']==6 and len(rows)==416
  state.update(status='COMPLETE',phase='DONE',score_rows=len(rows),saved_predictions=len(preds),replay_checks=len(replays))
 except BaseException as exc:
  state.update(status='PARTIAL',errors=[dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())]);print(traceback.format_exc(),flush=True)
 finally:
  if w:w.close()
  save(OUT/'status.json',state)
 print('STATUS',state,flush=True)
 if state['status']!='COMPLETE':raise SystemExit(1)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','status']);a=p.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='run':run()
 else:print(read(OUT/'status.json'))
