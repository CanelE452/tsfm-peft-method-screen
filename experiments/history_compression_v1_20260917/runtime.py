import os,time,random,traceback
import torch
from .common import *
from .model import Model,configure
class Runner:
 def __init__(self):
  validate_seal();configure();self.d=Data();self.state=read(OUT/'state.json') if (OUT/'state.json').exists() else dict(status='RUNNING',main_updates=0,smoke_updates=0,forwards=0,completed=0,attempts=0,started_at=time.time())
  self.fits=read(OUT/'fits.json') if (OUT/'fits.json').exists() else []
  if (OUT/'pending_update.json').exists():raise RuntimeError('AMBIGUOUS_UPDATE_REQUIRES_AUDIT')
  self.guard=Watch(OUT,'controller',wall_cap=14400);self.persist()
 def persist(self):save(OUT/'state.json',self.state);save(OUT/'fits.json',self.fits)
 def model(self,a,seed):
  m=Model(a,seed)
  def hook(*args):
   assert self.state['forwards']<40000,'FORWARD_BUDGET'
   self.state['forwards']+=1
  m.register_forward_pre_hook(hook);return m
 def predict(self,m,role,key):
  path=CACHE/'predictions'/f'{role}_{key}.npz';meta=path.with_suffix('.json')
  if path.exists():assert sha(path)==read(meta)['sha256'];return path
  pp=[];start=time.perf_counter()
  for i in range(len(self.d.x[role]['origins'])):
   self.guard.boundary()
   with torch.no_grad():pp.append(m(self.d.packet(role,i)).double().cpu().numpy())
  npz(path,pred=np.array(pp),origins=self.d.x[role]['origins']);save(meta,dict(sha256=sha(path),seconds=time.perf_counter()-start,role=role));self.persist();return path
 def teacher(self):
  s=read(OUT/'REUSE_RECEIPT.json')['teacher'];cp=s['selected'];assert sha(ROOT/cp['path'])==cp['sha256']
  m=self.model('LONG',s['seed']);restore(m,torch.load(ROOT/cp['path'],map_location='cpu',weights_only=True))
  p=self.predict(m,'TRAIN','TEACHER_'+cp['parameter_hash']);save(OUT/'teacher.json',dict(checkpoint=cp,path=str(p.relative_to(ROOT)),sha256=sha(p),extra_fits=0,extra_updates=0,uses_E=False,teacher_selection_uses_V=True,seconds=read(p.with_suffix('.json'))['seconds']))
  del m;cleanup();return np.load(p)['pred']
 def update(self,m,opt,teacher,i,step,fid,smoke=False):
  self.guard.boundary();key='smoke_updates' if smoke else 'main_updates';assert self.state[key]<(10 if smoke else 10240)
  save(OUT/'pending_update.json',dict(id=fid,step=step,smoke=smoke,at=time.time()))
  opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter();wall=time.time()
  loss,parts=m.objective(self.d.packet('TRAIN',i),self.d.y['TRAIN'][i],teacher[i]);loss.backward();ps=parameters(m)
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps.values()),'BAD_GRADIENT'
  extra={n:float(p.grad.norm()) for n,p in ps.items() if n.startswith('pool_')};norm=torch.nn.utils.clip_grad_norm_(list(ps.values()),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
  assert all(torch.isfinite(p).all() for p in ps.values()),'BAD_PARAMETER'
  row=dict(id=fid,step=step,origin_ordinal=i,origin=int(self.d.x['TRAIN']['origins'][i]),smoke=smoke,loss=float(loss.detach()),gradnorm=float(norm),extra_grad=extra,seconds=time.perf_counter()-start,contaminated=any(r['busy'] for r in self.guard.events if r['at']>=wall),**parts)
  self.state[key]+=1
  with (OUT/'optimizer.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
  self.persist();(OUT/'pending_update.json').unlink();return row
 def smoke(self,teacher):
  if (OUT/'smoke.json').exists():assert read(OUT/'smoke.json')['passed'];return
  checks=[];p=self.d.packet('TRAIN',0)
  # Native full-token and reconstructed encode/decode paths must agree.
  m=self.model('LONG',73100)
  with torch.no_grad():native=m(p);manual=m(p,manual_full=True)
  err=float(((native-manual).abs()/m.T(self.d.sigma)[:,None]).max());assert err<=1e-5,('NATIVE_PARITY',err)
  del m;cleanup()
  pool=self.model('POOL',73100)
  with torch.no_grad():qpool=pool(p).cpu();patch,_,_=pool.base._prepare_patched_context(pool.T(p['x']));hh=pool.base.input_patch_embedding(patch);pooled=pool.pooled(hh).cpu().numpy();ref=hh.cpu().numpy();expected=np.concatenate([(ref[:,:63].reshape(4,21,3,-1)/3).sum(2),ref[:,63:]],1)
  assert np.allclose(pooled,expected,atol=1e-6,rtol=1e-6);del pool;cleanup()
  for a in ARMS:
   m=self.model(a,73100);init=cpu_state(m);fh=frozen_hash(m);bh=tensor_hash(dict(m.named_buffers()))
   with torch.no_grad():q=m(p).cpu()
   if a!='STATS_SHORT':assert torch.allclose(q,qpool,rtol=1e-5,atol=1e-6),('INITIAL_POOL_PARITY',a)
   opt=torch.optim.AdamW(parameters(m).values(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
   rr=[self.update(m,opt,teacher,0,j+1,'SMOKE_'+a,True) for j in range(2)]
   assert frozen_hash(m)==fh and tensor_hash(dict(m.named_buffers()))==bh
   changed=sum(not torch.equal(init[n],p.detach().cpu()) for n,p in parameters(m).items());assert changed>0
   for n in rr[0]['extra_grad']:assert any(r['extra_grad'][n]>0 for r in rr),(a,n)
   with torch.no_grad():q=m(p).cpu().numpy()
   cp=CACHE/'smoke'/f'{a}.pt';atomic_torch(cp,cpu_state(m));count=sum(p.numel() for p in parameters(m).values());del m,opt;cleanup()
   m=self.model(a,73100);restore(m,torch.load(cp,map_location='cpu',weights_only=True))
   with torch.no_grad():qq=m(p).cpu().numpy()
   assert np.array_equal(q,qq),'RESTORE'
   checks.append(dict(arm=a,updates=2,parameters=count,changed_tensors=changed,frozen_unchanged=True,buffers_unchanged=True,restore_bitwise=True,extra_gradient=[r['extra_grad'] for r in rr]))
   del m;cleanup();print('SMOKE',a,flush=True)
  save(OUT/'smoke.json',dict(passed=True,native_full_error=err,independent_numpy_pooling=True,initial_pool_learn_equal=True,checks=checks))
 def checkpoint(self,m,fit,step):
  if any(c['step']==step for c in fit['checkpoints']):return
  state=cpu_state(m);ph=tensor_hash(state);cp=CACHE/'checkpoints'/f"{fit['id']}_{step}.pt";atomic_torch(cp,state)
  pred=self.predict(m,'V_SELECT',fit['arm']+'_'+ph);v=nrms(np.load(pred)['pred'],self.d.y['V_SELECT'],self.d.sigma)
  fit['checkpoints'].append(dict(step=step,path=str(cp.relative_to(ROOT)),sha256=sha(cp),parameter_hash=ph,prediction=str(pred.relative_to(ROOT)),accuracy=v));self.persist();print('CHECKPOINT',fit['id'],step,round(v,6),flush=True)
 def fit(self,a,seed,lr,teacher):
  fid=f'{a}_{seed}_{lr:.0e}';f=next((f for f in self.fits if f['id']==fid),None)
  if f and f['status']=='COMPLETE':return
  step0=0;resume=None
  if f:
   resume=torch.load(CACHE/'resume'/f'{fid}.pt',map_location='cpu',weights_only=False);step0=resume['step']
   logs=[json.loads(l) for l in (OUT/'optimizer.jsonl').read_text().splitlines() if json.loads(l)['id']==fid];assert len(logs)==step0,'PARTIAL_EPOCH_REQUIRES_AUDIT';assert resume['seal']==sha(OUT/'SEAL.json')
  else:
   assert self.state['attempts']<20;self.state['attempts']+=1;f=dict(id=fid,arm=a,seed=seed,lr=lr,status='RUNNING',updates=0,checkpoints=[],optimizer_seconds=0);self.fits.append(f);self.persist()
  m=self.model(a,seed);init=cpu_state(m);fh=frozen_hash(m);bh=tensor_hash(dict(m.named_buffers()));f['initial_lora_hash']=tensor_hash({n:p for n,p in init.items() if n.startswith('base.')});opt=torch.optim.AdamW(parameters(m).values(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0)
  if resume:
   restore(m,resume['parameters']);opt.load_state_dict(resume['optimizer']);torch.set_rng_state(resume['torch']);torch.cuda.set_rng_state_all(resume['cuda'])
  torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  if step0 in [0,256,512]:self.checkpoint(m,f,step0)
  for step in range(step0,512):
   epoch,pos=divmod(step,64);i=int(rng('ORDER',seed,epoch).permutation(64)[pos]);r=self.update(m,opt,teacher,i,step+1,fid);f['updates']=step+1;f['optimizer_seconds']+=r['seconds']
   if (step+1)%64==0:
    atomic_torch(CACHE/'resume'/f'{fid}.pt',dict(step=step+1,parameters=cpu_state(m),optimizer=opt.state_dict(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),seal=sha(OUT/'SEAL.json')));self.persist()
   if step+1 in [256,512]:self.checkpoint(m,f,step+1)
  assert frozen_hash(m)==fh and tensor_hash(dict(m.named_buffers()))==bh
  f.update(status='COMPLETE',seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),trainable_parameters=sum(p.numel() for p in parameters(m).values()),frozen_unchanged=True,buffers_unchanged=True)
  self.state['completed']+=1;self.persist();del m,opt,init;cleanup();print('FIT',fid,'COMPLETE',self.state['completed'],flush=True)
 def best(self,f):
  return dict(id=f['id'],arm=f['arm'],seed=f['seed'],lr=f['lr'],selected=min(f['checkpoints'],key=lambda c:(c['accuracy'],c['step'])),fixed512=f['checkpoints'][-1])
 def selections(self,seed):
  return [min([self.best(f) for f in self.fits if f['arm']==a and f['seed']==seed and f['status']=='COMPLETE'],key=lambda s:(s['selected']['accuracy'],s['lr'],s['selected']['step'])) for a in ARMS]
 def evaluate(self,selections):
  manifest=read(OUT/'predictions_manifest.json') if (OUT/'predictions_manifest.json').exists() else []
  checks=read(OUT/'restore.json') if (OUT/'restore.json').exists() else []
  for s in selections:
   m=self.model(s['arm'],s['seed'])
   for policy in ['selected','fixed512']:
    cp=s[policy];assert sha(ROOT/cp['path'])==cp['sha256'];restore(m,torch.load(ROOT/cp['path'],map_location='cpu',weights_only=True))
    with torch.no_grad():q=m(self.d.packet('V_SELECT',0)).double().cpu().numpy()
    expected=np.load(ROOT/cp['prediction'])['pred'][0];assert np.array_equal(q,expected),'RESTORE_V'
    if not any(r['id']==s['id'] and r['policy']==policy for r in checks):checks.append(dict(id=s['id'],policy=policy,restore_bitwise=True))
    pred=self.predict(m,'E_DISCOVERY',s['arm']+'_'+cp['parameter_hash'])
    row=dict(arm=s['arm'],seed=s['seed'],policy=policy,selected_step=cp['step'],path=str(pred.relative_to(ROOT)),sha256=sha(pred))
    if not any(r['arm']==s['arm'] and r['seed']==s['seed'] and r['policy']==policy for r in manifest):manifest.append(row)
    save(OUT/'predictions_manifest.json',manifest);save(OUT/'restore.json',checks)
   del m;cleanup()
 def benchmark(self,selections):
  # Inference only: no hidden extra training, identical4origins and cold model lifecycle.
  path=OUT/'inference_resources.json'
  rows=read(path) if path.exists() else []
  old=read(OUT/'REUSE_RECEIPT.json')['selections']
  ss=[dict(s,arm={'B0':'SHORT','B1':'LONG'}[s['arm']]) for s in old]+selections
  for s in sorted(ss,key=lambda s:(s['seed'],s['arm'])):
   if any(r['arm']==s['arm'] and r['seed']==s['seed'] for r in rows):continue
   self.guard.boundary();m=self.model(s['arm'],s['seed']);restore(m,torch.load(ROOT/s['selected']['path'],map_location='cpu',weights_only=True))
   with torch.no_grad():
    for i in range(2):m(self.d.packet('V_SELECT',i))
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
    for j in range(3):
     for i in range(4):
      self.guard.boundary();torch.cuda.synchronize();start=time.perf_counter();m(self.d.packet('V_SELECT',i));torch.cuda.synchronize();times.append(time.perf_counter()-start)
   rows.append(dict(arm=s['arm'],seed=s['seed'],median_seconds=float(np.median(times)),p95_seconds=float(np.quantile(times,.95)),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),measured_forwards=12,encoder_context_tokens=84 if s['arm']=='LONG' else 21 if s['arm'] in ['SHORT','STATS_SHORT'] else 42,extra_optimizer_updates=0))
   save(path,rows);del m;cleanup()
 def execute(self):
  self.guard.boundary(startup=True);teacher=self.teacher();self.smoke(teacher)
  for a in ARMS:
   for lr in LRS:self.fit(a,73100,lr,teacher)
  choices=self.selections(73100);save(OUT/'LR_selection.json',dict(selections=choices,uses_E=False));rates={s['arm']:s['lr'] for s in choices}
  for seed in [73101,73102]:
   for a in ARMS:self.fit(a,seed,rates[a],teacher)
  selections=self.selections(73101)+self.selections(73102)
  if not (OUT/'selections.json').exists():save(OUT/'selections.json',dict(selections=selections,uses_E=False,at=time.time()))
  if not (OUT/'EVALUATION_SEAL.json').exists():save(OUT/'EVALUATION_SEAL.json',dict(at=time.time(),selection_hash=sha(OUT/'selections.json'),LR_hash=sha(OUT/'LR_selection.json'),protocol_hash=sha(OUT/'SEAL.json'),new_E_labels_opened=False))
  assert sha(OUT/'selections.json')==read(OUT/'EVALUATION_SEAL.json')['selection_hash']
  self.evaluate(selections);self.benchmark(selections)
  self.state.update(status='FINISHED',finished_at=time.time());self.persist()
def run():
 if (OUT/'state.json').exists() and read(OUT/'state.json')['status']=='FINISHED':print('FINISHED; no duplicate execution');return
 r=Runner()
 try:r.execute()
 except BaseException as e:
  r.state.update(status='PAUSED_RESOURCE' if isinstance(e,ResourceError) else 'EXECUTION_ERROR',error=repr(e),traceback=traceback.format_exc());r.persist();raise
 finally:r.guard.close()
 from .analysis import verify,report
 verify();report()
