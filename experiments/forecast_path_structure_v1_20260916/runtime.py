"""Fixed FP32 native Chronos-2 LoRA; a single GPU guard, explicit counters and resume."""
import math,random,os,sys,time,subprocess,traceback
import torch
from torch import nn
from .common import *
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from experiments.peft_rank12_20260915.common import Watch,parameters,cpu_state,restore,tensor_hash,frozen_hash,cleanup,ResourceError
from tsfm_peft_screen.backbone import load_base,MODEL_ID,REVISION,QUANTILES
from tsfm_peft_screen.lora import MODULES
from chronos import Chronos2Pipeline
from chronos.chronos2.preprocess import from_list_of_dicts
class LoRA(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base;self.a=nn.Parameter(torch.empty(1,base.in_features,device=base.weight.device));nn.init.kaiming_uniform_(self.a,a=math.sqrt(5));self.b=nn.Parameter(torch.zeros(base.out_features,1,device=base.weight.device))
 def forward(self,x):return self.base(x)+2*nn.functional.linear(nn.functional.linear(x,self.a),self.b)
def make(seed):
 m=load_base()
 assert len(MODULES)==96 and list(m.chronos_config.quantiles)==list(TAUS)
 for i,name in enumerate(MODULES):
  torch.manual_seed(seed+1000+i);parent,leaf=name.rsplit('.',1);p=m.get_submodule(parent);setattr(p,leaf,LoRA(getattr(p,leaf)))
 ps=parameters(m);assert len(ps)==192 and sum(p.numel() for p in ps.values())==147456
 assert all(n.endswith('.a') or n.endswith('.b') for n in ps);return m

def configure():
 torch.set_num_threads(4);torch.manual_seed(SEEDS[0]);np.random.seed(SEEDS[0]);random.seed(SEEDS[0]);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
def tensors(item,y=None):
 a=from_list_of_dicts([item],prediction_length=24)[0];c=a['context'].to('cuda');f=a['future_covariates'].to('cuda');n=len(c)
 assert c.shape==(n,336) and f.shape==(n,24) and n in [1,4]
 labels=None
 if y is not None:assert n==4;labels=torch.full_like(f,float('nan'));labels[0]=torch.as_tensor(y,dtype=torch.float32,device='cuda')
 return dict(context=c,group_ids=torch.zeros(n,dtype=torch.long,device='cuda'),future_covariates=f,num_output_patches=2,future_target=labels)
def forward(m,item,y=None):
 v=m(**tensors(item,y));assert v.quantile_preds.shape==(len(item.get('past_covariates',{}))+1,21,32) and torch.isfinite(v.quantile_preds).all(),'NONFINITE_PREDICTION'
 if y is not None and not torch.isfinite(v.loss):raise FloatingPointError('NONFINITE_LOSS')
 return v

def atomic_torch(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(x,tmp);tmp.replace(p)
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def load_rng(r):random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
def source_files():
 import chronos.chronos2.model as mm,chronos.chronos2.preprocess as pp,chronos.chronos2.pipeline as pipe
 paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'scripts/run_forecast_path_structure.py',ROOT/'src/tsfm_peft_screen/backbone.py',ROOT/'src/tsfm_peft_screen/lora.py',ROOT/'experiments/peft_rank12_20260915/common.py',ROOT/'scripts/priority12/common.py',Path(mm.__file__),Path(pp.__file__),Path(pipe.__file__)]
 return {str(p):sha(p) for p in paths}

class Controller:
 def __init__(self,resume=False):
  self.data=read(OUT/'data_seal.json');self.jobs=self.data['jobs'];self.targets=self.data['targets'];self.scales=read(OUT/'train_scaling.json');self.schedule=dict(np.load(CACHE/'augmentation_schedule.npz'))
  assert self.targets,'NO_PREPARED_TARGETS'
  for p,h in self.data['files'].items():assert sha(ROOT/p)==h,('DATA_CHANGED',p)
  for j in self.jobs:
   assert sha(ROOT/j['input'])==j['input_sha256']
   if 'label' in j:assert sha(ROOT/j['label'])==j['label_sha256']
  self.x={j['id']:dict(np.load(ROOT/j['input'])) for j in self.jobs}
  self.ys={j['id']:np.load(ROOT/j['label']) for j in self.jobs if j['role']!='TEST'}
  if resume:
   self.state=read(OUT/'status.json');assert self.state['status'] not in ('EXECUTION_COMPLETE','PREDICTIONS_COMPLETE')
   self.fits=read(OUT/'fits.json') if (OUT/'fits.json').exists() else []
   self.preds=read(OUT/'prediction_manifest.json') if (OUT/'prediction_manifest.json').exists() else []
   sealed=read(OUT/'implementation_seal.json')
   for p,h in sealed['files'].items():assert sha(p)==h,('IMPLEMENTATION_CHANGED',p)
   if (OUT/'pending_update.json').exists():raise RuntimeError('AMBIGUOUS_UPDATE_NO_AUTOMATIC_REPLAY')
  else:
   assert not (OUT/'status.json').exists(),'Use status / exact resume; do not duplicate'
   self.state=dict(status='RUNNING',phase='SMOKE',at=time.time(),main_updates=0,smoke_updates=0,resource_updates=0,attempts=0,completed_fits=0,train_forwards=0,eval_forwards=0,verify_forwards=0,errors=[])
   self.fits=[];self.preds=[]
   snapshot=Path.home()/'.cache/huggingface/hub/models--amazon--chronos-2/snapshots'/REVISION
   save(OUT/'implementation_seal.json',dict(at=time.time(),files=source_files(),model_files={str(p):sha(p) for p in snapshot.iterdir() if p.is_file()},model_id=MODEL_ID,revision=REVISION,quantiles=QUANTILES,modules=list(MODULES),environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__),data_seal_sha256=sha(OUT/'data_seal.json')))
  self.w=None;self.countphase='verify';self.replays=read(OUT/'replay.json') if (OUT/'replay.json').exists() else []
 def persist(self):
  save(OUT/'status.json',self.state);save(OUT/'fits.json',self.fits);save(OUT/'prediction_manifest.json',self.preds)
  csvwrite(OUT/'fit_manifest.csv',[{k:v for k,v in f.items() if k!='checkpoints'} for f in self.fits])
 def counted(self,m):
  def hook(*args):self.state[self.countphase+'_forwards']+=1
  m.register_forward_hook(hook);return m
 def item(self,j,k=0,**kw):return payload(self.x[j['id']],self.scales[j['target']],k,**kw)
 def predict(self,m,j,k=0,history=False):
  self.w.boundary()
  with torch.no_grad():q=forward(m,self.item(j,k,history=history)).quantile_preds[0,:,:24].double().cpu().numpy()
  return q
 def predictions(self,m,target,role,weight_key,history=False):
  existing=next((r for r in self.preds if r['target']==target and r['role']==role and r['weight_key']==weight_key and r['history']==history),None)
  if existing:
   assert sha(ROOT/existing['path'])==existing['sha256'];return existing
  jobs=[j for j in self.jobs if j['target']==target and j['role']==role]
  self.countphase='eval';t0=time.perf_counter();qs=[]
  for j in jobs:
   if history:one=self.predict(m,j,history=True);q=np.repeat(one[None],4,axis=0)
   else:q=np.stack([self.predict(m,j,k) for k in range(4)])
   qs.append(q)
  p=CACHE/'predictions'/f'{target}_{role}_{weight_key}_{int(history)}.npz';p.parent.mkdir(exist_ok=True)
  np.savez_compressed(p,raw=np.stack(qs),ids=np.array([j['id'] for j in jobs]))
  r=dict(target=target,role=role,weight_key=weight_key,history=history,path=str(p.relative_to(ROOT)),sha256=sha(p),count=len(jobs),forwards=len(jobs)*(1 if history else 4),seconds=time.perf_counter()-t0,at=time.time())
  self.preds.append(r);self.persist();return r
 def update(self,m,opt,j,arm,seed,e,i,step,fid,smoke=False):
  self.w.boundary();self.countphase='train';key='smoke_updates' if smoke else 'main_updates'
  assert self.state['smoke_updates']<=24 and self.state['main_updates']<=24576
  assert self.state[key]<(24 if smoke else self.data['planned_updates'])
  pref=f"{j['target']}_{seed}";off=self.schedule[pref+'_offsets'][e//4,i];mask=self.schedule[pref+'_masks'][e,i] if arm=='DROP' else None
  item=self.item(j,arm=arm,epoch=e,offsets=off,mask=mask);y=self.ys[j['id']]
  save(OUT/'pending_update.json',dict(id=fid,step=step,smoke=smoke,at=time.time()))
  torch.cuda.synchronize();start=time.perf_counter();wallstart=time.time();opt.zero_grad(set_to_none=True);v=forward(m,item,y);v.loss.backward();ps=parameters(m)
  if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps.values()):raise FloatingPointError('NONFINITE_OR_MISSING_GRADIENT')
  norm=torch.nn.utils.clip_grad_norm_(list(ps.values()),1.,error_if_nonfinite=True);opt.step();self.state[key]+=1
  if not all(torch.isfinite(p).all() for p in ps.values()):raise FloatingPointError('NONFINITE_UPDATED_PARAMETER')
  torch.cuda.synchronize();seconds=time.perf_counter()-start;contam=any(r['busy'] for r in self.w.events if r['at']>=wallstart)
  r=dict(id=fid,target=j['target'],arm=arm,seed=seed,step=step,epoch=e,origin_id=j['id'],loss=float(v.loss.detach()),gradnorm=float(norm),seconds=seconds,smoke=smoke,contaminated=contam,free_mib=self.w.latest['free_mib'])
  with open(OUT/'update_log.jsonl','a') as f:f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
  (OUT/'pending_update.json').unlink();self.w.limits();return r
 def optimizer(self,m,lr):return torch.optim.AdamW(parameters(m).values(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0)
 def smoke(self):
  if (OUT/'smoke.json').exists():assert len(read(OUT/'smoke.json'))==4*len(self.targets);return
  records=[]
  for t in self.targets:
   tid=t['target'];j=next(j for j in self.jobs if j['target']==tid and j['role']=='TRAIN');item=self.item(j);sigma=self.scales[tid]['sigma_y']
   self.countphase='verify';base=self.counted(load_base());native=self.predict(base,j)
   with torch.no_grad():
    p=Chronos2Pipeline(base).predict([item],prediction_length=24,context_length=336,batch_size=4,cross_learning=False)[0][0].double().cpu().numpy()
    ordinary=forward(base,item,self.ys[j['id']]).quantile_preds.detach().clone();poison=forward(base,item,self.ys[j['id']]+12345678.).quantile_preds
   pipeline_error=float(np.max(abs(native-p))/sigma);assert pipeline_error<=1e-5;assert torch.equal(ordinary,poison),'LABEL_LEAKAGE'
   # Actual native supervision: only load's 24/32 positions, unchanged quantile reduction.
   a=tensors(item,self.ys[j['id']]);_,locscale=base.instance_norm(a['context']);yy,_=base.instance_norm(a['future_target'],locscale)
   mask=torch.zeros(4,2,16,device='cuda');mask[1:]=1;z=torch.zeros(4,21,32,device='cuda',requires_grad=True)
   loss=base._compute_loss(z,a['future_target'],None,mask,locscale,2);grad=torch.autograd.grad(loss,z)[0]
   yn=yy[0].double().cpu().numpy();taus=base.quantiles.double().cpu().numpy();expected=sum(2*(tau*y if y>=0 else (tau-1)*y) for tau in taus for y in yn)/(4*32)
   assert abs(float(loss)-expected)/max(1,abs(expected))<=1e-5
   assert not torch.count_nonzero(grad[1:]) and not torch.count_nonzero(grad[0,:,24:])
   del base,p,ordinary,poison,a,z,grad,loss;cleanup();initialhash=None
   for arm in ARMS:
    m=self.counted(make(SEEDS[0]));initial=cpu_state(m);ih=tensor_hash(initial)
    if initialhash is None:initialhash=ih
    assert ih==initialhash
    fh=frozen_hash(m);bh=tensor_hash(dict(m.named_buffers()));q=self.predict(m,j);err=float(np.max(abs(q-native))/sigma);assert err<=1e-5
    opt=self.optimizer(m,LRS[0]);curve=[]
    for step in range(2):curve.append(self.update(m,opt,j,arm,SEEDS[0],0,0,step+1,f'SMOKE_{tid}_{arm}',True))
    changed=[n for n,p in parameters(m).items() if not torch.equal(p.detach().cpu(),initial[n])];assert changed
    assert frozen_hash(m)==fh and tensor_hash(dict(m.named_buffers()))==bh
    cp=CACHE/'smoke'/f'{tid}_{arm}.pt';atomic_torch(cp,cpu_state(m));q=self.predict(m,j)
    del m,opt;cleanup();m=self.counted(make(SEEDS[0]));restore(m,torch.load(cp,map_location='cpu',weights_only=True));replay=self.predict(m,j)
    re=float(np.max(abs(q-replay))/sigma);assert re<=1e-5
    records.append(dict(target=tid,arm=arm,updates=2,initial_hash=ih,initial_normalized_maxabs=err,native_pipeline_normalized_maxabs=pipeline_error,native_loss_mask_verified=True,label_poison_invariant=True,changed_tensors=len(changed),frozen_unchanged=True,buffers_unchanged=True,restore_normalized_maxabs=re,restore_exact=bool(np.array_equal(q,replay)),checkpoint_sha256=sha(cp)))
    save(OUT/'smoke_progress.json',records);self.persist();del m;cleanup()
   print('SMOKE_COMPLETE',tid,flush=True)
  save(OUT/'smoke.json',records)
 def checkpoint(self,m,f,step):
  old=next((c for c in f['checkpoints'] if c['step']==step),None)
  if old:return old
  cp=CACHE/'checkpoints'/f"{f['id']}_{step}.pt";atomic_torch(cp,cpu_state(m));h=tensor_hash(cpu_state(m));weight_key='FROZEN_WEATHER' if step==0 else h
  r=self.predictions(m,f['target'],'V_SELECT',weight_key);z=np.load(ROOT/r['path']);raw=z['raw'];jobs=[j for j in self.jobs if j['target']==f['target'] and j['role']=='V_SELECT'];sigma=self.scales[f['target']]['sigma_y']
  scores=[metric(np.sort(raw[i,k],axis=0),self.ys[j['id']],sigma)['primary'] for i,j in enumerate(jobs) for k in range(4)]
  c=dict(step=step,path=str(cp.relative_to(ROOT)),sha256=sha(cp),parameter_hash=h,weight_key=weight_key,validation_primary=float(np.mean(scores)),prediction=r['path'],at=time.time());f['checkpoints'].append(c);self.persist()
  print('CHECKPOINT',f['id'],step,round(c['validation_primary'],6),flush=True);return c
 def train(self):
  self.state['phase']='TRAIN';self.persist()
  for t in self.targets:
   tid=t['target'];tr=[j for j in self.jobs if j['target']==tid and j['role']=='TRAIN']
   for seed in SEEDS:
    for lr in LRS:
     for arm in ARMS:
      fid=f'{tid}_{arm}_{seed}_{lr:.0e}';f=next((a for a in self.fits if a['id']==fid),None)
      if f and f['status']=='COMPLETE':continue
      if f and f['status']=='INCOMPLETE':continue
      startstep=0;resume=None
      if f:
       cp=CACHE/'resume'/f'{fid}.pt';assert cp.exists(),'MISSING_RESUME';resume=torch.load(cp,map_location='cpu',weights_only=False);startstep=resume['step']
       logs=[json.loads(l) for l in (OUT/'update_log.jsonl').read_text().splitlines() if json.loads(l)['id']==fid]
       assert len(logs)==startstep and (not logs or logs[-1]['step']==startstep),'PARTIAL_EPOCH_NO_EXACT_RESUME'
      else:
       assert self.state['attempts']<self.data['planned_fits'];self.state['attempts']+=1
       f=dict(id=fid,target=tid,arm=arm,seed=seed,lr=lr,status='RUNNING',updates=0,checkpoints=[],optimizer_seconds=0.,started_at=time.time());self.fits.append(f)
       with open(OUT/'fit_attempts.csv','a') as af:
        if af.tell()==0:af.write('id,at\n')
        af.write(f'{fid},{time.time()}\n')
      self.persist();m=self.counted(make(seed));initial=cpu_state(m);f['initial_hash']=tensor_hash(initial);fh=frozen_hash(m);bh=tensor_hash(dict(m.named_buffers()));opt=self.optimizer(m,lr)
      if resume:restore(m,resume['parameters']);opt.load_state_dict(resume['optimizer']);load_rng(resume['rng'])
      torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
      try:
       if startstep in (0,256,512):self.checkpoint(m,f,startstep)
       for step in range(startstep,512):
        e,pos=divmod(step,64);i=int(self.schedule[f'{tid}_{seed}_order'][e,pos]);j=tr[i]
        row=self.update(m,opt,j,arm,seed,e,i,step+1,fid);f['updates']=step+1;f['optimizer_seconds']+=row['seconds']
        if (step+1)%64==0:
         atomic_torch(CACHE/'resume'/f'{fid}.pt',dict(id=fid,step=step+1,epoch=(step+1)//64,stream_position=0,parameters=cpu_state(m),optimizer=opt.state_dict(),rng=rng(),schedule_sha256=sha(CACHE/'augmentation_schedule.npz'),implementation_sha256=sha(OUT/'implementation_seal.json')));self.persist()
        if step+1 in [256,512]:self.checkpoint(m,f,step+1)
       assert frozen_hash(m)==fh and tensor_hash(dict(m.named_buffers()))==bh
       f.update(status='COMPLETE',seconds=time.perf_counter()-t0,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),changed_tensors=sum(not torch.equal(initial[n],p.detach().cpu()) for n,p in parameters(m).items()),trainable_parameters=sum(p.numel() for p in parameters(m).values()),frozen_unchanged=True,buffers_unchanged=True)
       assert f['changed_tensors']>0;self.state['completed_fits']+=1
      except FloatingPointError as exc:
       f.update(status='INCOMPLETE',error=str(exc));self.state['errors'].append(dict(id=fid,type=type(exc).__name__,message=str(exc)))
       # No retry of an ambiguous failed optimizer step; retain evidence, compare remaining safe arms.
       if (OUT/'pending_update.json').exists():(OUT/'pending_update.json').rename(OUT/f'failed_update_{fid}.json')
      finally:
       self.persist();del m,opt,initial;cleanup()
      print('FIT',fid,f['status'],self.state['completed_fits'],'/',self.data['planned_fits'],flush=True)
 def select(self):
  assert all(f['status']=='COMPLETE' and f['updates']==512 for f in self.fits) and len(self.fits)==self.data['planned_fits'],'INCOMPLETE_COMPARISON_NO_COMPLETE_CLAIM'
  if (OUT/'selection_seal.json').exists():return read(OUT/'selection_seal.json')['selections']
  sels=[];lrrows=[]
  for t in self.targets:
   for arm in ARMS:
    ff=[f for f in self.fits if f['target']==t['target'] and f['arm']==arm]
    best={f['id']:min(f['checkpoints'],key=lambda c:(c['validation_primary'],c['step'])) for f in ff}
    scores={lr:np.mean([best[f['id']]['validation_primary'] for f in ff if f['lr']==lr]) for lr in LRS};lr=min(LRS,key=lambda x:(scores[x],x))
    lrrows += [dict(target=t['target'],arm=arm,lr=l,mean_best_validation=float(v),selected=l==lr) for l,v in scores.items()]
    for f in ff:
     if f['lr']==lr:sels.append(dict(id=f['id'],target=f['target'],arm=arm,seed=f['seed'],lr=lr,selected=best[f['id']],fixed512=f['checkpoints'][-1],budget_limited=best[f['id']]['step']==512))
  csvwrite(OUT/'lr_selection.csv',lrrows);save(OUT/'selection_seal.json',dict(at=time.time(),selections=sels,fit_sha256=sha(OUT/'fits.json'),implementation_sha256=sha(OUT/'implementation_seal.json'),rule='Per target/arm shared LR minimizing mean two-seed best V(S0..S3); ties lower LR and earlier step'))
  return sels
 def calibrate_and_test(self,sels):
  self.state['phase']='CALIBRATE';self.persist();cal=[];roles=[]
  # Each role references a weight key. Duplicate selected/fixed/INIT predictions are not re-run.
  for t in self.targets:
   for name,history in [('FROZEN_WEATHER',False),('FROZEN_HISTORY',True)]:roles.append(dict(id=f"{t['target']}_{name}",target=t['target'],arm=name,seed=-1,policy='SELECTED',weight_key=name,history=history,checkpoint=None))
  for s in sels:
   for policy,key in [('SELECTED','selected'),('FIXED512','fixed512')]:roles.append(dict(id=s['id'],target=s['target'],arm=s['arm'],seed=s['seed'],policy=policy,weight_key=s[key]['weight_key'],history=False,checkpoint=s[key]))
  for r in roles:
   if r['policy']!='SELECTED':continue
   m=self.counted(load_base() if r['checkpoint'] is None else make(r['seed']))
   if r['checkpoint'] is not None:assert sha(ROOT/r['checkpoint']['path'])==r['checkpoint']['sha256'];restore(m,torch.load(ROOT/r['checkpoint']['path'],map_location='cpu',weights_only=True))
   if r['checkpoint'] is not None:
    # Fresh restored model, fixed V inputs; no TEST labels used.
    j=next(j for j in self.jobs if j['target']==r['target'] and j['role']=='V_SELECT');self.countphase='verify';q=self.predict(m,j);old=np.load(ROOT/r['checkpoint']['prediction'])['raw'][0,0];err=float(np.max(abs(q-old))/self.scales[r['target']]['sigma_y']);assert err<=1e-5
    self.replays.append(dict(id=r['id'],phase='selected_restore',normalized_maxabs=err,exact=bool(np.array_equal(q,old))));save(OUT/'replay.json',self.replays)
   p=self.predictions(m,r['target'],'V_CALIBRATE',r['weight_key'],r['history']);raw=np.load(ROOT/p['path'])['raw'];ys=np.stack([self.ys[j['id']] for j in self.jobs if j['target']==r['target'] and j['role']=='V_CALIBRATE'])
   delta=calibrate(np.sort(raw,axis=-2),ys)
   cal.append(dict(id=r['id'],target=r['target'],arm=r['arm'],seed=r['seed'],delta=delta.tolist(),prediction=p,calibration_role='V_CALIBRATE ONLY'));del m;cleanup()
  save(OUT/'calibration_parameters.json',cal)
  # Seal code, choices, calibration, origins, evaluation rules BEFORE test prediction/label scoring.
  save(OUT/'evaluation_seal.json',dict(at=time.time(),roles=roles,selection_sha256=sha(OUT/'selection_seal.json'),calibration_sha256=sha(OUT/'calibration_parameters.json'),data_sha256=sha(OUT/'data_seal.json'),implementation_sha256=sha(OUT/'implementation_seal.json'),source_files=source_files(),test_labels_opened=False,scoring_rule='RAW primary; common 21-constant calibrated audit; S0..S3 equal weight; target/seed equal weight'))
  self.state['phase']='TEST_PREDICT';self.persist()
  for r in roles:
   m=self.counted(load_base() if r['checkpoint'] is None else make(r['seed']))
   if r['checkpoint'] is not None:restore(m,torch.load(ROOT/r['checkpoint']['path'],map_location='cpu',weights_only=True))
   before=frozen_hash(m);bh=tensor_hash(dict(m.named_buffers()))
   self.predictions(m,r['target'],'TEST',r['weight_key'],r['history'])
   assert frozen_hash(m)==before and tensor_hash(dict(m.named_buffers()))==bh
   del m;cleanup();print('TEST_PREDICTION',r['id'],r['policy'],flush=True)
  save(OUT/'test_prediction_seal.json',dict(at=time.time(),manifest_sha256=sha(OUT/'prediction_manifest.json'),evaluation_sha256=sha(OUT/'evaluation_seal.json'),predictions=[p for p in self.preds if p['role']=='TEST'],test_labels_opened=False))
 def run(self):
  try:
   configure();self.w=Watch(OUT,'controller',wall_cap=14400);self.w.boundary(startup=True);self.smoke();self.train();sels=self.select();self.calibrate_and_test(sels);self.state.update(status='PREDICTIONS_COMPLETE',phase='CPU_SCORE_VERIFY')
  except BaseException as exc:
   self.state.update(status='PARTIAL_EXECUTION',errors=self.state['errors']+[dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())]);print(traceback.format_exc(),flush=True)
  finally:
   if self.w:self.w.close()
   self.persist()
  return self.state['status']=='PREDICTIONS_COMPLETE'
