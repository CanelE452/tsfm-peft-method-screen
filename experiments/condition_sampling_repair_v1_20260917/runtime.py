"""Bounded single-worker training, atomic epoch resumes and independent candidate progression."""
import os,random,sys,time,traceback,shutil,subprocess
import torch
import psutil
from .common import *
from .model import *
from experiments.peft_rank12_20260915.common import Watch as ExistingWatch,ResourceError
from tsfm_peft_screen.backbone import MODEL_ID,REVISION
class Guard(ExistingWatch):
 def limits(self):
  if self.error:raise ResourceError('GPU_MONITOR_ERROR '+self.error)
  if time.time()-self.budget['started_at']>86400:raise ResourceError('PAUSED_BUDGET controller24h')
  if self.wait+self.budget['wait_seconds']>=1800:raise ResourceError('PAUSED_BUDGET wait1800s')
  if psutil.virtual_memory().available<2*2**30:raise ResourceError('RAM_BELOW_2GIB')
  if shutil.disk_usage(ROOT).free<10*2**30:raise ResourceError('DISK_BELOW_10GIB')
def atomic_torch(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');torch.save(x,q);q.replace(p)
def random_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(v):random.setstate(v['python']);np.random.set_state(v['numpy']);torch.set_rng_state(v['torch']);torch.cuda.set_rng_state_all(v['cuda'])

def primary(t,p,y,sigma,conds):
 # p = origin, condition, channel, H; R04 p has an extra early/late axis.
 if t=='R04':
  yy=np.stack([y[:,:,:48],y[:,:,24:72]],1);e=(p[:,0]-yy)/sigma[None,None,:,None];acc=float(np.sqrt(np.mean(e*e,axis=(0,1,3))).mean());rev=(p[:,0,1,:,:24]-p[:,0,0,:,24:])/sigma[None,:,None];return dict(accuracy=acc,revision=float(np.sqrt(np.mean(rev**2,axis=(0,2))).mean()))
 vals=[]
 for j,c in enumerate(conds):
  if t=='N01':
   if not c.startswith(('d6_','d24_')):continue
   ch=int(c.split('_c')[1]);vals.append(float(np.sqrt(np.mean(((p[:,j,ch]-y[:,ch])/sigma[ch])**2))))
  elif t=='N03':
   want=['delta1','delta4'] if set(conds)=={'delta1','delta4'} else ['delta2','irregular']
   if c in want:vals.append(nrms(p[:,j],y,sigma))
  else:vals.append(nrms(p[:,j],y,sigma))
 return dict(accuracy=float(np.mean(vals)))

class Controller:
 def __init__(self,resume=False):
  self.man=read(OUT/'MASTER_MANIFEST.json');self.seal=read(OUT/'MASTER_SEAL.json')
  def audited_hash(path,expected):
   current=sha(ROOT/path)
   if current!=expected:
    chain=read(OUT/'POST_SEAL_CORRECTIONS.json') if (OUT/'POST_SEAL_CORRECTIONS.json').exists() else []
    for amendment in chain:
     change=amendment.get('files',{}).get(path)
     if change and change['before_sha256']==expected:expected=change['after_sha256']
    assert current==expected,('SEALED_CHANGE_WITHOUT_AUDITED_CHAIN',path)
  for p,h in self.seal['files'].items():audited_hash(p,h)
  for p,h in self.seal['implementation'].items():audited_hash(p,h)
  self.state=read(OUT/'controller_state.json') if (OUT/'controller_state.json').exists() else dict(main_updates=0,smoke_updates=0,forwards=dict(train=0,eval=0,verify=0),attempts=0,completed_fits=0,started_at=time.time(),status='RUNNING')
  self.phase='verify';self.guard=None;self.track=None
  if (OUT/'pending_update.json').exists():raise RuntimeError('AMBIGUOUS_UPDATE_NO_AUTOMATIC_REPLAY')
  self.persist()
 def persist(self):save(OUT/'controller_state.json',self.state)
 def make(self,t,a,seed,aux,frozen=False):
  m=Adapted(t,a,seed,aux,frozen=frozen)
  def hook(*args):
   if sum(self.state['forwards'].values())>=200000:raise ResourceError('PAUSED_BUDGET forward200000')
   self.state['forwards'][self.phase]+=1
  m.base.register_forward_pre_hook(hook);return m
 def check(self):
  self.guard.boundary()
  if self.state['main_updates']>=49152 or self.state['smoke_updates']>48:raise ResourceError('PAUSED_BUDGET updates')
 def predict(self,m,data,role,key):
  file=CACHE/data.t/'predictions'/f'{role}_{key}.npz';meta=file.with_suffix('.json')
  if file.exists():assert meta.exists() and sha(file)==read(meta)['sha256'];return file
  self.phase='eval';pp=[];conds=conditions(data.t,role);start=time.perf_counter()
  for i in range(len(data.inputs[role]['origins'])):
   values=[];local={}
   for cond in conds:
    self.guard.boundary();p=data.packet(role,i,condition=cond)
    with torch.no_grad():
     if data.t=='R04':q=torch.stack([m(p),m(data.packet(role,i,condition=cond,late=True))]).double().cpu().numpy()
     elif data.t=='N01' and cond.startswith('d0_') and 'd0' in local:q=local['d0']
     else:q=m(p).double().cpu().numpy()
    if data.t=='N01' and cond.startswith('d0_'):local['d0']=q
    values.append(q)
   pp.append(values)
  npz(file,pred=np.array(pp),origins=data.inputs[role]['origins'],conditions=np.array(conds));save(meta,dict(sha256=sha(file),seconds=time.perf_counter()-start,rows=len(pp),phase=role,weight_key=key));self.persist();return file
 def frozen_cache(self,data):
  path=CACHE/data.t/'frozen.npz';meta=OUT/data.t/'frozen_parameters.json'
  if path.exists():assert sha(path)==read(meta)['cache_sha256'];return dict(np.load(path)),read(meta)
  t=data.t;m=self.make(t,ARMS[t][0],73100,data.aux,True);out={};self.phase='verify'
  for role in ROLES:
   early=[];late=[]
   for i in range(len(data.inputs[role]['origins'])):
    self.guard.boundary()
    with torch.no_grad():
     early.append(m(data.packet(role,i)).double().cpu().numpy())
     if t=='R04':late.append(m(data.packet(role,i,late=True)).double().cpu().numpy())
   out[role]=np.array(early)
   if late:out[role+'_late']=np.array(late)
  fixed={}
  if t=='R04':
   s=data.sigma[None,:,None];e=out['TRAIN'];l=out['TRAIN_late'];y=data.labels['TRAIN'];task=.5*np.mean(((e-y[:,:,:48])/s)**2,(1,2))+.5*np.mean(((l-y[:,:,24:72])/s)**2,(1,2));r=np.mean(((l[:,:,:24]-e[:,:,24:])/s)**2,(1,2));v=np.mean(((data.inputs['TRAIN']['innovation_observed']-e[:,:,:24])/s)**2,-1);fixed=dict(lambda_value=float(np.clip(.05*np.median(task[:16])/max(np.median(r[:16]),1e-8),1e-4,1)),mean_w=float(np.mean(1/(1+v))),innovation_upper_quartile=float(np.quantile(v,.75)),lambda_training_pairs=16,task_median=float(np.median(task[:16])),revision_median=float(np.median(r[:16])))
   fixed['lambda']=fixed.pop('lambda_value')
  npz(path,**out);fixed.update(cache_sha256=sha(path),inference_only=True,labels_used_for_parameters='TRAIN only',at=time.time());save(meta,fixed);del m;cleanup();return out,fixed
 def update(self,m,opt,data,i,epoch,fid,step,smoke=False,frozen=None,fixed=None):
  self.check();key='smoke_updates' if smoke else 'main_updates';cap=48 if smoke else 49152;assert self.state[key]<cap
  p=data.packet('TRAIN',i,epoch=epoch);yy=data.label('TRAIN',i);q0=late_q0=None;lp=None
  if data.t=='R04':p=dict(p,innovation_observed=data.inputs['TRAIN']['innovation_observed'][i]);lp=data.packet('TRAIN',i,epoch=epoch,late=True);q0=m.T(frozen['TRAIN'][i]);late_q0=m.T(frozen['TRAIN_late'][i])
  if data.t=='R09':q0=m.T(frozen['TRAIN'][i])
  save(OUT/'pending_update.json',dict(track=data.t,id=fid,step=step,smoke=smoke,at=time.time()));self.phase='train';opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter();wall=time.time();loss,parts=task_loss(m,p,yy,q0,lp,late_q0,fixed);loss.backward();ps=parameters(m)
  if any(v.grad is None or not torch.isfinite(v.grad).all() for v in ps.values()):raise FloatingPointError('MISSING_OR_NONFINITE_GRADIENT')
  extra={n:float(p.grad.norm()) for n,p in m.extra.items()};norm=torch.nn.utils.clip_grad_norm_(list(ps.values()),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();seconds=time.perf_counter()-start
  if any(not torch.isfinite(v).all() for v in ps.values()):raise FloatingPointError('NONFINITE_PARAMETER')
  self.state[key]+=1;row=dict(id=fid,step=step,epoch=epoch,origin=int(data.inputs['TRAIN']['origins'][i]),origin_ordinal=i,loss=float(loss.detach()),gradnorm=float(norm),clip_active=float(norm)>1,extra_grad=extra,loss_parts=parts,seconds=seconds,smoke=smoke,contaminated=any(e['busy'] for e in self.guard.events if e['at']>=wall),free_mib=self.guard.latest['free_mib'])
  if data.t=='N01':row.update(target=i%4,delay=[0,6,24][epoch%3])
  if data.t=='R09':row.update(detailed=p['detailed'])
  if data.t=='N02' and m.arm=='B3':row.update(a_saturation=float((m.extra['a'].detach().abs()>=2).float().mean()))
  with (OUT/data.t/'optimizer_log.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
  self.persist();(OUT/'pending_update.json').unlink();self.guard.limits();return row
 def smoke(self,data,arms,frozen,fixed):
  path=OUT/data.t/'smoke.json';records=read(path) if path.exists() else []
  for arm in arms:
   if any(r['arm']==arm for r in records):continue
   m=self.make(data.t,arm,73100,data.aux);initial=cpu_state(m);fh=frozen_hash(m);bh=tensor_hash(dict(m.named_buffers()));epoch=1 if data.t in ['N01','N03'] else 0;i=0
   if data.t=='R09':i=next(i for i,v in enumerate(data.inputs['TRAIN']['detailed']) if v)
   p=data.packet('TRAIN',i,epoch=epoch);self.phase='verify'
   from chronos.chronos2.preprocess import from_list_of_dicts
   with torch.no_grad():
    v=m.tensors(p);xx=v['context'].cpu().numpy();item=dict(target=xx[:4]);
    if len(xx)>4:item['past_covariates']={f'aux{j:02}':xx[j] for j in range(4,len(xx))}
    if v['future_covariates'] is not None:item['future_covariates']={f'aux{j:02}':v['future_covariates'][j].cpu().numpy() for j in range(4,len(xx))}
    official=from_list_of_dicts([item],prediction_length=p['H'])[0];assert np.array_equal(official['context'].numpy(),xx,equal_nan=True)
    native=m.base(context=official['context'].cuda(),future_covariates=official['future_covariates'].cuda(),group_ids=v['group_ids'],num_output_patches=v['num_output_patches']).quantile_preds[:4,m.mid,:p['H']];q=m(p);err=float(((q-native).abs()/m.T(data.sigma)[:,None]).max());assert err<=1e-5
   opt=torch.optim.AdamW(parameters(m).values(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0);curves=[]
   for s in range(2):curves.append(self.update(m,opt,data,i,epoch,f'SMOKE_{arm}',s+1,True,frozen,fixed))
   for n in m.extra:assert any(v['extra_grad'][n]>0 for v in curves),('DEGENERATE_AUX_GRADIENT',arm,n)
   assert frozen_hash(m)==fh and tensor_hash(dict(m.named_buffers()))==bh;changed=sum(not torch.equal(initial[n],p.cpu()) for n,p in parameters(m).items());assert changed
   self.phase='verify'
   with torch.no_grad():q=m(p).double().cpu().numpy()
   cp=CACHE/data.t/'smoke'/f'{arm}.pt';atomic_torch(cp,cpu_state(m));count=sum(v.numel() for v in parameters(m).values());extras={n:p.detach().cpu().tolist() for n,p in m.extra.items()};del m,opt;cleanup();m=self.make(data.t,arm,73100,data.aux);restore(m,torch.load(cp,map_location='cpu',weights_only=True))
   with torch.no_grad():qq=m(p).double().cpu().numpy()
   re=float(np.max(abs(q-qq)/data.sigma[:,None]));assert re<=1e-5
   records.append(dict(arm=arm,updates=2,trainable_parameters=count,extra_parameters=sum(np.size(v) for v in extras.values()),extra_gradients=[v['extra_grad'] for v in curves],extra_after=extras,frozen_unchanged=True,buffers_unchanged=True,changed_tensors=changed,official_preprocess_exact=True,native_initial_error=err,restore_error=re,restore_bitwise=bool(np.array_equal(q,qq)),median_slot=m.mid));save(path,records);del m;cleanup();print('SMOKE',data.t,arm,flush=True)
  return records
 def checkpoint(self,m,data,fit,step):
  existing=next((c for c in fit['checkpoints'] if c['step']==step),None)
  if existing:return existing
  cp=CACHE/data.t/'checkpoints'/f"{fit['id']}_{step}.pt";atomic_torch(cp,cpu_state(m));h=tensor_hash(cpu_state(m));pred=self.predict(m,data,'V_SELECT',fit['arm']+'_'+h);z=np.load(pred);scores=primary(data.t,z['pred'],data.labels['V_SELECT'],data.sigma,list(z['conditions']));v=dict(step=step,path=str(cp.relative_to(ROOT)),sha256=sha(cp),parameter_hash=h,prediction=str(pred.relative_to(ROOT)),**scores);fit['checkpoints'].append(v);save(OUT/data.t/'fits.json',self.fits);print('CHECKPOINT',data.t,fit['id'],step,round(scores['accuracy'],6),flush=True);return v
 def train_fit(self,data,arm,seed,lr,frozen,fixed):
  fid=f'{arm}_{seed}_{lr:.0e}';fit=next((f for f in self.fits if f['id']==fid),None)
  if fit and fit['status'] in ['COMPLETE','NUMERICAL_ERROR','INVALID_CONSTRUCT']:return fit
  resume=None;step0=0
  if fit:
   resume=torch.load(CACHE/data.t/'resume'/f'{fid}.pt',map_location='cpu',weights_only=False);step0=resume['step'];logs=[json.loads(l) for l in (OUT/data.t/'optimizer_log.jsonl').read_text().splitlines() if json.loads(l)['id']==fid];assert len(logs)==step0,'PARTIAL_EPOCH_NO_AUTOMATIC_REPLAY';assert resume['seal_sha256']==sha(OUT/'MASTER_SEAL.json')
  else:
   assert self.state['attempts']<96;self.state['attempts']+=1;fit=dict(id=fid,arm=arm,seed=seed,lr=lr,status='RUNNING',updates=0,checkpoints=[],optimizer_seconds=0,started_at=time.time());self.fits.append(fit)
  save(OUT/data.t/'fits.json',self.fits);m=self.make(data.t,arm,seed,data.aux);fh=frozen_hash(m);bh=tensor_hash(dict(m.named_buffers()));init=cpu_state(m);fit['initial_lora_hash']=tensor_hash({n:p for n,p in init.items() if n.startswith('base.')});opt=torch.optim.AdamW(parameters(m).values(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0)
  if resume:restore(m,resume['parameters']);opt.load_state_dict(resume['optimizer']);restore_rng(resume['rng'])
  torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  try:
   if step0 in [0,256,512]:self.checkpoint(m,data,fit,step0)
   for step in range(step0,512):
    epoch,pos=divmod(step,64);order=rng('ORDER',seed,epoch).permutation(64)
    if arm=='I0':d=np.flatnonzero(data.inputs['TRAIN']['detailed']);order=np.tile(rng('ORDER',seed,epoch).permutation(d),4)
    i=int(order[pos]);r=self.update(m,opt,data,i,epoch,fid,step+1,False,frozen,fixed);fit['updates']=step+1;fit['optimizer_seconds']+=r['seconds']
    if (step+1)%64==0:
     atomic_torch(CACHE/data.t/'resume'/f'{fid}.pt',dict(step=step+1,parameters=cpu_state(m),optimizer=opt.state_dict(),rng=random_state(),seal_sha256=sha(OUT/'MASTER_SEAL.json')));save(OUT/data.t/'fits.json',self.fits);self.persist()
     if sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())>100*2**30:raise ResourceError('PAUSED_BUDGET cache100GiB')
    if step+1 in [256,512]:self.checkpoint(m,data,fit,step+1)
   assert frozen_hash(m)==fh and tensor_hash(dict(m.named_buffers()))==bh
   fit.update(status='COMPLETE',seconds=time.perf_counter()-start,trainable_parameters=sum(p.numel() for p in parameters(m).values()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),frozen_unchanged=True,buffers_unchanged=True,extra_parameters={n:p.detach().cpu().tolist() for n,p in m.extra.items()});self.state['completed_fits']+=1
  except FloatingPointError as exc:
   fit.update(status='NUMERICAL_ERROR',error=str(exc));pending=OUT/'pending_update.json'
   if pending.exists():pending.rename(OUT/data.t/f'failed_update_{fid}.json')
  finally:save(OUT/data.t/'fits.json',self.fits);self.persist();del m,opt,init;cleanup()
  print('FIT',data.t,fid,fit['status'],'total',self.state['completed_fits'],flush=True);return fit
 def best(self,data,fit,reference=None):
  cps=fit['checkpoints'];feasible=True
  if data.t=='R04' and fit['arm']!='D0' and reference is not None:
   cc=[c for c in cps if c['accuracy']<=reference*1.01];feasible=bool(cc);chosen=min(cc,key=lambda c:(c['revision'],c['step'])) if cc else min(cps,key=lambda c:(c['accuracy'],c['step']))
  else:chosen=min(cps,key=lambda c:(c['accuracy'],c['step']))
  return dict(id=fit['id'],arm=fit['arm'],seed=fit['seed'],lr=fit['lr'],selected=chosen,fixed512=cps[-1],feasible=feasible)
 def select_seed(self,data,arms,seed,reference=None):
  rows=[]
  for arm in arms:
   fs=[f for f in self.fits if f['arm']==arm and f['seed']==seed and f['status']=='COMPLETE']
   if not fs:continue
   candidates=[self.best(data,f,reference) for f in fs]
   if data.t=='R04' and arm!='D0':
    good=[v for v in candidates if v['feasible']];best=min(good,key=lambda v:(v['selected']['revision'],v['lr'],v['selected']['step'])) if good else min(candidates,key=lambda v:(v['selected']['accuracy'],v['lr'],v['selected']['step']))
   else:best=min(candidates,key=lambda v:(v['selected']['accuracy'],v['lr'],v['selected']['step']))
   rows.append(best)
   if arm=='D0':reference=best['selected']['accuracy']
  return rows
 def context(self,t):
  self.track=t;self.fits=read(OUT/t/'fits.json') if (OUT/t/'fits.json').exists() else [];data=Packets(t);aliases=data.aux.get('aliases',{});arms=[a for a in ARMS[t] if a not in aliases];frozen,fixed=self.frozen_cache(data) if t in ['R04','R08'] else (None,None);return data,arms,frozen,fixed
 def stage(self,t,stage):
  data,arms,frozen,fixed=self.context(t)
  if stage=='smoke':self.smoke(data,arms,frozen,fixed)
  elif stage=='LR':
   for arm in arms:
    for lr in LRS:self.train_fit(data,arm,73100,lr,frozen,fixed)
   choices=self.select_seed(data,arms,73100);assert len(choices)==len(arms)
   save(OUT/t/'LR_selection.json',dict(selections=choices,learning_rates={v['arm']:v['lr'] for v in choices},at=time.time(),uses_E=False))
  elif stage=='repeat':
   rates=read(OUT/t/'LR_selection.json')['learning_rates'];selections=[]
   for seed in [73101,73102]:
    for arm in arms:self.train_fit(data,arm,seed,rates[arm],frozen,fixed)
    selections.extend(self.select_seed(data,arms,seed))
   assert len(selections)==2*len(arms)
   save(OUT/t/'selections.json',dict(selections=selections,aliases=data.aux.get('aliases',{}),at=time.time(),uses_E=False,LR_selection_sha256=sha(OUT/t/'LR_selection.json')))
  elif stage=='CPU':
   from .evaluate import select_cpu_controls
   selections=read(OUT/t/'selections.json')['selections'];select_cpu_controls(t,data,selections,frozen)
   save(OUT/t/'evaluation_seal.json',dict(at=time.time(),selections_sha256=sha(OUT/t/'selections.json'),CPU_sha256=sha(OUT/t/'CPU_selection.json'),master_sha256=sha(OUT/'MASTER_SEAL.json'),E_labels_opened=False))
  elif stage=='evaluation':
   global_seal=read(OUT/'EVALUATION_SEAL.json');assert sha(OUT/t/'evaluation_seal.json')==global_seal['track_seals'][t]
   selections=read(OUT/t/'selections.json')['selections'];preds=[];restores=[]
   for s in selections:
    m=self.make(t,s['arm'],s['seed'],data.aux)
    for policy in ['selected','fixed512']:
     cp=s[policy];assert sha(ROOT/cp['path'])==cp['sha256'];restore(m,torch.load(ROOT/cp['path'],map_location='cpu',weights_only=True));vp=np.load(ROOT/cp['prediction']);p=data.packet('V_SELECT',0,condition=str(vp['conditions'][0]));self.phase='verify'
     with torch.no_grad():q=(torch.stack([m(p),m(data.packet('V_SELECT',0,condition='paired',late=True))]) if t=='R04' else m(p)).double().cpu().numpy()
     err=float(np.max(abs(q-vp['pred'][0,0])/data.sigma.reshape((1,4,1) if t=='R04' else (4,1))));assert err<=1e-5;restores.append(dict(id=s['id'],policy=policy,error=err,bitwise=bool(np.array_equal(q,vp['pred'][0,0]))));file=self.predict(m,data,'E_DISCOVERY',s['arm']+'_'+cp['parameter_hash']);preds.append(dict(arm=s['arm'],seed=s['seed'],policy=policy,path=str(file.relative_to(ROOT)),sha256=sha(file),selected_step=cp['step']))
    del m;cleanup()
   save(OUT/t/'restore_verification.json',restores);save(OUT/t/'predictions_manifest.json',dict(predictions=preds,at=time.time(),E_labels_opened=False))
   from .evaluate import score_track
   score_track(t,data,selections,preds,frozen,data.aux.get('aliases',{}))
  elif stage=='cross':
   from .cross import run_cross
   run_cross(self,t,data)
 def run(self):
  configure();self.guard=Guard(OUT,'controller',wall_cap=86400)
  try:
   self.guard.boundary(startup=True)
   for stage in ['smoke','LR','repeat','CPU','evaluation','cross']:
    if stage=='evaluation' and not (OUT/'EVALUATION_SEAL.json').exists():
     track_seals={t:sha(OUT/t/'evaluation_seal.json') for t in ORDER if (OUT/t/'evaluation_seal.json').exists() and read(OUT/t/'STATUS.json')['EXECUTION'] not in ['INVALID_CONSTRUCT','NUMERICAL_ERROR']};save(OUT/'EVALUATION_SEAL.json',dict(at=time.time(),track_seals=track_seals,all_eligible_selections_frozen=True,E_labels_scored=False))
    for t in ORDER:
     state=read(OUT/t/'STATUS.json');done=state.get('completed_stages',[])
     if state['EXECUTION'] in ['BLOCKED_DIVERSITY','INVALID_CONSTRUCT','NUMERICAL_ERROR'] or stage in done:continue
     if stage=='cross' and state['EXECUTION']!='COMPLETE':continue
     save(OUT/'QUEUE_STATUS.json',dict(queue=ORDER,current=t,stage=stage,status='RUNNING',completed=[k for k in ORDER if read(OUT/k/'STATUS.json')['EXECUTION']=='COMPLETE']));print('STAGE_START',t,stage,flush=True)
     try:
      self.stage(t,stage);new=read(OUT/t/'STATUS.json');new['completed_stages']=done+[stage];new['phase']=stage;save(OUT/t/'STATUS.json',new);print('STAGE_END',t,stage,flush=True)
     except ResourceError:raise
     except (AssertionError,ValueError,RuntimeError,FloatingPointError,FileNotFoundError) as e:
      trace=traceback.format_exc();(OUT/t/f'error_{stage}.txt').write_text(trace);state.update(EXECUTION='NUMERICAL_ERROR' if isinstance(e,FloatingPointError) else 'INVALID_CONSTRUCT',reason=str(e),failed_stage=stage);save(OUT/t/'STATUS.json',state)
      pending=OUT/'pending_update.json'
      if pending.exists():pending.rename(OUT/t/f'uncertain_update_{int(time.time())}.json')
      cleanup();print('LOCAL_ERROR',t,stage,trace,flush=True)
     from .report import report_all
     report_all()
   self.state['status']='FINISHED';save(OUT/'QUEUE_STATUS.json',dict(queue=ORDER,current=None,status='FINISHED',completed=[t for t in ORDER if read(OUT/t/'STATUS.json')['EXECUTION']=='COMPLETE'],blocked=[t for t in ORDER if read(OUT/t/'STATUS.json')['EXECUTION']=='BLOCKED_DIVERSITY']))
  except ResourceError as e:
   self.state.update(status='PAUSED_BUDGET',reason=str(e));save(OUT/'QUEUE_STATUS.json',dict(queue=ORDER,current=self.track,status='PAUSED_BUDGET',reason=str(e),resume='scripts/with_cuda.sh .venv/bin/python scripts/run_condition_sampling_repair.py resume-all'));print('GLOBAL_PAUSE',e,flush=True)
  finally:
   self.guard.close();self.persist()
   from .report import report_all
   report_all()
