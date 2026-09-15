"""Shared finite runtime. A fit receives exposed history only, never evaluation target."""
import contextlib,hashlib,importlib.util,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];RUN='building_peft_topic_decision_20260916';OUT=ROOT/'results'/RUN;CACHE=ROOT/'.cache'/RUN;EXP=ROOT/'experiments'/RUN
sys.path.insert(0,str(ROOT/'scripts'))
from priority12.common import save,read,sha,csvwrite,parameters,cpu_state,restore,tensor_hash,frozen_hash,cleanup,Watch as BaseWatch,ResourceError
from tsfm_peft_screen.backbone import load_base,MODEL_ID,REVISION
from tsfm_peft_screen.lora import MODULES
POLICIES=['ZERO','EPOCH1','EPOCH4','EPOCH16','FIXED120'];LRS=[3e-5,1e-4]
class Watch(BaseWatch):
    def sample(self):
        r=super().sample()
        for a in r['apps']:a['allowed_desktop']=a['name']=='/usr/share/rustdesk/rustdesk'
        r['busy']=any(not a['own'] and not a['allowed_desktop'] for a in r['apps']) or r['free_mib']<1024
        return r
class Adapter(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.enabled=True
        self.lora_A=nn.Parameter(torch.empty(1,base.in_features,device=base.weight.device));nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,1,device=base.weight.device))
    def forward(self,x):return self.base(x)+(2*nn.functional.linear(nn.functional.linear(x,self.lora_A),self.lora_B) if self.enabled else 0)
@contextlib.contextmanager
def disabled(m):
    ll=[v for v in m.modules() if isinstance(v,Adapter)];previous=[v.enabled for v in ll]
    try:
        for v in ll:v.enabled=False
        yield
    finally:
        for v,b in zip(ll,previous):v.enabled=b

def setup(seed):
    torch.set_num_threads(4);torch.manual_seed(seed);np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
def make(seed):
    m=load_base()
    for i,name in enumerate(MODULES):
        torch.manual_seed(seed+1000+i);p,leaf=name.rsplit('.',1);parent=m.get_submodule(p);setattr(parent,leaf,Adapter(getattr(parent,leaf)))
    m.eval();ps=parameters(m);assert len(ps)==192 and sum(p.numel() for p in ps.values())==147456
    assert all('lora_A' in n or 'lora_B' in n for n in ps) and not any(p.requires_grad for p in m.output_patch_embedding.parameters())
    qs=np.array(m.chronos_config.quantiles);mi=np.flatnonzero(qs==.5);assert len(mi)==1
    metadata=dict(quantiles=qs.tolist(),median_index=int(mi[0]),trainable={n:dict(shape=list(p.shape),numel=p.numel()) for n,p in ps.items()},total=147456,rank=1,alpha=2,model=MODEL_ID,revision=REVISION)
    if (OUT/'parameter_audit.json').exists():assert read(OUT/'parameter_audit.json')==metadata
    else:save(OUT/'parameter_audit.json',metadata)
    return m

def tensor(x):return torch.as_tensor(np.array(x,copy=True),dtype=torch.float32,device='cuda')
def native(m,x,y=None):
    v=m(context=tensor(x).reshape(1,24),group_ids=torch.zeros(1,dtype=torch.long,device='cuda'),num_output_patches=2,future_target=None if y is None else tensor(y).reshape(1,24))
    assert v.quantile_preds.shape==(1,len(m.chronos_config.quantiles),32)
    if not torch.isfinite(v.quantile_preds).all() or (v.loss is not None and not torch.isfinite(v.loss)):raise FloatingPointError('NONFINITE_NATIVE')
    return v

def windows(h):
    assert len(h) in (72,336) and np.isfinite(h).all()
    xs=np.stack([h[i-24:i] for i in range(24,len(h)-23,24)]);ys=np.stack([h[i:i+24] for i in range(24,len(h)-23,24)])
    assert len(xs)==len(h)//24-1;return xs,ys

def step(policy,n):return 0 if policy=='ZERO' else 120 if policy=='FIXED120' else n*int(policy[5:])
def checkpoints(n):return sorted({step(p,n) for p in POLICIES})
def metric(q,y,h):
    meta=read(OUT/'parameter_audit.json');taus=np.array(meta['quantiles']);mi=meta['median_index'];q=np.asarray(q,dtype=np.float64);y=np.asarray(y,dtype=np.float64);h=np.asarray(h,dtype=np.float64)
    assert q.shape==(len(taus),24) and y.shape==(24,) and np.isfinite(q).all() and np.isfinite(y).all()
    sd=max(float(h.std(ddof=0)),1e-6);e=q[mi]-y;rmse=float(np.sqrt(np.mean(e*e)));d=y[None]-q
    return dict(primary=rmse/sd,raw_RMSE=rmse,raw_MAE=float(np.mean(abs(e))),scaled_2pinball=float((2*np.maximum(taus[:,None]*d,(taus[:,None]-1)*d)).mean()/sd),history_std=sd)

def predict(m,x,w,transform=None):
    w.boundary()
    with torch.no_grad():
        v=native(m,x).quantile_preds[0,:,:24]
        if transform is not None:v=transform(m,x,v)
        raw=v.double().cpu().numpy()
    assert np.isfinite(raw).all();return np.sort(raw,axis=0),raw

def affine(q,y):
    mi=read(OUT/'parameter_audit.json')['median_index'];x=q[:,mi].ravel();v=y.ravel();var=float(np.mean((x-x.mean())**2))
    if var<=np.finfo(float).eps*max(1,float(np.mean(x*x))):return dict(a=1.,b=0.,fallback=True,reason='OLS_ILL_CONDITIONED_F0_FIXED_FALLBACK')
    a=max(1e-6,float(np.mean((x-x.mean())*(v-v.mean()))/var));b=float(v.mean()-a*x.mean());assert np.isfinite([a,b]).all();return dict(a=a,b=b,fallback=False,reason=None)

def load_history(d):
    assert set(d)=={'path','sha256'};assert sha(ROOT/d['path'])==d['sha256'];h=np.load(ROOT/d['path']);windows(h);return h

def ledger():return read(OUT/'fits.json') if (OUT/'fits.json').exists() else []
def save_record(rec):
    records=ledger();matches=[i for i,v in enumerate(records) if v['id']==rec['id']]
    if matches:records[matches[0]]=rec
    else:records.append(rec)
    save(OUT/'fits.json',records);csvwrite(OUT/'fits.csv',[{k:v for k,v in r.items() if k not in ['checkpoints','losses']} for r in records])

def fit(job,h,w):
    """History array and whitelisted metadata only. No target path, array or episode manifest."""
    assert set(job)=={'id','method','role','building','days','origin','seed','lr'}
    assert len(h)==24*job['days'];x,y=windows(h);n=len(x);cps=checkpoints(n);maximum=max(cps);fid=job['id'];existing=[r for r in ledger() if r['id']==fid]
    if existing:
        rec=existing[0]
        if rec['status']=='COMPLETE':return rec
        assert rec['status']=='RUNNING','Failed attempts cannot be replaced or retried'
    else:
        assert len(ledger())<120
        rec=dict(**job,status='RUNNING',updates=0,planned_updates=maximum,windows=n,started_at=time.time(),training_seconds=0.,checkpoints={},losses=[],resumes=0);save_record(rec)
    m=None;opt=None;t0=time.monotonic()
    try:
        w.boundary();setup(job['seed']);m=make(job['seed']);frozen=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()));initial=tensor_hash(cpu_state(m));transform=None;compute_loss=None
        if job['method']!='STD':
            from candidate import build
            compute_loss,transform=build(job['method'],m,h,w)
        opt=torch.optim.AdamW(parameters(m).values(),lr=job['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0)
        folder=CACHE/'fits'/fid;folder.mkdir(parents=True,exist_ok=True);live=folder/'live.pt'
        if rec['updates']:
            z=torch.load(live,map_location='cpu',weights_only=True);assert z['updates']==rec['updates'];restore(m,z['model']);opt.load_state_dict(z['optimizer']);rec['resumes']+=1;save_record(rec)
        torch.cuda.reset_peak_memory_stats()
        for k in range(rec['updates'],maximum+1):
            if k in cps and str(k) not in rec['checkpoints']:
                statepath=folder/f'step{k}.pt';torch.save(cpu_state(m),statepath)
                preds=[predict(m,v,w,transform) for v in [h[-24:]]+list(x)];p=folder/f'prediction{k}.npz';np.savez_compressed(p,q=np.stack([v[0] for v in preds]),raw=np.stack([v[1] for v in preds]))
                rec['checkpoints'][str(k)]=dict(updates=k,path=str(p.relative_to(ROOT)),sha256=sha(p),weights=str(statepath.relative_to(ROOT)),weights_sha256=sha(statepath),train_seconds=rec['training_seconds']);save_record(rec)
            if k==maximum:break
            assert sum(r['updates'] for r in ledger())<19680,'MAIN_UPDATE_BUDGET'
            w.boundary();order=np.random.default_rng(job['seed']+k//n).permutation(n);j=int(order[k%n]);torch.cuda.synchronize();ts=time.monotonic();opt.zero_grad(set_to_none=True)
            loss=native(m,x[j],y[j]).loss if compute_loss is None else compute_loss(m,x[j],y[j])
            if not torch.isfinite(loss):raise FloatingPointError('NONFINITE_LOSS')
            loss.backward();ps=list(parameters(m).values())
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps):raise FloatingPointError('NONFINITE_GRADIENT')
            norm=torch.nn.utils.clip_grad_norm_(ps,1.,error_if_nonfinite=True);opt.step()
            if not all(torch.isfinite(p).all() for p in ps):raise FloatingPointError('NONFINITE_PARAMETERS')
            torch.cuda.synchronize();rec['training_seconds']+=time.monotonic()-ts;rec['updates']=k+1;rec['losses'].append(dict(update=k+1,window=j,loss=float(loss.detach()),grad_norm=float(norm)))
            tmp=live.with_suffix('.tmp');torch.save(dict(updates=k+1,model=cpu_state(m),optimizer=opt.state_dict()),tmp);tmp.replace(live);save_record(rec)
        assert frozen_hash(m)==frozen and tensor_hash(dict(m.named_buffers()))==buffers,'FROZEN_CHANGED'
        assert tensor_hash(cpu_state(m))!=initial,'NO_PARAMETER_UPDATE'
        rec.update(status='COMPLETE',frozen_unchanged=True,buffers_unchanged=True,trainable=147456,parameter_updated=True,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),wall_seconds=time.monotonic()-t0);save_record(rec);print('FIT_COMPLETE',fid,'updates',maximum,flush=True)
        return rec
    except BaseException as exc:
        rec.update(status='RESOURCE_PARTIAL' if isinstance(exc,ResourceError) else 'EXECUTION_ERROR',error=repr(exc),wall_seconds=time.monotonic()-t0);save_record(rec);raise
    finally:
        if opt is not None:del opt
        if m is not None:del m
        cleanup()
