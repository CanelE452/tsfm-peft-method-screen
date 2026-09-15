"""Finite new-building transfer study; no previous evaluation building access."""
import argparse, contextlib, csv, hashlib, importlib.util, json, math, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
RUN='building_transfer_subspace_v1_20260915'
OUT=ROOT/'results'/RUN; CACHE=ROOT/'.cache'/RUN; EXP=ROOT/'experiments'/RUN
OLD=ROOT/'results/building_coldstart_coverage_v1_20260915'; OC=ROOT/'.cache/building_coldstart_coverage_v1_20260915'
sys.path.insert(0,str(ROOT/'scripts'))
from priority12.common import save,read,sha,csvwrite,cpu_state,parameters,restore,tensor_hash,frozen_hash,cleanup,Watch as BaseWatch
spec=importlib.util.spec_from_file_location('coverage_pure_reference',ROOT/'experiments/building_coldstart_coverage_v1_20260915/core.py')
pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure)
metrics=pure.metrics; windows=pure.windows; affine=pure.affine
import torch
from torch import nn
from tsfm_peft_screen.backbone import load_base,MODEL_ID,REVISION
from tsfm_peft_screen.lora import MODULES,LowRank
SEED=61600
POLICIES=['epoch1','epoch4','epoch16','fixed120']

def steps(policy,n):return 120 if policy=='fixed120' else n*int(policy[5:])
def setup():
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
class Watch(BaseWatch):
    def sample(self):
        r=super().sample()
        for a in r['apps']:a['allowed_desktop']=a['name']=='/usr/share/rustdesk/rustdesk'
        r['busy']=any(not a['own'] and not a['allowed_desktop'] for a in r['apps']) or r['free_mib']<1024
        return r
class Adapter(nn.Module):
    def __init__(self,base,rank):
        super().__init__();self.base=base
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device));nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device))
    def forward(self,x):return self.base(x)+2*nn.functional.linear(nn.functional.linear(x,self.lora_A),self.lora_B)
class Bank(nn.Module):
    def __init__(self,base,name,pool,sources,coeff):
        super().__init__();self.base=base
        for key in ['lora_A','lora_B']:
            self.register_buffer('pool_'+key,pool[name+'.'+key].to(base.weight.device))
            self.register_buffer('bank_'+key,torch.stack([s[name+'.'+key] for s in sources]).to(base.weight.device))
        object.__setattr__(self,'coeff',coeff)
    def forward(self,x):
        common=2*nn.functional.linear(nn.functional.linear(x,self.pool_lora_A),self.pool_lora_B)
        # Shared scalar per source; each direction is source delta minus pooled delta.
        a=nn.functional.linear(x,self.bank_lora_A.flatten(0,1)).reshape(*x.shape[:-1],len(self.coeff),-1)
        response=2*torch.einsum('...kr,kor->...ko',a,self.bank_lora_B)
        correction=torch.einsum('...ko,k->...o',response,self.coeff)-self.coeff.sum()*common
        return self.base(x)+common+correction

def make(rank,pool=None,sources=None):
    m=load_base()
    if sources is not None:m.register_parameter('transfer_coefficients',nn.Parameter(torch.zeros(len(sources),device='cuda')))
    for i,name in enumerate(MODULES):
        torch.manual_seed(SEED+1000+i);parent,leaf=name.rsplit('.',1);p=m.get_submodule(parent);base=getattr(p,leaf)
        setattr(p,leaf,Bank(base,name,pool,sources,m.transfer_coefficients) if sources is not None else Adapter(base,rank))
    m.eval();assert not any(p.requires_grad for p in m.output_patch_embedding.parameters())
    assert sum(p.numel() for p in parameters(m).values())==(len(sources) if sources is not None else 147456*rank)
    if pool is not None and sources is None:restore(m,pool)
    return m

def tensor(a):return torch.as_tensor(np.array(a,copy=True),dtype=torch.float32,device='cuda')
def forward(m,x,y=None):
    r=m(context=tensor(x).reshape(1,24),group_ids=torch.zeros(1,dtype=torch.long,device='cuda'),num_output_patches=2,future_target=None if y is None else tensor(y).reshape(1,24))
    assert r.quantile_preds.shape==(1,21,32)
    if not torch.isfinite(r.quantile_preds).all() or (y is not None and not torch.isfinite(r.loss)):raise FloatingPointError('NONFINITE_FORWARD')
    return r

def predict(m,x,w):
    w.boundary()
    with torch.no_grad():raw=forward(m,x).quantile_preds[0,:,:24].double().cpu().numpy()
    return np.sort(raw,axis=0),raw

def datafile(name,a):
    p=CACHE/'data'/f'{name}.npy';p.parent.mkdir(parents=True,exist_ok=True);np.save(p,a)
    return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
def load(d):
    p=ROOT/d['path'];assert sha(p)==d['sha256'];return np.load(p)

def prepare():
    assert not (OUT/'seal.json').exists(),'Already sealed; do not re-prepare'
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    historical={}
    for folder in ['results','research']:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and OUT not in p.parents:historical[str(p.relative_to(ROOT))]=sha(p)
    save(OUT/'historical_hashes.json',historical)
    oldsplit=read(OLD/'building_split.json');excluded=sum([oldsplit[k] for k in ['recipe','dev','heldout']],[])
    rows=list(csv.DictReader((OLD/'eligibility.csv').open()))
    selected=sorted([r for r in rows if r['eligible']=='True' and r['building_id'] not in excluded],key=lambda r:r['hash'])[:24]
    assert len(selected)==24 and len({r['physical_group'] for r in selected})==24
    ids=[r['building_id'] for r in selected];assert not set(ids)&set(excluded)
    raw=pd.read_csv(OC/'official_source/electricity.csv',usecols=['timestamp']+ids,index_col='timestamp',parse_dates=True)
    clean=pd.read_csv(OC/'official_source/electricity_cleaned.csv',usecols=['timestamp']+ids,index_col='timestamp',parse_dates=True)
    frames=[];provenance={}
    for p in sorted((OC/'data/BuildingsBench/BDG-2').glob('*.csv')):
        columns=pd.read_csv(p,nrows=0).columns;use=['timestamp']+[bid for bid in ids if bid in columns]
        if len(use)>1:frames.append(pd.read_csv(p,usecols=use,index_col='timestamp',parse_dates=True))
        provenance[str(p.relative_to(ROOT))]=sha(p)
    sources=[];episodes=[];split={'excluded_previous_14':excluded,'previous_heldout_unopened':oldsplit['heldout'],'source':ids[:12],'tune':ids[12:16],'dev':ids[16:]}
    for i,r in enumerate(selected):
        bid=r['building_id'];s=pd.concat([d[bid] for d in frames if bid in d]).sort_index();assert s.index.is_unique
        start=pd.Timestamp(r['block_start']);end=start+pd.Timedelta(days=120)
        s=s.loc[(s.index>=start)&(s.index<end)];assert len(s)==2880 and np.all(np.diff(s.index.asi8)==3600*10**9)
        a=s.to_numpy(dtype=np.float64);rval=raw[bid].reindex(s.index).to_numpy();cval=clean[bid].reindex(s.index).to_numpy()
        assert np.isfinite(a).all() and np.isfinite(rval).all() and np.isfinite(cval).all()
        assert np.allclose(a,rval,rtol=0,atol=1e-7) and np.allclose(a,cval,rtol=0,atol=1e-7)
        if i<12:
            h=a[:56*24];v=a[56*24:70*24];assert h.std()>1e-6
            sources.append(dict(id=bid,physical_group=r['physical_group'],start=str(start),train=datafile(bid+'_source_train',h),validation=datafile(bid+'_source_validation',v)))
        else:
            role='tune' if i<16 else 'dev';local=i-12 if i<16 else i-16;day=2 if local%2==0 else 5
            j=next(j for j in range(14*24,119*24,24) if s.index[j].dayofweek==day)
            for days in [3,14]:
                eid=f'{bid}_H{days}';h=a[j-days*24:j];assert h.std()>1e-6
                episodes.append(dict(id=eid,building=bid,physical_group=r['physical_group'],role=role,days=days,origin=str(s.index[j]),weekday=day,history=datafile(eid+'_history',h),target=datafile(eid+'_target',a[j:j+24])))
    save(OUT/'split.json',split);save(OUT/'sources.json',sources);save(OUT/'episodes.json',episodes)
    for name in ['electricity.csv','electricity_cleaned.csv','metadata.csv']:provenance[str((OC/'official_source'/name).relative_to(ROOT))]=sha(OC/'official_source'/name)
    save(OUT/'data_provenance.json',dict(upstream=read(OLD/'observation_sources.json'),archive=read(OLD/'download_manifest.json'),files=provenance,eligibility_sha256=sha(OLD/'eligibility.csv'),old_code_sha256=sha(ROOT/'experiments/building_coldstart_coverage_v1_20260915/data.py'),rule='First 24 of previous performance-independent eligibility order excluding all previous 14. Only selected 24 raw values loaded; old heldout values not loaded. Same observed mask, no imputation.'))
    paths=list(EXP.glob('*.py'))+[OUT/'PROTOCOL.md',ROOT/'src/tsfm_peft_screen/backbone.py',ROOT/'src/tsfm_peft_screen/lora.py',ROOT/'scripts/priority12/common.py',ROOT/'experiments/building_coldstart_coverage_v1_20260915/core.py']
    save(OUT/'seal.json',dict(created_at=time.time(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),model=MODEL_ID,revision=REVISION,sources={str(p.relative_to(ROOT)):sha(p) for p in paths},data={str(p.relative_to(ROOT)):sha(p) for p in sorted((CACHE/'data').glob('*.npy'))},manifests={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'split.json',OUT/'episodes.json',OUT/'sources.json',OUT/'data_provenance.json']}))
    print('PREPARED',split,flush=True)

def checkseal():
    s=read(OUT/'seal.json')
    for kind in ['sources','data','manifests']:
        for p,h in s[kind].items():assert sha(ROOT/p)==h,('SEAL_CHANGED',p)

def checkpoint_path(fid,k):return CACHE/'checkpoints'/f'{fid}_s{k}.pt'
def fit(fid,rank,x,y,checkpoints,predx,w,pool=None,sources=None):
    dest=OUT/'fits'/f'{fid}.json';assert not dest.exists(),('DUPLICATE_FIT',fid)
    ledger=OUT/'fit_ledger.jsonl'
    prior=[json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    assert not any(a['id']==fid for a in prior),'No automatic retry of attempted fit'
    assert sum(a['event']=='START' for a in prior)<93
    m=make(rank,pool,sources);frozen=frozen_hash(m);initial=cpu_state(m)
    # Bank buffers are separately sealed; frozen_hash only covers parameters.
    buffers_before=tensor_hash({n:v for n,v in m.named_buffers()})
    ps=list(parameters(m).values());opt=torch.optim.AdamW(ps,lr=.01 if sources is not None else 1e-4,weight_decay=0,betas=(.9,.999),eps=1e-8)
    maxstep=max(checkpoints);n=len(x);assert x.shape==y.shape and x.shape[1]==24
    torch.cuda.reset_peak_memory_stats();trainseconds=0.;curves={};losses=[];started=time.time();t0=time.monotonic()
    def event(e,**kw):
        with ledger.open('a') as f:f.write(json.dumps(dict(id=fid,event=e,at=time.time(),**kw))+'\n')
    event('START',rank=rank,planned_updates=maxstep,trainable=sum(p.numel() for p in ps))
    try:
        for k in range(maxstep+1):
            if k in checkpoints:
                cp=checkpoint_path(fid,k);cp.parent.mkdir(parents=True,exist_ok=True);torch.save(cpu_state(m),cp)
                pr=[predict(m,v,w) for v in predx]
                pp=CACHE/'predictions'/f'{fid}_s{k}.npz';pp.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(pp,q=np.stack([p[0] for p in pr]),raw=np.stack([p[1] for p in pr]))
                curves[str(k)]=dict(step=k,train_seconds=trainseconds,prediction_file=str(pp.relative_to(ROOT)),prediction_sha256=sha(pp),checkpoint_file=str(cp.relative_to(ROOT)),checkpoint_sha256=sha(cp),checkpoint_tensor_hash=tensor_hash(cpu_state(m)))
            if k==maxstep:break
            w.boundary();torch.cuda.synchronize();t=time.monotonic();opt.zero_grad(set_to_none=True)
            # Fixed epoch-specific permutation. Prefix identical across ranks/initializations.
            epoch=k//n;order=np.random.default_rng(SEED+epoch).permutation(n);j=order[k%n]
            r=forward(m,x[j],y[j]);r.loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps),'NONFINITE_OR_MISSING_GRAD'
            norm=torch.nn.utils.clip_grad_norm_(ps,1.,error_if_nonfinite=True);opt.step()
            assert all(torch.isfinite(p).all() for p in ps),'NONFINITE_PARAMETER'
            torch.cuda.synchronize();trainseconds+=time.monotonic()-t
            losses.append(dict(step=k+1,loss=float(r.loss.detach()),gradient_norm=float(norm)))
            if (k+1)%100==0:print('UPDATE',fid,k+1,'/',maxstep,flush=True)
        assert frozen_hash(m)==frozen and tensor_hash({n:v for n,v in m.named_buffers()})==buffers_before,'FROZEN_CHANGED'
        assert tensor_hash(cpu_state(m))!=tensor_hash(initial),'NO_UPDATE'
        # Restore selected final saved tensors and verify exact prediction reload.
        q0=predict(m,predx[0],w)[1];restore(m,torch.load(checkpoint_path(fid,maxstep),map_location='cpu',weights_only=True));q1=predict(m,predx[0],w)[1]
        assert np.array_equal(q0,q1),'RELOAD_MISMATCH'
        rec=dict(id=fid,status='COMPLETED',rank=rank,updates=maxstep,trainable=sum(p.numel() for p in ps),windows=n,training_seconds=trainseconds,wall_seconds=time.monotonic()-t0,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),frozen_unchanged=True,buffers_unchanged=True,reload_exact=True,checkpoints=curves,losses=losses)
        save(dest,rec);event('COMPLETE',updates=maxstep);print('FIT_DONE',fid,maxstep,flush=True)
        del opt,m;cleanup();return rec
    except BaseException as e:
        save(dest,dict(id=fid,status='EXECUTION_ERROR',completed_updates=len(losses),error=repr(e),losses=losses,checkpoints=curves));event('ERROR',completed_updates=len(losses),error=repr(e));raise

def predictions(rec,k):
    d=rec['checkpoints'][str(k)];assert sha(ROOT/d['prediction_file'])==d['prediction_sha256'];return np.load(ROOT/d['prediction_file'])['q']
def state(rec,k):
    d=rec['checkpoints'][str(k)];assert sha(ROOT/d['checkpoint_file'])==d['checkpoint_sha256'];return torch.load(ROOT/d['checkpoint_file'],map_location='cpu',weights_only=True)
def ep_data(e):
    h=load(e['history']);x,y,_,_=windows(h,e['origin']);return h,x,y

def add_score(rows,e,arm,q,rec=None,k=0,extra=None):
    h=load(e['history']);target=load(e['target']);r=dict(episode=e['id'],building=e['building'],role=e['role'],days=e['days'],arm=arm,step=k,fit=rec['id'] if rec else None,train_seconds=rec['checkpoints'][str(k)]['train_seconds'] if rec else 0.,**metrics(q,target,h))
    if extra:r.update(extra)
    p=CACHE/'scores'/f"{e['role']}_{e['id']}_{arm}.npz";p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,q=q,y=target,h=h);r['replay_file']=str(p.relative_to(ROOT));r['replay_sha256']=sha(p)
    rows.append(r);csvwrite(OUT/'scores.csv',rows);return r

def select_recipe(rows,prefix):
    rr=[r for r in rows if r['role']=='tune' and r['arm'].startswith(prefix)]
    arms=sorted({r['arm'] for r in rr});summary=[]
    for arm in arms:
        v=[r for r in rr if r['arm']==arm];assert len(v)==8
        summary.append(dict(arm=arm,macro_scaled_RMSE=float(np.mean([r['scaled_RMSE'] for r in v])),mean_steps=float(np.mean([r['step'] for r in v])),rank=int(arm.split('_r')[1].split('_')[0])))
    best=min(summary,key=lambda s:(s['macro_scaled_RMSE'],s['mean_steps'],s['rank'],s['arm']))
    result=dict(selected=best,alternatives=summary,rule='Global equal-episode macro scaled RMSE on four tuning buildings only; tie fewer updates then smaller rank.')
    save(OUT/f'{prefix}selection.json',result);return best['rank'],best['arm'].split('_r')[1].split('_',1)[1]

def source_xy(s):
    h=load(s['train']);v=load(s['validation']);x=np.stack([h[i-24:i] for i in range(24,len(h)-23,24)]);y=np.stack([h[i:i+24] for i in range(24,len(h)-23,24)])
    a=np.r_[h[-24:],v];vx=np.stack([a[i-24:i] for i in range(24,len(a)-23,24)]);vy=np.stack([a[i:i+24] for i in range(24,len(a)-23,24)])
    assert len(x)==55 and len(vx)==14;return h,x,y,vx,vy

def source_select(rec,sources):
    curve=[]
    for key in rec['checkpoints']:
        k=int(key);qs=predictions(rec,k);idx=0;ms=[]
        for s in sources:
            h,_,_,vx,vy=source_xy(s)
            ms.append(np.mean([metrics(qs[idx+j],v,h)['scaled_RMSE'] for j,v in enumerate(vy)]));idx+=len(vx)
        curve.append(dict(step=k,source_validation_macro=float(np.mean(ms))))
    best=min(curve,key=lambda a:(a['source_validation_macro'],a['step']));save(OUT/(rec['id']+'_selection.json'),dict(selected=best,curve=curve));return best['step']

def smoke(w):
    e=read(OUT/'episodes.json')[0];h,x,y=ep_data(e);m=make(8)
    for i,name in enumerate(MODULES):
        layer=m.get_submodule(name);torch.manual_seed(SEED+1000+i);old=LowRank(layer.base)
        assert torch.equal(old.lora_A,layer.lora_A) and torch.equal(old.lora_B,layer.lora_B)
        v=torch.randn(1,2,layer.base.in_features,device='cuda');assert torch.equal(old(v),layer(v))
        del old
    with torch.no_grad():
        a=forward(m,h[-24:]).quantile_preds
        b=forward(m,h[-24:],np.ones(24)*99999).quantile_preds
        assert torch.equal(a,b),'TARGET_LEAK'
        from chronos import Chronos2Pipeline
        reference=Chronos2Pipeline(m).predict([torch.as_tensor(h[-24:].copy(),dtype=torch.float32)],prediction_length=24,context_length=24,batch_size=1)[0][0].double().cpu().numpy()
        err=float(np.max(np.abs(a[0,:,:24].double().cpu().numpy()-reference)));assert err==0
    for rank in [1,8]:
        z=Adapter(nn.Linear(7,5,device='cuda'),rank);z.base.requires_grad_(False);v=torch.randn(2,7,device='cuda');z(v).sum().backward();assert z.lora_B.grad.abs().sum()>0
    # Nonzero synthetic bank response, zero initialization identity, coefficient gradient and parameter ownership.
    name=MODULES[0];p=m.get_submodule(name);pool={name+'.lora_A':p.lora_A.detach(),name+'.lora_B':torch.randn_like(p.lora_B)*.01};sources=[{k:v.clone() for k,v in pool.items()} for _ in range(2)]
    sources[0][name+'.lora_B']*=2;sources[1][name+'.lora_B']*=-1;c=nn.Parameter(torch.zeros(2,device='cuda'));bank=Bank(p.base,name,pool,sources,c);v=torch.randn(2,p.base.in_features,device='cuda')
    expected=p.base(v)+2*nn.functional.linear(nn.functional.linear(v,pool[name+'.lora_A']),pool[name+'.lora_B'])
    assert torch.equal(bank(v),expected);bank(v).sum().backward();assert torch.isfinite(c.grad).all() and c.grad.abs().sum()>0
    with torch.no_grad():c.copy_(torch.tensor([.3,-.2],device='cuda'))
    direct=expected.clone()
    for j,s in enumerate(sources):direct+=c[j]*(2*nn.functional.linear(nn.functional.linear(v,s[name+'.lora_A']),s[name+'.lora_B'])-(expected-p.base(v)))
    torch.testing.assert_close(bank(v),direct,rtol=1e-5,atol=1e-5)
    save(OUT/'smoke.json',dict(status='PASS',native_pipeline_max_abs=err,target_poison_identity=True,rank8_matches_previous_adapter=True,rank1_8_gradients=True,bank_zero_identity=True,bank_formula_gradient=True,full_model_optimizer_updates=0,synthetic_backward_only=True))
    del m,bank;cleanup();print('SMOKE_PASS',flush=True)

def run():
    checkseal();assert not (OUT/'fit_ledger.jsonl').exists(),'Execution already attempted; no automatic duplicate/retry'
    setup();w=Watch(OUT,'study',wall_cap=7200);rows=[];terminal={}
    try:
        w.boundary(startup=True);smoke(w)
        episodes=read(OUT/'episodes.json');tune=[e for e in episodes if e['role']=='tune'];dev=[e for e in episodes if e['role']=='dev'];sources=read(OUT/'sources.json')
        # Stage 1: select one global local recipe, including properly normalized repetition budget.
        for e in tune:
            h,x,y=ep_data(e)
            for rank in [1,8]:
                cps=sorted({0}|{steps(p,len(x)) for p in POLICIES});rec=fit('local_tune_'+e['id']+f'_r{rank}',rank,x,y,cps,[h[-24:]]+list(x),w)
                for p in POLICIES:add_score(rows,e,f'local_r{rank}_{p}',predictions(rec,steps(p,len(x)))[0],rec,steps(p,len(x)))
                if rank==1:
                    f=predictions(rec,0);add_score(rows,e,'F0',f[0],rec,0);af=affine(f[1:],y);add_score(rows,e,'AFFINE',af['a']*f[0]+af['b'],rec,0,af)
        rank,policy=select_recipe(rows,'local_');save(OUT/'stage1_recipe.json',dict(rank=rank,policy=policy))
        # Pooled pretraining: 12 buildings x 55 windows, 1/2/4 epochs, source future validation only.
        sx=[];sy=[];vx=[]
        for s in sources:
            _,x,y,v,_=source_xy(s);sx.extend(x);sy.extend(y);vx.extend(v)
        n=len(sx);poolrec=fit('source_pooled',rank,np.array(sx),np.array(sy),[0,n,2*n,4*n],vx,w);poolstep=source_select(poolrec,sources);pool=state(poolrec,poolstep)
        for e in tune:
            h,x,y=ep_data(e);cps=sorted({0}|{steps(p,len(x)) for p in POLICIES});rec=fit('warm_tune_'+e['id'],rank,x,y,cps,[h[-24:]]+list(x),w,pool=pool)
            for p in POLICIES:add_score(rows,e,f'warm_r{rank}_{p}',predictions(rec,steps(p,len(x)))[0],rec,steps(p,len(x)))
            f=predictions(rec,0);add_score(rows,e,'POOLED',f[0],rec,0);af=affine(f[1:],y);add_score(rows,e,'POOLED_AFFINE',af['a']*f[0]+af['b'],rec,0,af)
        _,wp=select_recipe(rows,'warm_');save(OUT/'stage2_recipe.json',dict(rank=rank,policy=wp,source_step=poolstep))
        simple_arms=['F0','AFFINE',f'local_r{rank}_{policy}'];transfer_arms=['POOLED','POOLED_AFFINE',f'warm_r{rank}_{wp}']
        def tune_best(arms):return min(arms,key=lambda a:(np.mean([r['scaled_RMSE'] for r in rows if r['role']=='tune' and r['arm']==a]),a))
        simple=tune_best(simple_arms);trans=tune_best(transfer_arms)
        save(OUT/'deployment_selection.json',dict(simple=simple,transfer=trans,rule='One global arm chosen on tuning only; development never selects deployment.'))
        for e in dev:
            h,x,y=ep_data(e);k=steps(policy,len(x));rec=fit('local_dev_'+e['id'],rank,x,y,sorted({0,k,120}),[h[-24:]]+list(x),w)
            add_score(rows,e,'LOCAL',predictions(rec,k)[0],rec,k);add_score(rows,e,'LOCAL_FIXED120',predictions(rec,120)[0],rec,120)
            f=predictions(rec,0);add_score(rows,e,'F0',f[0],rec,0);af=affine(f[1:],y);add_score(rows,e,'AFFINE',af['a']*f[0]+af['b'],rec,0,af)
            k=steps(wp,len(x));rec=fit('warm_dev_'+e['id'],rank,x,y,[0,k],[h[-24:]]+list(x),w,pool=pool)
            add_score(rows,e,'WARM',predictions(rec,k)[0],rec,k);f=predictions(rec,0);add_score(rows,e,'POOLED',f[0],rec,0);af=affine(f[1:],y);add_score(rows,e,'POOLED_AFFINE',af['a']*f[0]+af['b'],rec,0,af)
        simple='LOCAL' if simple.startswith('local_') else simple;trans='WARM' if trans.startswith('warm_') else trans
        gate=transfer_gate(rows,simple,trans);save(OUT/'transfer_gate.json',gate)
        if not gate['pass']:
            terminal=dict(status='STOP_NO_TRANSFER_SIGNAL',candidate_fits=0,unrun='12 source directions + 8 coefficient tuning + 16 coefficient development fits; previous heldout untouched')
        else:
            # Exactly one conditional candidate. No extra candidate or seed after this branch.
            bankstates=[];selected=[]
            for s in sources:
                _,x,y,v,_=source_xy(s);rec=fit('direction_'+s['id'],rank,x,y,[0,55,110,220],v,w);k=source_select(rec,[s])
                if k>0:bankstates.append(state(rec,k));selected.append(dict(building=s['id'],step=k))
            save(OUT/'direction_selection.json',dict(selected=selected,minimum=2))
            if len(bankstates)<2:terminal=dict(status='STOP_INSUFFICIENT_SOURCE_DIRECTIONS',candidate_fits=0,unrun='Coefficient tuning/development 24 fits')
            else:
                for e in tune:
                    h,x,y=ep_data(e);cps=sorted({0}|{steps(p,len(x)) for p in POLICIES});rec=fit('coeff_tune_'+e['id'],rank,x,y,cps,[h[-24:]],w,pool=pool,sources=bankstates)
                    for p in POLICIES:add_score(rows,e,f'coeff_r{rank}_{p}',predictions(rec,steps(p,len(x)))[0],rec,steps(p,len(x)))
                _,cp=select_recipe(rows,'coeff_');save(OUT/'candidate_recipe.json',dict(rank=rank,policy=cp,coefficients=len(bankstates)))
                for e in dev:
                    h,x,y=ep_data(e);k=steps(cp,len(x));rec=fit('coeff_dev_'+e['id'],rank,x,y,[0,k],[h[-24:]],w,pool=pool,sources=bankstates)
                    add_score(rows,e,'COEFFICIENT',predictions(rec,k)[0],rec,k)
                candidate=candidate_gate(rows,trans);save(OUT/'candidate_gate.json',candidate);terminal=dict(status='DEVELOPMENT_SIGNAL' if candidate['pass'] else 'STOP_NO_COEFFICIENT_ADDED_VALUE',candidate_fits=24,unrun='All previous heldout; independent confirmation, extra seeds, other candidates not authorized')
        save(OUT/'terminal.json',terminal);print('TERMINAL',terminal,flush=True)
    except BaseException as e:
        from priority12.common import ResourceError
        terminal=dict(status='BLOCKED_COMMON_RESOURCE' if isinstance(e,ResourceError) else 'EXECUTION_ERROR',error=repr(e),traceback=traceback.format_exc(),performance_failure=False,no_automatic_retry=True)
        save(OUT/'terminal.json',terminal);print(traceback.format_exc(),flush=True)
    finally:w.close()

def transfer_gate(rows,simple,trans):
    d=[r for r in rows if r['role']=='dev'];lookup={(r['episode'],r['arm']):r for r in d};eps=sorted({r['episode'] for r in d});groups={}
    b=np.array([lookup[(e,simple)]['scaled_RMSE'] for e in eps]);t=np.array([lookup[(e,trans)]['scaled_RMSE'] for e in eps]);gain=100*(b.mean()-t.mean())/b.mean();h={}
    for day in [3,14]:
        mask=np.array([lookup[(e,simple)]['days']==day for e in eps]);h[str(day)]=float(100*(b[mask].mean()-t[mask].mean())/b[mask].mean())
    for i,e in enumerate(eps):groups.setdefault(lookup[(e,simple)]['building'],[]).append(i)
    bg={bid:float(100*(b[ix].mean()-t[ix].mean())/b[ix].mean()) for bid,ix in groups.items()};wins=sum(g>0 for g in bg.values())
    return dict(simple=simple,transfer=trans,macro_gain_percent=float(gain),history_gain_percent=h,building_gains=bg,positive_buildings=wins,**{'pass':bool(gain>=1 and wins>=6 and h['3']>0 and h['14']>=-1)},interpretation='Fixed development continuation gate; not publication success. Shared-source information and training cost are additional.')

def candidate_gate(rows,trans):
    # Candidate must improve BOTH selected simple and selected transfer; report all controls regardless.
    arms=['F0','AFFINE','LOCAL','POOLED','POOLED_AFFINE','WARM'];d=[r for r in rows if r['role']=='dev'];means={a:float(np.mean([r['scaled_RMSE'] for r in d if r['arm']==a])) for a in arms+['COEFFICIENT']};best=min(arms,key=lambda a:means[a]);gain=100*(means[best]-means['COEFFICIENT'])/means[best]
    out=transfer_gate(rows,best,'COEFFICIENT');out.update(best_reported_control=best,means=means,macro_gain_percent=float(gain));out['pass']=bool(gain>=.5 and out['positive_buildings']>=6 and out['history_gain_percent']['3']>=-1 and out['history_gain_percent']['14']>=-1);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);args=p.parse_args()
    if args.command=='prepare':prepare()
    else:run()
