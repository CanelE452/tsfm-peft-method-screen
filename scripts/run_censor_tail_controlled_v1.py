"""One bounded six-fit synthetic-censoring pilot; no recursive follow-up or retries."""
import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import torch
from tsfm_peft_screen.backbone import MODEL_ID, REVISION, QUANTILES, load_base, forecast
from tsfm_peft_screen.lora import attach, audit, disabled, snapshot, restore
from tsfm_peft_screen.metrics import score, independent
from tsfm_peft_screen.reproducibility import ROOT, sha, digest, write_json, seed_all, guard
from tsfm_peft_screen.candidates.censor_tail_v1 import objective, tail_diagnostics

OUT=ROOT/'results/censor_tail_controlled_v1'
CACHE=ROOT/'.cache/censor_tail_controlled_v1'
DOC=ROOT/'docs/CENSOR_TAIL_CONTROLLED_V1_PROTOCOL.md'
STEPS=[0,4,8,15,30,60,120,180,240,360]
ARMS=['NAIVE','DROP','TAIL']
SEEDS=[41000,41001]
BASE='e07ae8cd724fd8877734d730ec8cf3649c2d7591'

def read(p): return json.loads(Path(p).read_text())
def save(name,obj): write_json(OUT/name,obj)
def rel(p): return str(Path(p).relative_to(ROOT))
def csvwrite(p,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def gain(base,candidate): return 100*(base-candidate)/base if base else None
def treehash(items):
    h=hashlib.sha256()
    for n,p in sorted(items.items()):
        a=p.detach().cpu().contiguous().numpy();h.update(n.encode());h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
    return h.hexdigest()
def arrayhash(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()
def frozen(m):return treehash({n:p for n,p in m.named_parameters() if not p.requires_grad})
def build(seed):
    seed_all(seed);return attach(load_base(),seed)
def params(m):return [p for p in m.parameters() if p.requires_grad]
def optimizer(m):return torch.optim.AdamW(params(m),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
def cleanup():gc.collect();torch.cuda.empty_cache()

def source_files():
    paths=list((ROOT/'src').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'tests/test_censor_tail_controlled_v1.py',DOC]
    return {rel(p):sha(p) for p in sorted(paths)}

def prepare():
    if OUT.exists() or CACHE.exists():raise FileExistsError('Existing experiment: inspect status; no overwrite')
    OUT.mkdir();CACHE.mkdir()
    save('status.json',dict(stage='PREPARING',attempted_fits=0,completed_fits=0,updates=0,smoke_updates=0))
    meta=read(ROOT/'data/processed/m5/manifest.json')
    expected={'gate':[],'train':list(range(336,1400-47,24)),'validation':[1400,1450,1500,1550,1600],'evaluation':[1650,1698,1746,1794,1842,1890]}
    assert meta['origins']==expected and meta['bounds']==dict(end=1941,evaluation_start=1650,gate_start=0,train_start=336,validation_start=1400)
    for f,h in meta['files'].items():assert sha(ROOT/'data/processed/m5'/f)==h
    for p,h in meta['sources'].items():assert sha(p)==h
    schedule=read(ROOT/'results/candidate_07/sampling_manifest.json')
    assert len(schedule)==360
    for r in schedule:
        assert set(r)=={'origins','channels','variant'} and r['variant']=='clean'
        assert len(r['origins'])==len(r['channels'])==8
        assert all(o in expected['train'] for o in r['origins'])
        assert all(len(c)==1 and 0<=c[0]<256 for c in r['channels'])
    with np.load(ROOT/'data/processed/m5/fit.npz') as z:
        values=z['values'];caps=z['caps'];channels=z['channels']
    assert values.shape==(1650,256) and channels.tolist()==meta['channels']
    train=values[336:1400]
    assert np.isfinite(values).all() and ((train==0).mean(0)>=.5).all() and ((train>0).sum(0)>=40).all()
    rebuilt=np.array([max(1.,np.quantile(c[c>0],.6)) for c in train.T])
    assert np.array_equal(rebuilt,caps)
    # Preserve exact existing cap values. They were generated under the pinned NumPy version.
    observed=np.minimum(values,caps[None,:]).astype(np.float32)
    mask=values>caps[None,:]
    s_obs=np.std(observed[336:1400].astype(np.float64),axis=0,ddof=0)
    assert np.isfinite(s_obs).all() and (s_obs>0).all()
    changed=values.copy();changed[mask]+=10000
    assert np.array_equal(np.minimum(changed,caps).astype(np.float32),observed)
    assert np.array_equal(changed>caps,mask)
    np.savez_compressed(CACHE/'train_observed.npz',observed=observed[:1400],censored=mask[:1400],caps=caps,scale=s_obs,channels=channels)
    vo=expected['validation']
    np.savez_compressed(CACHE/'validation_scoring.npz',context=np.stack([observed[o-336:o].T for o in vo]),
        target=np.stack([values[o:o+48].T for o in vo]),scale=s_obs,caps=caps,origins=np.array(vo),channels=channels)
    del values,train,changed
    oldfiles=[ROOT/p for p in subprocess.check_output(['git','ls-files','results'],cwd=ROOT,text=True).splitlines()]
    oldfiles += list((ROOT/'.cache/candidate_07').rglob('*.pt'))
    oldfiles += [ROOT/'data/processed/m5'/p for p in ['fit.npz','evaluation.npz','manifest.json']]
    oldfiles += [Path(p) for p in meta['sources']]
    from huggingface_hub import try_to_load_from_cache
    model_files={}
    for f,h in read(ROOT/'results/screening_summary/common_integrity.json')['model_files'].items():
        p=try_to_load_from_cache(MODEL_ID,f,revision=REVISION);assert isinstance(p,str) and sha(p)==h
        model_files[p]=h;oldfiles.append(Path(p))
    history={str(p):sha(p) for p in oldfiles}
    save('preserved_hashes.json',history)
    config=dict(arms=ARMS,seeds=SEEDS,lr=1e-4,steps=360,checkpoints=STEPS,batch=8,context=336,horizon=48,
        trainable_parameters=1179648,trainable_tensors=192,rank=8,lora_scale=2,precision='FP32 model; FP64 log-survival',
        lambda_rule='clip(0.1*mean(first16 F0 DROP)/mean(first16 F0 survival),0.001,100)',
        max_fits=6,max_updates=2160,max_smoke_updates=6,wall_cap_seconds=7200,fit_cap_seconds=1800,wait_cap_seconds=900,
        primary='V-selected E_dev equal-series scaled 2-pinball',secondary='fixed step360 E_dev',
        bootstrap=dict(iterations=2000,seed=41002,unit='series; retain both optimizer seeds and all origins/horizons',interval='percentile95'),
        selection='minimum clean V primary; exact ties earliest step',E_scope='previously used synthetic-censoring development evaluation',
        schema=dict(train_observed='observed[T,I],censored[T,I],caps[I],scale[I],channels[I]; T=1400; no original target',
            validation_scoring='context[O,I,336] capped; target[O,I,48] original; scale,caps,origins,channels',
            prediction='prediction[O,I,21,48],target[O,I,48],censored[O,I,48],scale[I],caps[I],origins[O]'))
    save('config.json',config)
    save('manifest.json',dict(prepared_at=time.time(),base_commit=BASE,current_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        baseline_diff=subprocess.check_output(['git','diff',BASE,'--','src','scripts','docs/CANDIDATE_07.md'],text=True),
        sources=source_files(),model_id=MODEL_ID,model_revision=REVISION,model_files=model_files,
        data_manifest_sha256=sha(ROOT/'data/processed/m5/manifest.json'),sampling_sha256=sha(ROOT/'results/candidate_07/sampling_manifest.json'),
        ids=channels.tolist(),ids_sha256=arrayhash(channels),caps=caps.tolist(),caps_sha256=arrayhash(caps),scale=s_obs.tolist(),scale_sha256=arrayhash(s_obs),
        cache_hashes={rel(p):sha(p) for p in CACHE.glob('*.npz')},config_sha256=sha(OUT/'config.json'),
        leakage=dict(hidden_excess_changed_by=10000,capped_inputs_and_masks_exactly_unchanged=True,training_schema_contains_no_original_sales=True,clean_V_permission=True,E_dev_reused=True),
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=sys.platform)))
    (OUT/'PROTOCOL.md').write_text(DOC.read_text())
    save('status.json',dict(stage='PREPARED',attempted_fits=0,completed_fits=0,updates=0,smoke_updates=0))
    print('PREPARED: input/model/sampling hashes verified; no E_dev arrays opened',flush=True)


class Monitor:
    def __init__(self,phase,allow_rustdesk=False):
        self.phase=phase;self.allow_rustdesk=allow_rustdesk;self.stop=threading.Event();self.rows=[];self.error=None;self.latest=None
        self.fit_start=None;self.started=time.monotonic();self.wait=0.;self.io=0.;self.scoring=0.;self.validation=0.;self.forward=0;self.backward=0;self.output_backward=0
        self.lock=open(ROOT/'.cache/gpu.lock','a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=read(OUT/'budget.json') if (OUT/'budget.json').exists() else dict(started_at=time.time(),gpu_wait_seconds=0.)
        self.budget=previous
        self.thread=threading.Thread(target=self.poll,daemon=True);self.thread.start()
    def poll(self):
        while not self.stop.is_set():
            try:
                g=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3).strip().split(',')
                s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True,timeout=3)
                own={os.getpid()}|{p.pid for p in psutil.Process().children(recursive=True)}
                apps=[]
                for line in s.splitlines():
                    p,n,mem=[v.strip() for v in line.split(',')];a=dict(pid=int(p),name=n,memory_mib=int(mem))
                    a['allowed']=a['pid'] in own or (self.allow_rustdesk and n=='/usr/share/rustdesk/rustdesk' and int(mem)<=512);apps.append(a)
                row=dict(at=time.time(),phase=self.phase,gpu=g[0],free_mib=int(g[1]),utilization=int(g[2]),apps=apps)
                self.rows.append(row);self.latest=row
                save(f'gpu_{self.phase}.json',self.rows)
            except BaseException as e:self.error=repr(e);return
            self.stop.wait(2)
    def bounds(self):
        if self.error:raise RuntimeError('GPU monitor failed: '+self.error)
        if time.time()-self.budget['started_at']>7200:raise TimeoutError('Whole execution wall cap 2 hours')
        if self.fit_start and time.monotonic()-self.fit_start>1800:raise TimeoutError('Fit wall cap 30 minutes')
        if self.budget['gpu_wait_seconds']+self.wait>900:raise TimeoutError('Cumulative GPU wait cap 15 minutes')
        if psutil.virtual_memory().available<2*2**30:raise MemoryError('Available RAM below 2 GiB')
    def boundary(self,startup=False):
        stable=None
        while True:
            self.bounds();r=self.latest
            fresh=r is not None and time.time()-r['at']<=5
            ok=fresh and not any(not a['allowed'] for a in r['apps']) and r['free_mib']>=(4096 if startup else 1024)
            if ok:
                if not startup:break
                if stable is None:stable=time.monotonic()
                if time.monotonic()-stable>=30:break
            else:stable=None
            t=time.monotonic();time.sleep(.25);self.wait+=time.monotonic()-t
        guard()
    def finish(self):
        self.stop.set();self.thread.join(timeout=7)
        self.budget['gpu_wait_seconds']+=self.wait;save('budget.json',self.budget)
        info=dict(seconds=time.monotonic()-self.started,gpu_wait_seconds=self.wait,model_forward=self.forward,model_backward=self.backward,output_tensor_backward=self.output_backward,
            validation_seconds=self.validation,scoring_seconds=self.scoring,io_seconds=self.io,
            gpu_peak_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            gpu_peak_reserved=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            peak_cpu_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            rustdesk_exception_authorized=self.allow_rustdesk)
        save(f'resources_{self.phase}.json',info);fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close();return info


class ObservedTraining:
    """Only capped observations and exact event bits; no Panel or original-sales access."""
    def __init__(self):
        with np.load(CACHE/'train_observed.npz') as z:
            assert set(z.files)=={'observed','censored','caps','scale','channels'}
            self.observed=z['observed'];self.censored=z['censored'];self.scale=z['scale']
        assert self.observed.shape==(1400,256)
        self.schedule=read(ROOT/'results/candidate_07/sampling_manifest.json')
    def batch(self,index):
        r=self.schedule[index];cs=[v[0] for v in r['channels']];oo=r['origins']
        x=np.stack([self.observed[o-336:o,c] for o,c in zip(oo,cs)])
        sale=np.stack([self.observed[o:o+48,c] for o,c in zip(oo,cs)])
        mask=np.stack([self.censored[o:o+48,c] for o,c in zip(oo,cs)])
        return (torch.tensor(x,device='cuda'),torch.tensor(sale,device='cuda'),torch.tensor(mask,device='cuda'),
                torch.tensor(self.scale[cs],device='cuda'),torch.arange(8,device='cuda'))


def forward(m,x,g,w):
    w.boundary();w.forward+=1;return forecast(m,x,g)

def predict_validation(m,w):
    """Scoring-only path, never returns clean targets to a training update."""
    started=time.monotonic();p=[]
    with np.load(CACHE/'validation_scoring.npz') as z:
        contexts=z['context'];y=z['target'];sc=z['scale']
    for x in contexts:
        chunks=[]
        for start in range(0,256,32):
            with torch.no_grad():chunks.append(forward(m,torch.tensor(x[start:start+32],device='cuda'),torch.arange(32,device='cuda'),w)[1].cpu().numpy())
        p.append(np.concatenate(chunks))
    p=np.stack(p);value=score(p,y,sc)['scaled_2pinball'];w.validation+=time.monotonic()-started
    assert np.isfinite(value);return p,value

def store_prediction(name,p,y,sc,caps,origins,w=None):
    t=time.monotonic();path=CACHE/(name+'.npz')
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,prediction=p,target=y,scale=sc,caps=caps,censored=y>caps[None,:,None],origins=np.asarray(origins))
    if w:w.io+=time.monotonic()-t
    return dict(file=rel(path),sha256=sha(path))


def assert_sources():
    m=read(OUT/'manifest.json')
    assert source_files()==m['sources'],'Source changed after prepare'
    assert sha(OUT/'config.json')==m['config_sha256']
    assert sha(ROOT/'results/candidate_07/sampling_manifest.json')==m['sampling_sha256']
    for p,h in m['cache_hashes'].items():assert sha(ROOT/p)==h


def preflight(allow_rustdesk=False):
    assert read(OUT/'status.json')['stage']=='PREPARED','No preflight replay or smoke retry'
    assert_sources()
    r=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_censor_tail_controlled_v1.py'],cwd=ROOT,text=True,capture_output=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    save('cpu_tests.json',dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
    if r.returncode:raise RuntimeError('CPU tests failed; no model smoke started')
    w=Monitor('preflight',allow_rustdesk);record=dict(status='RUNNING',smoke_updates=0,arms=[])
    record['execution_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    save('preflight.json',record);state=read(OUT/'status.json');state['stage']='PREFLIGHT';save('status.json',state)
    try:
        w.boundary(startup=True);seed_all(41000);data=ObservedTraining();m=load_base();tasks=[];tails=[]
        for i in range(16):
            x,sale,mask,sc,g=data.batch(i)
            with torch.no_grad():
                z,p,loc,scale=forward(m,x,g,w)
                _,task,tail=objective('TAIL',z,p,sale,mask,loc,scale,sc,1.)
            tasks.append(float(task));tails.append(float(tail))
        t0=float(np.mean(tasks));c0=float(np.mean(tails));assert t0>0 and c0>0 and np.isfinite(t0+c0)
        lc=float(np.clip(.1*t0/c0,.001,100))
        save('lambda.json',dict(T0=t0,C0=c0,lambda_c=lc,unclipped=.1*t0/c0,clipped=lc in (.001,100),batches=list(range(16)),tasks=tasks,survival=tails))
        del m;cleanup();init_hash=None
        for arm in ARMS:
            m=build(41000);opt=optimizer(m);before=frozen(m);initial=snapshot(m);ih=treehash(initial)
            if init_hash is None:init_hash=ih
            assert ih==init_hash
            x,sale,mask,sc,g=data.batch(0)
            with torch.no_grad():
                p=forward(m,x,g,w)[1]
                with disabled(m):f0=forward(m,x,g,w)[1]
            error=float((p-f0).abs().max());assert error<=1e-6
            norms=[]
            for i in range(2):
                x,sale,mask,sc,g=data.batch(i);opt.zero_grad(set_to_none=True)
                z,p,loc,scale=forward(m,x,g,w);loss,task,tail=objective(arm,z,p,sale,mask,loc,scale,sc,lc)
                assert torch.isfinite(loss);w.backward+=1;loss.backward()
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params(m))
                norm=float(torch.nn.utils.clip_grad_norm_(params(m),1.));assert norm>0
                assert record['smoke_updates']<6;opt.step();record['smoke_updates']+=1;norms.append(norm);save('preflight.json',record)
            assert frozen(m)==before and treehash(snapshot(m))!=ih
            # Actual model DROP/TAIL shared task graph and hidden-value isolation: only capped batch is passed.
            z,p,loc,scale=forward(m,x,g,w)
            a,ta,_=objective('DROP',z,p,sale,mask,loc,scale,sc,lc)
            b,tb,_=objective('TAIL',z,p,sale.clone(),mask.clone(),loc,scale,sc,0.)
            ga=torch.autograd.grad(a,params(m),retain_graph=True);gb=torch.autograd.grad(b,params(m));w.backward+=2
            assert torch.equal(a,b) and torch.equal(ta,tb) and all(torch.equal(u,v) for u,v in zip(ga,gb))
            vpred,vscore=predict_validation(m,w);path=CACHE/f'smoke_{arm}.pt';torch.save(snapshot(m),path)
            restore(initial,m);restore(torch.load(path,weights_only=True),m)
            replay,rscore=predict_validation(m,w);assert np.array_equal(vpred,replay) and vscore==rscore
            record['arms'].append(dict(arm=arm,initial_hash=ih,identity_max_abs=error,gradient_norms=norms,frozen_unchanged=True,
                validation_reload_max_abs=float(np.max(abs(vpred-replay))),shared_task_and_gradient_exact=True,**audit(m)))
            save('preflight.json',record);del m,opt,initial,z,p,ga,gb;cleanup()
        record['status']='VERIFIED';record['historical_checkpoint_replays']=0
        record['leakage_scope']='Prepare metamorphic generator test + isolated capped training schema + actual model shared task/gradient; clean V scoring separate'
        save('preflight.json',record);state.update(stage='PREFLIGHT_VERIFIED',smoke_updates=record['smoke_updates']);save('status.json',state)
    except BaseException as e:
        record.update(status='IMPLEMENTATION_OR_ENVIRONMENT_UNRESOLVED',error=repr(e));save('preflight.json',record)
        state.update(stage='STOPPED_BEFORE_FITS',smoke_updates=record['smoke_updates'],reason=repr(e));save('status.json',state);raise
    finally:w.finish()


def train_fit(arm,seed,data,lc,w,entry,allfits,trajectories,diagnostics,state,initial_hashes):
    w.fit_start=time.monotonic();t=time.monotonic();v0=w.validation;io0=w.io;wait0=w.wait
    m=build(seed);opt=optimizer(m);before=frozen(m);ih=treehash(snapshot(m));initial_hashes.setdefault(str(seed),ih);assert initial_hashes[str(seed)]==ih
    entry.update(initial_hash=ih,parameter_audit=audit(m));torch.cuda.reset_peak_memory_stats()
    directory=CACHE/f'{seed}_{arm}';directory.mkdir();best=float('inf');beststep=None;nonzero=0;clipped=0;train_seconds=0.
    last_loss=None;nonzero_updates=0
    try:
        for step in range(361):
            w.boundary()
            if step in STEPS:
                p,val=predict_validation(m,w)
                trajectories.append(dict(seed=seed,arm=arm,step=step,V_primary=val,train_loss=last_loss))
                tick=time.monotonic();torch.save(snapshot(m),directory/f'step_{step}.pt')
                np.savez_compressed(directory/f'V_{step}.npz',prediction=p)
                w.io+=time.monotonic()-tick
                if val<best:best=val;beststep=step
                csvwrite(OUT/'trajectories.csv',trajectories)
                print(json.dumps(dict(seed=seed,arm=arm,step=step,V=val)),flush=True)
            if step==360:break
            tick=time.monotonic();x,sale,mask,sc,g=data.batch(step);opt.zero_grad(set_to_none=True)
            z,p,loc,scale=forward(m,x,g,w);loss,task,tail=objective(arm,z,p,sale,mask,loc,scale,sc,lc)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            diagnostic=tail_diagnostics(p,sale,mask,sc) if arm=='TAIL' else {}
            w.output_backward+=int(diagnostic.get('strong_right',0)>0)
            w.backward+=1;loss.backward()
            if any(v.grad is None or not torch.isfinite(v.grad).all() for v in params(m)):raise FloatingPointError('Nonfinite/missing gradient')
            norm=float(torch.nn.utils.clip_grad_norm_(params(m),1.));clipped+=int(norm>1);nonzero+=int(norm>0)
            old_parameters=[v.detach().clone() for v in params(m)]
            assert state['updates']<2160;opt.step();entry['updates']+=1;state['updates']+=1
            changed=any(not torch.equal(a,b) for a,b in zip(old_parameters,params(m)))
            nonzero_updates+=int(changed);del old_parameters
            last_loss=float(loss.detach());train_seconds+=time.monotonic()-tick
            diagnostics.append(dict(seed=seed,arm=arm,step=step+1,loss=last_loss,task=float(task.detach()),survival=float(tail.detach()),lambda_c=lc,
                gradient_norm=norm,clipped=norm>1,nonzero_parameter_update=changed,**diagnostic))
            if (step+1)%10==0:
                save('fits.json',allfits);save('status.json',state);csvwrite(OUT/'train_diagnostics.csv',diagnostics)
        assert frozen(m)==before
        entry.update(status='COMPLETE',selected_step=beststep,V_primary=best,nonzero_gradient_updates=nonzero,nonzero_parameter_updates=nonzero_updates,clipping_count=clipped,frozen_unchanged=True,
            selected_checkpoint=rel(directory/f'step_{beststep}.pt'),selected_sha256=sha(directory/f'step_{beststep}.pt'),
            endpoint_checkpoint=rel(directory/'step_360.pt'),endpoint_sha256=sha(directory/'step_360.pt'),
            checkpoints={str(s):sha(directory/f'step_{s}.pt') for s in STEPS},
            wall_seconds=time.monotonic()-t,training_seconds=train_seconds,validation_seconds=w.validation-v0,io_seconds=w.io-io0,gpu_wait_seconds=w.wait-wait0,
            gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved(),peak_cpu_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    except BaseException:
        # Save exact interrupted state once; do not silently retry from initialization.
        torch.save(dict(model=snapshot(m),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),updates=entry['updates']),directory/'interrupted_state.pt')
        entry['status']='INTERRUPTED';raise
    finally:
        save('fits.json',allfits);csvwrite(OUT/'train_diagnostics.csv',diagnostics);del m,opt;cleanup();w.fit_start=None


def evaluation_data():
    """Called only after all selections and their disk-reload V checks are sealed."""
    seal=read(OUT/'selection_seal.json');assert seal['V_reloads_passed'] and len(seal['winners'])==6
    assert seal['manifest_sha256']==sha(OUT/'manifest.json') and seal['config_sha256']==sha(OUT/'config.json')
    if (OUT/'E_dev_access.json').exists():raise RuntimeError('No repeated E_dev access')
    save('E_dev_access.json',dict(opened_at=time.time(),selection_sha256=sha(OUT/'selection_seal.json'),role='reused E_dev; V-selected primary and prespecified step360 secondary only'))
    with np.load(ROOT/'data/processed/m5/fit.npz') as z:past=z['values'];caps=z['caps']
    with np.load(ROOT/'data/processed/m5/evaluation.npz') as z:values=np.concatenate([past,z['tail']])
    origins=[1650,1698,1746,1794,1842,1890]
    contexts=np.stack([np.minimum(values[o-336:o],caps).astype(np.float32).T for o in origins])
    target=np.stack([values[o:o+48].T for o in origins]);sc=np.array(read(OUT/'manifest.json')['scale'])
    return contexts,target,sc,caps,origins


def predict_contexts(m,contexts,w):
    result=[]
    for x in contexts:
        pieces=[]
        for start in range(0,256,32):
            with torch.no_grad():pieces.append(forward(m,torch.tensor(x[start:start+32],device='cuda'),torch.arange(32,device='cuda'),w)[1].cpu().numpy())
        result.append(np.concatenate(pieces))
    return np.stack(result)


def run(allow_rustdesk=False):
    state=read(OUT/'status.json');assert state['stage']=='PREFLIGHT_VERIFIED' and not (OUT/'fits.json').exists(),'No automatic fit retry'
    assert_sources();assert read(OUT/'preflight.json')['status']=='VERIFIED'
    w=Monitor('run',allow_rustdesk);fits=[];traj=[];diags=[];initial_hashes={}
    try:
        w.boundary(startup=True);data=ObservedTraining();lc=read(OUT/'lambda.json')['lambda_c']
        state['stage']='TRAINING';save('status.json',state)
        for seed in SEEDS:
            for arm in ARMS:
                assert state['attempted_fits']<6
                entry=dict(seed=seed,arm=arm,status='STARTED',updates=0,started_at=time.time());fits.append(entry)
                state['attempted_fits']+=1;save('fits.json',fits);save('status.json',state)
                train_fit(arm,seed,data,lc,w,entry,fits,traj,diags,state,initial_hashes)
                state['completed_fits']+=1;save('fits.json',fits);save('status.json',state)
        assert state['updates']==2160 and len(fits)==6
        # All six choices fixed before scoring-only E_dev access. Reload clean V from disk first.
        winners=[dict(seed=f['seed'],arm=f['arm'],step=f['selected_step'],V_primary=f['V_primary'],checkpoint=f['selected_checkpoint'],sha256=f['selected_sha256']) for f in fits]
        save('selection_choices.json',dict(winners=winners,fixed_at=time.time(),config_sha256=sha(OUT/'config.json'),manifest_sha256=sha(OUT/'manifest.json')))
        reloads=[]
        for c in winners:
            assert sha(ROOT/c['checkpoint'])==c['sha256'];m=build(c['seed']);restore(torch.load(ROOT/c['checkpoint'],weights_only=True),m)
            p,val=predict_validation(m,w)
            with np.load(CACHE/f"{c['seed']}_{c['arm']}"/f"V_{c['step']}.npz") as z:err=float(np.max(abs(p-z['prediction'])))
            assert err==0 and val==c['V_primary'];reloads.append(dict(seed=c['seed'],arm=c['arm'],prediction_max_abs=err,primary_abs=abs(val-c['V_primary'])))
            del m;cleanup()
        save('selection_seal.json',dict(winners=winners,sealed_at=time.time(),choices_sha256=sha(OUT/'selection_choices.json'),V_reloads_passed=True,V_reloads=reloads,
            config_sha256=sha(OUT/'config.json'),manifest_sha256=sha(OUT/'manifest.json'),preflight_sha256=sha(OUT/'preflight.json'),lambda_sha256=sha(OUT/'lambda.json')))
        state['stage']='EVALUATING';save('status.json',state);tick=time.monotonic();contexts,y,sc,caps,origins=evaluation_data()
        index=[];seed_all(41000);m=load_base();p=predict_contexts(m,contexts,w)
        index.append(dict(arm='F0',seed=None,role='F0',step=0,**store_prediction('E_F0',p,y,sc,caps,origins,w)));f0=p.copy();del m;cleanup()
        for f in fits:
            m=build(f['seed'])
            for role,step,path in [('selected',f['selected_step'],f['selected_checkpoint']),('endpoint',360,f['endpoint_checkpoint'])]:
                restore(torch.load(ROOT/path,weights_only=True),m);p=predict_contexts(m,contexts,w)
                if step==0:assert np.array_equal(p,f0),'Selected F0 identity'
                index.append(dict(arm=f['arm'],seed=f['seed'],role=role,step=step,**store_prediction(f"E_{f['seed']}_{f['arm']}_{role}",p,y,sc,caps,origins,w)))
                save('prediction_index.json',index)
            del m;cleanup()
        w.scoring+=time.monotonic()-tick;state['stage']='EXECUTION_COMPLETE';save('status.json',state)
    except BaseException as e:
        state.update(stage='STOPPED',reason=repr(e));save('status.json',state);raise
    finally:w.finish()


def metric_vectors(p,y,sc,caps):
    p=np.sort(p.astype(np.float64),axis=2);y=y.astype(np.float64)
    e=y[:,:,None,:]-p;t=np.array(QUANTILES)[None,None,:,None]
    pin=(2*np.maximum(t*e,(t-1)*e)).mean(2)/sc[None,:,None]
    cens=y>caps[None,:,None];all_count=y.shape[0]*y.shape[2];rows={};vectors={}
    med=p[:,:,QUANTILES.index(.5)];lo=p[:,:,QUANTILES.index(.1)];hi=p[:,:,QUANTILES.index(.9)]
    for name,mask in [('all',np.ones_like(cens)),('censored',cens),('uncensored',~cens)]:
        n=mask.sum((0,2));active=n>0;total=np.where(mask,pin,0).sum((0,2))
        v=np.divide(total,n,out=np.zeros_like(total),where=active)
        def mean(values):return float((np.where(mask,values,0).sum((0,2))[active]/n[active]).mean()) if active.any() else None
        rows[name]=dict(primary=float(v[active].mean()) if active.any() else None,contribution=float((total/all_count).mean()),active_series=int(active.sum()),positions=int(mask.sum()),
            median_mae=mean(abs(med-y)),signed_bias=mean(med-y),interval80_coverage=mean((y>=lo)&(y<=hi)),interval80_width=mean(hi-lo))
        vectors[name]=v;vectors[name+'_active']=active
    assert abs(rows['all']['primary']-rows['censored']['contribution']-rows['uncensored']['contribution'])<1e-10
    return rows,vectors


def verify():
    assert read(OUT/'status.json')['stage'] in ['EXECUTION_COMPLETE','VERIFIED','REPORTED']
    assert_sources();history=read(OUT/'preserved_hashes.json')
    for p,h in history.items():assert sha(p)==h,p
    fits=read(OUT/'fits.json');assert len(fits)==6 and all(f['status']=='COMPLETE' and f['updates']==360 for f in fits)
    assert len({f['initial_hash'] for f in fits if f['seed']==41000})==len({f['initial_hash'] for f in fits if f['seed']==41001})==1
    seal=read(OUT/'selection_seal.json');access=read(OUT/'E_dev_access.json');assert access['selection_sha256']==sha(OUT/'selection_seal.json') and access['opened_at']>=seal['sealed_at']
    assert seal['lambda_sha256']==sha(OUT/'lambda.json') and seal['preflight_sha256']==sha(OUT/'preflight.json')
    checkpoints=0;vreplays=0;maximum=0.;rows=[];series={}
    with np.load(CACHE/'validation_scoring.npz') as z:vy=z['target'];sc=z['scale']
    trajectory=list(csv.DictReader(open(OUT/'trajectories.csv')))
    for f in fits:
        directory=CACHE/f"{f['seed']}_{f['arm']}"
        for step,h in f['checkpoints'].items():
            assert sha(directory/f'step_{step}.pt')==h;checkpoints+=1
            with np.load(directory/f'V_{step}.npz') as z:p=z['prediction']
            v=independent(p,vy,sc);r=next(r for r in trajectory if int(r['seed'])==f['seed'] and r['arm']==f['arm'] and int(r['step'])==int(step))
            delta=abs(v-float(r['V_primary']));maximum=max(maximum,delta);assert delta<=1e-10;vreplays+=1
        candidates=[r for r in trajectory if int(r['seed'])==f['seed'] and r['arm']==f['arm']]
        best=min(candidates,key=lambda r:(float(r['V_primary']),int(r['step'])))
        assert int(best['step'])==f['selected_step']
    with np.load(ROOT/'data/processed/m5/fit.npz') as z:original_prefix=z['values']
    with np.load(ROOT/'data/processed/m5/evaluation.npz') as z:original_values=np.concatenate([original_prefix,z['tail']])
    expected_target=np.stack([original_values[o:o+48].T for o in [1650,1698,1746,1794,1842,1890]])
    for item in read(OUT/'prediction_index.json'):
        assert sha(ROOT/item['file'])==item['sha256']
        with np.load(ROOT/item['file']) as z:p=z['prediction'];y=z['target'];sc=z['scale'];caps=z['caps'];mask=z['censored']
        assert np.array_equal(y,expected_target),'E_dev origin/target alignment'
        assert np.array_equal(mask,y>caps[None,:,None])
        primary=independent(p,y,sc);metrics,vectors=metric_vectors(p,y,sc,caps)
        maximum=max(maximum,abs(primary-metrics['all']['primary']));assert abs(primary-metrics['all']['primary'])<=1e-10
        for subgroup,v in metrics.items():
            # independent existing metric checks primary, MAE, interval for each subgroup.
            mask=None if subgroup=='all' else (y>caps[None,:,None] if subgroup=='censored' else y<=caps[None,:,None])
            reference=score(p,y,sc,mask)
            for k,rk in [('primary','scaled_2pinball'),('median_mae','median_mae'),('interval80_coverage','interval80_coverage'),('interval80_width','interval80_width')]:
                assert abs(v[k]-reference[rk])<=1e-10,(subgroup,k)
            rows.append(dict(arm=item['arm'],seed=item['seed'],role=item['role'],step=item['step'],subgroup=subgroup,**v))
        series[(item['arm'],item['seed'],item['role'])]=vectors
    assert len(read(OUT/'prediction_index.json'))==13
    # Shared paired resamples; retain optimizer seeds, origins and horizons together.
    rng=np.random.default_rng(41002);indices=rng.integers(0,256,(2000,256));comparisons=[]
    f0=series[('F0',None,'F0')]['all']
    for role in ['selected','endpoint']:
        tail=np.mean([series[('TAIL',s,role)]['all'] for s in SEEDS],axis=0)
        for base in ['DROP','NAIVE','F0']:
            b=f0 if base=='F0' else np.mean([series[(base,s,role)]['all'] for s in SEEDS],axis=0)
            gs=100*(b[indices].mean(1)-tail[indices].mean(1))/b[indices].mean(1)
            comparisons.append(dict(role=role,comparison='TAIL_vs_'+base,baseline_primary=float(b.mean()),TAIL_primary=float(tail.mean()),
                absolute_reduction=float((b-tail).mean()),gain_percent=gain(float(b.mean()),float(tail.mean())),
                CI95_lower=float(np.quantile(gs,.025)),CI95_upper=float(np.quantile(gs,.975))))
    f0_primary=float(f0.mean())
    for r in rows:
        if r['subgroup']=='all':r['E_model_over_E_F0']=r['primary']/f0_primary
    csvwrite(OUT/'metrics.csv',rows);csvwrite(OUT/'comparisons.csv',comparisons)
    means=[]
    for role in ['selected','endpoint']:
        for arm in ['F0']+ARMS:
            for subgroup in ['all','censored','uncensored']:
                rr=[r for r in rows if r['arm']==arm and r['subgroup']==subgroup and (r['role']==role or arm=='F0')]
                means.append(dict(role=role,arm=arm,subgroup=subgroup,**{k:float(np.mean([r[k] for r in rr])) for k in ['primary','contribution','median_mae','signed_bias','interval80_coverage','interval80_width']}))
    csvwrite(OUT/'seed_mean_metrics.csv',means)
    seed_gains=[]
    for s in SEEDS:
        tail=float(series[('TAIL',s,'selected')]['all'].mean())
        seed_gains.append(dict(seed=s,TAIL_primary=tail,**{f'gain_vs_{b}_percent':gain(float((f0 if b=='F0' else series[(b,s,'selected')]['all']).mean()),tail) for b in ['DROP','NAIVE','F0']}))
    save('analysis.json',dict(seed_gains=seed_gains,comparisons=comparisons,bootstrap_indices_sha256=arrayhash(indices)))
    save('verification.json',dict(status='VERIFIED',fits=6,updates=sum(f['updates'] for f in fits),smoke_updates=read(OUT/'preflight.json')['smoke_updates'],
        checkpoint_hashes=checkpoints,V_prediction_replays=vreplays,E_prediction_replays=13,independent_primary_max_abs=maximum,
        historical_files_unchanged=len(history),selected_V_disk_reloads=seal['V_reloads'],sources_unchanged=True,
        E_access_after_all_selections=True,bootstrap_samples=2000,additional_model_forward=0,additional_updates=0,
        limitations='Cached tensor/scalar replay; selected checkpoint GPU reloads were performed before E access. Paired series CI is descriptive; no store/date clustering.'))
    state=read(OUT/'status.json');state['stage']='VERIFIED';save('status.json',state)
    print(json.dumps(read(OUT/'verification.json')),flush=True)


def report():
    assert read(OUT/'verification.json')['status']=='VERIFIED'
    a=read(OUT/'analysis.json');fits=read(OUT/'fits.json');metrics=list(csv.DictReader(open(OUT/'metrics.csv')))
    signs=[r['gain_vs_DROP_percent'] for r in a['seed_gains']]
    naive=next(r for r in a['comparisons'] if r['role']=='selected' and r['comparison']=='TAIL_vs_NAIVE')
    if all(v>0 for v in signs) and naive['gain_percent']>0:decision='제한된 후속 검토 신호'
    elif min(signs)<0<max(signs) and next(r for r in a['comparisons'] if r['role']=='selected' and r['comparison']=='TAIL_vs_DROP')['gain_percent']>0:decision='불안정한 개발 신호'
    else:decision='현재 설정에서 추가 가치 미확보'
    means=list(csv.DictReader(open(OUT/'seed_mean_metrics.csv')))
    mean=lambda arm,subgroup:float(next(r['primary'] for r in means if r['role']=='selected' and r['arm']==arm and r['subgroup']==subgroup))
    if mean('TAIL','all')>mean('DROP','all') and mean('TAIL','censored')<mean('DROP','censored'):decision='회복/과대예측 간 절충'
    qualifiers=[]
    if mean('TAIL','all')>mean('F0','all'):
        qualifiers.append('적응 자체의 실용적 이득 미확보')
        if mean('TAIL','all')<min(mean('DROP','all'),mean('NAIVE','all')):qualifiers.append('비교군 대비 적응 손해 완화')
    if all(f['selected_step']==0 for f in fits):qualifiers.append('학습된 수정이 선택되지 않음')
    if all(mean(arm,'all')>mean('F0','all') for arm in ARMS):qualifiers.append('세 학습 방식 모두 F0보다 나쁨; 이 조건의 적응이 불리했으며 모든 PEFT로 일반화하지 않음')
    selected=[r for r in metrics if r['role']=='selected' and r['subgroup']=='all']
    lines=['# 품절·검열 관측 PEFT: 꼬리 수정 통제 파일럿', '',
        f'본학습 {len(fits)} fits / {sum(f["updates"] for f in fits)} updates, smoke {read(OUT/"preflight.json")["smoke_updates"]} updates 완료. 판정: **{decision}**. 자동 후속 실행 없음.', '',
        '## 1. 문제와 비교', '',
        '기존 유한 지지범위와 survival clipping은 강한 하한 위반에서 큰 손실과 0 gradient를 함께 만든다. NAIVE(관측량 감독), DROP(검열 target 제외), TAIL(DROP + 하한 초과 사건 벌점)을 동일 Chronos-2 rank8 LoRA와 360-step 예산으로 비교했다.', '',
        '## 2. 수정과 데이터 계약', '',
        '정렬된 분위수 사이를 선형 보간하고 양끝에 지수 꼬리를 붙여 log-survival을 직접 계산한다. 이는 연속 분포 proxy 가정이며 정확한 count likelihood, ISQF 전체 재현 또는 새 PEFT 구조가 아니다. 원래 CDF와 과거 결과는 수정하지 않았다.', '',
        '| 항목 | 과거 Candidate07 | 이번 파일럿 |','| --- | --- | --- |',
        '| 꼬리 | 유한 연장 + 확률 clipping | 무한 지수 꼬리 + FP64 log-domain |',
        '| 정규화 scale | 원판매량 학습 표준편차 | capped 관측 학습 표준편차 |',
        '| lambda | 첫 batch | F0의 첫 16개 train batch |',
        '| 보존항 | 비교 arm에 포함 | 없음 |',
        '| 예산 | 기존 2 LR | LR1e-4, seeds41000/41001, 6 fits |', '',
        '기존 M5 256개 계열과 상한을 유지했다. 숨긴 초과 크기는 학습 데이터 객체에 들어가지 않는다. 모든 arm에는 원판매량 clean V 선택 권한을 동일하게 부여했다. E_dev는 이미 사용된 개발 평가 구간이며 독립 test가 아니다. 원판매량 자체가 실제 잠재 수요라는 보장도 없다.', '',
        '## 3. 전체 결과', '',
        '| arm | seed | 선택 step | 전체 primary | E_model/E_F0 |','| --- | --- | --- | --- | --- |']
    for r in [r for r in metrics if r['subgroup']=='all' and r['role'] in ['F0','selected']]:
        lines.append(f"| {r['arm']} | {r['seed'] or 'shared'} | {r['step']} | {float(r['primary']):.9f} | {float(r['E_model_over_E_F0']):.9f} |")
    lines += ['', '주 지표는 계열별 동일 가중 scaled 2-pinball이며 CRPS/WQL이 아니다. primary는 V-selected, step360은 별도 secondary로 고정했다. 두 seed 평균은 아래 TAIL 비교표에 포함된다.', '',
        '| 역할 | 비교 | baseline 평균 primary | TAIL 평균 primary | 개선율 % | 기술적 95% CI % |','| --- | --- | --- | --- | --- | --- |']
    for r in a['comparisons']:lines.append(f"| {r['role']} | {r['comparison']} | {r['baseline_primary']:.9f} | {r['TAIL_primary']:.9f} | {r['gain_percent']:+.6f} | [{r['CI95_lower']:+.6f}, {r['CI95_upper']:+.6f}] |")
    lines += ['', '개선율은 100*(baseline−TAIL)/baseline이다. 2,000회 paired series bootstrap은 같은 계열의 모든 시점과 두 seed를 함께 유지했다. 같은 상품/상점·날짜의 상관을 완전히 처리하지 못하므로 유의성이나 일반화를 확증하지 않는다.', '',
        '## 4. 하위집단과 학습 신호', '',
        '[metrics.csv](metrics.csv)에 전체·synthetic-censored·나머지 위치의 primary, 전체 오차 기여도, median MAE와 signed bias, q10–q90 coverage/width를 기록했다. 검열/비검열 기여도는 전체 위치 분모를 유지하여 전체 primary와 합이 일치한다. subgroup primary는 해당 위치가 있는 계열의 동일 가중 평균이다.', '',
        '[train_diagnostics.csv](train_diagnostics.csv)에 매 update의 task/survival, gradient norm/clipping과 TAIL의 꼬리 사용·gap floor·knot 이동·강한 upper-tail common-shift 미분을 기록했다. 모든 개별 knot 미분의 부호를 동일하게 요구하지 않는다.', '',
        '## 5. 자원과 검증', '',
        '[fits.json](fits.json)에 fit별 wall/training/V/I/O/대기 시간과 최대 allocated/reserved GPU·CPU RAM, nonzero gradient updates, 선택 step을 기록했다. [verification.json](verification.json)에 독립 수치 재생, checkpoint reload, 원자료/과거 결과/source 해시 검사 및 실제 counts를 기록했다.', '',
        'TAIL의 확률 계산은 추가 비용이다. 동일 파라미터·updates는 동일 시간·메모리를 의미하지 않는다. 자원 우위를 주장하지 않는다. CPU/smoke 합격은 성능 성공 판정과 별개다.', '',
        '## 6. 한계와 다음 결정', '',
        decision+'. 이 결과로 실제 재고 수익·잠재 수요 복원·새 방법 신규성·독립 test 성능을 주장하지 않는다. 과거 buggy loss의 동조건 재학습을 하지 않았으므로 꼬리 구현만이 과거 실패의 원인이었다고 결론 내릴 수 없다. 고정 상한 초과량의 분포는 하한 사건만으로 식별되지 않으며 사전학습과 꼬리 가정에 의존한다.', '',
        '6 fits 이후 추가 학습은 실행하지 않는다. 새 PEFT로 연결하려면 별도 사용자 판단과 신규성·필요성 검토가 필요하다.', '',
        '참고: [Park et al., AISTATS 2022](https://proceedings.mlr.press/v151/park22a.html), [GluonTS ISQF exponential-tail 문서](https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.distributions.isqf.html), [FreshRetailNet-50K](https://arxiv.org/html/2505.16319v5). 이번에는 FreshRetailNet 다운로드나 학습을 하지 않았다.', '']
    lines += ['부가 판정: '+('; '.join(qualifiers) if qualifiers else '추가 해당 사항 없음'), '', '### 선택 모델의 seed 평균과 하위집단', '', '| arm | 집단 | primary | 전체 기여도 | median MAE | signed bias | coverage80 | width80 |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in means:
        if r['role']=='selected':lines.append('| '+r['arm']+' | '+r['subgroup']+' | '+' | '.join(f"{float(r[k]):.7f}" for k in ['primary','contribution','median_mae','signed_bias','interval80_coverage','interval80_width'])+' |')
    lines += ['', '자원 총계: '+json.dumps(read(OUT/'resources_run.json'),ensure_ascii=False)+'. 전처리/CPU 구현 시간은 모델 실행 wall time에 포함되지 않는다. GPU startup 안정화 대기 30초도 대기 비용에 포함한다.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    # At most two figures, generated only after independent metric verification.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,subgroup in zip(axes,['all','censored']):
        values=[mean(arm,subgroup) for arm in ['F0']+ARMS]
        ax.bar(['F0']+ARMS,values,color=['#888888','#4477aa','#66aabb','#cc6677'])
        ax.set(title=subgroup+' positions',ylabel='Equal-series scaled 2-pinball')
        ax.set_ylim(bottom=0)
    fig.suptitle('Reused E_dev: V-selected models, mean of two seeds')
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    diagnostics=list(csv.DictReader(open(OUT/'train_diagnostics.csv')))
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for seed in SEEDS:
        rr=[r for r in diagnostics if r['arm']=='TAIL' and int(r['seed'])==seed]
        steps=[int(r['step']) for r in rr]
        axes[0].plot(steps,[float(r['survival']) for r in rr],label=str(seed),alpha=.8)
        axes[1].plot(steps,[float(r['gradient_norm']) for r in rr],label=str(seed),alpha=.8)
    axes[0].set(xlabel='Optimizer step',ylabel='Unweighted survival loss / valid positions')
    axes[1].set(xlabel='Optimizer step',ylabel='LoRA gradient L2 before clipping',yscale='log')
    for ax in axes:ax.legend()
    fig.suptitle('TAIL training diagnostics; no smoothing')
    fig.savefig(OUT/'tail_diagnostics.png',dpi=160);plt.close(fig)
    with open(OUT/'REPORT.md','a') as f:
        f.write('\n![E_dev comparison](comparison.png)\n\n![TAIL diagnostics](tail_diagnostics.png)\n')
    save('decision.json',dict(decision=decision,qualifiers=qualifiers,automatic_followup=False,seed_gains=a['seed_gains']))
    state=read(OUT/'status.json');state['stage']='REPORTED';state['decision']=decision;save('status.json',state)
    print(decision,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','preflight','run','verify','report','status'])
    p.add_argument('--allow-rustdesk',action='store_true',help='Only use with explicit user authorization; exact executable <=512 MiB')
    a=p.parse_args()
    if a.stage=='status':print(json.dumps(read(OUT/'status.json') if (OUT/'status.json').exists() else dict(stage='NOT_PREPARED'),ensure_ascii=False,indent=2));return
    if a.stage in ['preflight','run']:globals()[a.stage](a.allow_rustdesk)
    else:globals()[a.stage]()

if __name__=='__main__':main()
