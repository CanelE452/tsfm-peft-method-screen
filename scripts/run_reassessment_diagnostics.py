"""Bounded serial diagnostics; deliberately has no evaluation/test-data loader."""
import argparse
from contextlib import nullcontext
import csv
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from tsfm_peft_screen.reassessment import DiagnosticModel, objective, schedule, choose
from tsfm_peft_screen.reproducibility import ROOT, sha, digest, write_json, seed_all, guard
from tsfm_peft_screen.metrics import score, independent
from tsfm_peft_screen.forecast_query.gpu import GPUWatch
from tsfm_peft_screen.forecast_query.equal_time import preserve_rng
from run_calibration_anchor_followup import parameters, restore, tensor_hash, frozen_hash, trainable

CONFIG_PATH = ROOT/'configs/reassessment_diagnostics_20260914.json'
CFG = json.loads(CONFIG_PATH.read_text())
OUT = ROOT/'results'/CFG['run_id']
CACHE = ROOT/'.cache'/CFG['run_id']
RESEARCH = ROOT/'research'/CFG['run_id']


def read(path):
    return json.loads(path.read_text())


def sources():
    return {str(p.relative_to(ROOT)): sha(p) for folder in ('src','scripts','configs','tests')
            for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}


def input_receipts():
    old = read(ROOT/'results/overnight_20260913/contract.json')
    rows = {}
    for name in ('etth1','traffic'):
        path = ROOT/'.cache/overnight_20260913/data'/f'{name}_development.npz'
        assert sha(path) == old['staged_data'][name]['development_sha256']
        rows[name] = dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
    path = ROOT/'data/processed/m5/fit.npz'
    manifest = ROOT/'data/processed/m5/manifest.json'
    assert sha(path) == read(manifest)['files']['fit.npz']
    rows['m5'] = dict(path=str(path.relative_to(ROOT)),sha256=sha(path),
                      manifest=str(manifest.relative_to(ROOT)),manifest_sha256=sha(manifest))
    return rows


def load_data(dataset):
    receipt = input_receipts()[dataset]
    with np.load(ROOT/receipt['path'],allow_pickle=False) as z:
        values, scale = z['values'], z['scale']
        zero_fraction = z['zero_fraction'] if dataset=='m5' else None
    if dataset=='m5':
        meta = read(ROOT/receipt['manifest'])
        train_origins, validation_origins = meta['origins']['train'], meta['origins']['validation']
        assert len(values)==1650 and len(scale)==256
    else:
        c=CFG['topics']['anchor']
        train_origins, validation_origins = [list(range(c[k][0],c[k][1]+1,c[k][2]))
                                             for k in ('train_origins','validation_origins')]
        assert len(scale)==4 and len(values)==max(validation_origins)+48
    assert max(train_origins)+48<=min(validation_origins)
    assert max(validation_origins)+48<=len(values)
    return dict(values=values,scale=scale,train=train_origins,validation=validation_origins,
                zero_fraction=zero_fraction)


def batch(data, sampling, context, topic):
    xs,ys,sc,groups=[],[],[],[]
    for index,(origin,channels) in enumerate(zip(sampling['origins'],sampling['channels'])):
        assert context<=origin and origin+48<=len(data['values'])
        channels=np.asarray(channels)
        xs.append(data['values'][origin-context:origin,channels].T.copy())
        ys.append(data['values'][origin:origin+48,channels].T.copy())
        sc.extend(data['scale'][channels])
        groups.extend([index]*len(channels) if topic=='anchor' else range(len(groups),len(groups)+len(channels)))
    return (torch.tensor(np.concatenate(xs),device='cuda',dtype=torch.float32),
            torch.tensor(np.concatenate(ys),device='cuda',dtype=torch.float32),
            torch.tensor(groups,device='cuda'),torch.tensor(sc,device='cuda',dtype=torch.float32))


def precision(topic):
    return torch.autocast('cuda',dtype=torch.bfloat16) if topic=='anchor' else nullcontext()


def check_contract():
    c=read(OUT/'contract.json')
    assert c['config']==CFG
    for name,h in {**c['source_hashes'],**c['historical_result_hashes']}.items():
        assert sha(ROOT/name)==h,name
    assert input_receipts()==c['inputs']
    return c


class Worker:
    def __init__(self,topic,smoke=False):
        self.topic,self.c=topic,CFG['topics'][topic]
        self.folder=(RESEARCH if smoke else OUT)/topic
        self.cache=(ROOT/'.cache'/f'{CFG["run_id"]}_smoke' if smoke else CACHE)/topic
        self.folder.mkdir(parents=True,exist_ok=True)
        self.cache.mkdir(parents=True,exist_ok=True)
        self.watch=GPUWatch(self.folder/'gpu.json')
        self.start=time.monotonic()
        self.state=dict(status='WAITING_FOR_GPU',fits_attempted=0,fits_completed=0,training_updates=0)

    def emit(self,**kw):
        self.state.update(kw)
        self.state.update(wall_seconds=time.monotonic()-self.start,heartbeat_unix=time.time())
        write_json(self.folder/'status.json',self.state)

    def check(self,label,force=False):
        self.watch.check(label,force)
        guard()

    def wait(self):
        idle=None
        while True:
            if time.monotonic()-self.start>CFG['max_queue_wall_seconds']:
                raise TimeoutError('GPU wait exceeded diagnostic wall budget')
            r=self.watch.read('startup_wait')
            good=not r['external_pids'] and r['free_mib']>=CFG['startup_free_mib'] and r['utilization_percent']<90
            idle=(idle or time.monotonic()) if good else None
            self.emit(status='WAITING_FOR_GPU',free_mib=r['free_mib'],external_pids=r['external_pids'])
            if idle is not None and time.monotonic()-idle>=CFG['startup_idle_seconds']: return
            time.sleep(5)

    def evaluate(self,model,data,origins,tag,frozen=False,ablation='normal'):
        pp,yy=[],[]
        start=time.monotonic()
        with preserve_rng(),torch.no_grad():
            if self.topic=='anchor':
                for j in range(0,len(origins),2):
                    self.check('validation_'+tag)
                    oo=origins[j:j+2]
                    x,y,g,_=batch(data,dict(origins=oo,channels=[list(range(4)) for _ in oo]),1024,self.topic)
                    with precision(self.topic):p=model(x,g,frozen=frozen)[1]
                    pp.append(p.cpu().numpy().reshape(-1,4,21,48));yy.append(y.cpu().numpy().reshape(-1,4,48))
                p,y=np.concatenate(pp),np.concatenate(yy)
            else:
                for origin in origins:
                    pc,yc=[],[]
                    for channels in np.array_split(np.arange(256),8):
                        self.check('validation_'+tag)
                        x,y,g,_=batch(data,dict(origins=[origin],channels=[channels]),336,self.topic)
                        p=model(x,g,frozen=frozen,ablation=ablation)[1]
                        pc.append(p.cpu().numpy());yc.append(y.cpu().numpy())
                    pp.append(np.concatenate(pc));yy.append(np.concatenate(yc))
                p,y=np.stack(pp),np.stack(yy)
        assert np.isfinite(p).all()
        path=self.cache/(tag+'.npz');assert not path.exists(),path
        np.savez_compressed(path,prediction=p,target=y,scale=data['scale'],origins=origins)
        return dict(metrics=score(p,y,data['scale']),prediction_file=str(path.relative_to(ROOT)),
                    prediction_sha256=sha(path),wall_seconds=time.monotonic()-start)

    def fit(self,dataset,arm,seed,lr,recipe,data,teacher):
        fid=f'{dataset}_{seed}_{arm}_{recipe}'
        assert not (self.folder/(fid+'_attempt.json')).exists(),'No implicit retry'
        self.state['fits_attempted']+=1
        assert self.state['fits_attempted']<=self.c['fit_cap']
        self.emit(status='TRAINING',fit_id=fid,step=0)
        write_json(self.folder/(fid+'_attempt.json'),dict(fit_id=fid,started_unix=time.time()))
        seed_all(seed);self.check('model_load',True)
        model=DiagnosticModel(self.topic,arm,seed)
        ps=trainable(model);initial=parameters(model);frozen=frozen_hash(model)
        optimizer=torch.optim.AdamW(ps.values(),lr=lr,weight_decay=0.)
        samples=schedule(self.topic,data['train'],seed,self.c['steps'])
        samplemap={o:i for i,o in enumerate(data['train'])}
        start=time.monotonic();rows=[];ledger=[]
        for step in range(self.c['steps']+1):
            if time.monotonic()-start>CFG['max_fit_wall_seconds']:raise TimeoutError('Fit cap; no retry')
            if step in self.c['checkpoints']:
                row=self.evaluate(model,data,data['validation'],fid+f'_V_{step}')
                cp=self.cache/(fid+f'_V_{step}.pt');torch.save(parameters(model),cp)
                row.update(fit_id=fid,dataset=dataset,arm=arm,seed=seed,lr=lr,recipe=recipe,step=step,
                           checkpoint_file=str(cp.relative_to(ROOT)),checkpoint_sha256=sha(cp))
                rows.append(row);write_json(self.folder/(fid+'_trajectory.json'),rows)
                print('V',fid,step,row['metrics']['scaled_2pinball'],flush=True)
            if step==self.c['steps']:break
            self.check('train_'+fid)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
            x,y,g,sc=batch(data,samples[step],self.c['context'],self.topic)
            with precision(self.topic):z,p,loc,scale=model(x,g)
            target=None
            if arm.endswith('_anchor'):
                idx=[samplemap[o] for o in samples[step]['origins']]
                target=torch.tensor(teacher[idx].reshape(len(x),21,48),device='cuda')
            task,reg=objective(self.topic,arm,z,p,y,loc,scale,sc,target,self.c.get('anchor_coefficient',.1))
            loss=task+reg
            assert torch.isfinite(loss) and torch.isfinite(p).all()
            loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in ps.values())
            norm=torch.nn.utils.clip_grad_norm_(list(ps.values()),1.,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize()
            ledger.append(dict(step=step+1,seconds=time.perf_counter()-tick,loss=float(task.detach()),
                regularizer=float(reg.detach()),gradient_norm=float(norm),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            self.state['training_updates']+=1
            if (step+1)%25==0:self.emit(step=step+1)
        final_hash=tensor_hash(parameters(model))
        assert frozen_hash(model)==frozen and final_hash!=tensor_hash(initial)
        best=choose(rows)
        restore(model,torch.load(ROOT/best['checkpoint_file'],map_location='cpu',weights_only=True))
        replay=self.evaluate(model,data,data['validation'],fid+'_best_reload')
        with np.load(ROOT/best['prediction_file']) as a,np.load(ROOT/replay['prediction_file']) as b:
            assert np.array_equal(a['prediction'],b['prediction'])
        branch={n:float((p.detach().cpu()-initial[n]).norm()) for n,p in ps.items() if n.startswith('adapter.')}
        result=dict(fit_id=fid,topic=self.topic,dataset=dataset,arm=arm,seed=seed,lr=lr,recipe=recipe,
                    updates=len(ledger),active_seconds=sum(r['seconds'] for r in ledger),
                    wall_seconds=time.monotonic()-start,trainable_parameters=sum(p.numel() for p in ps.values()),
                    frozen_sha256=frozen,initial_parameters_sha256=tensor_hash(initial),final_parameters_sha256=final_hash,
                    sample_sequence_sha256=digest(samples),best=best,reload=replay,
                    selected_adapter_parameter_change_norms=branch,resources=ledger)
        write_json(self.folder/(fid+'_fit.json'),result)
        self.state['fits_completed']+=1;self.emit()
        del model,ps,optimizer,initial;gc.collect();torch.cuda.empty_cache()
        return {k:v for k,v in result.items() if k!='resources'} | {'fit_receipt_sha256':sha(self.folder/(fid+'_fit.json'))}

    def train(self):
        check_contract();assert not (self.folder/'fits.json').exists()
        self.wait();fits=[];baselines=[]
        for dataset in self.c['datasets']:
            data=load_data(dataset);seed_all(self.c['seeds'][0])
            model=DiagnosticModel(self.topic,self.c['arms'][0],self.c['seeds'][0]);teacher=None
            f0=self.evaluate(model,data,data['validation'],dataset+'_F0_V',frozen=True)
            baselines.append(dict(dataset=dataset,split='validation',**f0))
            if self.topic=='anchor':
                cached=self.evaluate(model,data,data['train'],dataset+'_F0_train',frozen=True)
                baselines.append(dict(dataset=dataset,split='train',**cached))
                with np.load(ROOT/cached['prediction_file']) as z:teacher=z['prediction']
            del model;gc.collect();torch.cuda.empty_cache()
            write_json(self.folder/'baselines.json',baselines)
            for seed in self.c['seeds']:
                for arm in self.c['arms']:
                    for recipe,lr in enumerate(self.c['learning_rates']):
                        fits.append(self.fit(dataset,arm,seed,lr,recipe,data,teacher))
        write_json(self.folder/'fits.json',fits)
        selected=[]
        for ds in self.c['datasets']:
            for seed in self.c['seeds']:
                for arm in self.c['arms']:
                    selected.append(choose([f['best'] for f in fits if (f['dataset'],f['seed'],f['arm'])==(ds,seed,arm)]))
        seal=dict(selections=selected,contract_sha256=sha(OUT/'contract.json'),fits_sha256=sha(self.folder/'fits.json'),
                  created_unix=time.time(),scope='V diagnostics only; not authorized to open E')
        seal['sha256']=digest(seal);write_json(self.folder/'selection_seal.json',seal)
        if self.topic=='dualclock':
            ablations=[];data=load_data('m5')
            for s in selected:
                if s['arm']=='standard':continue
                seed_all(s['seed']);self.check('ablation_load',True)
                model=DiagnosticModel(self.topic,s['arm'],s['seed'])
                restore(model,torch.load(ROOT/s['checkpoint_file'],map_location='cpu',weights_only=True))
                for mode in self.c['validation_ablations']:
                    row=self.evaluate(model,data,data['validation'],s['fit_id']+'_'+mode,ablation=mode)
                    ablations.append(dict(arm=s['arm'],seed=s['seed'],mode=mode,selected=s,**row))
                    write_json(self.folder/'ablations.json',ablations)
                del model;gc.collect();torch.cuda.empty_cache()
        check_contract();self.emit(status='COMPLETE_V_DIAGNOSTIC',new_evaluation_accesses=0)


def smoke():
    RESEARCH.mkdir(parents=True,exist_ok=True)
    assert not (RESEARCH/'smoke.json').exists(),'Preserve existing smoke receipt'
    lock=open(ROOT/'.cache/gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=[]
    for topic,c in CFG['topics'].items():
        w=Worker(topic,smoke=True);w.wait();data=load_data(c['datasets'][0])
        samples=schedule(topic,data['train'],c['seeds'][0],2)
        for arm in c['arms']:
            seed_all(c['seeds'][0]);w.check('smoke_load',True)
            model=DiagnosticModel(topic,arm,c['seeds'][0]);init=parameters(model);frozen=frozen_hash(model)
            ps=trainable(model);opt=torch.optim.AdamW(ps.values(),lr=1e-4,weight_decay=0)
            x,y,g,sc=batch(data,samples[0],c['context'],topic)
            with torch.no_grad(),precision(topic):
                initial=model(x,g)[1];f0=model(x,g,frozen=True)[1]
            assert torch.equal(initial,f0),(topic,arm,float((initial-f0).abs().max()))
            for step,s in enumerate(samples):
                w.check('smoke_step');opt.zero_grad(set_to_none=True)
                x,y,g,sc=batch(data,s,c['context'],topic)
                with torch.no_grad(),precision(topic):teacher=model(x,g,frozen=True)[1]
                with precision(topic):z,p,loc,scale=model(x,g)
                loss,reg=objective(topic,arm,z,p,y,loc,scale,sc,teacher)
                (loss+reg).backward()
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in ps.values())
                torch.nn.utils.clip_grad_norm_(list(ps.values()),1.,error_if_nonfinite=True);opt.step()
            assert tensor_hash(init)!=tensor_hash(parameters(model)) and frozen_hash(model)==frozen
            if model.adapter is not None:
                assert any(float((p.detach().cpu()-init[n]).abs().max())>0 for n,p in ps.items() if n.startswith('adapter.encoder.'))
            rows.append(dict(topic=topic,arm=arm,updates=2,step0_exact=True,frozen_unchanged=True,
                             trainable_parameters=sum(p.numel() for p in ps.values()),finite_gradient=True))
            write_json(RESEARCH/'smoke_progress.json',rows)
            del model,ps,opt,init;gc.collect();torch.cuda.empty_cache()
    write_json(RESEARCH/'smoke.json',dict(status='PASS',updates=14,rows=rows,source_hashes=sources(),inputs=input_receipts()))


def initialize():
    assert not OUT.exists() and not CACHE.exists(),'No overwrite/retry'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'Commit before execution'
    s=read(RESEARCH/'smoke.json');assert s['status']=='PASS' and s['source_hashes']==sources()
    from huggingface_hub import hf_hub_download
    from tsfm_peft_screen.backbone import MODEL_ID,REVISION
    model_files=read(ROOT/'results/screening_summary/common_integrity.json')['model_files']
    for name,h in model_files.items():assert sha(hf_hub_download(MODEL_ID,name,revision=REVISION,local_files_only=True))==h
    history={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'results').rglob('*') if p.is_file()}
    OUT.mkdir();CACHE.mkdir()
    write_json(OUT/'contract.json',dict(config=CFG,source_hashes=sources(),historical_result_hashes=history,
        execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        model_revision=REVISION,model_files=model_files,inputs=input_receipts(),smoke_sha256=sha(RESEARCH/'smoke.json')))


def finalize(verify_only=False):
    contract=check_contract();cache_count=0;max_error=0.;seen=set();fits_all=[];comparisons=[];prefix=[]
    def replay(row):
        nonlocal cache_count,max_error
        path=ROOT/row['prediction_file'];assert sha(path)==row['prediction_sha256']
        if str(path) not in seen:
            with np.load(path,allow_pickle=False) as z:
                err=abs(independent(z['prediction'],z['target'],z['scale'])-row['metrics']['scaled_2pinball'])
            assert err<1e-10;max_error=max(max_error,err);cache_count+=1;seen.add(str(path))
    for topic,c in CFG['topics'].items():
        folder=OUT/topic;fits=read(folder/'fits.json');assert len(fits)==c['fit_cap']
        assert {(f['dataset'],f['seed'],f['arm'],f['recipe']) for f in fits}=={
            (d,s,a,r) for d in c['datasets'] for s in c['seeds'] for a in c['arms'] for r in range(2)}
        seal=read(folder/'selection_seal.json');h=seal.pop('sha256');assert digest(seal)==h
        assert seal['contract_sha256']==sha(OUT/'contract.json') and seal['fits_sha256']==sha(folder/'fits.json')
        allrows=[]
        for f in fits:
            full=folder/(f['fit_id']+'_fit.json');assert sha(full)==f['fit_receipt_sha256']
            ledger=read(full)['resources'];assert len(ledger)==f['updates']==c['steps']
            assert [r['step'] for r in ledger]==list(range(1,c['steps']+1))
            assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in ledger)
            assert abs(sum(r['seconds'] for r in ledger)-f['active_seconds'])<1e-7
            data=load_data(f['dataset'])
            assert digest(schedule(topic,data['train'],f['seed'],c['steps']))==f['sample_sequence_sha256']
            rows=read(folder/(f['fit_id']+'_trajectory.json'));assert [r['step'] for r in rows]==c['checkpoints']
            assert choose(rows)==f['best']
            for row in rows:
                replay(row);assert sha(ROOT/row['checkpoint_file'])==row['checkpoint_sha256']
            replay(f['reload'])
            with np.load(ROOT/f['best']['prediction_file']) as a,np.load(ROOT/f['reload']['prediction_file']) as b:
                assert np.array_equal(a['prediction'],b['prediction'])
            allrows.extend(rows)
        fits_all.extend(fits)
        for r in read(folder/'baselines.json'):replay(r)
        for s in seal['selections']:
            rr=[r for r in allrows if (r['dataset'],r['seed'],r['arm'])==(s['dataset'],s['seed'],s['arm'])]
            assert choose(rr)==s
            if topic=='dualclock':
                short=choose(rr,360)
                comparisons.append(dict(topic=topic,dataset=s['dataset'],seed=s['seed'],contrast=s['arm']+'_long_vs_short',
                    scope='V_selected',reference_loss=short['metrics']['scaled_2pinball'],
                    method_loss=s['metrics']['scaled_2pinball'],method_step=s['step'],reference_step=short['step']))
                for lr in c['learning_rates']:
                    short_fixed=next(r for r in rr if r['step']==360 and r['lr']==lr)
                    long_fixed=next(r for r in rr if r['step']==1440 and r['lr']==lr)
                    comparisons.append(dict(topic=topic,dataset='m5',seed=s['seed'],contrast=s['arm']+'_long_vs_short',
                        scope=f'fixed_endpoints_lr_{lr}',reference_loss=short_fixed['metrics']['scaled_2pinball'],
                        method_loss=long_fixed['metrics']['scaled_2pinball'],method_step=1440,reference_step=360))
        if topic=='anchor':
            for ds in c['datasets']:
                for seed in c['seeds']:
                    for obj in ('native','raw'):
                        for scope in ['V_selected']+[f'fixed_step_{step}_lr_{lr}' for step in c['checkpoints'] for lr in c['learning_rates']]:
                            pool=seal['selections'] if scope=='V_selected' else [r for r in allrows if r['step']==int(scope.split('_')[2]) and r['lr']==float(scope.split('_')[4])]
                            a=next(r for r in pool if (r['dataset'],r['seed'],r['arm'])==(ds,seed,obj+'_anchor'))
                            b=next(r for r in pool if (r['dataset'],r['seed'],r['arm'])==(ds,seed,obj))
                            comparisons.append(dict(topic=topic,dataset=ds,seed=seed,contrast=obj+'_anchor_vs_plain',scope=scope,
                                reference_loss=b['metrics']['scaled_2pinball'],method_loss=a['metrics']['scaled_2pinball'],
                                method_step=a['step'],reference_step=b['step']))
        else:
            for seed in c['seeds']:
                for budget in (360,1440):
                    selected={a:choose([r for r in allrows if r['seed']==seed and r['arm']==a],budget) for a in c['arms']}
                    for baseline in ('standard','summary'):
                        a,b=selected['dualclock'],selected[baseline]
                        comparisons.append(dict(topic=topic,dataset='m5',seed=seed,contrast='dualclock_vs_'+baseline,
                            scope=f'V_selected_budget_{budget}',reference_loss=b['metrics']['scaled_2pinball'],
                            method_loss=a['metrics']['scaled_2pinball'],method_step=a['step'],reference_step=b['step']))
            for r in read(folder/'ablations.json'):
                replay(r);s=r['selected'];assert s in seal['selections']
                comparisons.append(dict(topic=topic,dataset='m5',seed=r['seed'],contrast=r['arm']+'_'+r['mode'],
                    scope='V_ablation',reference_loss=s['metrics']['scaled_2pinball'],method_loss=r['metrics']['scaled_2pinball'],
                    method_step=s['step'],reference_step=s['step']))
            original=list(csv.DictReader((ROOT/'results/candidate_02/trajectories.csv').open()))
            names={'standard':'STANDARD_LORA','summary':'EVENT_SUMMARY_LORA','dualclock':'DUALCLOCK_ADAPTER'}
            for r in allrows:
                if r['seed']==30000 and r['step']<=360:
                    old=next(v for v in original if v['arm']==names[r['arm']] and float(v['lr'])==r['lr'] and int(v['step'])==r['step'])
                    prefix.append(dict(arm=r['arm'],lr=r['lr'],step=r['step'],historical_validation_loss=float(old['validation_loss']),
                        current_validation_loss=r['metrics']['scaled_2pinball'],absolute_difference=abs(float(old['validation_loss'])-r['metrics']['scaled_2pinball'])))
    assert len(fits_all)==44 and sum(f['updates'] for f in fits_all)==46080
    for r in comparisons:r['improvement_percent']=100*(1-r['method_loss']/r['reference_loss'])
    result=dict(status='VERIFIED_V_DIAGNOSTICS',completed_fits=44,training_updates=46080,
                new_evaluation_accesses=0,prediction_caches_replayed=cache_count,max_primary_error=max_error,
                historical_files_unchanged=len(contract['historical_result_hashes']),
                historical_360_prefix_max_validation_difference=max(r['absolute_difference'] for r in prefix),
                scope=CFG['scope'])
    if not verify_only:
        write_json(OUT/'verification.json',result);write_json(OUT/'historical_prefix.json',prefix)
        with (OUT/'comparisons.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(comparisons[0]),lineterminator='\n');w.writeheader();w.writerows(comparisons)
        write_json(OUT/'fit_inventory.json',[{k:f[k] for k in ('fit_id','topic','updates','active_seconds','wall_seconds')} for f in fits_all])
    print(json.dumps(result,indent=2));return result


def queue():
    lock=open(ROOT/'.cache/gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    initialize();start=time.monotonic();state=dict(status='RUNNING',pid=os.getpid(),jobs=[],started_unix=time.time())
    child=None
    def emit(**kw):
        state.update(kw);state['wall_seconds']=time.monotonic()-start;write_json(OUT/'queue_status.json',state)
    def stop(signum,frame):raise KeyboardInterrupt(f'Queue signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for topic in CFG['topics']:
            log=CACHE/(topic+'.log')
            with log.open('x') as f:
                child=subprocess.Popen([str(ROOT/'scripts/with_cuda.sh'),sys.executable,__file__,'train','--topic',topic],
                    cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                record=dict(topic=topic,pid=child.pid,started_unix=time.time());emit(current_job=record)
                while child.poll() is None:
                    if time.monotonic()-start>CFG['max_queue_wall_seconds']:raise TimeoutError('Whole queue wall cap')
                    emit();time.sleep(5)
                record.update(exit_code=child.returncode,finished_unix=time.time());state['jobs'].append(record)
                write_json(OUT/(topic+'_job_exit.json'),record);child=None
        if any(j['exit_code'] for j in state['jobs']):
            emit(status='INCONCLUSIVE_EXECUTION',finished_unix=time.time());return
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        done=subprocess.run([sys.executable,__file__,'finalize'],cwd=ROOT,env=env,timeout=600)
        emit(status='COMPLETE' if done.returncode==0 else 'VERIFICATION_ERROR',finished_unix=time.time())
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        emit(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'INCONCLUSIVE_EXECUTION',error=str(exc),finished_unix=time.time())
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['smoke','start','queue','train','status','finalize'])
    parser.add_argument('--topic',choices=CFG['topics']);parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    if args.command=='smoke':return smoke()
    if args.command=='queue':return queue()
    if args.command=='train':
        assert args.topic is not None
        def terminate(signum,frame):raise TimeoutError(f'Worker signal {signum}')
        signal.signal(signal.SIGTERM,terminate)
        w=Worker(args.topic)
        try:return w.train()
        except BaseException as exc:
            w.emit(status='INCONCLUSIVE_EXECUTION',error=str(exc));(w.folder/'traceback.txt').write_text(traceback.format_exc());raise
    if args.command=='finalize':return finalize(args.verify_only)
    if args.command=='status':
        for p in [OUT/'queue_status.json']+[OUT/t/'status.json' for t in CFG['topics']]:
            print(str(p.relative_to(ROOT)),read(p) if p.exists() else 'NOT_STARTED')
        return
    assert not OUT.exists(),'Immutable run already exists'
    with (ROOT/'.cache'/f'{CFG["run_id"]}_queue.log').open('x') as f:
        p=subprocess.Popen([str(ROOT/'scripts/with_cuda.sh'),sys.executable,__file__,'queue'],cwd=ROOT,
            stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    time.sleep(2)
    assert p.poll() is None,'Queue exited; inspect its retained log'
    print('Queue launched',p.pid)


if __name__=='__main__':main()
