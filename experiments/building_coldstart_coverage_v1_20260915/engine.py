"""Guarded bounded fit execution, stage ledger, data access and checkpoint provenance."""
import sys,time,traceback,importlib.metadata
from core import *
from data import prepare_data,episodes,history,target

def append(name,row):
    p=OUT/(name+'.json');rr=read(p) if p.exists() else [];rr.append(row);save(p,rr);csvwrite(OUT/(name+'.csv'),rr);return rr

def prediction(e,method,q,raw,y,h,suffix='',extra=None):
    folder=CACHE/'predictions';folder.mkdir(exist_ok=True);key=e['id']+'_'+method+suffix;p=folder/(key+'.npz');assert not p.exists(),('DUPLICATE_PREDICTION',key)
    np.savez(p,q=q,raw=raw,target=y,history=h)
    row=dict(id=key,episode=e['id'],building_id=e['building_id'],role=e['role'],history_days=e['history_days'],forecast_type=e['forecast_type'],coverage_count=e['coverage_count'],method=method,path=str(p.relative_to(ROOT)),sha256=sha(p),raw_crossings=int(np.sum(np.diff(raw,axis=0)<0)),**metrics(q,y,h),**(extra or {}));append('prediction_manifest',row);return row

def prepare():
    d=prepare_data()
    if d['status']!='DATA_READY':save(OUT/'status.json',dict(status=d['status'],fits=0,updates=0));return
    if (OUT/'prepare_seal.json').exists():check_seal();print('Already prepared');return
    import subprocess
    p=subprocess.run([str(ROOT/'scripts/with_cuda.sh'),str(ROOT/'.venv/bin/python'),str(EXP/'test_correctness.py')],cwd=ROOT,capture_output=True,text=True)
    save(OUT/'cpu_checks.json',dict(tag='확인',exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,p.stderr
    from model import setup,Watch,make,predict,forward,update,audit,parameters,cpu_state,tensor_hash,frozen_hash,restore,cleanup,torch,disabled,tensor
    from chronos import Chronos2Pipeline
    setup();records=[];w=Watch(OUT,'smoke')
    try:
        w.boundary(startup=True)
        for role in ['recipe','dev']:
            e=episodes(role)[0];h=history(e);x,y,ds,c=windows(h,e['origin']);m=make();initial=cpu_state(m);fh=frozen_hash(m)
            q0,raw0=predict(m,h[-24:],f0=True);q,raw=predict(m,h[-24:]);assert np.array_equal(q0,q) and np.array_equal(raw0,raw)
            with torch.no_grad(),disabled(m):native=Chronos2Pipeline(m).predict([torch.as_tensor(h[-24:].copy(),dtype=torch.float32)],prediction_length=24,context_length=24,batch_size=1)[0][0].double().cpu().numpy()
            error=float(np.max(np.abs(native-raw0)));assert np.allclose(native,raw0,rtol=1e-5,atol=1e-5),error
            with torch.no_grad():
                a=forward(m,x[0],y[0]).quantile_preds;b=forward(m,x[0],y[0]+1234).quantile_preds;assert torch.equal(a,b),'Target leaked to prediction'
            opt=torch.optim.AdamW(parameters(m).values(),lr=3e-5,weight_decay=0);r=update(m,opt,x[0],y[0],w)
            # Record an applied smoke update before remaining correctness checks.
            append('smoke_updates',dict(role=role,episode=e['id'],updates=1,**r))
            state=cpu_state(m);assert tensor_hash(state)!=tensor_hash(initial);assert frozen_hash(m)==fh
            after,rawafter=predict(m,h[-24:]);path=CACHE/('smoke_'+role+'.pt');torch.save(state,path);count=audit(m)
            del opt,m;cleanup();m=make();restore(m,torch.load(path,weights_only=True));again,_=predict(m,h[-24:]);assert np.array_equal(after,again)
            records.append(dict(role=role,episode=e['id'],identity_exact=True,pipeline_max_abs=error,reload_max_abs=float(np.max(np.abs(after-again))),frozen_unchanged=True,nonzero_update=True,native_crossings_before=int((np.diff(raw0,axis=0)<0).sum()),native_crossings_after=int((np.diff(rawafter,axis=0)<0).sum()),shape=list(q.shape),**count));del m;cleanup()
        save(OUT/'model_smoke.json',dict(tag='확인',status='PASS',records=records,updates=2,forecasting_fits=0))
    finally:w.close()
    import torch
    modelroot=Path.home()/'.cache/huggingface/hub/models--amazon--chronos-2/snapshots'/read(CONFIG)['revision']
    modelhash={str(p):sha(p) for p in modelroot.rglob('*') if p.is_file()}
    assert modelhash
    save(OUT/'environment.json',dict(tag='확인',python=sys.version,packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},torch=torch.__version__,cuda=torch.version.cuda,driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()))
    save(OUT/'prepare_seal.json',dict(tag='확인',at=time.time(),sources=source_hashes(),staged=d['staged'],model_files=modelhash,config=read(CONFIG),data_manifest_sha256=sha(OUT/'data_manifest.json')))
    save(OUT/'status.json',dict(status='PREPARED',fits=0,updates=0,smoke_updates=2));print('PREPARED SMOKE PASS',flush=True)

def one_fit(e,lr,steps,phase,w):
    from model import make,predict,update,parameters,cpu_state,tensor_hash,frozen_hash,restore,cleanup,torch
    fid=e['id']+('_lr'+str(lr) if phase=='recipe' else '');ledger=read(OUT/'fit_attempts.json') if (OUT/'fit_attempts.json').exists() else []
    old=[r for r in ledger if r['fit']==fid]
    if old:
        assert old[0]['status']=='COMPLETE',('NO_AUTOMATIC_RETRY',old[0]);return old[0]
    assert len(ledger)<48;assert sum(r['phase']==phase for r in ledger)<{'recipe':8,'screen':16,'heldout':24}[phase]
    row=dict(fit=fid,episode=e['id'],phase=phase,building_id=e['building_id'],history_days=e['history_days'],forecast_type=e['forecast_type'],lr=lr,planned_updates=steps,actual_updates=0,status='STARTED',started_at=time.time())
    ledger.append(row)
    def flush():save(OUT/'fit_attempts.json',ledger);csvwrite(OUT/'fit_attempts.csv',ledger)
    flush();started=time.monotonic();active=0;trajectory=[];torch.cuda.reset_peak_memory_stats();m=None;opt=None
    try:
        m=make();fh=frozen_hash(m);initial=tensor_hash(cpu_state(m));h=history(e);xs,ys,dates,c=windows(h,e['origin']);assert c==e['coverage_count'];v=[]
        if phase=='recipe':
            assert e['history_days']==14 and e['forecast_type']=='Wednesday' and len(xs)==13
            vx,vy=xs[-2:],ys[-2:];xs,ys=xs[:-2],ys[:-2];scalehist=h[:-48]
        else:
            q0,raw0=predict(m,h[-24:],f0=True)
            # Fit affine only to exposed target-history windows before episode target is opened.
            fq=np.stack([predict(m,x,f0=True)[0] for x in xs]);aff=affine(fq,ys)
            ap=CACHE/'affine';ap.mkdir(exist_ok=True);np.savez(ap/(fid+'.npz'),q=fq,y=ys);row['affine']=aff;row['affine_data_sha256']=sha(ap/(fid+'.npz'))
        opt=torch.optim.AdamW(parameters(m).values(),lr=lr,weight_decay=0,betas=(.9,.999),eps=1e-8)
        rng=np.random.default_rng(61601);order=[]
        while len(order)<steps:order.extend(rng.permutation(len(xs)).tolist())
        row.update(trainable_count=sum(p.numel() for p in parameters(m).values()),frozen_hash_before=fh,initial_lora_hash=initial,fit_windows=len(xs),order_sha256=__import__('hashlib').sha256(np.asarray(order[:steps],dtype=np.int64).tobytes()).hexdigest());flush()
        cpdir=CACHE/'checkpoints'/fid;cpdir.mkdir(parents=True,exist_ok=True)
        def checkpoint(step):
            path=cpdir/(str(step)+'.pt');torch.save(cpu_state(m),path)
            if phase=='recipe':
                rr=[]
                for j,(xx,yy) in enumerate(zip(vx,vy)):
                    q,raw=predict(m,xx);r=prediction(e,'RECIPE',q,raw,yy,scalehist,suffix=f'_lr{lr}_u{step}_v{j}',extra=dict(lr=lr,updates=step,inner_v=j,checkpoint=str(path.relative_to(ROOT))));rr.append(r['scaled_RMSE'])
                v.append(dict(building_id=e['building_id'],lr=lr,updates=step,primary=float(np.mean(rr)),checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path)));csvwrite(OUT/(fid+'_recipe.csv'),v)
            return path
        if phase=='recipe':checkpoint(0)
        for i in range(steps):
            j=order[i];r=update(m,opt,xs[j],ys[j],w);row['actual_updates']+=1;active+=r['seconds'];trajectory.append(dict(fit=fid,phase=phase,update=i+1,window=j,**r));flush()
            if phase=='recipe' and i+1 in [15,30,60,120]:checkpoint(i+1)
        path=cpdir/(str(steps)+'.pt') if phase=='recipe' else checkpoint(steps)
        row['frozen_hash_after']=frozen_hash(m);assert row['frozen_hash_after']==fh;row['final_lora_hash']=tensor_hash(cpu_state(m));assert row['final_lora_hash']!=initial
        probe=vx[0] if phase=='recipe' else h[-24:];q,raw=predict(m,probe)
        del opt,m;opt=None;m=None;cleanup();m=make();restore(m,torch.load(path,weights_only=True));qr,rr=predict(m,probe);assert np.array_equal(q,qr) and np.array_equal(raw,rr),'RELOAD_MISMATCH'
        row.update(reload_max_abs=float(np.max(np.abs(q-qr))),checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path),frozen_unchanged=True)
        if phase=='recipe':row['recipe_records']=v
        else:
            y=target(e)
            for method,qq,rawq in [('F0',q0,raw0),('STANDARD',q,raw),('AFFINE',aff['a']*q0+aff['b'],aff['a']*raw0+aff['b'])]:prediction(e,method,qq,rawq,y,h)
        row['status']='COMPLETE';print('FIT COMPLETE',phase,len([r for r in ledger if r['phase']==phase]),fid,'updates',steps,flush=True)
    except BaseException as exc:
        row.update(status='EXECUTION_ERROR',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        row.update(active_train_seconds=active,wall_seconds=time.monotonic()-started,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved());flush();csvwrite(OUT/(fid+'_trajectory.csv'),trajectory)
        if m is not None:del m
        if opt is not None:del opt
        cleanup()
    return row

def run_fits(phase):
    check_seal()
    from model import setup,Watch
    setup();role={'recipe':'recipe','screen':'dev','heldout':'heldout'}[phase]
    if phase!='recipe':recipe=read(OUT/'recipe_seal.json')
    if phase=='heldout':
        rule=read(OUT/'stageB_selection.json');assert rule['status']=='METHOD_SIGNAL'
        if not (OUT/'heldout_seal.json').exists():save(OUT/'heldout_seal.json',dict(tag='확인',at=time.time(),episodes=episodes('heldout'),recipe=recipe,rules=rule,config=read(CONFIG)))
    w=Watch(OUT,phase)
    try:
        w.boundary(startup=True)
        for e in episodes(role):
            if phase=='recipe':
                if e['history_days']!=14 or e['forecast_type']!='Wednesday':continue
                for lr in [3e-5,1e-4]:one_fit(e,lr,120,phase,w)
            else:one_fit(e,recipe['lr'],recipe['updates'],phase,w)
    finally:w.close()
    if phase=='recipe':
        fits=[f for f in read(OUT/'fit_attempts.json') if f['phase']=='recipe'];assert len(fits)==8 and all(f['status']=='COMPLETE' for f in fits)
        records=[r for f in fits for r in f['recipe_records']];csvwrite(OUT/'recipe_selection.csv',records)
        choices=[dict(lr=lr,updates=s,primary=float(np.mean([r['primary'] for r in records if r['lr']==lr and r['updates']==s]))) for lr in [3e-5,1e-4] for s in [15,30,60,120]]
        best=min(choices,key=lambda r:(r['primary'],r['updates'],r['lr']))
        save(OUT/'recipe_seal.json',dict(tag='확인',at=time.time(),**best,choices=choices,step0_macro=float(np.mean([r['primary'] for r in records if r['updates']==0])),selection_scope='4 recipe buildings, 2 inner validation days, no episode future'))
        print('RECIPE SEALED',best,flush=True)
