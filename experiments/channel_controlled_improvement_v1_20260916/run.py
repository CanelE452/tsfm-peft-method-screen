"""Finite stage machine: fixed training -> sealed selection -> development evaluation."""
import argparse, traceback
from runtime import *

def smoke(w):
    c=contract();s=state_json();assert s['status']=='PREPARED' and s['smoke_updates']==0
    status(status='SMOKE');records=[]
    for d,dc in c['data'].items():
        values,std=load(DATA,d,'train');common=None
        for arm in ARMS:
            w.boundary();m=make(arm,dc['channel_ids'],41000,'cuda');initial=cpu_state(m)
            frozen=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()))
            m.eval();x,y=batch(values,dc['panels']['TRAIN'][:8])
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):p=m(x).forecast.float().cpu()
            if common is None:common=p.clone()
            assert torch.equal(common,p),'INITIAL_UP0_COMMON_PATH'
            del x,y,p
            opt=optimizer(m,.001);steps=[];m.train()
            for k in range(2):
                def applied():
                    ss=state_json();assert ss['smoke_updates']<12;status(smoke_updates=ss['smoke_updates']+1)
                steps.append(step(m,opt,values,dc['panels']['TRAIN'][8*k:8*k+8],8,w,applied))
            now=cpu_state(m);changes={n:float((now[n].double()-v.double()).norm()) for n,v in initial.items()}
            assert frozen_hash(m)==frozen and tensor_hash(dict(m.named_buffers()))==buffers
            for prefix in ['head.','encoder.' if arm=='LH' else 'side.']:
                assert any(v>0 for n,v in changes.items() if n.startswith(prefix))
            records.append(dict(dataset=d,arm=arm,steps=steps,changes=changes,initial_common_exact=True,frozen_unchanged=True,buffers_unchanged=True))
            save(OUT/'smoke.json',records);del m,opt;cleanup()
    assert state_json()['smoke_updates']==12;status(status='SMOKE_COMPLETE')

def train(w,resuming=False):
    c=contract();s=state_json();assert s['status'] in (['SMOKE_COMPLETE','TRAINING','EXECUTION_ERROR'] if resuming else ['SMOKE_COMPLETE'])
    schedules=read(OUT/'schedules.json');reuse=read(OUT/'reuse.json');initials=read(OUT/'initial_hashes.json')
    fits=read(OUT/'fits.json') if (OUT/'fits.json').exists() else []
    curves=read(OUT/'training_curves.json') if (OUT/'training_curves.json').exists() else []
    def ledger():
        save(OUT/'fits.json',fits);csvwrite(OUT/'fit_attempts.csv',fits)
        save(OUT/'training_curves.json',curves)
        csvwrite(OUT/'training_curves.csv',[dict(**{k:v for k,v in r.items() if k not in ['metrics']},mse=r['metrics']['mse'],mae=r['metrics']['mae']) for r in curves])
        status(new_fits_started=sum(f['mode']=='NEW' for f in fits),new_fits_completed=sum(f['mode']=='NEW' and f['status']=='COMPLETE' for f in fits),reused_fits_completed=sum(f['mode']=='REUSE' and f['status']=='COMPLETE' for f in fits),training_updates=sum(f['updates'] for f in fits if f['mode']=='NEW'),reused_updates=sum(f['updates'] for f in fits if f['mode']=='REUSE'))
    for cell in c['cells']:
        fid=cell['fit'];existing=next((f for f in fits if f['fit']==fid),None)
        if existing and existing['status']=='COMPLETE':continue
        if existing:assert resuming and (CACHE/(fid+'_resume.pt')).exists(), 'NO_COMPLETE_STATE'
        d,arm,seed=cell['dataset'],cell['arm'],cell['seed'];dc=c['data'][d]
        train_values,_=load(DATA,d,'train');v_values,std=load(DATA,d,'development')
        w.boundary();tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
        m=make(arm,dc['channel_ids'],seed,'cuda')
        load_peak=torch.cuda.max_memory_allocated();load_seconds=time.monotonic()-tick
        initial=cpu_state(m);assert tensor_hash(initial)==initials[f'{d}_{seed}_{arm}']
        frozen=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()))
        f=existing or dict(**cell,mode='REUSE' if cell['reuse_key'] else 'NEW',status='RUNNING',updates=0,epochs=0,active_seconds=0.,validation_seconds=0.,checkpoint_seconds=0.,resume_save_seconds=0.,load_seconds=load_seconds,load_peak=load_peak,train_peak_allocated=0,train_peak_reserved=0,validation_peak_allocated=0,wall_seconds=0.)
        if not existing:fits.append(f)
        status(status='TRAINING',current_fit=fid);ledger()
        def checkpoint(epoch,oldcp=None):
            cp=CACHE/f'{fid}_{epoch:02}.pt'
            t=time.monotonic()
            if oldcp:
                cp=ROOT/oldcp['checkpoint_path'];assert sha(cp)==oldcp['checkpoint_sha256']
                restore(m,torch.load(cp,weights_only=True,map_location='cpu'))
            elif not cp.exists():atomic_torch(cp,cpu_state(m))
            else:assert tensor_hash(torch.load(cp,weights_only=True,map_location='cpu'))==tensor_hash(cpu_state(m))
            cp_hash=sha(cp);f['checkpoint_seconds']+=time.monotonic()-t
            for panel in ['V_FIXED','V_MIXED']:
                if any(r['fit']==fid and r['epoch']==epoch and r['panel']==panel for r in curves):continue
                r=predict(m,v_values,std,dc['panels'][panel],f'{panel}_{fid}_{epoch:02}',w)
                row=dict(**cell,epoch=epoch,updates=epoch*(len(train_values) and len(dc['panels']['TRAIN'])//8),panel=panel,checkpoint_path=str(cp.relative_to(ROOT)),checkpoint_sha256=cp_hash,**r)
                curves.append(row);f['validation_seconds']+=r['seconds'];f['validation_peak_allocated']=max(f['validation_peak_allocated'],r['peak_allocated']);ledger()
            print('EPOCH',fid,epoch,[(p,next(r['metrics']['mse'] for r in curves if r['fit']==fid and r['epoch']==epoch and r['panel']==p)) for p in ['V_FIXED','V_MIXED']],flush=True)
        if cell['reuse_key']:
            old=reuse[cell['reuse_key']]
            for cp in old['checkpoints']:checkpoint(cp['epoch'],cp)
            f.update(updates=old['updates'],epochs=20,historical_steps_path=old['old_directory']+'/'+old['old_fit']+'_steps.json',initial_hash=tensor_hash(initial),original_training_cost='REUSED; separate old step records')
        else:
            opt=optimizer(m,cell['lr']);sched=torch.optim.lr_scheduler.StepLR(opt,5,.5)
            steps=read(OUT/(fid+'_steps.json')) if (OUT/(fid+'_steps.json')).exists() else []
            rp=CACHE/(fid+'_resume.pt');jp=CACHE/(fid+'_journal.json')
            epoch=1;offset=0
            if existing:
                st=torch.load(rp,map_location='cpu',weights_only=False)
                assert st['fit']==fid and st['contract_sha256']==sha(OUT/'seal.json')
                j=read(jp);assert j['phase']=='COMMITTED' and j['updates']==st['fit_record']['updates'],'AMBIGUOUS_UPDATE_NO_AUTOMATIC_REPLAY'
                restore(m,st['parameters']);opt.load_state_dict(st['optimizer']);sched.load_state_dict(st['scheduler']);rng_restore(st['rng']);epoch=st['epoch'];offset=st['offset'];f.update(st['fit_record']);steps=st['steps']
                status(resume_count=state_json().get('resume_count',0)+1)
            def persist(ep,off):
                t=time.monotonic()
                atomic_torch(rp,dict(fit=fid,contract_sha256=sha(OUT/'seal.json'),parameters=cpu_state(m),optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng_state(),epoch=ep,offset=off,fit_record=copy.deepcopy(f),steps=steps))
                f['resume_save_seconds']+=time.monotonic()-t
                save(jp,dict(phase='COMMITTED',updates=f['updates']))
            if not existing:checkpoint(0);persist(1,0)
            for ep in range(epoch,21):
                oo=schedules[f'{d}_{seed}'][ep-1];m.train()
                for i in range(offset if ep==epoch else 0,len(oo),8):
                    assert state_json()['training_updates']<c['update_cap']
                    save(jp,dict(phase='UPDATE_PENDING',previous_updates=f['updates'],epoch=ep,offset=i))
                    r=step(m,opt,train_values,oo[i:i+8],8,w)
                    f['updates']+=1;f['active_seconds']+=r['seconds'];f['train_peak_allocated']=max(f['train_peak_allocated'],r['peak_allocated']);f['train_peak_reserved']=max(f['train_peak_reserved'],r['peak_reserved'])
                    steps.append(dict(epoch=ep,update=f['updates'],**r));persist(ep,i+8)
                checkpoint(ep);sched.step();f['epochs']=ep;persist(ep+1,0)
                save(OUT/(fid+'_steps.json'),steps);ledger();offset=0
            assert f['updates']==20*len(dc['panels']['TRAIN'])//8
            assert frozen_hash(m)==frozen and tensor_hash(dict(m.named_buffers()))==buffers
            now=cpu_state(m);changes={n:float((v-initial[n]).double().norm()) for n,v in now.items()}
            assert any(v>0 for n,v in changes.items() if n.startswith('head.'))
            assert any(v>0 for n,v in changes.items() if n.startswith('encoder.' if arm=='LH' else 'side.'))
            save(OUT/(fid+'_changes.json'),changes);f['resume_path']=str(rp.relative_to(ROOT));f['frozen_and_buffers_unchanged']=True
            del opt,sched
        f.update(status='COMPLETE',wall_seconds=f['wall_seconds']+time.monotonic()-tick,trainable=761952)
        ledger();del m;cleanup()
    assert len(fits)==36 and all(f['status']=='COMPLETE' for f in fits)
    assert sum(f['updates'] for f in fits)==45720 and len(curves)==36*21*2
    for panel in ['V_FIXED','V_MIXED']:
        rows=[r for r in curves if r['panel']==panel];save(OUT/(panel+'_predictions.json'),rows)
        csvwrite(OUT/(panel+'_scores.csv'),[dict(fit=r['fit'],dataset=r['dataset'],arm=r['arm'],seed=r['seed'],lr=r['lr'],epoch=r['epoch'],mse=r['metrics']['mse'],mae=r['metrics']['mae']) for r in rows])
    status(status='TRAIN_COMPLETE',current_fit=None)

def select():
    c=contract();assert state_json()['status']=='TRAIN_COMPLETE'
    curves=read(OUT/'training_curves.json');sels=[];lrtable=[]
    for d in c['data']:
        for arm in ARMS:
            rows=[r for r in curves if r['dataset']==d and r['arm']==arm]
            def best(lr,seed,panel):return min([r for r in rows if r['lr']==lr and r['seed']==seed and r['panel']==panel],key=lambda r:(r['metrics']['mse'],r['epoch']))
            scores={lr:float(np.mean([best(lr,s,'V_MIXED')['metrics']['mse'] for s in SEEDS])) for lr in LRS}
            chosen=min(LRS,key=lambda lr:(scores[lr],lr))
            lrtable += [dict(dataset=d,arm=arm,lr=lr,three_seed_minimum_mean=scores[lr],chosen=lr==chosen) for lr in LRS]
            for seed in SEEDS:
                options={'P_MAIN':best(chosen,seed,'V_MIXED'),'P_VFIXED':best(chosen,seed,'V_FIXED'),'P_LR_FIXED':best(.001,seed,'V_MIXED'),'P_LAST20':next(r for r in rows if r['lr']==chosen and r['seed']==seed and r['epoch']==20 and r['panel']=='V_MIXED')}
                for policy,r in options.items():sels.append(dict(policy=policy,**r))
    assert len(sels)==72
    save(OUT/'selection_seal.json',dict(at=time.time(),contract_sha256=sha(OUT/'seal.json'),curves_sha256=sha(OUT/'training_curves.json'),fits_sha256=sha(OUT/'fits.json'),lr_comparison=lrtable,selections=sels,exposure='DISCOVERY_REUSED_PERIODS'))
    csvwrite(OUT/'lr_selection.csv',lrtable);status(status='SELECTED')

def evaluate(w):
    c=contract();ss=read(OUT/'selection_seal.json');assert state_json()['status']=='SELECTED'
    assert sha(OUT/'training_curves.json')==ss['curves_sha256'] and sha(OUT/'fits.json')==ss['fits_sha256']
    status(status='EVALUATING');rows=[];replay=[];dependencies=[]
    selected=ss['selections'];done={}
    for sel in selected:
        d,arm,seed=sel['dataset'],sel['arm'],sel['seed'];dc=c['data'][d];key=(sel['fit'],sel['epoch'])
        if key not in done:
            w.boundary();m=make(arm,dc['channel_ids'],seed,'cuda')
            assert sha(ROOT/sel['checkpoint_path'])==sel['checkpoint_sha256'];restore(m,torch.load(ROOT/sel['checkpoint_path'],map_location='cpu',weights_only=True))
            vv,std=load(DATA,d,'development');vr=predict(m,vv,std,dc['panels'][sel['panel']],f'REPLAY_{sel["fit"]}_{sel["epoch"]:02}_{sel["panel"]}',w)
            with np.load(ROOT/sel['prediction_path']) as z:a=z['prediction']
            with np.load(ROOT/vr['prediction_path']) as z:b=z['prediction']
            same=np.array_equal(a,b);torch.testing.assert_close(torch.from_numpy(a),torch.from_numpy(b),rtol=1e-5,atol=1e-6)
            replay.append(dict(fit=sel['fit'],epoch=sel['epoch'],bitwise_equal=same,max_abs=float(np.max(abs(a-b))),mismatch_count=int(np.sum(a!=b)),reference=sel['prediction_path'],replay=vr))
            # E file is first opened here, after all 36 paths and selection have been sealed.
            ev,std=load(DATA,d,'evaluation');computed={}
            for panel in ['E_FIXED','E_MIXED']:computed[panel]=predict(m,ev,std,dc['panels'][panel],f'{panel}_{sel["fit"]}_{sel["epoch"]:02}',w)
            done[key]=computed;del m;cleanup()
        for panel,r in done[key].items():rows.append(dict(dataset=d,arm=arm,seed=seed,policy=sel['policy'],panel=panel,fit=sel['fit'],lr=sel['lr'],epoch=sel['epoch'],updates=sel['updates'],checkpoint_path=sel['checkpoint_path'],checkpoint_sha256=sel['checkpoint_sha256'],**r))
        save(OUT/'evaluation.json',rows);save(OUT/'replays.json',replay)
    for d,dc in c['data'].items():
        ev,std=load(DATA,d,'evaluation')
        for baseline in ['SEASONAL_NAIVE','LAST_VALUE']:
            for panel in ['E_FIXED','E_MIXED']:
                r=predict(None,ev,std,dc['panels'][panel],f'{panel}_{d}_{baseline}',w,baseline)
                rows.append(dict(dataset=d,arm=baseline,seed=None,policy='BASELINE',panel=panel,epoch=0,updates=0,**r))
        for seed in SEEDS:
            m=make('LH',dc['channel_ids'],seed,'cuda')
            for panel in ['E_FIXED','E_MIXED']:
                r=predict(m,ev,std,dc['panels'][panel],f'{panel}_{d}_{seed}_INIT',w)
                rows.append(dict(dataset=d,arm='INIT',seed=seed,policy='BASELINE',panel=panel,epoch=0,updates=0,**r))
            del m;cleanup()
    for sel in [r for r in selected if r['arm']=='PRIOR' and r['policy']=='P_MAIN']:
        d,seed=sel['dataset'],sel['seed'];dc=c['data'][d];m=make('PRIOR',dc['channel_ids'],seed,'cuda');restore(m,torch.load(ROOT/sel['checkpoint_path'],weights_only=True,map_location='cpu'))
        vv,std=load(DATA,d,'development')
        for layer in m.side:layer.arm='SIDE'
        off=predict(m,vv,std,dc['panels']['V_MIXED'],f'PRIOR_OFF_{d}_{seed}',w)
        dependencies.append(dict(dataset=d,seed=seed,on_mse=sel['metrics']['mse'],off_mse=off['metrics']['mse'],off_relative_harm=100*(off['metrics']['mse']/sel['metrics']['mse']-1),on_prediction=sel['prediction_path'],**off));del m;cleanup()
    save(OUT/'evaluation.json',rows);save(OUT/'prior_dependency.json',dependencies)
    csvwrite(OUT/'prior_dependency.csv',[{k:v for k,v in r.items() if k!='metrics'} for r in dependencies])
    for panel in ['E_FIXED','E_MIXED']:
        csvwrite(OUT/(panel+'_scores.csv'),[dict(**{k:v for k,v in r.items() if k not in ['metrics']},mse=r['metrics']['mse'],mae=r['metrics']['mae'],raw_mae=r['metrics']['raw_mae']) for r in rows if r['panel']==panel])
    status(status='EVALUATION_COMPLETE')

def dispatch(stage):
    if stage=='prepare':prepare();return
    if stage=='status':print(json.dumps(state_json(),indent=2));return
    if stage=='select':select();return
    if stage in ['verify','report']:
        from reporting import verify_report
        verify_report();return
    w=None
    try:
        c=contract();configure();w=Watch(OUT,stage,c['wall_cap']);w.boundary(startup=True)
        if stage=='smoke':smoke(w)
        elif stage in ['train','resume']:train(w,resuming=stage=='resume')
        elif stage=='evaluate':evaluate(w)
        elif stage=='resources':
            from resources import measure
            measure(w)
        elif stage=='all':
            s=state_json()['status']
            if s=='PREPARED':smoke(w)
            if state_json()['status']=='SMOKE_COMPLETE':train(w)
            if state_json()['status']=='TRAIN_COMPLETE':select()
            if state_json()['status']=='SELECTED':evaluate(w)
            if state_json()['status']=='EVALUATION_COMPLETE':
                from resources import measure
                measure(w)
    except BaseException as e:
        previous=state_json();save(OUT/f'error_{int(time.time())}.json',dict(previous=previous,error=repr(e),traceback=traceback.format_exc()))
        status(status='EXECUTION_ERROR',failed_phase=previous['status'],error=repr(e));raise
    finally:
        cleanup()
        if w:w.close()
    if stage=='all':
        from reporting import verify_report
        verify_report()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','smoke','train','select','evaluate','resources','verify','report','status','resume','all']);a=p.parse_args();dispatch(a.stage)
