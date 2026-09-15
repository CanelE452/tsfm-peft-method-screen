"""R2 prepare, dry-fit/smoke, bounded epoch training, sealed E and verification."""
import argparse,copy,json,math,os,subprocess,sys,time,traceback,importlib.metadata
from contextlib import nullcontext
import numpy as np
import torch
from common import ROOT,EXP,RESEARCH,R2 as OUT,C2 as CACHE,save,read,sha,digest,csvwrite,parameters,cpu_state,tensor_hash,frozen_hash,restore,preserve_rng,cleanup,Watch,ResourceError,source_hashes,verify_contract,audit_history
from rank2_model import ARMS,make,inventory,up
from rank2_data import stage,load,batch,loss,metrics,independent

def configure():
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False;torch.use_deterministic_algorithms(True)
def prepare():
    if OUT.exists():
        assert (OUT/'prepare_repair.json').exists() and not (OUT/'contract.json').exists() and not (OUT/'status.json').exists(),'No duplicate run overwrite'
        data=stage(CACHE,allow_existing=True);return finish_prepare(data)
    assert not CACHE.exists();OUT.mkdir();CACHE.mkdir();data=stage(CACHE)
    receipt=read(ROOT/'results/priority12_20260915/moment_source.json');snapshot=__import__('pathlib').Path(receipt['snapshot']);model_files={str(p):sha(p) for p in snapshot.iterdir() if p.is_file()}
    tests=subprocess.run([sys.executable,'-m','pytest','-q',str(EXP/'test_rank12.py')],cwd=ROOT,capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});save(OUT/'cpu_tests.json',dict(exit_code=tests.returncode,stdout=tests.stdout,stderr=tests.stderr));assert tests.returncode==0,tests.stdout+tests.stderr
    inventory_rows=[];parity=[];common_hash={}
    for seed in [41000,41001]:
        for arm in ARMS:
            m=make(arm,data['electricity']['channel_ids'],seed);inv=inventory(m);inventory_rows.append(dict(seed=seed,**inv))
            key_state={n:p for n,p in parameters(m).items() if not n.startswith('channel_adapter.')};shared={k:v for k,v in key_state.items() if not k.startswith('frequency_adapter.')};h=tensor_hash(shared)
            if seed not in common_hash:common_hash[seed]=h
            assert common_hash[seed]==h,'COMMON_INIT_MISMATCH'
            if arm=='INDIV_REF':
                official=up.TimePEFTPipeline(m,copy.deepcopy(m.frequency_adapter),up.ChannelAdapter(512,32,2));official.channel_adapter.load_state_dict(m.channel_adapter.state_dict());m.eval();official.eval();values,_=load(CACHE,'electricity','train');x,y=batch(values,data['electricity']['origins']['train'][:2],'cpu')
                with torch.no_grad():a=m(x).forecast;b=official(x).forecast
                err=float((a-b).abs().max());assert err<=1e-6;torch.testing.assert_close(loss(a,y),loss(b,y),atol=1e-7,rtol=1e-6);parity.append(dict(seed=seed,max_abs_error=err,official_same_state=True));del official,x,y,a,b
            del m;cleanup()
    csvwrite(OUT/'PARAMETER_BUDGET.csv',[dict(seed=r['seed'],arm=r['arm'],channel_block=r['groups']['channel_adapter'],total_trainable=r['trainable'],total_model=r['total'],**r['groups']) for r in inventory_rows]);save(OUT/'parameter_inventory.json',inventory_rows);save(OUT/'official_parity.json',parity);save(OUT/'common_initial_hashes.json',common_hash)
    cells=[dict(dataset=d,seed=z,arm=a) for d in data for z in [41000,41001] for a in ARMS];order=np.random.default_rng(61510).permutation(len(cells));cells=[dict(attempt_order=i,**cells[int(j)]) for i,j in enumerate(order)];save(OUT/'fit_manifest.json',cells)
    schedules={}
    for d in data:
        for seed in [41000,41001]:
            rng=np.random.default_rng(seed);schedules[f'{d}_{seed}']=[rng.permutation(data[d]['origins']['train']).tolist() for _ in range(20)]
    save(OUT/'schedules.json',schedules)
    finish_prepare(data)

def finish_prepare(data):
    receipt=read(ROOT/'results/priority12_20260915/moment_source.json');snapshot=__import__('pathlib').Path(receipt['snapshot']);model_files={str(p):sha(p) for p in snapshot.iterdir() if p.is_file()}
    save(OUT/'environment.json',dict(python=sys.version,torch=torch.__version__,dependencies=sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions()),model=receipt,matmul='highest; TF32 off; deterministic; no checkpoint',override='Existing isolated .venv-channel reused; transformers4.44.2 is upstream-required override to momentfm metadata4.33.3; Q environment unchanged'))
    save(OUT/'contract.json',dict(data=data,source_hashes=source_hashes(),model=receipt,model_files=model_files,fit_manifest_hash=sha(OUT/'fit_manifest.json'),schedules_hash=sha(OUT/'schedules.json'),historical_hashes=read(RESEARCH/'historical_hashes.json'),recipe=dict(L=96,H=96,C=32,precision='bf16',FFT='fp32',effective_batch=8,epochs=20,patience=5,min_delta=1e-4,lr=.001,weight_decay=0,stepLR_epochs=5,stepLR_gamma=.5,seeds=[41000,41001],micro_candidates=[8,4,2,1],fit_cap=24,smoke_cap=24,wall_cap_seconds=43200),mechanism_seed=61530,bootstrap='np.array_split chronological E origins into min(8,N) blocks; sample blocks to N then truncate; 1000 seed61520',budget_signal='BASIS wins both seeds vs V-selected same-budget baseline, mean gain>=0.5%, and mean MSE below LH on at least one source',compression='>=20% total trainable reduction vs INDIV_REF and mean MSE <=1.005 REF',novelty='KNOWN_PARAMETERIZATION: affine weight bank equals low rank factorization in vectorized weight space; insertion differs from official C-LoRA. No first-method claim.'))
    save(OUT/'status.json',dict(status='PREPARED',smoke_attempts=0,smoke_updates=0,fit_attempts=0,fits_completed=0,training_updates=0));(OUT/'PROTOCOL.md').write_bytes((RESEARCH/'PROTOCOL.md').read_bytes());print('R2 PREPARED',flush=True)

def step(m,opt,values,origins,micro,w,on_applied=None):
    start=w.before();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();opt.zero_grad(set_to_none=True);x,y=batch(values,origins);counts=torch.isfinite(y).sum((0,2));total=0.
    for i in range(0,len(origins),micro):
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=m(x[i:i+micro]).forecast;value=loss(pred,y[i:i+micro],counts)
        assert torch.isfinite(value) and torch.isfinite(pred).all();value.backward();total+=float(value.detach());del pred,value
    ps=parameters(m);assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps.values());norm=torch.nn.utils.clip_grad_norm_(list(ps.values()),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();seconds=time.perf_counter()-tick
    if on_applied:on_applied()
    assert all(torch.isfinite(p).all() for p in ps.values());end,contaminated=w.after(start)
    return dict(seconds=seconds,loss=total,gradient_norm=float(norm),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),nvml_used_mib=end['used_mib'],free_mib=min(start['free_mib'],end['free_mib']),external_compute_contaminated=contaminated,lr=opt.param_groups[0]['lr'],origins=len(origins))
def evaluate(m,values,std,origins,tag,w,micro,seasonal=False):
    pp=[];yy=[];started=time.monotonic();mode=m.training if m is not None else None
    with preserve_rng(),torch.no_grad():
        if m is not None:m.eval()
        for i in range(0,len(origins),micro):
            if w:w.boundary()
            x,y=batch(values,origins[i:i+micro])
            with torch.autocast('cuda',dtype=torch.bfloat16):p=x[:,:,-24:].repeat(1,1,4) if seasonal else m(x).forecast
            pp.append(p.float().cpu().numpy());yy.append(y.cpu().numpy())
        if m is not None:m.train(mode)
    p=np.concatenate(pp);y=np.concatenate(yy);path=CACHE/(tag+'.npz');assert not path.exists();np.savez_compressed(path,prediction=p,target=y,std=std,origins=np.array(origins));return dict(metrics=metrics(p,y,std),prediction_path=str(path.relative_to(ROOT)),prediction_sha256=sha(path),wall_seconds=time.monotonic()-started)

def smoke():
    c=verify_contract(OUT);s=read(OUT/'status.json');assert s['status']=='PREPARED';w=Watch(OUT,'smoke',43200);dry=[];checks=[]
    try:
        s['status']='SMOKE_WAITING';save(OUT/'status.json',s);w.boundary(startup=True);chosen=None
        for micro in [8,4,2,1]:
            feasible=True
            for d in c['data']:
                values,_=load(CACHE,d,'train')
                for arm in ARMS:
                    w.boundary();m=None
                    try:
                        m=make(arm,c['data'][d]['channel_ids'],41000,'cuda');m.train();torch.cuda.reset_peak_memory_stats();x,y=batch(values,c['data'][d]['origins']['train'][:micro]);start=time.monotonic()
                        with torch.autocast('cuda',dtype=torch.bfloat16):p=m(x).forecast;ll=loss(p,y)
                        ll.backward();torch.cuda.synchronize();assert torch.isfinite(ll) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters(m).values())
                        adam_bytes=sum(p.numel()*8 for p in parameters(m).values());peak=torch.cuda.max_memory_allocated();sample=w.sample();ok=peak+adam_bytes < sample['total_mib']*2**20-(sample['used_mib']*2**20-torch.cuda.memory_reserved())-1024*2**20
                        dry.append(dict(dataset=d,arm=arm,micro=micro,passed=ok,peak_allocated=peak,estimated_adam_state_bytes=adam_bytes,seconds=time.monotonic()-start,optimizer_updates=0));feasible &= ok
                    except torch.cuda.OutOfMemoryError as e:dry.append(dict(dataset=d,arm=arm,micro=micro,passed=False,error=repr(e),optimizer_updates=0));feasible=False
                    finally:
                        m=None
                        # release graph references explicitly before the next model
                        x=y=p=ll=None;cleanup();save(OUT/'dry_fit.json',dry)
            if feasible:chosen=micro;break
        if chosen is None:s['status']='BLOCKED_RESOURCE';return
        save(OUT/'resource_strategy.json',dict(micro=chosen,effective_batch=8,precision='bf16',FFT='fp32',selection='Largest common feasible micro before any score selection',dry_hash=sha(OUT/'dry_fit.json')))
        for d in c['data']:
            values,_=load(CACHE,d,'train')
            for arm in ARMS:
                m=make(arm,c['data'][d]['channel_ids'],41000,'cuda');m.train();opt=torch.optim.AdamW(parameters(m).values(),lr=.001,weight_decay=0,betas=(.9,.999),eps=1e-8);initial=cpu_state(m);frozen=frozen_hash(m);records=[]
                for j in range(2):
                    assert s['smoke_attempts']<24;s['smoke_attempts']+=1;save(OUT/'status.json',s)
                    def applied():s['smoke_updates']+=1;save(OUT/'status.json',s)
                    records.append(step(m,opt,values,c['data'][d]['origins']['train'][j*8:(j+1)*8],chosen,w,applied))
                now=cpu_state(m);changed={n:float((now[n].double()-initial[n].double()).norm()) for n in now};assert frozen_hash(m)==frozen and any(v>0 for n,v in changed.items() if n.startswith('head.'))
                if arm=='BASIS_BUDGET':assert changed['channel_adapter.coefficients']>0 and all(changed[f'channel_adapter.up_projections.{k}.weight']>0 for k in range(1,4))
                checks.append(dict(dataset=d,arm=arm,passed=True,records=records,parameter_changes=changed,frozen_unchanged=True));save(OUT/'smoke_checks.json',checks);m=opt=None;initial=now=None;cleanup();print('R2 SMOKE',d,arm,s['smoke_updates'],flush=True)
        assert s['smoke_updates']==24;s['status']='SMOKE_PASSED'
    except BaseException as e:s.update(status='BLOCKED_GPU_BUSY' if 'BLOCKED_GPU_BUSY' in str(e) else 'BLOCKED_RESOURCE' if isinstance(e,torch.cuda.OutOfMemoryError) else 'INCONCLUSIVE_EXECUTION',error=repr(e),traceback=traceback.format_exc())
    finally:save(OUT/'status.json',s);w.close();print('R2 SMOKE END',s['status'],flush=True)

def train_evaluate():
    c=verify_contract(OUT);s=read(OUT/'status.json');assert s['status']=='SMOKE_PASSED' and s['fit_attempts']==0;strategy=read(OUT/'resource_strategy.json');micro=strategy['micro'];schedules=read(OUT/'schedules.json');assert sha(OUT/'schedules.json')==c['schedules_hash'];assert sha(OUT/'fit_manifest.json')==c['fit_manifest_hash'];w=Watch(OUT,'training',43200);fits=[];trajectory=[];evaluation=[]
    try:
        w.boundary(startup=True)
        for cell in read(OUT/'fit_manifest.json'):
            w.boundary();d,arm,seed=cell['dataset'],cell['arm'],cell['seed'];fid=f"{cell['attempt_order']:02}_{d}_{seed}_{arm}";s['fit_attempts']+=1;s.update(status='TRAINING',current_fit=fid);fit=dict(fit=fid,**cell,status='RUNNING',updates=0,epochs=0);fits.append(fit);save(OUT/'status.json',s);save(OUT/'fits.json',fits);m=opt=None;records=[];fr=[];started=time.monotonic()
            try:
                m=make(arm,c['data'][d]['channel_ids'],seed,'cuda');m.train();opt=torch.optim.AdamW(parameters(m).values(),lr=.001,weight_decay=0,betas=(.9,.999),eps=1e-8);scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=5,gamma=.5);frozen=frozen_hash(m);initial=cpu_state(m);values,std=load(CACHE,d,'development');best=None
                def checkpoint(epoch):
                    nonlocal best
                    r=evaluate(m,values,std,c['data'][d]['origins']['validation'],f'V_{fid}_{epoch:02}',w,micro);path=CACHE/f'{fid}_epoch{epoch:02}.pt';torch.save(cpu_state(m),path);row=dict(fit=fid,dataset=d,seed=seed,arm=arm,epoch=epoch,updates=fit['updates'],checkpoint_path=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path),**r);fr.append(row);trajectory.append(row);save(OUT/'trajectories.json',trajectory)
                    if best is None or r['metrics']['mse']<best['metrics']['mse']:best=row
                    print('R2 V',fid,epoch,r['metrics']['mse'],flush=True);return r['metrics']['mse']
                patience_best=checkpoint(0);bad=0
                for epoch,oo in enumerate(schedules[f'{d}_{seed}'],1):
                    m.train()
                    for i in range(0,len(oo),8):
                        def applied():fit['updates']+=1;s['training_updates']+=1
                        r=step(m,opt,values,oo[i:i+8],micro,w,applied);records.append(dict(epoch=epoch,update=fit['updates'],**r))
                    fit['epochs']=epoch;v=checkpoint(epoch)
                    if v<patience_best-1e-4:patience_best=v;bad=0
                    else:bad+=1
                    scheduler.step();save(OUT/'status.json',s);save(OUT/'fits.json',fits);save(OUT/(fid+'_steps.json'),records)
                    if bad>=5:break
                assert fit['updates']==fit['epochs']*math.ceil(len(oo)/8) and fit['updates']<={'electricity':1280,'traffic':1260}[d]
                assert frozen_hash(m)==frozen;now=cpu_state(m);save(OUT/(fid+'_parameter_updates.json'),{n:float((now[n].double()-initial[n].double()).norm()) for n in now})
                # A newly constructed model, not merely an in-place restoration, replays best V.
                m=opt=None;cleanup();m=make(arm,c['data'][d]['channel_ids'],seed,'cuda');restore(m,torch.load(ROOT/best['checkpoint_path'],weights_only=True,map_location='cpu'));replay=evaluate(m,values,std,c['data'][d]['origins']['validation'],'RELOAD_'+fid,w,micro)
                with np.load(ROOT/best['prediction_path']) as z:a=z['prediction']
                with np.load(ROOT/replay['prediction_path']) as z:b=z['prediction']
                assert np.array_equal(a,b) and replay['metrics']==best['metrics'];fit.update(status='COMPLETE',best=best,reload=replay,reload_max_abs=0.,frozen_unchanged=True,active_seconds=sum(r['seconds'] for r in records),wall_seconds=time.monotonic()-started,peak_allocated=max(r['peak_allocated'] for r in records),peak_reserved=max(r['peak_reserved'] for r in records),median_step_seconds=float(np.median([r['seconds'] for r in records])),budget_limited=fit['epochs']==20 and fr[-1]['metrics']['mse']<fr[-2]['metrics']['mse'],early_stopped=bad>=5);s['fits_completed']+=1
            except BaseException as e:
                fit.update(status='EXECUTION_ERROR',error=repr(e),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-started)
                if isinstance(e,ResourceError) or any(v in str(e).lower() for v in ['illegal memory','device-side assert']):raise
            finally:save(OUT/(fid+'_steps.json'),records);save(OUT/'fits.json',fits);save(OUT/'status.json',s);m=opt=None;initial=now=None;cleanup()
        if s['fits_completed']!=24:s['status']='INCONCLUSIVE_EXECUTION';return
        controls=[]
        for d in c['data']:
            for seed in [41000,41001]:
                candidates=[f for f in fits if f['dataset']==d and f['seed']==seed and f['arm'] in ['INDIV_BUDGET','SHARED_BUDGET','FACTOR_BUDGET']];f=min(candidates,key=lambda f:(f['best']['metrics']['mse'],f['arm']));controls.append(dict(dataset=d,seed=seed,arm=f['arm'],fit=f['fit'],validation_mse=f['best']['metrics']['mse'],selection_cost_fits=3))
        seal=dict(contract_hash=sha(OUT/'contract.json'),selections=[f['best'] for f in fits],controls=controls,planned_attempts_finished=True,sealed_at=time.time());seal['seal_hash']=digest(seal);save(OUT/'selection_seal.json',seal);verify_contract(OUT);audit_history()
        # V-only mechanisms; no use for checkpoint/control selection.
        interventions=[]
        for f in fits:
            if f['arm']!='BASIS_BUDGET':continue
            d,seed=f['dataset'],f['seed'];m=make(f['arm'],c['data'][d]['channel_ids'],seed,'cuda');restore(m,torch.load(ROOT/f['best']['checkpoint_path'],weights_only=True,map_location='cpu'));values,std=load(CACHE,d,'development');coeff=m.channel_adapter.coefficients.detach().clone();perm=np.random.default_rng(61530).permutation(32)
            for role,new in [('permuted',coeff[torch.tensor(perm,device='cuda')]),('channel_mean',coeff.mean(0,keepdim=True).expand_as(coeff))]:
                with torch.no_grad():m.channel_adapter.coefficients.copy_(new)
                rr=evaluate(m,values,std,c['data'][d]['origins']['validation'],f'V_DIAG_{f["fit"]}_{role}',w,micro);interventions.append(dict(dataset=d,seed=seed,arm=f['arm'],role=role,base_validation_mse=f['best']['metrics']['mse'],coefficient_channel_variance=coeff.var(0,unbiased=False).cpu().tolist(),**rr))
            m=None;cleanup()
        save(OUT/'mechanism.json',interventions);save(OUT/'evaluation_access.json',dict(opened_at=time.time(),selection_hash=sha(OUT/'selection_seal.json')))
        for d in c['data']:
            values,std=load(CACHE,d,'evaluation');r=evaluate(None,values,std,c['data'][d]['origins']['evaluation'],f'E_{d}_seasonal',w,micro,True);evaluation.append(dict(dataset=d,seed=None,arm='SEASONAL_NAIVE',role='seasonal_naive',**r))
            for f in [f for f in fits if f['dataset']==d]:
                m=make(f['arm'],c['data'][d]['channel_ids'],f['seed'],'cuda')
                for role in ['INIT','selected']:
                    if role=='selected':restore(m,torch.load(ROOT/f['best']['checkpoint_path'],weights_only=True,map_location='cpu'))
                    r=evaluate(m,values,std,c['data'][d]['origins']['evaluation'],f'E_{f["fit"]}_{role}',w,micro);evaluation.append(dict(dataset=d,seed=f['seed'],arm=f['arm'],role=role,**r));save(OUT/'evaluation.json',evaluation)
                m=None;cleanup()
        s['status']='COMPLETE'
    except BaseException as e:s.update(status='BLOCKED_GPU_BUSY' if 'BLOCKED_GPU_BUSY' in str(e) else 'INCONCLUSIVE_EXECUTION',error=repr(e),traceback=traceback.format_exc())
    finally:save(OUT/'status.json',s);w.close();print('R2 END',s['status'],s['fits_completed'],flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','smoke','train-evaluate','verify','status']);a=p.parse_args();configure()
    if a.stage=='status':print(json.dumps(read(OUT/'status.json'),indent=2))
    elif a.stage=='verify':
        from report import verify_rank2
        verify_rank2()
    else:globals()[a.stage.replace('-','_')]()
if __name__=='__main__':main()
