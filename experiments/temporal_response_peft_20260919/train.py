import signal
from .common import *
from experiments.outlier_signal_peft_v1_20260917.reference_core import rng
STOP=False

def request_stop(*_):
    global STOP
    STOP=True

def fit(source,arm,seed,watch):
    check_seal();fid=fit_id(source,arm,seed);out=OUT/'fits'/fid;out.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json');assert r['status']=='COMPLETE'
        for c in r['checkpoints']:assert sha(ROOT/c['checkpoint'])==c['sha256']
        return r
    folder=CACHE/'fits'/fid;folder.mkdir(parents=True,exist_ok=True);resume=folder/'resume.pt';journal=OUT/'UPDATE_LEDGER.jsonl'
    ledger=[json.loads(l) for l in open(journal)] if journal.exists() else [];records=[r for r in ledger if r['fit']==fid]
    assert len(ledger)+1024-len(records)<=MAIN_CAP,'MAIN_UPDATE_CAP'
    a=arrays(source);watch.boundary();m=build(arm,seed,source);frozen=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()));opt=optimizer_for(m)
    initial=cpu_state(m);r=read(OUT/'fits'/fit_id(source,'PLAIN',seed)/'receipt.json');st=torch.load(ROOT/r['checkpoints'][0]['checkpoint'],map_location='cpu',weights_only=True)
    assert tensor_hash(st)==tensor_hash(initial),'PAIRED_INITIAL_MISMATCH'
    assert frozen==r['frozen_sha256'],'REUSED_B0_MISMATCH'
    x,_,s=batch(a,arm,0,np.arange(32))
    with torch.no_grad():assert torch.equal(m(x,s),m(x,s,residual_mode='off'))
    save(out/'INITIAL.json',dict(initial_exact_B0=True,initial_exact_reused_PLAIN=True,baseline=old.baseline_row(source,seed),frozen_sha256=frozen))
    step=0;checks=[];train_time=0.;val_time=0.;io_time=0.
    if resume.exists():
        z=torch.load(resume,weights_only=False,map_location='cpu');step=z['step'];assert step%32==0 and len(records)==step,'PARTIAL_EPOCH_NO_EXACT_RECOVERY'
        intent=read(out/'intent.json');assert intent==dict(status='JOURNALED',completed_step=step),'AMBIGUOUS_OPTIMIZER'
        restore(m,z['model']);opt.load_state_dict(z['optimizer']);checks=z['checks'];train_time=z['train_time'];val_time=z['val_time'];io_time=z['io_time'];torch.set_rng_state(z['rng_cpu']);torch.cuda.set_rng_state_all(z['rng_cuda'])
    else:assert not records and not (out/'intent.json').exists(),'MISSING_RESUME_AMBIGUOUS_UPDATES'
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    def checkpoint():
        nonlocal val_time,io_time
        if step not in [0,256,512,768,1024] or any(c['step']==step for c in checks):return
        v,elapsed=validation(m,source,32,watch)
        if step==0:np.testing.assert_allclose(v,r['checkpoints'][0]['objective'],rtol=1e-10,atol=1e-12)
        val_time+=elapsed;t=time.perf_counter();p=folder/f'checkpoint_{step}.pt';atomic_torch(p,cpu_state(m));io_time+=time.perf_counter()-t
        checks.append(dict(step=step,objective=v,checkpoint=str(p.relative_to(ROOT)),sha256=sha(p)));save(out/'validation.json',checks)
        print('CHECKPOINT',fid,step,v,flush=True);status(execution='TRAINING',current_fit=fid,current_step=step)
    checkpoint()
    for epoch in range(step//32,32):
        order=rng(84100,source,seed,epoch).permutation(1024)
        for lo in range(0,1024,32):
            watch.boundary();idx=order[lo:lo+32];x,y,s=batch(a,arm,epoch,idx);sync();t=time.perf_counter();opt.zero_grad(set_to_none=True)
            values=backward(m,x,y,s,arm,source,epoch,idx);norm=torch.nn.utils.clip_grad_norm_(parameters(m).values(),1.);assert torch.isfinite(norm);sync();fb=time.perf_counter()-t
            save(out/'intent.json',dict(status='BEFORE_OPTIMIZER',next_step=step+1));t=time.perf_counter();opt.step();sync();elapsed=fb+time.perf_counter()-t;train_time+=elapsed;step+=1
            append(journal,dict(fit=fid,source=source,arm=arm,seed=seed,lr=LR,step=step,epoch=epoch,gradient_norm=float(norm),seconds=elapsed,**values));save(out/'intent.json',dict(status='JOURNALED',completed_step=step))
        checkpoint();t=time.perf_counter();atomic_torch(resume,dict(step=step,model=cpu_state(m),optimizer=opt.state_dict(),checks=checks,train_time=train_time,val_time=val_time,io_time=io_time,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()));io_time+=time.perf_counter()-t
        status(execution='TRAINING',current_fit=fid,current_step=step)
        if STOP:raise ResourceError('USER_STOP_AT_EPOCH_BOUNDARY')
    assert frozen_hash(m)==frozen and tensor_hash(dict(m.named_buffers()))==buffers and step==1024
    assert [c['step'] for c in checks]==[0,256,512,768,1024]
    selected=min(checks,key=lambda c:(c['objective'],c['step']))
    result=dict(fit=fid,source=source,arm=arm,seed=seed,lr=LR,status='COMPLETE',updates=step,new_updates=step,reused=False,selected=selected,checkpoints=checks,optimization_limited=selected['step']==1024,microbatch=32,frozen_sha256=frozen,frozen_unchanged=True,buffers_unchanged=True,optimizer_seconds=train_time,validation_seconds=val_time,checkpoint_io_seconds=io_time,wall_seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),extra_labelled_examples=0,adapted_train_forwards=2*step,teacher_train_forwards=2*step)
    save(out/'receipt.json',result);print('FIT_COMPLETE',fid,'selected',selected['step'],flush=True);del m,opt;cleanup();return result
