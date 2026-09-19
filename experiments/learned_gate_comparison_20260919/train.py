import signal
from .common import *
from experiments.outlier_signal_peft_v1_20260917.reference_core import rng
STOP=False
def request_stop(*_):
    global STOP
    STOP=True
def fit(source,arm,seed,lr,watch):
    fid=fit_id(source,arm,seed,lr);out=OUT/'fits'/fid;folder=CACHE/'fits'/fid
    out.mkdir(parents=True,exist_ok=True);folder.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json');assert r['status']=='COMPLETE'
        for c in r['checkpoints']:assert sha(ROOT/c['checkpoint'])==c['sha256']
        return r
    records=[json.loads(s) for s in (OUT/'UPDATE_LEDGER.jsonl').read_text().splitlines()] if (OUT/'UPDATE_LEDGER.jsonl').exists() else []
    own=[r for r in records if r['fit']==fid];assert len(records)+1024-len(own)<=MAIN_CAP
    m=build(arm,seed,source);before=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()))
    opt=torch.optim.AdamW(parameters(m).values(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0)
    step=0;checks=[];train_time=0.;val_time=0.;a=arrays(source);resume=folder/'resume.pt'
    if resume.exists():
        z=torch.load(resume,map_location='cpu',weights_only=False);step=z['step']
        assert step%32==0 and len(own)==step
        assert read(out/'intent.json')==dict(status='JOURNALED',step=step)
        restore(m,z['model']);opt.load_state_dict(z['optimizer']);checks=z['checks'];train_time=z['train_time'];val_time=z['val_time']
        torch.set_rng_state(z['rng_cpu']);torch.cuda.set_rng_state_all(z['rng_cuda'])
    else:assert not own and not (out/'intent.json').exists()
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    def checkpoint():
        nonlocal val_time
        if step not in [0,256,512,768,1024] or any(c['step']==step for c in checks):return
        value,elapsed=validation(m,source,32,watch);val_time+=elapsed
        p=folder/f'checkpoint_{step}.pt';atomic_torch(p,cpu_state(m))
        checks.append(dict(step=step,objective=value,checkpoint=str(p.relative_to(ROOT)),sha256=sha(p)))
        save(out/'validation.json',checks);print('CHECKPOINT',fid,step,value,flush=True)
    checkpoint()
    for epoch in range(step//32,32):
        order=rng(84100,source,seed,epoch).permutation(1024)
        for lo in range(0,1024,32):
            begin=watch.before();idx=order[lo:lo+32];x,y,s=batch(a,arm,epoch,idx);sync();t=time.perf_counter();opt.zero_grad(set_to_none=True)
            forecast_loss=loss_2pinball(m(x,s),y,s,m.base.quantiles);entropy=m.entropy();loss=forecast_loss+m.regularizer();assert torch.isfinite(loss);loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters(m).values())
            norm=torch.nn.utils.clip_grad_norm_(parameters(m).values(),1.);assert torch.isfinite(norm)
            save(out/'intent.json',dict(status='BEFORE_OPTIMIZER',step=step+1));opt.step();sync();elapsed=time.perf_counter()-t;train_time+=elapsed;step+=1
            append(OUT/'UPDATE_LEDGER.jsonl',dict(fit=fid,step=step,source=source,arm=arm,seed=seed,lr=lr,epoch=epoch,loss=float(loss.detach()),forecast_loss=float(forecast_loss.detach()),entropy=float(entropy.detach()),gate_mean=float(m.adapter.last_gate.detach().mean()),gradient_norm=float(norm),seconds=elapsed))
            save(out/'intent.json',dict(status='JOURNALED',step=step));_,bad=watch.after(begin)
            if bad:raise ResourceError('EXTERNAL_COMPUTE_DURING_UPDATE')
        checkpoint();atomic_torch(resume,dict(step=step,model=cpu_state(m),optimizer=opt.state_dict(),checks=checks,train_time=train_time,val_time=val_time,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()))
        status(execution='TRAINING',current_fit=fid,current_step=step)
        if STOP:raise ResourceError('REQUESTED_STOP_EPOCH_BOUNDARY')
    assert step==1024 and frozen_hash(m)==before and tensor_hash(dict(m.named_buffers()))==buffers
    assert [c['step'] for c in checks]==[0,256,512,768,1024]
    r=dict(status='COMPLETE',fit=fid,source=source,arm=arm,seed=seed,lr=lr,updates=step,checkpoints=checks,selected=min(checks,key=lambda c:(c['objective'],c['step'])),frozen_unchanged=True,frozen_sha256=before,buffers_unchanged=True,trainable_parameters=sum(v.numel() for v in parameters(m).values()),optimizer_seconds=train_time,validation_seconds=val_time,invocation_wall_seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    save(out/'receipt.json',r);print('FIT_COMPLETE',fid,flush=True);del m,opt;cleanup();return r
def train_all(watch):
    choices={}
    for source in SOURCES:
        choices[source]={}
        for arm in ARMS:
            runs=[fit(source,arm,81550,lr,watch) for lr in LRS]
            chosen=min(runs,key=lambda r:(r['selected']['objective'],r['lr'],r['selected']['step']))
            choices[source][arm]=dict(lr=chosen['lr'],selection_fit=chosen['fit'])
    save(OUT/'LR_SELECTION.json',choices)
    for source in SOURCES:
        for arm in ARMS:
            for seed in SEEDS:fit(source,arm,seed,choices[source][arm]['lr'],watch)
    assert sum(1 for _ in open(OUT/'UPDATE_LEDGER.jsonl'))==MAIN_CAP
