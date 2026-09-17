"""Fixed-budget fits, V-only selection, epoch-boundary exact resume."""
import signal
from .common import *
from .checks import arrays,batch,backward
from .reference_core import STATES,rng
STOP=False
def request_stop(*_):
    global STOP
    STOP=True

def fit_id(source,arm,seed,lr):return f'{source}_{arm}_s{seed}_lr{lr:g}'

@torch.no_grad()
def predict(model,x,s,micro,watch):
    chunks=[];model.eval()
    for lo in range(0,len(x),micro):
        watch.boundary()
        xx=torch.as_tensor(np.array(x[lo:lo+micro],copy=True),device='cuda');ss=torch.as_tensor(np.array(s[lo:lo+micro],copy=True),device='cuda')
        p=model(xx,ss);assert torch.isfinite(p).all(),'NONFINITE_PREDICTIONS'
        chunks.append(p.cpu().numpy())
    return np.concatenate(chunks)

def validation(model,source,micro,watch):
    f=CACHE/'conditions'/source;n=256
    x=np.load(f/'V_SELECT_x.npy',mmap_mode='r');s=np.load(f/'V_SELECT_sigma.npy',mmap_mode='r');y=np.load(f/'V_SELECT_y.npy',mmap_mode='r')
    indices=np.concatenate([np.arange(STATES.index(k)*2*n,(STATES.index(k)+1)*2*n) for k in ['REFERENCE','POINT8','BURST8','SHIFT4','SHIFT8']])
    sync();t=time.perf_counter();p=predict(model,x[indices],s[indices],micro,watch)
    metric_sigma=np.tile(read(OUT/'DATA_MANIFEST.json')[source]['sigma_train_population'],len(x)//4)
    score=float(np.mean(np.abs(p[:,4].astype(float)-y[indices])/metric_sigma[indices,None]));sync()
    return score,time.perf_counter()-t

def fit(source,arm,seed,lr,watch):
    check_seal();fid=fit_id(source,arm,seed,lr);out=OUT/'fits'/fid;out.mkdir(parents=True,exist_ok=True)
    receipt=out/'receipt.json'
    if receipt.exists() and read(receipt)['status']=='COMPLETE':return read(receipt)
    folder=CACHE/'fits'/fid;folder.mkdir(parents=True,exist_ok=True)
    journal=OUT/'UPDATE_LEDGER.jsonl';resume=folder/'resume.pt'
    records=[json.loads(line) for line in open(journal) if json.loads(line)['fit']==fid] if journal.exists() else []
    total=sum(1 for _ in open(journal)) if journal.exists() else 0
    assert total<=59392 and 1024-len(records)<=59392-total,'MAIN_UPDATE_CAP'
    a=arrays(source);micro=read(OUT/'microbatch.json')[source]['microbatch']
    model=build(arm,seed,source=source);initial_check(model,source,arm,seed,out,watch);buffer_before=tensor_hash(dict(model.named_buffers()));frozen_before=frozen_hash(model);optimizer=torch.optim.AdamW(parameters(model).values(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0)
    checks=[];step=0;train_time=0.;val_time=0.;io_time=0.;gradclip=[]
    if resume.exists():
        state=torch.load(resume,weights_only=False,map_location='cpu')
        step=state['step'];assert step%32==0 and len(records)==step,'PARTIAL_EPOCH_NO_EXACT_RECOVERY'
        intent=read(out/'update_intent.json')
        assert intent['status']=='JOURNALED' and intent['completed_step']==step,'UNRESOLVED_OPTIMIZER_INTENT'
        restore(model,state['model']);optimizer.load_state_dict(state['optimizer'])
        checks=state['checks'];train_time=state['train_time'];val_time=state['val_time'];io_time=state['io_time']
        torch.set_rng_state(state['rng_cpu']);torch.cuda.set_rng_state_all(state['rng_cuda'])
    else:
        assert not records,'NO_RESUME_FOR_COMPLETED_UPDATES'
        assert not (out/'update_intent.json').exists(),'UNRESOLVED_OPTIMIZER_INTENT'
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    def checkpoint():
        nonlocal val_time,io_time
        if step not in [0,256,512,768,1024]:return
        if any(c['step']==step for c in checks):return
        v,elapsed=validation(model,source,micro,watch);val_time+=elapsed
        t=time.perf_counter();p=folder/f'checkpoint_{step}.pt';atomic_torch(p,cpu_state(model));io_time+=time.perf_counter()-t
        checks.append(dict(step=step,objective=v,checkpoint=str(p.relative_to(ROOT)),sha256=sha(p)))
        save(out/'validation.json',checks)
        print('CHECKPOINT',fid,step,v,flush=True);status(current_fit=fid,current_step=step)
    checkpoint()
    for epoch in range(step//32,32):
        order=rng(84100,source,seed,epoch).permutation(1024)
        for offset in range(0,1024,32):
            watch.boundary();x,y,s=batch(a,arm,epoch,order[offset:offset+32])
            sync();t=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            value=backward(model,x,y,s,micro);norm=torch.nn.utils.clip_grad_norm_(parameters(model).values(),1.)
            assert torch.isfinite(norm),'NONFINITE_GRAD_NORM'
            # Intent prevents claiming an interrupted optimizer update never happened.
            sync();forward_backward_seconds=time.perf_counter()-t
            save(out/'update_intent.json',dict(fit=fid,next_step=step+1,status='BEFORE_OPTIMIZER'))
            t=time.perf_counter();optimizer.step();sync();elapsed=forward_backward_seconds+time.perf_counter()-t;train_time+=elapsed;step+=1
            append(journal,dict(fit=fid,source=source,arm=arm,seed=seed,lr=lr,step=step,epoch=epoch,
                                loss=value,gradient_norm=float(norm),optimizer_seconds=elapsed))
            save(out/'update_intent.json',dict(fit=fid,completed_step=step,status='JOURNALED'))
        if step in [256,512,768,1024]:checkpoint()
        t=time.perf_counter();atomic_torch(resume,dict(step=step,model=cpu_state(model),optimizer=optimizer.state_dict(),checks=checks,
            train_time=train_time,val_time=val_time,io_time=io_time,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()))
        io_time+=time.perf_counter()-t
        if STOP:raise ResourceError('USER_STOP_AT_EXACT_EPOCH_BOUNDARY')
    assert step==1024 and frozen_before==frozen_hash(model) and buffer_before==tensor_hash(dict(model.named_buffers()))
    assert sorted(c['step'] for c in checks)==[0,256,512,768,1024],'INVALID_CHECKPOINT_SCHEDULE'
    selected=min(checks,key=lambda v:(v['objective'],v['step']))
    result=dict(fit=fid,source=source,arm=arm,seed=seed,lr=lr,status='COMPLETE',updates=step,
        selected=selected,checkpoints=checks,optimization_limited=selected['step']==1024,
        frozen_unchanged=True,frozen_sha256=frozen_before,microbatch=micro,
        optimizer_seconds=train_time,validation_seconds=val_time,checkpoint_io_seconds=io_time,
        invocation_wall_seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    save(receipt,result);print('FIT_COMPLETE',fid,'selected',selected['step'],flush=True)
    del model,optimizer,a;cleanup();status();return result

def initial_check(model,source,arm,seed,out,watch):
    if arm=='B0':return
    row=baseline_row(source,seed)
    base=build('C0',seed,source);a=arrays(source);x,y,s=batch(a,arm,0,np.arange(4))
    with torch.no_grad():assert torch.equal(model(x,s),base(x,s)),'INITIAL_NOT_MATCHED_B0'
    save(out/'initial_parity.json',dict(baseline=row,exact=True,optimizer_updates=0))
    del base;cleanup()

def choice(source,arm,seed):
    rows=[read(OUT/'fits'/fit_id(source,arm,seed,lr)/'receipt.json') for lr in [1e-4,3e-4]]
    r=min(rows,key=lambda v:(v['selected']['objective'],v['lr'],v['selected']['step']))
    return dict(lr=r['lr'],fit=r['fit'],selected=r['selected'])

def register_base(r):
    p=OUT/'NEW_BASELINES.json';d=read(p) if p.exists() else {}
    d[f"{r['source']}_{r['seed']}"]=dict(source=r['source'],seed=r['seed'],fit=r['fit'],**r['selected'])
    save(p,d)

def selected_row(r):return dict(source=r['source'],arm='C0' if r['arm']=='B0' else r['arm'],seed=r['seed'],fit=r['fit'],lr=r['lr'],**r['selected'])

def train_all(watch):
    models=read(OUT/'REUSED_MODELS.json');choices={}
    # Fixed original source third seed; never repeat old training.
    for source in ['electricity','ettm1']:
        r=fit(source,'B0',81553,read(V2/'LR_SELECTION.json')[source]['B0']['lr'],watch);register_base(r);models.append(selected_row(r))
        for arm in CORE:
            r=fit(source,arm,81553,read(OLD/'LR_SELECTION.json')[source][arm]['lr'],watch);models.append(selected_row(r))
    for source in ['electricity','ettm1']:
        choices[source]={}
        for arm in CONTROLS:
            for lr in [1e-4,3e-4]:fit(source,arm,81550,lr,watch)
            c=choice(source,arm,81550);choices[source][arm]=c
            for seed in [81551,81552,81553]:models.append(selected_row(fit(source,arm,seed,c['lr'],watch)))
    source='ettm2';choices[source]={}
    for lr in [1e-4,3e-4]:fit(source,'B0',85550,lr,watch)
    c=choice(source,'B0',85550);choices[source]['B0']=c;register_base(read(OUT/'fits'/c['fit']/'receipt.json'))
    for seed in [85551,85552,85553]:
        r=fit(source,'B0',seed,c['lr'],watch);register_base(r);models.append(selected_row(r))
    for arm in CORE:
        for lr in [1e-4,3e-4]:fit(source,arm,85550,lr,watch)
        c=choice(source,arm,85550);choices[source][arm]=c
        for seed in [85551,85552,85553]:models.append(selected_row(fit(source,arm,seed,c['lr'],watch)))
    assert len(models)==54 and len({(m['source'],m['arm'],m['seed']) for m in models})==54
    save(OUT/'LR_SELECTION.json',choices);save(OUT/'MODEL_SELECTION.json',models)
