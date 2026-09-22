from model import *
from tasks import *
from data import Data
from vendor.cosa_core import SimpleOutputAdapter
import gc,sys,traceback

CAPS=dict(offline=8192,A_local=6144,B_online=12288,smoke=32)
budget=None
def ledger_entries():
    p=RESULTS/'UPDATE_LEDGER.jsonl';return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def update(optimizer,parameters,key,step,phase,lossval):
    global budget
    if budget is None:
        entries=[x for x in ledger_entries() if x['kind']=='intent'];budget=dict(counts={p:sum(x['phase']==p for x in entries) for p in CAPS},seen={(x['key'],x['step']) for x in entries})
    assert budget['counts'][phase]<CAPS[phase] and (key,step) not in budget['seen']
    assert torch.isfinite(lossval);lossval.backward()
    grads=[p.grad for p in parameters if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads)
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(norm)
    append(RESULTS/'UPDATE_LEDGER.jsonl',dict(kind='intent',key=key,step=step,phase=phase,loss=float(lossval.detach()),gradient_norm=float(norm)))
    budget['counts'][phase]+=1;budget['seen'].add((key,step));optimizer.step()
    append(RESULTS/'UPDATE_LEDGER.jsonl',dict(kind='commit',key=key,step=step,phase=phase))
def snapshot(m,g=None):return dict(bank=m.bank(),generator=None if g is None else {k:v.detach().cpu().clone() for k,v in g.state_dict().items()})
def restore(m,g,state):
    m.restore(state['bank'])
    if g is not None:g.load_state_dict(state['generator'])
def path(arm,seed,step):return CACHE/'checkpoints'/f'{arm}_{seed}_{step}.pt'
def load_state(arm,seed,step):return torch.load(path(arm,seed,step),weights_only=True)['state']
def reference(seed):
    m=Model(seed);m.restore(load_state('G0',seed,512)['bank']);m.bank_grad(False);return Reference(seed,m)
def fit(d,arm,seed):
    sealcheck();key=f'{arm}_{seed}';assert not any(x.get('key')==key for x in ledger_entries()),'No ambiguous fit replay'
    m=Model(seed);g=None;ref=None
    if arm!='G0':m.restore(load_state('G0',seed,512)['bank'])
    if arm not in ['G0','STATIC']:g=generator(seed,len(m.targets),arm);ref=reference(seed)
    params=[p for p in m.parameters() if p.requires_grad]+([] if g is None else list(g.parameters()));optimizer=opt(params)
    folder=CACHE/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=snapshot(m,g),step=0),path(arm,seed,0));frozen=m.frozen();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();trace=[];compute=0.
    packets=d.base_schedule(seed) if arm=='G0' else d.meta_schedule(seed)
    for step,pairs in enumerate(packets,1):
        records_batch=[];xs=[];ys=[];scales=[]
        for series,t in pairs:
            x=d.values[t-256:t,series].copy();xs.append(x);ys.append(d.values[t:t+64,series].copy());scales.append(scale(x))
            if g is not None:
                ep=make_episode(d.asof(int(series),int(t)),int(t),arm[0]);qr=ref.support(int(series),ep);records_batch.append(records(ep,qr,arm))
        torch.cuda.synchronize();tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        c=None if g is None else g(torch.stack(records_batch));q=m(tensor(np.stack(xs)),c);lv=loss(q,tensor(np.stack(ys)),tensor(scales));update(optimizer,params,key,step,'offline',lv)
        if g is not None and step==2:
            assert any(p.grad is not None and float(p.grad.norm())>0 for p in g.encoder.parameters()),'Generator gradient missing'
        torch.cuda.synchronize();compute+=time.perf_counter()-tick
        trace.append(dict(step=step,loss=float(lv.detach()),coefficient_absmax=None if c is None else float(c.detach().abs().max()),coefficient_std=None if c is None else float(c.detach().std())))
        if step in [128,512]:
            torch.save(dict(state=snapshot(m,g),optimizer=optimizer.state_dict(),step=step,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),path(arm,seed,step))
            event('offline_checkpoint',key=key,step=step,loss=float(lv.detach()),wall_seconds=time.perf_counter()-start)
    assert m.frozen()==frozen
    save(RESULTS/'fits'/f'{key}.json',dict(arm=arm,seed=seed,updates=512,bank_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),generator_parameters=0 if g is None else sum(p.numel() for p in g.parameters()),compute_seconds=compute,wall_seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated(),reference_new_calls=0 if ref is None else ref.calls,reference_new_seconds=0 if ref is None else ref.seconds,frozen_unchanged=True,trace=trace,checkpoints={str(s):sha(path(arm,seed,s)) for s in STEPS}))
    del m,g,ref,optimizer;gc.collect();torch.cuda.empty_cache()

def query_scores(d,track,role,arm,seed,tag):
    pairs=d.episodes(track,role);q=read_predictions(track,role,arm,seed,tag,pairs);vals=[]
    for (series,t),v in zip(pairs,q):
        vals.append(metrics(v,d.values[t:t+64,series],scale(d.values[t-256:t,series]))['pinball'].mean())
    return float(np.mean(vals))
def resource(track,**kw):append(RESULTS/track/'RESOURCE_EVENTS.jsonl',kw)

def warm_reference_stream(d,role,seed):
    ref=reference(seed);pairs=d.episodes('B',role)
    for series in d.role_indices[role]:
        times=pairs[pairs[:,0]==series,1]
        for issue in range(int(times[0])-64,int(times[-1])+1,8):
            ref.get(series,issue,d.asof(series,issue)[-256:])
    event('reference_stream_sealed',role=role,seed=seed,new_forecasts=ref.calls)
    del ref;gc.collect();torch.cuda.empty_cache()

def predict_fixed(d,track,role,arm,seed,step,state_arm=None):
    m=Model(seed);genarm=arm.startswith(('A_','B_')) and arm.endswith(('TIME','SET','NOERROR','FULLGEN'))
    g=generator(seed,len(m.targets),arm) if genarm else None
    actual=state_arm or ('STATIC' if 'LONG' in arm or arm.endswith('AFFINE') else arm)
    restore(m,g,load_state(actual,seed,step));m.bank_grad(False);ref=reference(seed) if g is not None else None
    issuer=Issuer(track,role,arm,seed,step);pairs=d.episodes(track,role);costs=[];coeffstats=[]
    for series,t in pairs:
        prefix=d.asof(int(series),int(t));ep=make_episode(prefix,int(t),track);qr=ref.support(int(series),ep) if ref else None
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.no_grad():
            c=g(records(ep,qr,arm)[None]) if g else None
            length=(512 if track=='A' else 320) if 'LONG' in arm else 256
            q=m(tensor(prefix[-length:][None]),c).cpu().numpy()[0]
            if arm.endswith('AFFINE'):
                qs=m(tensor(ep['support'])).cpu().numpy();q=apply_affine(q,affine(qs,ep),ep['scale'])
            if c is not None:
                coeffstats.append(dict(series=int(series),issue=int(t),absmax=float(c.abs().max()),std=float(c.std())))
        torch.cuda.synchronize();costs.append(time.perf_counter()-start);issuer.put(series,t,q)
    resource(track,role=role,arm=arm,seed=seed,tag=step,kind='fixed',query_count=len(pairs),compute_seconds=sum(costs),median_seconds=float(np.median(costs)),peak_allocated=torch.cuda.max_memory_allocated(),coefficient_stats=coeffstats)
    del m,g,ref;gc.collect();torch.cuda.empty_cache()

def local_a(d,role,arm,seed,step):
    m=Model(seed);base=load_state('STATIC' if arm=='A_LOCAL' else 'A_TIME',seed,step);m.bank_grad(arm=='A_LOCAL');frozen=m.frozen();issuers={k:Issuer('A',role,arm,seed,k) for k in [0,1,4,8]};costs=[]
    for series,t in d.episodes('A',role):
        m.restore(base['bank']);prefix=d.asof(int(series),int(t));ep=make_episode(prefix,int(t),'A');c=torch.nn.Parameter(torch.ones(1,len(m.targets),8,device='cuda')) if arm=='A_COEFF' else None
        params=[c] if c is not None else [p for p in m.parameters() if p.requires_grad];optimizer=opt(params);key=f'{role}_{arm}_{seed}_{series}_{t}';elapsed=0.
        for k in range(9):
            if k:
                torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                q=m(tensor(ep['support']),None if c is None else c.expand(4,-1,-1));lv=safe_masked_pinball(q,tensor(ep['y']),torch.as_tensor(ep['mask'],device='cuda'),tensor(np.full(4,ep['scale'])))
                update(optimizer,params,key,k,'A_local',lv);torch.cuda.synchronize();elapsed+=time.perf_counter()-start
            if k in issuers:
                torch.cuda.synchronize();start=time.perf_counter()
                with torch.no_grad():q=m(tensor(ep['x'][None]),c).cpu().numpy()[0]
                torch.cuda.synchronize();forward=time.perf_counter()-start;issuers[k].put(series,t,q,optimizer_updates=k)
                costs.append(dict(k=k,seconds=elapsed+forward,series=int(series),issue=int(t)))
    assert m.frozen()==frozen
    resource('A',role=role,arm=arm,seed=seed,kind='local',costs=costs,peak_allocated=torch.cuda.max_memory_allocated(),episodes_reset=True)
    del m,optimizer;gc.collect();torch.cuda.empty_cache()

def online_b(d,role,arm,seed,step):
    m=Model(seed);base=load_state('B_TIME' if arm=='B_COEFF' else 'STATIC',seed,step);m.bank_grad(arm in ['B_WAIT','B_MASKED']);frozen=m.frozen();issuer=Issuer('B',role,arm,seed,'online');pairs=d.episodes('B',role)
    for series in d.role_indices[role]:
        m.restore(base['bank']);torch.manual_seed(seed)
        cell=SimpleOutputAdapter(64,5,9,False,'Linear').cuda().eval() if arm=='B_COSA_CELL' else None
        c=torch.nn.Parameter(torch.ones(1,len(m.targets),8,device='cuda')) if arm=='B_COEFF' else None
        params=list(cell.parameters()) if cell is not None else [c] if c is not None else [p for p in m.parameters() if p.requires_grad];optimizer=opt(params,1e-3 if cell is not None else 1e-4)
        costs=[];unique_cells=set();unique_targets=set();uses=0
        for tick,t in enumerate(pairs[pairs[:,0]==series,1]):
            ep=make_episode(d.asof(int(series),int(t)),int(t),'B');mask=ep['mask'].copy()
            if arm=='B_WAIT':mask[1:]=False
            active=np.flatnonzero(mask.any(1));x=ep['support'][active];y=ep['y'][active];visible=mask[active];s=tensor(np.full(len(active),ep['scale']))
            for i in active:
                for j in np.flatnonzero(mask[i]):unique_cells.add((int(ep['issues'][i]),int(j)));unique_targets.add(int(ep['issues'][i]+j));uses+=2
            torch.cuda.synchronize();start=time.perf_counter()
            fixed=None
            if cell is not None:
                with torch.no_grad():fixed=m(tensor(x))
            for j in range(2):
                optimizer.zero_grad(set_to_none=True)
                q=cosa_forward(cell,fixed,x) if cell is not None else m(tensor(x),None if c is None else c.expand(len(x),-1,-1))
                lv=safe_masked_pinball(q,tensor(y),torch.as_tensor(visible,device='cuda'),s);update(optimizer,params,f'{role}_{arm}_{seed}_{series}',tick*2+j+1,'B_online',lv)
            with torch.no_grad():
                q=m(tensor(ep['x'][None]),c)
                if cell is not None:q=cosa_forward(cell,q,ep['x'][None])
                q=q.cpu().numpy()[0]
            torch.cuda.synchronize();costs.append(time.perf_counter()-start);issuer.put(series,t,q,optimizer_updates=(tick+1)*2)
        state=dict(bank=m.bank(),coefficient=None if c is None else c.detach().cpu(),cell=None if cell is None else {n:v.detach().cpu() for n,v in cell.state_dict().items()},optimizer=optimizer.state_dict())
        p=CACHE/'B'/'stream_states'/f'{role}_{arm}_{seed}_{series}.pt';p.parent.mkdir(parents=True,exist_ok=True);torch.save(state,p)
        resource('B',role=role,arm=arm,seed=seed,series=int(series),kind='online',compute_seconds=sum(costs),median_seconds=float(np.median(costs)),updates=96,unique_target_lead_cells=len(unique_cells),unique_target_indices=len(unique_targets),cell_uses=uses,state_sha=sha(p),peak_allocated=torch.cuda.max_memory_allocated())
        event('online_stream_complete',role=role,arm=arm,seed=seed,series=int(series))
    assert m.frozen()==frozen
    del m,optimizer;gc.collect();torch.cuda.empty_cache()

def select_and_run(d,track):
    arms=[f'{track}_'+x for x in (['NOERROR','SET','TIME'] if track=='A' else ['FULLGEN','SET','TIME'])]
    for seed in SEEDS:
        for arm in (arms if seed==SEEDS[0] else arms[::-1]):fit(d,arm,seed)
    if track=='B':
        for seed in SEEDS:warm_reference_stream(d,'DEV',seed)
    selection={};devrows=[]
    for arm in ['STATIC']+arms:
        selection[arm]={}
        for seed in SEEDS:
            scores={}
            for step in STEPS:
                predict_fixed(d,track,'DEV',arm,seed,step);scores[step]=query_scores(d,track,'DEV',arm,seed,step)
            chosen=min(scores,key=lambda s:(scores[s],s));selection[arm][str(seed)]=chosen;devrows.append(dict(arm=arm,seed=seed,scores=scores,selected=chosen))
    save(RESULTS/track/'MODEL_SELECTION.json',dict(checkpoints=selection,validation=devrows))
    for seed in SEEDS:
        predict_fixed(d,track,'DEV','G0',seed,512)
        for arm in ['STATIC_LONG'+('512' if track=='A' else '320'),track+'_AFFINE']:predict_fixed(d,track,'DEV',arm,seed,selection['STATIC'][str(seed)])
        methods=[track+'_LOCAL',track+'_COEFF'] if track=='A' else ['B_WAIT','B_MASKED','B_COEFF','B_COSA_CELL']
        for arm in (methods if seed==SEEDS[0] else methods[::-1]):
            step=selection[track+'_TIME' if arm.endswith('COEFF') else 'STATIC'][str(seed)]
            if track=='A':local_a(d,'DEV',arm,seed,step)
            else:online_b(d,'DEV',arm,seed,step)
    tags={arm:{str(s):selection[arm][str(s)] for s in SEEDS} for arm in ['STATIC']+arms}
    tags['G0']={str(s):512 for s in SEEDS}
    for arm in ['STATIC_LONG'+('512' if track=='A' else '320'),track+'_AFFINE']:tags[arm]={str(s):selection['STATIC'][str(s)] for s in SEEDS}
    kselected={}
    for arm in methods:
        if track=='A':
            means={k:float(np.mean([query_scores(d,track,'DEV',arm,s,k) for s in SEEDS])) for k in [0,1,4,8]};kselected[arm]=min(means,key=lambda k:(means[k],k));tags[arm]={str(s):kselected[arm] for s in SEEDS}
        else:tags[arm]={str(s):'online' for s in SEEDS}
    scores={arm:float(np.mean([query_scores(d,track,'DEV',arm,s,tags[arm][str(s)]) for s in SEEDS])) for arm in tags}
    lawful=[a for a in tags if a!=track+'_TIME'];baseline=min(lawful,key=lambda a:(scores[a],a))
    localwinner=min(['A_LOCAL','A_COEFF'],key=scores.get) if track=='A' else None
    save(RESULTS/track/'BASELINE_SELECTION.json',dict(baseline=baseline,local_baseline=localwinner,dev_scores=scores,tags=tags,kselected=kselected))
    save(RESULTS/track/'SELECTION_SEAL.json',dict(files={p.name:sha(p) for p in [RESULTS/track/'MODEL_SELECTION.json',RESULTS/track/'BASELINE_SELECTION.json']},source_seal=sha(RESULTS/'SOURCE_SEAL.json')));event('selection_sealed',track=track,baseline=baseline)
    if track=='B':
        for seed in SEEDS:warm_reference_stream(d,'TEST',seed)
    for seed in SEEDS:
        fixed=['G0','STATIC']+arms+['STATIC_LONG'+('512' if track=='A' else '320'),track+'_AFFINE']
        for arm in (fixed if seed==SEEDS[0] else fixed[::-1]):predict_fixed(d,track,'TEST',arm,seed,tags[arm][str(seed)])
        for arm in (methods if seed==SEEDS[0] else methods[::-1]):
            step=selection[track+'_TIME' if arm.endswith('COEFF') else 'STATIC'][str(seed)]
            if track=='A':local_a(d,'TEST',arm,seed,step)
            else:online_b(d,'TEST',arm,seed,step)
    save(RESULTS/track/'ALL_TEST_SAVED.json',dict(status='COMPLETE',test_query_scoring_started=False));event('test_predictions_saved',track=track)
    readonly_checks(d,track,selection)

def readonly_checks(d,track,selection):
    rows=[];ablation=[]
    for seed in SEEDS:
        ref=reference(seed);series,t=d.episodes(track,'TEST')[0];prefix=d.asof(int(series),int(t));ep=make_episode(prefix,int(t),track)
        support_times=[]
        with torch.no_grad():
            for repeat in range(5):
                torch.cuda.synchronize();start=time.perf_counter();qr=ref.model(tensor(ep['support'])).cpu().numpy();torch.cuda.synchronize();support_times.append(time.perf_counter()-start)
        rows.append(dict(seed=seed,arm='G0_SUPPORT4',seconds=support_times,median=float(np.median(support_times)),min=min(support_times),max=max(support_times),query_count=4))
        arms=['G0','STATIC','STATIC_LONG'+('512' if track=='A' else '320'),track+'_AFFINE']+list(a for a in selection if a!='STATIC')
        for arm in (arms if seed==SEEDS[0] else arms[::-1]):
            m=Model(seed);g=generator(seed,len(m.targets),arm) if arm in selection and arm!='STATIC' else None
            state_arm=arm if arm in selection or arm=='G0' else 'STATIC';step=512 if arm=='G0' else selection[state_arm][str(seed)];restore(m,g,load_state(state_arm,seed,step));m.bank_grad(False);times=[]
            with torch.no_grad():
                rr=records(ep,qr,arm)
                for repeat in range(5):
                    torch.cuda.synchronize();start=time.perf_counter();c=g(rr[None]) if g else None
                    length=(512 if track=='A' else 320) if 'LONG' in arm else 256
                    q=m(tensor(prefix[-length:][None]),c).cpu().numpy()[0]
                    if arm.endswith('AFFINE'):q=apply_affine(q,affine(m(tensor(ep['support'])).cpu().numpy(),ep),ep['scale'])
                    torch.cuda.synchronize();times.append(time.perf_counter()-start)
                if arm==track+'_TIME':
                    c1=g(rr[None]);c2=g(rr.flip(0)[None]);q1=m(tensor(ep['x'][None]),c1);q2=m(tensor(ep['x'][None]),c2)
                    ablation.append(dict(seed=seed,coefficient_max_difference=float((c1-c2).abs().max()),prediction_max_difference=float((q1-q2).abs().max()),scope='First fixed TEST episode read-only; no selection or training'))
            rows.append(dict(seed=seed,arm=arm,seconds=times,median=float(np.median(times)),min=min(times),max=max(times),query_count=1,needs_G0_support=g is not None));del m,g;gc.collect();torch.cuda.empty_cache()
        del ref;gc.collect();torch.cuda.empty_cache()
    save(RESULTS/track/'FORWARD_BENCHMARK.json',dict(repeats=5,scope='First predetermined TEST context, fixed snapshots, no optimizer; support4 batched; exclude checkpoint load and file IO',rows=rows))
    save(RESULTS/track/'ORDER_ABLATION.json',ablation)

def run():
    d=Data();assert read(RESULTS/'PREFLIGHT.json')['status']=='PASS';assert not (RESULTS/'SOURCE_SEAL.json').exists()
    save(RESULTS/'SOURCE_SEAL.json',dict(files=source_files(),data={p.name:sha(p) for p in RESULTS.glob('*.json')},main_updates=0))
    for arm in ['G0','STATIC']:
        for seed in SEEDS:fit(d,arm,seed)
    save(RESULTS/'SHARED_CHECKPOINTS.json',{f'{a}_{s}':{str(k):sha(path(a,s,k)) for k in STEPS} for a in ['G0','STATIC'] for s in SEEDS})
    for track in ['A','B']:select_and_run(d,track)
    save(RESULTS/'ALL_MAIN_COMPLETE.json',dict(status='COMPLETE',counts=budget['counts']));event('all_main_complete',counts=budget['counts'])

if __name__=='__main__':
    setup();CACHE.mkdir(parents=True,exist_ok=True);lock=CACHE/'EXECUTOR.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:run()
    except Exception as e:
        save(RESULTS/f'ERROR_{int(time.time())}.json',dict(classification='BLOCKED_RESOURCE' if 'out of memory' in str(e).lower() else 'BLOCKED_IMPLEMENTATION',error=repr(e),traceback=traceback.format_exc()));raise
    finally:lock.unlink()
