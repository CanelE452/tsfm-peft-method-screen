from model import *
import gc,sys,subprocess,traceback,platform,importlib.metadata
from data import Data
from evaluation import *

def source_hashes():return {str(p.relative_to(EXP)):sha(p) for p in EXP.iterdir() if p.suffix in ['.py','.md']}
def sealcheck():
    for name,h in read(RESULTS/'SOURCE_SEAL.json')['files'].items():assert sha(EXP/name)==h,name
def ledger(kind,**kw):
    with (RESULTS/'UPDATE_LEDGER.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(kind=kind,utc=time.time(),**kw))+'\n');f.flush()
def spent():
    p=RESULTS/'UPDATE_LEDGER.jsonl';lines=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    return lines
_budget=None
def update(opt,key,step,phase):
    global _budget
    if _budget is None:
        entries=[x for x in spent() if x['kind']=='intent'];_budget=dict(main=sum(x['phase']=='main' for x in entries),smoke=sum(x['phase']=='smoke' for x in entries),seen={(x['key'],x['step']) for x in entries})
    cap=8192 if phase=='main' else 8;assert _budget[phase]<cap and (key,step) not in _budget['seen']
    ledger('intent',key=key,step=step,phase=phase);_budget[phase]+=1;_budget['seen'].add((key,step))
    torch.cuda.synchronize();tick=time.perf_counter();opt.step();torch.cuda.synchronize();seconds=time.perf_counter()-tick
    ledger('commit',key=key,step=step,phase=phase);return seconds
def optimizer(m):return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.)
def batch_tensors(b):return {k:torch.as_tensor(b[k],device='cuda') for k in ['x','y','sigma']}
def initialize(m,arm,bases):
    tick=time.perf_counter()
    if arm!='RANDOM':m.structured(bases[arm])
    torch.cuda.synchronize();return time.perf_counter()-tick

def preflight():
    assert not (RESULTS/'PREFLIGHT.json').exists();RESULTS.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    info=read(ROOT/'results/ab_path_ensemble_screen_v1_20260922/A/SOURCE_AND_MODEL_MANIFEST.json')
    for model in info['models'].values():
        for f in model['files']:assert sha(Path(model['path'])/f['name'])==f['sha256']
    save(RESULTS/'MODEL_AND_ENVIRONMENT.json',dict(models=info['models'],python=sys.version,platform=platform.platform(),gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,packages={k:importlib.metadata.version(k) for k in ['torch','chronos-forecasting','peft','transformers','numpy','pandas','scipy','matplotlib']}))
    (RESULTS/'requirements-lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    for source in SOURCES:
        d=Data(source);save(RESULTS/source/'DATA_AUDIT.json',d.audit);event('data_pass',source=source)
        bases,receipt=build_bases(d);torch.save(bases,CACHE/f'{source}_bases.pt');receipt['basis_file_sha256']=sha(CACHE/f'{source}_bases.pt');save(RESULTS/source/'BASIS_AUDIT.json',receipt)
        np.savez_compressed(CACHE/f'{source}_basis_packets.npz',pairs=d.basis_pairs)
        for seed in SEEDS:np.savez_compressed(CACHE/f'{source}_schedule_{seed}.npz',pairs=d.schedule(seed))
        event('basis_complete',source=source,prep_seconds=receipt['charged_prep_seconds']);gc.collect();torch.cuda.empty_cache()
    # Actual native parity, zero-output initialization, two-update gradients and restore.
    d=Data(SOURCES[0]);bases=torch.load(CACHE/f'{SOURCES[0]}_bases.pt',weights_only=False);b=batch_tensors(d.batch(d.schedule(SEEDS[0])[0]));reference=None;smoke=[]
    initial_reference=None
    for arm in ARMS:
        m=Model();initialize(m,arm,bases);initial=m.state();frozen=m.frozen()
        if initial_reference is None:initial_reference=initial
        else:
            for n,v in initial.items():
                if '.encoder.' in n and 'lora_A' in n:assert torch.allclose(v.norm(dim=1),initial_reference[n].norm(dim=1),atol=1e-6)
                else:assert torch.equal(v,initial_reference[n]),n
        with torch.no_grad():
            q=m(b['x']);native=m.pipeline.predict(b['x'],prediction_length=64).transpose(1,2)
            assert torch.allclose(q.cpu(),native.cpu(),atol=1e-5,rtol=1e-5)
            if reference is None:reference=q.detach().cpu()
            else:assert torch.equal(q.cpu(),reference)
        opt=optimizer(m);norms=[]
        for step in [1,2]:
            opt.zero_grad(set_to_none=True);loss=pinball(m(b['x']),b['y'],b['sigma']);loss.backward()
            g={n:float(p.grad.norm()) for n,p in m.named_parameters() if p.requires_grad};assert all(np.isfinite(v) for v in g.values())
            if step==2:assert any(v>0 for n,v in g.items() if '.encoder.' in n and 'lora_A' in n)
            norm=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.);assert norm>0
            update(opt,'smoke_'+arm,step,'smoke');norms.append(float(norm))
        assert m.frozen()==frozen;trained=m.state()
        with torch.no_grad():after=m(b['x']).cpu()
        m.restore(initial)
        with torch.no_grad():assert torch.equal(m(b['x']).cpu(),reference)
        m.restore(trained)
        with torch.no_grad():assert torch.equal(m(b['x']).cpu(),after)
        smoke.append(dict(arm=arm,native_parity=True,initial_equal=True,restore=True,gradient_norms=norms,frozen_unchanged=True));del m,opt;gc.collect();torch.cuda.empty_cache()
    save(RESULTS/'PREFLIGHT.json',dict(status='PASS',smoke=smoke,smoke_updates=8,main_updates=0,actual_qv_parity=True))
    event('preflight_pass')

def fit(source,arm,seed,bases):
    key=f'{source}_{arm}_{seed}';assert not any(x.get('key')==key for x in spent()),'Fit replay prohibited';sealcheck();d=Data(source)
    tick=time.perf_counter();m=Model(seed);model_load_seconds=time.perf_counter()-tick;apply_seconds=initialize(m,arm,bases)
    prep=read(RESULTS/source/'BASIS_AUDIT.json')['charged_prep_seconds'][arm]+apply_seconds
    init=m.state();frozen=m.frozen();opt=optimizer(m);schedule=d.schedule(seed);folder=CACHE/'fits'/key;folder.mkdir(parents=True,exist_ok=False)
    ledger('fit_start',key=key,phase='main');curves={};cps={0:init};times={0:0.};steps_seconds=0.;trace=[];wall=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    curves[0]=score(predict(m,d,'VAL',predpath(source,seed,arm,'VAL',0)))
    for step,pairs in enumerate(schedule,1):
        torch.cuda.synchronize();t=time.perf_counter();b=batch_tensors(d.batch(pairs));opt.zero_grad(set_to_none=True);loss=pinball(m(b['x']),b['y'],b['sigma']);assert torch.isfinite(loss);loss.backward()
        for p in m.parameters():
            if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all()
        norm=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.);assert norm>0
        # Intent logging I/O is excluded from pure adaptation compute, but included in wall.
        torch.cuda.synchronize();before=time.perf_counter();optimizer_seconds=update(opt,key,step,'main')
        steps_seconds+=before-t+optimizer_seconds;trace.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm)))
        if step in STEPS:
            cps[step]=m.state();times[step]=prep+steps_seconds
            torch.save(dict(model=cps[step],optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),step=step),folder/f'checkpoint_{step}.pt')
            curves[step]=score(predict(m,d,'VAL',predpath(source,seed,arm,'VAL',step)));event('checkpoint',key=key,step=step,val=curves[step],adaptation_seconds=times[step])
    assert frozen==m.frozen();selected=min(STEPS,key=lambda s:(curves[s],s));m.restore(cps[selected]);torch.save(cps,folder/'states.pt')
    pred=predict(m,d,'CAL',predpath(source,seed,arm,'CAL',selected));cal=calibrate(pred)
    result=dict(key=key,source=source,arm=arm,seed=seed,steps=512,selected=selected,curves=curves,times=times,prep_seconds=prep,model_load_seconds=model_load_seconds,fit_wall_seconds=time.perf_counter()-wall,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),trainable_parameters=294912,initial_sha=hash_tensors(init.items()),selected_sha=hash_tensors(cps[selected].items()),frozen_unchanged=True,cal=cal,trace=trace)
    save(RESULTS/'fits'/f'{key}.json',result);ledger('fit_complete',key=key,phase='main');del m,opt;gc.collect();torch.cuda.empty_cache()
    return result

def run():
    assert read(RESULTS/'PREFLIGHT.json')['status']=='PASS';assert not (RESULTS/'SOURCE_SEAL.json').exists()
    save(RESULTS/'SOURCE_SEAL.json',dict(files=source_hashes(),artifacts={str(p.relative_to(RESULTS)):sha(p) for p in RESULTS.rglob('*.json')},main_updates=0))
    fits=[]
    for source in SOURCES:
        bases=torch.load(CACHE/f'{source}_bases.pt',weights_only=False)
        for seed in SEEDS:
            ordering=ARMS if seed==SEEDS[0] else ARMS[::-1]
            for arm in ordering:fits.append(fit(source,arm,seed,bases))
    selection={}
    for source in SOURCES:
        d=Data(source);m=Model()
        for role in ['CAL','VAL']:p=predict(m,d,role,predpath(source,0,'F0',role,0))
        cal=calibrate(load(predpath(source,0,'F0','CAL',0)));scores={a:float(np.mean([x['curves'][128] for x in fits if x['source']==source and x['arm']==a])) for a in ['RANDOM','PCA']}
        selection[source]=dict(baseline=min(scores,key=scores.get),baseline_validation128=scores,f0_cal=cal);del m;gc.collect();torch.cuda.empty_cache()
    save(RESULTS/'SELECTION.json',selection);save(RESULTS/'SELECTION_SEAL.json',dict(files={str(p.relative_to(RESULTS)):sha(p) for p in list((RESULTS/'fits').glob('*.json'))+[RESULTS/'SELECTION.json']},test_started=False));event('all_selection_sealed')
    sealcheck()
    for source in SOURCES:
        d=Data(source)
        for x in [x for x in fits if x['source']==source]:
            m=Model(x['seed']);states=torch.load(CACHE/'fits'/x['key']/'states.pt',weights_only=True)
            for step in sorted(set([128,x['selected']])):
                m.restore(states[step]);predict(m,d,'TEST',predpath(source,x['seed'],x['arm'],'TEST',step))
            del m,states;gc.collect();torch.cuda.empty_cache()
        m=Model();predict(m,d,'TEST',predpath(source,0,'F0','TEST',0));del m;gc.collect();torch.cuda.empty_cache()
    save(RESULTS/'PREDICTIONS_MANIFEST.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in CACHE.rglob('*.npz')},test_scoring_started=False))
    save(RESULTS/'ALL_TEST_SAVED.json',dict(status='COMPLETE'));event('all_test_saved')

if __name__=='__main__':
    setup();CACHE.mkdir(parents=True,exist_ok=True);lock=CACHE/'EXECUTOR.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        if sys.argv[1]=='preflight':preflight()
        elif sys.argv[1]=='main':run()
        else:raise ValueError('Expected preflight or main')
    except Exception as e:
        save(RESULTS/f'ERROR_{int(time.time())}.json',dict(classification='RESOURCE_BLOCK' if 'out of memory' in str(e).lower() else 'IMPLEMENTATION_FAILURE',error=repr(e),traceback=traceback.format_exc()));raise
    finally:lock.unlink()
