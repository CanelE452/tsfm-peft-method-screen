"""Explicit controller for the finite contract; no silent continuation to other topics."""
import argparse,os,subprocess,traceback
from runtime import *

def snapshot_code(paths):return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def prepare():
    assert (OUT/'audit.json').exists() and (OUT/'EVIDENCE.md').exists()
    if (OUT/'std_seal.json').exists():verify_seal();return
    code=[EXP/'runtime.py',EXP/'runner.py',EXP/'audit_evidence.py',OUT/'STD_PROTOCOL.md',OUT/'EXECUTION_CONTRACT.txt',ROOT/'scripts/priority12/common.py',ROOT/'src/tsfm_peft_screen/backbone.py']
    model=read(ROOT/'results/building_coldstart_coverage_v1_20260915/prepare_seal.json')['model_files']
    for p,h in model.items():assert sha(p)==h
    save(OUT/'std_seal.json',dict(at=time.time(),code=snapshot_code(code),episodes_sha256=sha(OUT/'episodes.json'),model=model,max_attempts=120,max_main_updates=19680,max_smoke_updates=24,seeds=[61680,61681],lrs=LRS,policies=POLICIES))
    save(OUT/'gpu_budget.json',dict(started_at=read(OUT/'audit.json')['at'],wait_seconds=0.))
    print('STD_PREPARED',flush=True)
def verify_seal():
    s=read(OUT/'std_seal.json')
    for p,h in s['code'].items():assert sha(ROOT/p)==h,('STD_CODE_CHANGED',p)
    assert sha(OUT/'episodes.json')==s['episodes_sha256']
    return s

def smoke(w):
    if (OUT/'std_smoke.json').exists():assert read(OUT/'std_smoke.json')['status']=='PASS';return
    e=next(e for e in read(OUT/'episodes.json') if e['role']=='DISCOVERY');h=load_history(e['history']);xs,ys=windows(h);m=make(61680);fh=frozen_hash(m);before=cpu_state(m)
    with disabled(m):q0,raw0=predict(m,h[-24:],w)
    q,raw=predict(m,h[-24:],w);assert np.array_equal(q0,q) and np.array_equal(raw0,raw)
    with torch.no_grad():
        a=native(m,xs[0],ys[0]).quantile_preds;b=native(m,xs[0],ys[0]+1234).quantile_preds;assert torch.equal(a,b)
        from chronos import Chronos2Pipeline
        pip=Chronos2Pipeline(m).predict([torch.as_tensor(h[-24:].copy(),dtype=torch.float32)],prediction_length=24,context_length=24,batch_size=1)[0][0].double().cpu().numpy()
        error=float(np.max(abs(pip-raw)));scale=max(float(h.std()),1e-6);normalized=error/scale;assert np.allclose(pip/scale,raw/scale,atol=1e-5,rtol=1e-5)
    opt=torch.optim.AdamW(parameters(m).values(),lr=3e-5,betas=(.9,.999),eps=1e-8,weight_decay=0);sr=read(OUT/'smoke_ledger.json') if (OUT/'smoke_ledger.json').exists() else []
    for k in range(2):
        assert sum(v['updates'] for v in sr)<24;w.boundary();opt.zero_grad(set_to_none=True);loss=native(m,xs[k%len(xs)],ys[k%len(ys)]).loss;loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters(m).values());torch.nn.utils.clip_grad_norm_(parameters(m).values(),1.,error_if_nonfinite=True);opt.step();sr.append(dict(stage='STD',updates=1,loss=float(loss.detach()),at=time.time()));save(OUT/'smoke_ledger.json',sr)
    assert frozen_hash(m)==fh and tensor_hash(cpu_state(m))!=tensor_hash(before)
    path=CACHE/'std_smoke.pt';torch.save(cpu_state(m),path);post=predict(m,h[-24:],w)[1];del opt,m;cleanup();m=make(61680);restore(m,torch.load(path,map_location='cpu',weights_only=True));replay=predict(m,h[-24:],w)[1];assert np.array_equal(post,replay)
    save(OUT/'std_smoke.json',dict(status='PASS',updates=2,step0_identity=True,target_poison_equal=True,pipeline_max_abs=error,pipeline_normalized_abs=normalized,fresh_reload_max_abs=0.,frozen_unchanged=True,first_A_zero_allowed=True));del m;cleanup()

def target(e):
    if e['role']=='LOCKED':assert (OUT/'selection_seal.json').exists(),'LOCKED_TARGET_CLOSED'
    d=e['target'];assert sha(ROOT/d['path'])==d['sha256'];return np.load(ROOT/d['path'])
def score_fit(rec,e):
    path=OUT/'scores.json';rows=read(path) if path.exists() else []
    if any(r['fit']==rec['id'] for r in rows):return
    h=load_history(e['history']);y=target(e);x,ys=windows(h)
    for key,d in rec['checkpoints'].items():
        assert sha(ROOT/d['path'])==d['sha256'];z=np.load(ROOT/d['path']);q=z['q'][0]
        rows.append(dict(fit=rec['id'],episode=e['id'],building=e['building'],role=e['role'],days=e['days'],method=rec['method'],lr=rec['lr'],seed=rec['seed'],step=int(key),train_seconds=d['train_seconds'],prediction_path=d['path'],prediction_sha256=d['sha256'],**metric(q,y,h)))
    if rec['method']=='STD' and not any(r['episode']==e['id'] and r['method']=='F0' for r in rows):
        d=rec['checkpoints']['0'];z=np.load(ROOT/d['path']);af=affine(z['q'][1:],ys)
        references=[('F0',z['q'][0],z['raw'][0]),('AFFINE',af['a']*z['q'][0]+af['b'],af['a']*z['raw'][0]+af['b']),('SEASONAL24',np.tile(h[-24:],(len(z['q'][0]),1)),np.tile(h[-24:],(len(z['q'][0]),1)))]
        for arm,qq,raw in references:
            dest=CACHE/(e['id']+'_'+arm+'.npz');np.savez_compressed(dest,q=qq,raw=raw)
            rows.append(dict(fit='REFERENCE_'+e['id'],episode=e['id'],building=e['building'],role=e['role'],days=e['days'],method=arm,lr=0.,seed=0,step=0,train_seconds=0.,prediction_path=str(dest.relative_to(ROOT)),prediction_sha256=sha(dest),affine=af if arm=='AFFINE' else None,**metric(qq,y,h)))
    save(path,rows);csvwrite(OUT/'scores.csv',rows)

def select(method):
    rows=read(OUT/'scores.json');choices=[]
    for lr in LRS:
        for policy in POLICIES:
            v=[r for r in rows if r['role']=='DISCOVERY' and r['method']==method and r['lr']==lr and r['step']==step(policy,r['days']-1)];assert len(v)==8
            choices.append(dict(lr=lr,policy=policy,primary=float(np.mean([r['primary'] for r in v])),mean_updates=float(np.mean([r['step'] for r in v]))))
    best=min(choices,key=lambda r:(r['primary'],r['mean_updates'],r['lr']));save(OUT/f'{method}_selection.json',dict(selected=best,choices=choices,scope='DISCOVERY4 only; global LR and budget kind across H3/H14; seed61680'))

def std():
    verify_seal();setup(61680);w=Watch(OUT,'std',wall_cap=14400)
    try:
        w.boundary(startup=True);smoke(w)
        for e in read(OUT/'episodes.json'):
            if e['role']!='DISCOVERY':continue
            h=load_history(e['history'])
            for lr in LRS:
                job=dict(id=f"STD_{e['id']}_lr{lr}_seed61680",method='STD',role='DISCOVERY',building=e['building'],days=e['days'],origin=e['origin'],seed=61680,lr=lr)
                rec=fit(job,h,w);score_fit(rec,e)
        select('STD');save(OUT/'status.json',dict(status='STD_COMPLETE_AWAITING_TOPIC_SEAL',fits=len(ledger()),updates=sum(r['updates'] for r in ledger()),locked_targets_opened=False));print('STD_COMPLETE',read(OUT/'STD_selection.json')['selected'],flush=True)
    except BaseException as exc:
        save(OUT/'status.json',dict(status='PARTIAL' if isinstance(exc,ResourceError) else 'INCONCLUSIVE_IMPLEMENTATION',error=repr(exc),traceback=traceback.format_exc(),fits=len(ledger()),updates=sum(r['updates'] for r in ledger())));raise
    finally:w.close()

def status():
    print(json.dumps(dict(state=read(OUT/'status.json') if (OUT/'status.json').exists() else 'NOT_STARTED',fits=[{k:r[k] for k in ['id','status','updates']} for r in ledger()]),indent=2))
def all_stages():
    prepare()
    if not (OUT/'STD_selection.json').exists():std()
    if not (OUT/'topic_seal.json').exists():print('STD complete. Topic mechanism must be sealed before candidate comparison; LOCKED remains closed.');return
    from compare import run_comparison
    run_comparison()
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Finite 120-fit topic contract; explicit stages and no duplicate completed fits.');parser.add_argument('command',choices=['prepare','std','all','status','verify']);args=parser.parse_args()
    {'prepare':prepare,'std':std,'all':all_stages,'status':status,'verify':verify_seal}[args.command]()
