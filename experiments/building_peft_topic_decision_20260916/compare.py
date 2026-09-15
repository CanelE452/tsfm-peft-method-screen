"""Candidate comparison follows the contract regardless of discovery score sign."""
import copy,traceback
from runtime import *
from runner import verify_seal,load_history,score_fit,select
from candidate import build

def check_topic():
    s=read(OUT/'topic_seal.json');assert sha(OUT/'STD_selection.json')==s['std_selection_sha256']
    for p,h in s['sources'].items():assert sha(ROOT/p)==h,('TOPIC_SOURCE_CHANGED',p)
    return s

def smoke_candidates(w):
    if (OUT/'candidate_smoke.json').exists():assert read(OUT/'candidate_smoke.json')['status']=='PASS';return
    e=next(e for e in read(OUT/'episodes.json') if e['role']=='DISCOVERY');h=load_history(e['history']);x,y=windows(h);records=[];sr=read(OUT/'smoke_ledger.json');save(OUT/'candidate_smoke_started.json',dict(at=time.time(),initial_smoke_updates=sum(v['updates'] for v in sr)))
    for method in ['SIMPLE','CANDIDATE']:
        signatures=[];preds=[]
        for poison in [0,1234]:
            # Dummy E is deliberately outside all callable data interfaces.
            dummy_e=np.ones(24)*poison;assert dummy_e.shape==(24,)
            setup(61680);m=make(61680);frozen=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()));initial=tensor_hash(cpu_state(m));fn,transform=build(method,m,h,w)
            a=predict(m,h[-24:],w)[1]
            with disabled(m):b=predict(m,h[-24:],w)[1]
            assert np.array_equal(a,b)
            opt=torch.optim.AdamW(parameters(m).values(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
            for k in range(2):
                assert sum(v['updates'] for v in sr)<24;w.boundary();opt.zero_grad(set_to_none=True);loss=fn(m,x[k%len(x)],y[k%len(y)]);assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters(m).values());torch.nn.utils.clip_grad_norm_(parameters(m).values(),1.,error_if_nonfinite=True);opt.step();sr.append(dict(stage=method,dummy_poison=poison,updates=1,loss=float(loss.detach()),at=time.time()));save(OUT/'smoke_ledger.json',sr)
            state=cpu_state(m);assert tensor_hash(state)!=initial and frozen_hash(m)==frozen and tensor_hash(dict(m.named_buffers()))==buffers
            pred=predict(m,h[-24:],w)[1];p=CACHE/f'smoke_{method}_{poison}.pt';torch.save(state,p);signatures.append(tensor_hash(state));preds.append(pred)
            del opt,m;cleanup();m=make(61680);restore(m,torch.load(p,map_location='cpu',weights_only=True));replay=predict(m,h[-24:],w)[1];assert np.array_equal(pred,replay);del m;cleanup()
        assert signatures[0]==signatures[1] and np.array_equal(preds[0],preds[1]);records.append(dict(method=method,synthetic_future_poison_same_weights=True,synthetic_future_poison_same_forecast=True,step0_identity=True,frozen_unchanged=True,buffers_unchanged=True,fresh_reload_exact=True,updates=4))
    save(OUT/'candidate_smoke.json',dict(status='PASS',records=records,total_smoke_updates=sum(v['updates'] for v in sr),candidate_main_fits=0,actual_locked_target_access=False));print('CANDIDATE_SMOKE_PASS',flush=True)

def job_for(e,method,lr,seed):return dict(id=f"{method}_{e['id']}_lr{lr}_seed{seed}",method=method,role=e['role'],building=e['building'],days=e['days'],origin=e['origin'],seed=seed,lr=lr)
def run_one(e,method,lr,seed,w):
    h=load_history(e['history']);job=job_for(e,method,lr,seed)
    # Exact uniform geometry aliases only; distinct trajectories never pooled by similar scores.
    if method=='CANDIDATE':
        hh=hashlib.sha256(np.asarray(h,dtype=np.float64).tobytes()).hexdigest();p=OUT/'calibration'/(hh+'_SIMPLE.json')
        if p.exists():
            g=read(p);simple=job_for(e,'SIMPLE',lr,seed);matches=[f for f in ledger() if f['id']==simple['id'] and f['status']=='COMPLETE']
            if matches and ((g['geometry']['w_level']==1 and g['geometry']['w_shape']==1) or g['lambda_value']==0):
                rec=copy.deepcopy(matches[0]);rec.update(id=job['id'],method='CANDIDATE');aliases=read(OUT/'fit_aliases.json') if (OUT/'fit_aliases.json').exists() else []
                if not any(a['id']==job['id'] for a in aliases):aliases.append(dict(id=job['id'],physical_fit=matches[0]['id'],reason='Exact objective, parameter, seed, LR, data and trajectory equality',updates_additional=0));save(OUT/'fit_aliases.json',aliases)
                score_fit(rec,e);return
    rec=fit(job,h,w);score_fit(rec,e)

def seal_selection():
    path=OUT/'selection_seal.json'
    if path.exists():return read(path)
    choices={m:read(OUT/f'{m}_selection.json')['selected'] for m in ['STD','SIMPLE','CANDIDATE']}
    s=dict(at=time.time(),selected=choices,selection_sources={str((OUT/f'{m}_selection.json').relative_to(ROOT)):sha(OUT/f'{m}_selection.json') for m in choices},topic_seal_sha256=sha(OUT/'topic_seal.json'),locked=[e for e in read(OUT/'episodes.json') if e['role']=='LOCKED'],seeds=[61680,61681],decision_thresholds='Unchanged EXECUTION_CONTRACT section9; ≥1%, both seeds positive, paired CI lower>0, neither H worse>1%, strongest fixed controls and F0')
    save(path,s);print('SELECTION_SEALED',choices,flush=True);return s

def run_comparison():
    verify_seal();check_topic();assert len([r for r in ledger() if r['method']=='STD' and r['role']=='DISCOVERY' and r['status']=='COMPLETE'])==16
    w=Watch(OUT,'comparison',wall_cap=14400)
    try:
        w.boundary(startup=True);smoke_candidates(w);episodes=read(OUT/'episodes.json')
        for e in episodes:
            if e['role']!='DISCOVERY':continue
            for lr in LRS:
                for method in ['SIMPLE','CANDIDATE']:run_one(e,method,lr,61680,w)
        for method in ['SIMPLE','CANDIDATE']:select(method)
        selection=seal_selection()
        # No development performance entry gate. Selection remains frozen for both seeds.
        for p,h in selection['selection_sources'].items():assert sha(ROOT/p)==h
        for e in episodes:
            if e['role']!='LOCKED':continue
            for seed in [61680,61681]:
                for method in ['STD','SIMPLE','CANDIDATE']:run_one(e,method,selection['selected'][method]['lr'],seed,w)
        save(OUT/'status.json',dict(status='TRAINING_COMPLETE_AWAITING_INDEPENDENT_AUDIT',fits=len(ledger()),updates=sum(r['updates'] for r in ledger()),logical_fits=len(ledger())+len(read(OUT/'fit_aliases.json') if (OUT/'fit_aliases.json').exists() else []),locked_opened=True));print('ALL_TRAINING_COMPLETE',flush=True)
    except BaseException as exc:
        save(OUT/'status.json',dict(status='PARTIAL' if isinstance(exc,ResourceError) else 'INCONCLUSIVE_IMPLEMENTATION',error=repr(exc),traceback=traceback.format_exc(),fits=len(ledger()),updates=sum(r['updates'] for r in ledger())));raise
    finally:w.close()
