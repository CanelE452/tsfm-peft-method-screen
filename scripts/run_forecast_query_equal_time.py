"""Execute the a33f245 equal-time plan. No old results or plan files mutated."""
import copy
import fcntl
import gc
import json
import signal
import subprocess
import time
import traceback
from contextlib import nullcontext

import numpy as np
import torch
from huggingface_hub import hf_hub_download

import run_forecast_query_checkpoint_diagnostic as shared
from tsfm_peft_screen.backbone import native_loss, REVISION
from tsfm_peft_screen.forecast_query.equal_time import EqualTimeModel, TrainingClock, choose_storage, preserve_rng
from tsfm_peft_screen.forecast_query.gpu import GPUWatch
from tsfm_peft_screen.metrics import score
from tsfm_peft_screen.reproducibility import ROOT, sha, digest, write_json, source_hashes, seed_all, guard

CFG = json.loads((ROOT/'configs/forecast_query_equal_time_plan.json').read_text())
assert CFG['numerical_equivalence_tolerances'] == shared.CFG['tolerances']
OUT = ROOT/'results/forecast_query_equal_time'
CACHE = ROOT/'.cache/forecast_query_equal_time'


def origins(key):
    d=CFG[key+'_origins']
    return list(range(d['start'],d['stop_inclusive']+1,d['stride']))


def batch(values, oo):
    assert min(oo)>=CFG['context'] and max(oo)+48<=len(values)
    x=np.concatenate([values[o-4096:o,:4].T for o in oo]).astype(np.float32)
    y=np.concatenate([values[o:o+48,:4].T for o in oo]).astype(np.float32)
    return torch.tensor(x,device='cuda'),torch.tensor(y,device='cuda'),torch.arange(len(oo),device='cuda').repeat_interleave(4)


def params(m):
    return shared.params(m)


def state(m):
    return shared.cpu_tree(params(m))


def restore_parameters(m, saved):
    assert params(m).keys()==saved.keys()
    with torch.no_grad():
        for n,p in params(m).items():
            p.copy_(saved[n])


def frozen_hash(m):
    return shared.tree_hash({n:p for n,p in m.named_parameters() if not p.requires_grad})


def new_model(arm, seed, lr):
    seed_all(seed)
    m=EqualTimeModel(arm,seed)
    opt=torch.optim.AdamW(list(params(m).values()),lr=lr,weight_decay=0.)
    return m,opt


def step(m,opt,values,oo,precision='bf16',export_state=None):
    ps=params(m)
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    tick=time.perf_counter()
    x,y,g=batch(values,oo)
    with torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext():
        z,p,l,s=m(x,g)
        loss=native_loss(z,y,l,s)
    assert torch.isfinite(loss) and torch.isfinite(p).all()
    loss.backward()
    assert all(v.grad is not None for v in ps.values())
    norm=torch.nn.utils.clip_grad_norm_(list(ps.values()),1.,error_if_nonfinite=True)
    opt.step()
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-tick
    resource=dict(seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  loss=float(loss.detach()),gradient_norm=float(norm))
    artifact=None
    if export_state is not None:
        now=state(m)
        adam=torch.cat([v.detach().cpu().reshape(-1) for _,d in sorted(opt.state_dict()['state'].items())
                        for _,v in sorted(d.items()) if isinstance(v,torch.Tensor)])
        artifact=dict(z=z.detach().cpu(),raw=p.detach().cpu(),loss=loss.detach().cpu().reshape(1),
                      gradient=shared.flatten({n:v.grad.detach().cpu() for n,v in ps.items()}),
                      update=shared.flatten({n:now[n]-export_state['parameters'][n] for n in now}),
                      adam=adam,gradient_norm=float(norm),rng_after_sha256=shared.tree_hash(shared.rng()))
        assert artifact['update'].abs().sum()>0
    return resource,artifact


def stage_data():
    receipts={}
    for name in CFG['datasets']:
        folder=ROOT/'data/processed'/name
        meta=json.loads((folder/'manifest.json').read_text())
        assert all(sha(folder/f)==h for f,h in meta['files'].items())
        with np.load(folder/'fit.npz',allow_pickle=False) as z:
            prefix=z['values'][:,:4]
        with np.load(folder/'evaluation.npz',allow_pickle=False) as z:
            values=np.concatenate([prefix,z['tail'][:,:4]])
        # Mechanical staging only. No E statistics, losses, model calls or selection.
        lo,hi=CFG['train_scale_interval_half_open']
        scale=np.maximum(np.nanstd(values[lo:hi],axis=0,dtype=np.float64),1e-6)
        dev=CACHE/(name+'_development.npz')
        held=CACHE/(name+'_heldout.npz')
        np.savez_compressed(dev,values=values[:max(origins('validation'))+48],scale=scale)
        np.savez_compressed(held,values=values[:max(origins('evaluation'))+48],scale=scale)
        receipts[name]=dict(source_hashes=meta['files'],development_sha256=sha(dev),
                            heldout_sha256=sha(held),scale=scale.tolist(),
                            staging_scope='Raw arrays mechanically staged; E statistics/scoring not computed.')
    return receipts


def open_development(name):
    with np.load(CACHE/(name+'_development.npz'),allow_pickle=False) as z:
        values,scale=z['values'],z['scale']
    assert len(values)==23760
    return values,scale


def require_selection_seal():
    seal=json.loads((OUT/'selection_seal.json').read_text())
    saved=seal.pop('seal_sha256')
    assert digest(seal)==saved and len(seal['selections'])==16
    expected={(d,s,a) for d in CFG['datasets'] for s in CFG['seeds'] for a in CFG['arms']}
    assert {(r['dataset'],r['seed'],r['arm']) for r in seal['selections']}==expected
    assert seal['contract_sha256']==sha(OUT/'contract.json')
    for r in seal['selections']:
        assert sha(CACHE/r['checkpoint_file'])==r['checkpoint_sha256']
    return seal


def evaluate(m,values,scale,oo,tag,watch,f0=False):
    pp,yy=[],[]
    with preserve_rng(),torch.no_grad():
        for j in range(0,len(oo),2):
            watch.check('evaluation_'+tag)
            x,y,g=batch(values,oo[j:j+2])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z,p,l,s=m.f0(x,g) if f0 else m(x,g)
            pp.append(p.cpu().numpy().reshape(-1,4,21,48))
            yy.append(y.cpu().numpy().reshape(-1,4,48))
    pred,target=np.concatenate(pp),np.concatenate(yy)
    path=CACHE/(tag+'.npz')
    np.savez_compressed(path,prediction=pred,target=target,scale=scale)
    return dict(metrics=score(pred,target,scale),prediction_file=path.name,prediction_sha256=sha(path))


def main():
    assert not OUT.exists() and not CACHE.exists(),'Immutable experiment already exists'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'Commit before execution'
    lock=open(ROOT/'.cache/gpu.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def signal_stop(signum,frame):
        raise TimeoutError(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,signal_stop)
    OUT.mkdir()
    CACHE.mkdir()
    counts=dict(preflight_warmup_updates=0,preflight_bf16_updates=0,preflight_fp32_updates=0,
                fit_attempts=0,completed_fits=0,training_optimizer_updates=0,evaluation_open_events=0)
    start=time.monotonic()
    watch=GPUWatch(OUT/'gpu_monitor.json')
    fits,trajectories,preflight,parity,compatibility,storage,evaluation=[],[],[],[],[],[],[]
    def status(label,**kw):
        write_json(OUT/'status.json',dict(status=label,counts=counts,elapsed_wall_seconds=time.monotonic()-start,**kw))
    def boundary(label,force=False):
        if time.monotonic()-start>CFG['gpu']['experiment_timeout_seconds']:
            raise TimeoutError('Experiment timeout')
        watch.check(label,force=force)
        guard(start)
    historical={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'results').rglob('*')
                if p.is_file() and OUT not in p.parents}
    try:
        contract=dict(execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                      plan_commit='a33f2456fe573df513a3e062f999ff0ef4d93cc6',
                      config=CFG,config_sha256=sha(ROOT/'configs/forecast_query_equal_time_plan.json'),
                      source_hashes=source_hashes(),runner_sha256=sha(__file__),
                      shared_helper_sha256=sha(ROOT/'scripts/run_forecast_query_checkpoint_diagnostic.py'),
                      historical_result_hashes=historical,model_revision=REVISION,torch_version=torch.__version__)
        expected=json.loads((ROOT/'results/screening_summary/common_integrity.json').read_text())['model_files']
        actual={f:sha(hf_hub_download('amazon/chronos-2',f,revision=REVISION,local_files_only=True)) for f in expected}
        assert expected==actual
        contract['model_files']=actual
        contract['data']=stage_data()
        write_json(OUT/'contract.json',contract)
        status('WAITING_FOR_GPU')
        idle=None
        while True:
            r=watch.read('startup_wait')
            good=not r['external_pids'] and r['free_mib']>=4096 and r['utilization_percent']<90
            idle=(idle or time.monotonic()) if good else None
            if idle and time.monotonic()-idle>=30:
                break
            if time.monotonic()-start>CFG['gpu']['startup_timeout_seconds']:
                raise TimeoutError('Startup wait timeout')
            time.sleep(5)

        status('PREFLIGHT')
        for arm in CFG['arms']:
            boundary('preflight_load',True)
            m,opt=new_model(arm,30000,min(CFG['learning_rates'][arm]))
            frozen=frozen_hash(m)
            values,_=open_development('ettm2')
            for _ in range(2):
                boundary('state_warmup')
                step(m,opt,values,CFG['storage_preflight']['origins'])
                counts['preflight_warmup_updates']+=1
            warmed=shared.snapshot(m,opt)
            warm_hash=shared.tree_hash(warmed)
            torch.save(warmed,CACHE/(arm+'_warm_state.pt'))
            for cp in (False,True):
                shared.restore(m,opt,warmed)
                m.checkpoint_enabled=cp
                boundary('kernel_warmup')
                step(m,opt,values,CFG['storage_preflight']['origins'])
                counts['preflight_warmup_updates']+=1
            for name in CFG['datasets']:
                values,_=open_development(name)
                for precision in ('fp32','bf16'):
                    shared.restore(m,opt,warmed)
                    m.checkpoint_enabled=False
                    data=batch(values,CFG['storage_preflight']['origins'])
                    differences=shared.old_forward_parity(m,data,precision)
                    compatibility.append(dict(arm=arm,dataset=name,precision=precision,differences=differences))
                    del data
                    for rep in range(1 if precision=='fp32' else 3):
                        pair={}
                        for cp in ([False,True] if rep%2==0 else [True,False]):
                            shared.restore(m,opt,warmed)
                            m.checkpoint_enabled=cp
                            assert shared.tree_hash(shared.snapshot(m,opt))==warm_hash
                            boundary('preflight_measurement',True)
                            r,a=step(m,opt,values,CFG['storage_preflight']['origins'],precision,warmed)
                            key='preflight_fp32_updates' if precision=='fp32' else 'preflight_bf16_updates'
                            counts[key]+=1
                            tag=f'pre_{name}_{arm}_{precision}_{rep}_{int(cp)}'
                            path=CACHE/(tag+'.pt')
                            torch.save(a,path)
                            preflight.append(dict(tag=tag,dataset=name,arm=arm,precision=precision,repeat=rep,
                                                  checkpoint=cp,initial_state_sha256=warm_hash,cache_sha256=sha(path),**r))
                            pair[cp]=a
                            write_json(OUT/'preflight_measurements.json',preflight)
                            status('PREFLIGHT')
                        eq=dict(arm=arm,dataset=name,precision=precision,repeat=rep,
                                **shared.compare(pair[False],pair[True],precision))
                        parity.append(eq)
                        write_json(OUT/'preflight_parity.json',parity)
                        assert eq['passed'],('CP parity failed',eq)
                        del pair,a
                rows=[r for r in preflight if r['dataset']==name and r['arm']==arm and r['precision']=='bf16']
                selected=choose_storage(rows,CFG['storage_preflight']['max_peak_allocated_bytes'])
                storage.append(dict(dataset=name,arm=arm,**selected))
                print('STORAGE',name,arm,selected,flush=True)
            assert frozen_hash(m)==frozen
            del m,opt,warmed,values
            gc.collect()
            torch.cuda.empty_cache()
        assert (counts['preflight_warmup_updates'],counts['preflight_bf16_updates'],counts['preflight_fp32_updates'])==(16,48,16)
        write_json(OUT/'historical_forward_parity.json',compatibility)
        storage_seal=dict(choices=storage,measurements_sha256=sha(OUT/'preflight_measurements.json'))
        storage_seal['seal_sha256']=digest(storage_seal)
        write_json(OUT/'storage_seal.json',storage_seal)
        manifest=json.loads((ROOT/'research/forecast_query_equal_time_plan/fit_manifest.json').read_text())
        assert len(manifest)==CFG['fit_attempt_cap']
        write_json(OUT/'fit_manifest.json',manifest)
        schedules={str(seed):np.random.default_rng(seed).choice(origins('train'),size=(4096,2),replace=True).tolist()
                   for seed in CFG['seeds']}
        write_json(OUT/'train_schedules.json',schedules)

        for recipe in manifest:
            assert counts['fit_attempts']<32
            counts['fit_attempts']+=1
            name,arm,seed,lr=[recipe[k] for k in ('dataset','arm','seed','lr')]
            fid=f"{recipe['attempt_order']:02}_{name}_{seed}_{arm}"
            status('TRAINING',current_fit=fid)
            boundary('fit_start',True)
            fitstart=time.monotonic()
            values,scale=open_development(name)
            m,opt=new_model(arm,seed,lr)
            m.checkpoint_enabled=next(r['checkpoint'] for r in storage if r['dataset']==name and r['arm']==arm)
            frozen=frozen_hash(m)
            clock=TrainingClock(CFG['checkpoint_training_seconds'][1:],CFG['maximum_boundary_overshoot_seconds'],CFG['maximum_updates_per_fit'])
            resources=[]
            fitrows=[]
            validation_seconds=0.
            checkpoint_seconds=0.
            best=None
            def checkpoint_model(nominal):
                nonlocal validation_seconds,checkpoint_seconds,best
                tick=time.monotonic()
                out=evaluate(m,values,scale,origins('validation'),f'V_{fid}_{int(nominal)}',watch)
                validation_seconds+=time.monotonic()-tick
                tick=time.monotonic()
                path=CACHE/(fid+f'_{int(nominal)}s.pt')
                torch.save(state(m),path)
                checkpoint_hash=sha(path)
                checkpoint_seconds+=time.monotonic()-tick
                row=dict(fit=fid,dataset=name,arm=arm,seed=seed,lr=lr,nominal_seconds=nominal,
                         active_seconds=clock.elapsed,updates=clock.updates,
                         checkpoint_file=path.name,checkpoint_sha256=checkpoint_hash,**out)
                trajectories.append(row)
                fitrows.append(row)
                key=(out['metrics']['scaled_2pinball'],nominal,lr)
                if best is None or key<(best['metrics']['scaled_2pinball'],best['nominal_seconds'],best['lr']):
                    best=row
                write_json(OUT/'trajectories.json',trajectories)
                print('FIT',fid,'budget',nominal,'actual',round(clock.elapsed,3),'steps',clock.updates,
                      'V',round(out['metrics']['scaled_2pinball'],7),flush=True)
            checkpoint_model(0)
            while not clock.complete:
                boundary(fid)
                r,_=step(m,opt,values,schedules[str(seed)][clock.updates])
                resources.append(r)
                counts['training_optimizer_updates']+=1
                b=clock.add(r['seconds'])
                if b is not None:
                    checkpoint_model(b)
                    status('TRAINING',current_fit=fid)
            assert all(torch.isfinite(p).all() for p in params(m).values())
            assert frozen_hash(m)==frozen
            record=dict(fit=fid,**recipe,checkpoint=m.checkpoint_enabled,updates=clock.updates,
                        active_seconds=clock.elapsed,total_fit_wall_seconds=time.monotonic()-fitstart,
                        validation_seconds=validation_seconds,checkpoint_seconds=checkpoint_seconds,
                        peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in resources),
                        peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in resources),
                        step_resources=resources,best=best)
            fits.append(record)
            counts['completed_fits']+=1
            write_json(OUT/'fits.json',fits)
            print('FIT COMPLETE',len(fits),'/32',fid,flush=True)
            del m,opt,values,resources
            gc.collect()
            torch.cuda.empty_cache()

        assert len(fits)==32
        selections=[]
        for name in CFG['datasets']:
            for seed in CFG['seeds']:
                for arm in CFG['arms']:
                    candidates=[r['best'] for r in fits if r['dataset']==name and r['seed']==seed and r['arm']==arm]
                    selections.append(min(candidates,key=lambda r:(r['metrics']['scaled_2pinball'],r['nominal_seconds'],r['lr'])))
        seal=dict(selections=selections,contract_sha256=sha(OUT/'contract.json'),
                  storage_seal_sha256=sha(OUT/'storage_seal.json'),trajectories_sha256=sha(OUT/'trajectories.json'))
        seal['seal_sha256']=digest(seal)
        write_json(OUT/'selection_seal.json',seal)
        status('SEALED_BEFORE_E')
        replay=[]
        for choice in selections:
            name,arm,seed,lr=[choice[k] for k in ('dataset','arm','seed','lr')]
            boundary('selected_checkpoint_replay',True)
            m,opt=new_model(arm,seed,lr)
            m.checkpoint_enabled=next(r['checkpoint'] for r in storage if r['dataset']==name and r['arm']==arm)
            restore_parameters(m,torch.load(CACHE/choice['checkpoint_file'],weights_only=True))
            values,scale=open_development(name)
            r=evaluate(m,values,scale,origins('validation'),f"REPLAY_{choice['fit']}",watch)
            with np.load(CACHE/r['prediction_file']) as a,np.load(CACHE/choice['prediction_file']) as b:
                error=float(np.max(np.abs(a['prediction']-b['prediction'])))
            assert error==0
            replay.append(dict(fit=choice['fit'],prediction_max_abs_error=error,**r))
            del m,opt,values
            gc.collect()
            torch.cuda.empty_cache()
        write_json(OUT/'checkpoint_replay.json',replay)
        for name in CFG['datasets']:
            require_selection_seal()
            path=CACHE/(name+'_heldout.npz')
            assert sha(path)==contract['data'][name]['heldout_sha256']
            counts['evaluation_open_events']+=1
            status('EVALUATING',dataset=name)
            with np.load(path,allow_pickle=False) as z:
                values,scale=z['values'],z['scale']
            for seed in CFG['seeds']:
                for arm in CFG['arms']:
                    boundary('E_model',True)
                    choice=next(r for r in selections if r['dataset']==name and r['seed']==seed and r['arm']==arm)
                    m,opt=new_model(arm,seed,choice['lr'])
                    m.checkpoint_enabled=next(r['checkpoint'] for r in storage if r['dataset']==name and r['arm']==arm)
                    if arm=='standard' and seed==30000:
                        r=evaluate(m,values,scale,origins('evaluation'),f'E_{name}_F0',watch,f0=True)
                        evaluation.append(dict(dataset=name,arm='F0',seed=None,**r))
                    if arm=='query':
                        r=evaluate(m,values,scale,origins('evaluation'),f'E_{name}_{seed}_query_initial',watch)
                        evaluation.append(dict(dataset=name,arm='query_initial',seed=seed,**r))
                    restore_parameters(m,torch.load(CACHE/choice['checkpoint_file'],weights_only=True))
                    r=evaluate(m,values,scale,origins('evaluation'),f'E_{name}_{seed}_{arm}',watch)
                    evaluation.append(dict(dataset=name,arm=arm,seed=seed,fit=choice['fit'],
                                           nominal_seconds=choice['nominal_seconds'],**r))
                    write_json(OUT/'evaluation.json',evaluation)
                    del m,opt
                    gc.collect()
                    torch.cuda.empty_cache()
            del values
        assert source_hashes()==contract['source_hashes']
        assert all(sha(ROOT/p)==h for p,h in historical.items())
        boundary('complete',True)
        status('COMPLETE',historical_results_unchanged=True,expanded=False,
               verdict='AWAITING_INDEPENDENT_FINALIZATION')
    except Exception as e:
        status('INCONCLUSIVE_EXECUTION',error=str(e),traceback=traceback.format_exc(),
               original_forecast_query_verdict='FAIL')
        raise


if __name__=='__main__':
    main()
