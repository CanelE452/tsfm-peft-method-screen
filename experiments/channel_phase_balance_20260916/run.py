"""Finite four-fit phase sampling control; no candidate generation or retries."""
import argparse, ast, copy, json, math, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from model import ROOT, make, original_make
from common import read, save, sha, cpu_state, parameters, tensor_hash, frozen_hash, restore, cleanup, preserve_rng, Watch
from rank2_data import load, batch, metrics, independent
from run_rank2 import configure, step

OUT = ROOT/'results/channel_phase_balance_20260916'
EXP = ROOT/'experiments/channel_phase_balance_20260916'
CACHE = ROOT/'.cache/channel_phase_balance_20260916'
OLD = ROOT/'results/channel_sharing_screen_v1_20260915'
DATA = ROOT/'.cache/channel_sharing_screen_v1_20260915'


def check_hashes(mapping):
    for p, expected in mapping.items():
        assert sha(ROOT/p) == expected, ('HASH_CHANGED', p)


def prepare():
    assert not (OUT/'seal.json').exists() and not CACHE.exists(), 'No duplicate prepare/run'
    configure(); CACHE.mkdir()
    old=read(OLD/'contract.json');check_hashes(old['source_hashes'])
    for d in old['data'].values():check_hashes(d['staged'])
    for p,h in old['model_files'].items():assert sha(p)==h
    hist={str(p.relative_to(ROOT)):sha(p) for directory in ['results','research'] for p in (ROOT/directory).rglob('*') if p.is_file() and OUT not in p.parents}
    save(OUT/'historical_hashes.json',hist)
    data=copy.deepcopy(old['data']);audit=[];schedules={}
    for d,dc in data.items():
        previous=np.asarray(dc['origins']['train'],dtype=np.int64)
        assert np.all(previous%24==0)
        balanced=previous+np.arange(len(previous))%24
        balanced=np.where(balanced+96>dc['t1'],balanced-24,balanced)
        assert len(np.unique(balanced))==len(previous) and np.all(np.diff(balanced)>0)
        assert balanced.min()>=96 and balanced.max()+96<=dc['t1'] and np.max(abs(balanced-previous))<=23
        counts=np.bincount(balanced%24,minlength=24);assert counts.min()>0 and counts.max()-counts.min()<=1
        dc['origins']['train']=balanced.tolist()
        for seed in [41000,41001]:
            rng=np.random.default_rng(seed);schedules[f'{d}_{seed}']=[rng.permutation(balanced).tolist() for _ in range(20)]
        values,std=load(DATA,d,'development');poison=values.copy();poison[dc['t1']:]=123456.
        x,y=batch(values,balanced,'cpu');xx,yy=batch(poison,balanced,'cpu')
        assert torch.equal(x,xx) and torch.equal(y,yy)
        audit.append(dict(dataset=d,old_origin_phase_counts=np.bincount(previous%24,minlength=24).tolist(),new_origin_phase_counts=counts.tolist(),old_origins=previous.tolist(),new_origins=balanced.tolist(),max_shift_hours=int(np.max(abs(balanced-previous))),future_poison_unchanged=True))
    save(OUT/'sampler_audit.json',audit);save(OUT/'schedules.json',schedules)
    references=[]
    for r in read(OLD/'evaluation.json'):
        if r['arm'] in ['LH','SEASONAL_NAIVE']:
            assert sha(ROOT/r['prediction_path'])==r['prediction_sha256']
            with np.load(ROOT/r['prediction_path']) as z:actual=independent(z['prediction'],z['target'])
            assert math.isclose(actual,r['metrics']['mse'],rel_tol=1e-12,abs_tol=1e-12)
            references.append(r)
    save(OUT/'reused_controls.json',references)
    oldfits=[f for f in read(OLD/'fits.json') if f['arm']=='LH'];assert len(oldfits)==4
    oldtrajectory=read(OLD/'trajectories.json');checks=[];initial_rows=[]
    for f in oldfits:
        init=next(r for r in oldtrajectory if r['fit']==f['fit'] and r['epoch']==0)
        assert sha(ROOT/init['checkpoint_path'])==init['checkpoint_sha256']
        assert sha(ROOT/init['prediction_path'])==init['prediction_sha256']
        m=make(data[f['dataset']]['channel_ids'],f['seed']).eval()
        state=torch.load(ROOT/init['checkpoint_path'],weights_only=True,map_location='cpu')
        assert tensor_hash(cpu_state(m))==tensor_hash(state)
        assert sum(p.numel() for p in parameters(m).values())==761952
        checks.append(dict(dataset=f['dataset'],seed=f['seed'],initial_state_equal_to_old_epoch0=True,initial_state_sha256=tensor_hash(state),trainable=761952))
        initial_rows.append(init);del m,state;cleanup()
    save(OUT/'cpu_checks.json',checks);save(OUT/'old_initial_states.json',initial_rows)
    files=list(EXP.glob('*.py'))+[OUT/'PROTOCOL.md',OUT/'schedules.json',OUT/'sampler_audit.json',OUT/'old_initial_states.json',OLD/'contract.json',OLD/'fits.json',OLD/'trajectories.json',OLD/'evaluation.json',OUT/'reused_controls.json']
    sources={str(p.relative_to(ROOT)):sha(p) for p in files};sources.update(old['source_hashes'])
    save(OUT/'seal.json',dict(created_at=time.time(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hashes=sources,data=data,model_files=old['model_files'],historical_manifest_sha256=sha(OUT/'historical_hashes.json'),cells=[dict(dataset=d,seed=s) for d in ['electricity','traffic'] for s in [41000,41001]],old_selected_epochs={f"{f['dataset']}_{f['seed']}":f['best']['epoch'] for f in oldfits},fit_cap=4,update_cap=5080,smoke_cap=4,wall_cap=3600,exposure='DISCOVERY_REUSED_E',novelty='KNOWN_SAMPLING_CONTROL'))
    save(OUT/'status.json',dict(status='PREPARED',fit_attempts=0,fits_completed=0,training_updates=0,smoke_updates=0))
    print('PREPARED; 4 phase-balanced LH fits maximum',flush=True)


def evaluate(m, values, std, origins, tag, w):
    path=CACHE/(tag+'.npz'); assert not path.exists()
    pp=[]; yy=[]
    with preserve_rng(),torch.no_grad():
        m.eval()
        for i in range(0,len(origins),8):
            w.boundary(); x,y=batch(values,origins[i:i+8])
            with torch.autocast('cuda',dtype=torch.bfloat16): p=m(x).forecast
            assert torch.isfinite(p).all()
            pp.append(p.float().cpu().numpy()); yy.append(y.cpu().numpy())
    p=np.concatenate(pp); y=np.concatenate(yy); met=metrics(p,y,std)
    scalar=independent(p,y); assert math.isclose(scalar,met['mse'],rel_tol=1e-12,abs_tol=1e-12)
    scalar_mae=math.fsum(math.fsum(abs(float(a)-float(b)) for a,b in zip(p[:,c].flat,y[:,c].flat) if np.isfinite(b))/int(np.isfinite(y[:,c]).sum()) for c in range(32))/32
    assert math.isclose(scalar_mae,met['mae'],rel_tol=1e-12,abs_tol=1e-12)
    np.savez_compressed(path,prediction=p,target=y,std=std,origins=np.array(origins))
    return dict(metrics=met,prediction_path=str(path.relative_to(ROOT)),prediction_sha256=sha(path),scalar_mse=scalar,scalar_mae=scalar_mae)


def optimizer(m):
    return torch.optim.AdamW(parameters(m).values(),lr=.001,weight_decay=0,betas=(.9,.999),eps=1e-8)


def run():
    c=read(OUT/'seal.json'); check_hashes(c['source_hashes'])
    s=read(OUT/'status.json'); assert s['status']=='PREPARED' and s['fit_attempts']==0
    for d in c['data'].values(): check_hashes(d['staged'])
    configure(); w=None; fits=[]; trajectory=[]; results=[]; smoke=[]; m=opt=None
    try:
        w=Watch(OUT,'finite_diagnostic',c['wall_cap']); w.boundary(startup=True)
        s['status']='SMOKE';save(OUT/'status.json',s)
        for d in c['data']:
            ids=c['data'][d]['channel_ids'];m=make(ids,41000,'cuda');m.eval()
            vals,std=load(DATA,d,'development');x,y=batch(vals,c['data'][d]['origins']['validation'][:8])
            init=next(r for r in read(OUT/'old_initial_states.json') if r['dataset']==d and r['seed']==41000)
            with np.load(ROOT/init['prediction_path']) as z:b=z['prediction'][:8]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):a=m(x).forecast
            assert np.array_equal(a.float().cpu().numpy(),b),'INITIAL_HISTORICAL_LH_PARITY'
            del x,y,a,b;cleanup()
            frozen=frozen_hash(m); buffers=tensor_hash(dict(m.named_buffers())); initial=cpu_state(m); opt=optimizer(m); records=[];m.train()
            for k in range(2):
                assert s['smoke_updates']<4
                def applied(): s['smoke_updates']+=1;save(OUT/'status.json',s)
                records.append(step(m,opt,vals,c['data'][d]['origins']['train'][8*k:8*(k+1)],8,w,applied))
            assert frozen_hash(m)==frozen and tensor_hash(dict(m.named_buffers()))==buffers
            now=cpu_state(m); changes={n:float((now[n].double()-v.double()).norm()) for n,v in initial.items()}
            for prefix in ['head.','encoder.']:
                assert any(v>0 for n,v in changes.items() if n.startswith(prefix)),prefix
            smoke.append(dict(dataset=d,initial_bf16_LH_exact=True,frozen_and_buffers_unchanged=True,changes=changes,steps=records))
            save(OUT/'smoke.json',smoke);m=opt=None;cleanup()
        schedules=read(OUT/'schedules.json')
        for index,cell in enumerate(c['cells']):
            d,seed=cell['dataset'],cell['seed'];fid=f'{index:02}_{d}_{seed}_BALANCED_LH'
            fit=dict(fit=fid,**cell,status='RUNNING',updates=0,epochs=0);fits.append(fit);s['fit_attempts']+=1
            s.update(status='TRAINING',current_fit=fid);save(OUT/'status.json',s);save(OUT/'fits.json',fits)
            tick=time.monotonic();w.boundary();m=make(c['data'][d]['channel_ids'],seed,'cuda');opt=optimizer(m)
            scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=5,gamma=.5)
            frozen=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()));initial=cpu_state(m)
            vals,std=load(DATA,d,'development');best=None;steps=[]
            def checkpoint(epoch):
                nonlocal best
                r=evaluate(m,vals,std,c['data'][d]['origins']['validation'],f'V_{fid}_{epoch:02}',w)
                path=CACHE/f'{fid}_{epoch:02}.pt';torch.save(cpu_state(m),path)
                row=dict(fit=fid,**cell,epoch=epoch,updates=fit['updates'],checkpoint_path=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path),**r)
                trajectory.append(row);save(OUT/'trajectory.json',trajectory)
                if best is None or r['metrics']['mse']<best['metrics']['mse']:best=row
                print('V',fid,epoch,r['metrics']['mse'],flush=True)
                return r['metrics']['mse']
            patience_best=checkpoint(0);bad=0
            for epoch,oo in enumerate(schedules[f'{d}_{seed}'],1):
                m.train()
                for i in range(0,len(oo),8):
                    assert s['training_updates']<5080
                    def applied():fit['updates']+=1;s['training_updates']+=1
                    r=step(m,opt,vals,oo[i:i+8],8,w,applied);steps.append(dict(epoch=epoch,update=fit['updates'],**r))
                fit['epochs']=epoch;v=checkpoint(epoch)
                if v<patience_best-1e-4:patience_best=v;bad=0
                else:bad+=1
                scheduler.step();save(OUT/(fid+'_steps.json'),steps);save(OUT/'fits.json',fits);save(OUT/'status.json',s)
                if bad>=5:break
            assert frozen_hash(m)==frozen and tensor_hash(dict(m.named_buffers()))==buffers
            now=cpu_state(m);save(OUT/(fid+'_changes.json'),{n:float((now[n].double()-v.double()).norm()) for n,v in initial.items()})
            m=opt=None;cleanup();m=make(c['data'][d]['channel_ids'],seed,'cuda')
            assert sha(ROOT/best['checkpoint_path'])==best['checkpoint_sha256'];restore(m,torch.load(ROOT/best['checkpoint_path'],weights_only=True,map_location='cpu'))
            replay=evaluate(m,vals,std,c['data'][d]['origins']['validation'],'REPLAY_'+fid,w)
            with np.load(ROOT/best['prediction_path']) as z: aa=z['prediction']
            with np.load(ROOT/replay['prediction_path']) as z: bb=z['prediction']
            assert np.array_equal(aa,bb)
            fit.update(status='COMPLETE',best=best,replay=replay,frozen_and_buffers_unchanged=True,replay_max_abs=0,
                wall_seconds=time.monotonic()-tick,active_seconds=sum(r['seconds'] for r in steps),peak_allocated=max(r['peak_allocated'] for r in steps),
                median_step_seconds=float(np.median([r['seconds'] for r in steps])),trainable=761952,early_stopped=bad>=5)
            s['fits_completed']+=1;save(OUT/'fits.json',fits);save(OUT/'status.json',s);m=opt=None;cleanup()
        save(OUT/'selection_seal.json',dict(at=time.time(),selections=[f['best'] for f in fits],exposure=c['exposure'],source_seal_sha256=sha(OUT/'seal.json')))
        s['status']='EVALUATING_REUSED_E';save(OUT/'status.json',s)
        for f in fits:
            d,seed=f['dataset'],f['seed'];vals,std=load(DATA,d,'evaluation')
            matched=next(r for r in trajectory if r['fit']==f['fit'] and r['epoch']==c['old_selected_epochs'][f'{d}_{seed}'])
            computed={}
            for role,cp in [('selected',f['best']),('matched_old_epoch',matched)]:
                epoch=cp['epoch']
                if epoch not in computed:
                    m=make(c['data'][d]['channel_ids'],seed,'cuda')
                    assert sha(ROOT/cp['checkpoint_path'])==cp['checkpoint_sha256']
                    restore(m,torch.load(ROOT/cp['checkpoint_path'],weights_only=True,map_location='cpu'))
                    computed[epoch]=evaluate(m,vals,std,c['data'][d]['origins']['evaluation'],f'E_{f["fit"]}_{epoch:02}',w)
                    m=None;cleanup()
                r=computed[epoch]
                for ctrl in read(OUT/'reused_controls.json'):
                    if ctrl['dataset']==d and (ctrl['seed']==seed or ctrl['arm']=='SEASONAL_NAIVE') and ctrl['role']!='INIT':
                        with np.load(ROOT/ctrl['prediction_path']) as z:yy=z['target'];oo=z['origins']
                        with np.load(ROOT/r['prediction_path']) as z:assert np.array_equal(yy,z['target'],equal_nan=True) and np.array_equal(oo,z['origins'])
                results.append(dict(dataset=d,seed=seed,arm='BALANCED_LH',role=role,epoch=epoch,updates=cp['updates'],**r));save(OUT/'evaluation.json',results)
        check_hashes(c['source_hashes']);check_hashes(read(OUT/'historical_hashes.json'))
        s['status']='COMPLETE';s['historical_files_preserved']=len(read(OUT/'historical_hashes.json'))
    except BaseException as exc:
        s.update(status='INCONCLUSIVE_EXECUTION',error=repr(exc),traceback=traceback.format_exc())
        if fits and fits[-1]['status']=='RUNNING': fits[-1].update(status='EXECUTION_ERROR',error=repr(exc))
        print(traceback.format_exc(),flush=True)
    finally:
        save(OUT/'fits.json',fits);save(OUT/'status.json',s)
        m=opt=None;cleanup()
        if w:w.close()
        print('END',s,flush=True)
    return 0 if s['status']=='COMPLETE' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run']);args=p.parse_args()
    if args.stage=='prepare':prepare()
    else:sys.exit(run())
