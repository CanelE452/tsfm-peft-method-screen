"""Finite four-fit known-control diagnostic; no candidate generation or retries."""
import argparse, ast, copy, json, math, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from model import ROOT, make, ResidualGate, original_make
from common import read, save, sha, cpu_state, parameters, tensor_hash, frozen_hash, restore, cleanup, preserve_rng, Watch
from rank2_data import load, batch, metrics, independent
from run_rank2 import configure, step

OUT = ROOT/'results/channel_identity_diagnostic_20260916'
EXP = ROOT/'experiments/channel_identity_diagnostic_20260916'
CACHE = ROOT/'.cache/channel_identity_diagnostic_20260916'
OLD = ROOT/'results/channel_sharing_screen_v1_20260915'
DATA = ROOT/'.cache/channel_sharing_screen_v1_20260915'


def check_hashes(mapping):
    for p, expected in mapping.items():
        assert sha(ROOT/p) == expected, ('HASH_CHANGED', p)


def structural():
    source = ROOT/'sources/timepeft_ea4e7e1/run.py'
    tree = ast.parse(source.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ['FrequencyAdapter', 'ChannelAdapter']]
    assert len(classes) == 2
    ns = dict(torch=torch, nn=nn)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), 'exec'), ns)
    torch.manual_seed(61692)
    f = ns['FrequencyAdapter'](3, 16).double().eval()
    c = ns['ChannelAdapter'](16, 4, 2).double().eval()
    h = torch.randn(2, 4, 12, 16, dtype=torch.float64)
    baseline = c(h, f(h))
    changed = h.clone(); changed[:, 1] += torch.randn_like(changed[:, 1])
    intervention = c(changed, f(changed))
    off_channel = float((baseline[:, [0,2,3]]-intervention[:, [0,2,3]]).abs().max())
    own_channel = float((baseline[:, 1]-intervention[:, 1]).abs().max())
    assert off_channel == 0 and own_channel > 0
    perm = torch.tensor([2, 0, 3, 1])
    reordered = c(h[:, perm], f(h[:, perm]))
    slot_sensitive = float((baseline[:, perm]-reordered).abs().max())
    c2 = copy.deepcopy(c); c2.up_projections = nn.ModuleList([copy.deepcopy(c.up_projections[int(i)]) for i in perm])
    coupled = float((baseline[:, perm]-c2(h[:, perm], f(h[:, perm]))).abs().max())
    assert slot_sensitive > 0 and coupled == 0
    try:
        c(h[:, :3], f(h[:, :3])); raise AssertionError('Expected C mismatch')
    except ValueError as exc:
        mismatch = str(exc)
    # Known gate equation: output and derivatives against an explicit scalar expression.
    class Tiny(nn.Module):
        def forward(self, x, y, *args): return torch.tanh(x+y)
    g = ResidualGate(Tiny()).double()
    assert torch.equal(g(h, h/3), h)
    g.gate.data.fill_(.2)
    x = h.clone().requires_grad_(); y = h.clone().requires_grad_()
    a = g(x, y).square().sum(); ga = torch.autograd.grad(a, (x,y,g.gate))
    xx=x.detach().clone().requires_grad_(); yy=y.detach().clone().requires_grad_(); aa=g.gate.detach().clone().requires_grad_()
    b=(xx+aa*torch.tanh(xx+yy)).square().sum(); gb=torch.autograd.grad(b,(xx,yy,aa))
    assert torch.equal(a,b) and all(torch.equal(u,v) for u,v in zip(ga,gb))
    return dict(source=str(source.relative_to(ROOT)), source_sha256=sha(source), off_channel_max_abs=off_channel,
                perturbed_channel_max_abs=own_channel, input_only_permutation_error=slot_sensitive,
                input_and_assigned_weights_permutation_error=coupled, different_channel_count_error=mismatch,
                zero_gate_identity=True, explicit_fp64_output_gradient_exact=True,
                scope='Untrained public modules only; no accuracy claim; channel IDs must move with their assigned weights.')


def prepare():
    assert not (OUT/'seal.json').exists() and not CACHE.exists(), 'No duplicate prepare/run'
    configure(); CACHE.mkdir()
    old = read(OLD/'contract.json')
    check_hashes(old['source_hashes'])
    for d in old['data'].values(): check_hashes(d['staged'])
    for p,h in old['model_files'].items(): assert sha(p)==h
    hist = {str(p.relative_to(ROOT)):sha(p) for directory in ['results','research']
            for p in (ROOT/directory).rglob('*') if p.is_file() and OUT not in p.parents}
    save(OUT/'historical_hashes.json', hist)
    references = []
    for r in read(OLD/'evaluation.json'):
        if r['arm'] in ['LH','SHARED_BUDGET','SEASONAL_NAIVE']:
            assert sha(ROOT/r['prediction_path']) == r['prediction_sha256']
            with np.load(ROOT/r['prediction_path']) as z:
                actual=independent(z['prediction'],z['target'])
            assert math.isclose(actual,r['metrics']['mse'],rel_tol=1e-12,abs_tol=1e-12)
            references.append(r)
    save(OUT/'reused_controls.json',references)
    save(OUT/'structural_probe.json',structural())
    checks=[]
    for seed in [41000,41001]:
        ids=old['data']['electricity']['channel_ids']
        m=make(ids,seed).eval(); lh=original_make('LH',ids,seed).eval()
        vals,_=load(DATA,'electricity','train'); x,_=batch(vals,old['data']['electricity']['origins']['train'][:2],'cpu')
        with torch.no_grad(): a=m(x).forecast; b=lh(x).forecast
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        n=sum(p.numel() for p in parameters(m).values()); assert n==1319712
        checks.append(dict(seed=seed,initial_LH_max_abs=float((a-b).abs().max()),trainable=n))
        del m,lh,x,a,b;cleanup()
    save(OUT/'cpu_checks.json',checks)
    files = list(EXP.glob('*.py'))+[OUT/'PROTOCOL.md',OLD/'contract.json',OLD/'schedules.json',OLD/'fits.json',OLD/'evaluation.json',OUT/'reused_controls.json']
    sources={str(p.relative_to(ROOT)):sha(p) for p in files}; sources.update(old['source_hashes'])
    save(OUT/'seal.json',dict(created_at=time.time(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes=sources,data=old['data'],model_files=old['model_files'],historical_manifest_sha256=sha(OUT/'historical_hashes.json'),
        cells=[dict(dataset=d,seed=s) for d in ['electricity','traffic'] for s in [41000,41001]],
        fit_cap=4,update_cap=5080,smoke_cap=4,wall_cap=3600,exposure='DISCOVERY_REUSED_E',novelty='DIRECT_EQUIVALENCE_REZERO'))
    save(OUT/'status.json',dict(status='PREPARED',fit_attempts=0,fits_completed=0,training_updates=0,smoke_updates=0))
    print('PREPARED; 4 new fits maximum; known control only',flush=True)


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
            ids=c['data'][d]['channel_ids']; m=make(ids,41000,'cuda'); lh=original_make('LH',ids,41000,'cuda');m.eval();lh.eval()
            vals,std=load(DATA,d,'train');x,y=batch(vals,c['data'][d]['origins']['train'][:8])
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):a=m(x).forecast;b=lh(x).forecast
            assert torch.equal(a,b),'INITIAL_LH_PARITY'; del lh,x,y,a,b; cleanup()
            frozen=frozen_hash(m); buffers=tensor_hash(dict(m.named_buffers())); initial=cpu_state(m); opt=optimizer(m); records=[];m.train()
            for k in range(2):
                assert s['smoke_updates']<4
                def applied(): s['smoke_updates']+=1;save(OUT/'status.json',s)
                records.append(step(m,opt,vals,c['data'][d]['origins']['train'][8*k:8*(k+1)],8,w,applied))
            assert frozen_hash(m)==frozen and tensor_hash(dict(m.named_buffers()))==buffers
            now=cpu_state(m); changes={n:float((now[n].double()-v.double()).norm()) for n,v in initial.items()}
            assert changes['channel_adapter.gate']>0
            for prefix in ['channel_adapter.base.','frequency_adapter.','head.','encoder.']:
                assert any(v>0 for n,v in changes.items() if n.startswith(prefix)),prefix
            smoke.append(dict(dataset=d,initial_bf16_LH_exact=True,frozen_and_buffers_unchanged=True,changes=changes,steps=records))
            save(OUT/'smoke.json',smoke);m=opt=None;cleanup()
        schedules=read(OLD/'schedules.json')
        for index,cell in enumerate(c['cells']):
            d,seed=cell['dataset'],cell['seed'];fid=f'{index:02}_{d}_{seed}_REZERO_SHARED'
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
                row=dict(fit=fid,**cell,epoch=epoch,updates=fit['updates'],gate=float(m.channel_adapter.gate.detach()),checkpoint_path=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path),**r)
                trajectory.append(row);save(OUT/'trajectory.json',trajectory)
                if best is None or r['metrics']['mse']<best['metrics']['mse']:best=row
                print('V',fid,epoch,r['metrics']['mse'],'gate',row['gate'],flush=True)
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
                median_step_seconds=float(np.median([r['seconds'] for r in steps])),trainable=1319712,early_stopped=bad>=5)
            s['fits_completed']+=1;save(OUT/'fits.json',fits);save(OUT/'status.json',s);m=opt=None;cleanup()
        save(OUT/'selection_seal.json',dict(at=time.time(),selections=[f['best'] for f in fits],exposure=c['exposure'],source_seal_sha256=sha(OUT/'seal.json')))
        s['status']='EVALUATING_REUSED_E';save(OUT/'status.json',s)
        for f in fits:
            d=f['dataset'];m=make(c['data'][d]['channel_ids'],f['seed'],'cuda');restore(m,torch.load(ROOT/f['best']['checkpoint_path'],weights_only=True,map_location='cpu'))
            vals,std=load(DATA,d,'evaluation');r=evaluate(m,vals,std,c['data'][d]['origins']['evaluation'],'E_'+f['fit'],w)
            for ctrl in read(OUT/'reused_controls.json'):
                if ctrl['dataset']==d and (ctrl['seed']==f['seed'] or ctrl['arm']=='SEASONAL_NAIVE') and ctrl['role']!='INIT':
                    with np.load(ROOT/ctrl['prediction_path']) as z: yy=z['target'];oo=z['origins']
                    with np.load(ROOT/r['prediction_path']) as z: assert np.array_equal(yy,z['target'],equal_nan=True) and np.array_equal(oo,z['origins'])
            results.append(dict(dataset=d,seed=f['seed'],arm='REZERO_SHARED',role='selected',epoch=f['best']['epoch'],**r));save(OUT/'evaluation.json',results)
            m=None;cleanup()
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
