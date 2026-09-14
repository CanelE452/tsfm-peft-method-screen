"""Preregistered window-budget intervention, sealed heldout evaluation and archival.

Reuse the frozen diagnostic worker only inside these isolated processes. Earlier
source files and result receipts remain byte-for-byte intact.
"""
import argparse
import gc
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import zipfile
import fcntl

import numpy as np
import pandas as pd
import torch
import run_reassessment_diagnostics as B
from tsfm_peft_screen.reproducibility import ROOT, sha, digest, write_json, seed_all
from tsfm_peft_screen.metrics import score, independent

CONFIG = ROOT/'configs/anchor_window_study_20260914.json'
CFG = json.loads(CONFIG.read_text())
OUT = ROOT/'results'/CFG['run_id']
CACHE = ROOT/'.cache'/CFG['run_id']
STAGED = ROOT/'data/processed'/CFG['run_id']
RESEARCH = ROOT/'research'/CFG['run_id']


def origins(spec):
    return list(range(spec[0], spec[1]+1, spec[2]))


def case_parts(case):
    source, budget = case.rsplit('_', 1)
    assert source in CFG['data'] and int(budget) in (32, 233)
    return source, int(budget)


def train_origins(source, budget):
    dense = origins(CFG['data'][source]['train'])
    assert len(dense) == 233
    return [dense[i] for i in np.linspace(0, 232, budget, dtype=int)]


def stage():
    assert not STAGED.exists(), 'Keep any previous staging attempt'
    STAGED.mkdir(parents=True)
    manifest = {}
    for source, spec in CFG['data'].items():
        path = ROOT/spec['raw_path']
        assert sha(path) == spec['raw_sha256'], source
        if source == 'beijing':
            with zipfile.ZipFile(path) as z:
                df = pd.read_csv(z.open('PRSA_data_2010.1.1-2014.12.31.csv'))
            dates = pd.to_datetime(df[['year','month','day','hour']])
            assert len(df) == 43824 and dates.is_monotonic_increasing and dates.is_unique
            assert (dates.diff().dropna() == pd.Timedelta(hours=1)).all()
            values = df[spec['channels']].to_numpy(dtype=np.float32)
        elif source == 'ettm2_later':
            df = pd.read_csv(path)
            dates = pd.to_datetime(df['date'])
            assert len(df) == 69680 and dates.is_monotonic_increasing and dates.is_unique
            assert (dates.diff().dropna() == pd.Timedelta(minutes=15)).all()
            values = df[spec['channels']].to_numpy(dtype=np.float32)
        else:
            values = np.loadtxt(gzip.open(path, 'rt'), delimiter=',', usecols=spec['channels'], dtype=np.float32)
            assert len(values) == 26304
        assert values.ndim == 2 and values.shape[1] == 4 and not np.isinf(values).any()
        tr, va, ev = [origins(spec[k]) for k in ('train','validation','evaluation')]
        assert tr[0] >= 1024 and tr[-1]+48 <= va[0] and va[-1]+48 <= ev[0]
        assert ev[-1]+48 <= len(values) and len(va) == 16 and len(ev) == 32
        # A common train-only scale isolates window selection, not label scarcity.
        scale = np.nanstd(values[tr[0]-1024:tr[-1]+48].astype(np.float64), axis=0)
        assert np.isfinite(scale).all() and (scale > 0).all()
        for oo in (tr, va, ev):
            yy = np.stack([values[o:o+48] for o in oo])
            assert np.isfinite(yy).sum((0,1)).min() > 0
        dev = STAGED/(source+'_development.npz')
        held = STAGED/(source+'_heldout.npz')
        np.savez_compressed(dev, values=values[:va[-1]+48], scale=scale)
        np.savez_compressed(held, values=values[:ev[-1]+48], scale=scale)
        manifest[source] = dict(raw_path=spec['raw_path'], raw_sha256=sha(path),
            development_path=str(dev.relative_to(ROOT)), development_sha256=sha(dev),
            heldout_path=str(held.relative_to(ROOT)), heldout_sha256=sha(held),
            rows=len(values), channels=spec['channels'], train_scale=scale.tolist(),
            missing_counts=np.isnan(values).sum(0).tolist(),
            note='Mechanical schema/missingness checks only; no evaluation forecasts or metric selection.')
    write_json(STAGED/'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


def input_receipts():
    rows = B.read(STAGED/'manifest.json')
    for row in rows.values():
        for prefix in ('development','heldout'):
            assert sha(ROOT/row[prefix+'_path']) == row[prefix+'_sha256']
        assert sha(ROOT/row['raw_path']) == row['raw_sha256']
    return rows


def load_dev(case):
    source, budget = case_parts(case)
    row = B.read(STAGED/'manifest.json')[source]
    assert sha(ROOT/row['development_path']) == row['development_sha256']
    with np.load(ROOT/row['development_path'], allow_pickle=False) as z:
        values, scale = z['values'], z['scale']
    va = origins(CFG['data'][source]['validation'])
    assert len(values) == va[-1]+48
    return dict(values=values, scale=scale, train=train_origins(source,budget), validation=va, zero_fraction=None)


def require_e_barrier():
    seal = B.read(OUT/'evaluation_seal.json')
    payload = {k:v for k,v in seal.items() if k != 'sha256'}
    assert digest(payload) == seal['sha256']
    assert seal['contract_sha256'] == sha(OUT/'contract.json')
    assert seal['development_seal_sha256'] == sha(OUT/'anchor/selection_seal.json')
    assert seal['fits_sha256'] == sha(OUT/'anchor/fits.json')
    fits = B.read(OUT/'anchor/fits.json')
    assert len(fits) == 48 and sum(f['updates'] for f in fits) == 43200
    assert len(seal['selections']) == 24 and len(seal['validation_policies']) == 12
    assert seal['selections'] == B.read(OUT/'anchor/selection_seal.json')['selections']
    return seal


def load_heldout(case):
    require_e_barrier()  # Must run before reading any heldout values.
    source, _ = case_parts(case)
    row = B.read(STAGED/'manifest.json')[source]
    path = ROOT/row['heldout_path']
    assert sha(path) == row['heldout_sha256']
    with np.load(path, allow_pickle=False) as z:
        data = dict(values=z['values'], scale=z['scale'], zero_fraction=None)
    data['evaluation'] = origins(CFG['data'][source]['evaluation'])
    return data


def check_contract():
    c = B.read(OUT/'contract.json')
    assert c['config'] == CFG and c['inputs'] == input_receipts()
    for name, h in {**c['source_hashes'], **c['historical_result_hashes']}.items():
        assert sha(ROOT/name) == h, name
    return c


def wire():
    B.CFG, B.OUT, B.CACHE, B.RESEARCH = CFG, OUT, CACHE, RESEARCH
    B.load_data, B.input_receipts, B.check_contract = load_dev, input_receipts, check_contract


def smoke():
    assert not (RESEARCH/'smoke_progress.json').exists(), 'Keep failed smoke attempts'
    lock = open(ROOT/'.cache/gpu.lock','a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    w = B.Worker('anchor', smoke=True); w.wait(); rows = []
    for source in CFG['data']:
        data = load_dev(source+'_32')
        samples = B.schedule('anchor',data['train'],34000,2)
        for arm in ('native','native_anchor'):
            seed_all(34000); w.check('smoke_load',True)
            model = B.DiagnosticModel('anchor',arm,34000)
            initial, frozen = B.parameters(model), B.frozen_hash(model)
            ps = B.trainable(model); opt = torch.optim.AdamW(ps.values(),lr=1e-4,weight_decay=0.)
            for step, sample in enumerate(samples):
                w.check('smoke_step'); opt.zero_grad(set_to_none=True)
                x,y,g,sc = B.batch(data,sample,1024,'anchor')
                with torch.no_grad(), B.precision('anchor'): teacher = model(x,g,frozen=True)[1]
                with B.precision('anchor'): z,p,loc,scale = model(x,g)
                if step == 0: assert torch.equal(p,teacher)
                loss, reg = B.objective('anchor',arm,z,p,y,loc,scale,sc,teacher,.1)
                assert torch.isfinite(loss+reg)
                (loss+reg).backward()
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in ps.values())
                torch.nn.utils.clip_grad_norm_(list(ps.values()),1.,error_if_nonfinite=True); opt.step()
            assert B.frozen_hash(model) == frozen and B.tensor_hash(B.parameters(model)) != B.tensor_hash(initial)
            rows.append(dict(source=source,arm=arm,updates=2,step0_exact=True,frozen_unchanged=True,
                finite_gradient=True,trainable_parameters=sum(v.numel() for v in ps.values())))
            write_json(RESEARCH/'smoke_progress.json',rows)
            del model,ps,opt,initial; gc.collect(); torch.cuda.empty_cache()
    write_json(RESEARCH/'smoke.json',dict(status='PASS',updates=12,rows=rows,
        source_hashes=B.sources(),inputs=input_receipts()))


def initialize():
    assert not OUT.exists() and not CACHE.exists(), 'No overwrite or implicit retry'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(), 'Commit before execution'
    s = B.read(RESEARCH/'smoke.json')
    assert s['status'] == 'PASS' and s['updates'] == 12 and s['source_hashes'] == B.sources()
    assert s['inputs'] == input_receipts()
    from huggingface_hub import hf_hub_download
    from tsfm_peft_screen.backbone import MODEL_ID, REVISION
    model_files = B.read(ROOT/'results/screening_summary/common_integrity.json')['model_files']
    for name,h in model_files.items():
        assert sha(hf_hub_download(MODEL_ID,name,revision=REVISION,local_files_only=True)) == h
    history = {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'results').rglob('*') if p.is_file()}
    OUT.mkdir(); CACHE.mkdir()
    write_json(OUT/'contract.json',dict(config=CFG,source_hashes=B.sources(),historical_result_hashes=history,
        execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        model_revision=REVISION,model_files=model_files,inputs=input_receipts(),smoke_sha256=sha(RESEARCH/'smoke.json')))


def validation_policies(selections):
    result = []
    for ds in CFG['topics']['anchor']['datasets']:
        for seed in CFG['topics']['anchor']['seeds']:
            pool = [r for r in selections if r['dataset'] == ds and r['seed'] == seed]
            assert len(pool) == 2
            # Plain wins exact ties. The policy has a larger arm-selection budget.
            chosen = min(pool,key=lambda r:(r['metrics']['scaled_2pinball'],r['arm']!='native'))
            result.append(dict(dataset=ds,seed=seed,chosen_arm=chosen['arm'],selection=chosen))
    return result


def seal_evaluation():
    check_contract()
    fits = B.read(OUT/'anchor/fits.json')
    assert len(fits) == 48 and sum(f['updates'] for f in fits) == 43200
    dev = B.read(OUT/'anchor/selection_seal.json')
    assert digest({k:v for k,v in dev.items() if k!='sha256'}) == dev['sha256']
    assert dev['contract_sha256'] == sha(OUT/'contract.json')
    assert dev['fits_sha256'] == sha(OUT/'anchor/fits.json')
    selections = dev['selections']
    for s in selections:
        assert s == B.choose([f['best'] for f in fits if (f['dataset'],f['seed'],f['arm']) == (s['dataset'],s['seed'],s['arm'])])
    seal = dict(selections=selections,validation_policies=validation_policies(selections),
        contract_sha256=sha(OUT/'contract.json'),fits_sha256=sha(OUT/'anchor/fits.json'),
        development_seal_sha256=sha(OUT/'anchor/selection_seal.json'),created_unix=time.time(),
        scope='All 48 fits finished and V choices frozen before any E forecast; no E tuning.')
    seal['sha256'] = digest(seal)
    assert not (OUT/'evaluation_seal.json').exists()
    write_json(OUT/'evaluation_seal.json',seal)


def evaluate():
    check_contract(); seal = require_e_barrier()
    assert not (OUT/'evaluation_attempt.json').exists(), 'No implicit E retry'
    write_json(OUT/'evaluation_attempt.json',dict(started_unix=time.time(),seal_sha256=sha(OUT/'evaluation_seal.json')))
    w = B.Worker('anchor'); w.wait(); records = []
    for ds in CFG['topics']['anchor']['datasets']:
        data = load_heldout(ds); seed_all(34000)
        model = B.DiagnosticModel('anchor','native',34000)
        row = w.evaluate(model,data,data['evaluation'],ds+'_F0_E',frozen=True)
        records.append(dict(dataset=ds,arm='F0',seed=None,split='evaluation',**row))
        del model; gc.collect(); torch.cuda.empty_cache()
        for s in [r for r in seal['selections'] if r['dataset'] == ds]:
            w.check('E_load',True); seed_all(s['seed'])
            assert sha(ROOT/s['checkpoint_file']) == s['checkpoint_sha256']
            model = B.DiagnosticModel('anchor',s['arm'],s['seed'])
            B.restore(model,torch.load(ROOT/s['checkpoint_file'],map_location='cpu',weights_only=True))
            row = w.evaluate(model,data,data['evaluation'],s['fit_id']+'_selected_E')
            records.append(dict(dataset=ds,arm=s['arm'],seed=s['seed'],split='evaluation',selection=s,**row))
            write_json(OUT/'evaluation_progress.json',records)
            del model; gc.collect(); torch.cuda.empty_cache()
    assert len(records) == 30
    write_json(OUT/'evaluation.json',records); check_contract()
    w.emit(status='COMPLETE_HELDOUT_EVALUATION',evaluation_forecasts=30)


def channel_origin_loss(row):
    from tsfm_peft_screen.backbone import QUANTILES
    with np.load(ROOT/row['prediction_file']) as z:
        p = np.sort(z['prediction'].astype(np.float64),axis=2)
        y,sc = z['target'].astype(np.float64),z['scale']
    valid = np.isfinite(y)
    e = y[:,:,None,:]-p; q = np.array(QUANTILES)[None,None,:,None]
    losses = np.where(valid[:,:,None,:],2*np.maximum(q*e,(q-1)*e),0.)
    return losses.sum((2,3))/21/sc[None,:],valid.sum(2)


def block_interval(plain, anchor):
    # Same sampled chronological blocks for both seeds, channels, methods.
    pairs = [(channel_origin_loss(p),channel_origin_loss(a)) for p,a in zip(plain,anchor)]
    rng = np.random.default_rng(81421); deltas = []
    for _ in range(2000):
        idx = np.concatenate([np.arange(b*8,(b+1)*8) for b in rng.integers(0,4,4)])
        errors = []
        for (pn,n),(an,m) in pairs:
            assert np.array_equal(n,m)
            den = n[idx].sum(0)
            assert (den > 0).all()
            errors.append(((pn[idx].sum(0)/den).mean(),(an[idx].sum(0)/den).mean()))
        ref,method = np.mean(errors,axis=0); deltas.append(100*(1-method/ref))
    return np.quantile(deltas,[.025,.975]).tolist()


def finalize(verify_only=False):
    contract = check_contract(); seal = require_e_barrier()
    fits = B.read(OUT/'anchor/fits.json'); evals = B.read(OUT/'evaluation.json')
    expected = {(d,s,a,r) for d in CFG['topics']['anchor']['datasets'] for s in (34000,34001)
                for a in ('native','native_anchor') for r in (0,1)}
    assert {(f['dataset'],f['seed'],f['arm'],f['recipe']) for f in fits} == expected
    assert len(evals) == 30 and seal['validation_policies'] == validation_policies(seal['selections'])
    seen = set(); errors = []; allrows = []
    def replay(row,ds,split):
        path = ROOT/row['prediction_file']; assert sha(path) == row['prediction_sha256']
        if str(path) in seen: return
        data = load_heldout(ds) if split == 'evaluation' else load_dev(ds)
        oo = data[split]
        expected_y = np.stack([data['values'][o:o+48].T for o in oo])
        with np.load(path,allow_pickle=False) as z:
            assert np.array_equal(z['origins'],oo) and np.array_equal(z['scale'],data['scale'])
            assert np.array_equal(z['target'],expected_y,equal_nan=True)
            err = abs(independent(z['prediction'],z['target'],z['scale'])-row['metrics']['scaled_2pinball'])
        assert err < 1e-10; errors.append(err); seen.add(str(path))
    for f in fits:
        path = OUT/'anchor'/(f['fit_id']+'_fit.json')
        assert sha(path) == f['fit_receipt_sha256']
        full = B.read(path); ledger = full['resources']
        assert len(ledger) == f['updates'] == 900
        assert [r['step'] for r in ledger] == list(range(1,901))
        assert all(np.isfinite(r['loss']) and np.isfinite(r['regularizer']) and np.isfinite(r['gradient_norm']) for r in ledger)
        assert abs(sum(r['seconds'] for r in ledger)-f['active_seconds']) < 1e-7
        assert f['trainable_parameters'] == 1179648 and f['initial_parameters_sha256'] != f['final_parameters_sha256']
        data = load_dev(f['dataset'])
        assert digest(B.schedule('anchor',data['train'],f['seed'],900)) == f['sample_sequence_sha256']
        rows = B.read(OUT/'anchor'/(f['fit_id']+'_trajectory.json'))
        assert [r['step'] for r in rows] == [0,150,450,900] and B.choose(rows) == f['best']
        for r in rows:
            replay(r,f['dataset'],'validation')
            assert sha(ROOT/r['checkpoint_file']) == r['checkpoint_sha256']
        replay(f['reload'],f['dataset'],'validation')
        with np.load(ROOT/f['best']['prediction_file']) as a,np.load(ROOT/f['reload']['prediction_file']) as b:
            assert np.array_equal(a['prediction'],b['prediction'])
        allrows.extend(rows)
    for r in B.read(OUT/'anchor/baselines.json'): replay(r,r['dataset'],r['split'])
    for s in seal['selections']:
        assert B.choose([r for r in allrows if (r['dataset'],r['seed'],r['arm']) == (s['dataset'],s['seed'],s['arm'])]) == s
    for r in evals:
        replay(r,r['dataset'],'evaluation')
        if r['arm'] != 'F0': assert r['selection'] in seal['selections']
    assert {(r['dataset'],r['seed'],r['arm']) for r in evals} == {
        (d,s,a) for d in CFG['topics']['anchor']['datasets'] for s,a in [(None,'F0'),(34000,'native'),(34001,'native'),(34000,'native_anchor'),(34001,'native_anchor')]}
    cells = []
    for ds in CFG['topics']['anchor']['datasets']:
        source,budget = case_parts(ds)
        def rows(arm): return sorted([r for r in evals if r['dataset']==ds and r['arm']==arm],key=lambda r:r['seed'] or 0)
        p,a = rows('native'),rows('native_anchor')
        pm,am = [np.mean([r['metrics']['scaled_2pinball'] for r in rr]) for rr in (p,a)]
        f0 = rows('F0')[0]['metrics']['scaled_2pinball']
        policy = [next(r for r in evals if (r['dataset'],r['seed'],r['arm'])==(ds,v['seed'],v['chosen_arm']))
                  for v in seal['validation_policies'] if v['dataset']==ds]
        policy_loss = np.mean([r['metrics']['scaled_2pinball'] for r in policy])
        cells.append(dict(dataset=ds,source=source,training_windows=budget,plain_loss=float(pm),anchor_loss=float(am),
            f0_loss=f0,anchor_gain_percent=float(100*(1-am/pm)),plain_vs_f0_percent=float(100*(1-pm/f0)),
            anchor_vs_f0_percent=float(100*(1-am/f0)),validation_policy_loss=float(policy_loss),
            policy_gain_percent=float(100*(1-policy_loss/pm)),
            seed_gains_percent=[100*(1/y['metrics']['scaled_2pinball']*x['metrics']['scaled_2pinball']) for x,y in zip(a,p)],
            descriptive_block_interval_percent=block_interval(p,a)))
    contrast = {source:next(c['anchor_gain_percent'] for c in cells if c['source']==source and c['training_windows']==32)
                -next(c['anchor_gain_percent'] for c in cells if c['source']==source and c['training_windows']==233) for source in CFG['data']}
    macro = float(np.mean([c['anchor_gain_percent'] for c in cells]))
    sparse = float(np.mean([c['anchor_gain_percent'] for c in cells if c['training_windows']==32]))
    interaction = float(np.mean(list(contrast.values())))
    support = sparse>0 and interaction>0 and sum(v>0 for v in contrast.values())>=2
    summary = dict(scope=CFG['scope'],cells=cells,macro_anchor_gain_percent=macro,
        sparse_macro_gain_percent=sparse,sparse_minus_dense_gain_percentage_points=contrast,
        mean_interaction_percentage_points=interaction,
        decision='CONDITIONAL_FOLLOWUP_SIGNAL' if support else 'WINDOW_BUDGET_HYPOTHESIS_NOT_SUPPORTED',
        warning='Descriptive adaptive research follow-up; neither outcome is a publication or novelty PASS.')
    verified = dict(status='VERIFIED_HELDOUT_STUDY',completed_fits=48,training_updates=43200,
        smoke_updates_separate=12,evaluation_forecast_caches=30,prediction_caches_replayed=len(seen),
        max_primary_error=max(errors),historical_files_unchanged=len(contract['historical_result_hashes']),
        execution_commit=contract['execution_commit'])
    if not verify_only:
        write_json(OUT/'verification.json',verified); write_json(OUT/'summary.json',summary)
        lines = ['# Anchoring × optimization window budget','',f"Decision: **{summary['decision']}**. This is not a paper PASS.",'',
            '48 fits, 43,200 updates, two seeds; all validation choices sealed before 30 heldout forecasts.',
            'Positive gain means lower scaled 2-pinball than plain native LoRA. No all-dataset improvement gate.','',
            '| Source | Windows | Plain | Anchor | Gain % | Seed gains % | Descriptive block interval % |',
            '|---|---:|---:|---:|---:|---|---|']
        for c in cells:
            lines.append(f"| {c['source']} | {c['training_windows']} | {c['plain_loss']:.6f} | {c['anchor_loss']:.6f} | {c['anchor_gain_percent']:+.3f} | {c['seed_gains_percent'][0]:+.3f}, {c['seed_gains_percent'][1]:+.3f} | {c['descriptive_block_interval_percent'][0]:+.3f}, {c['descriptive_block_interval_percent'][1]:+.3f} |")
        lines += ['',f'Macro gain: {macro:+.3f}%. Sparse-minus-dense interaction: {interaction:+.3f} percentage points.','',
            'The two budgets reuse the same evaluation period; they are not independent datasets. Four chronological blocks with two fixed seeds give descriptive uncertainty only. Beijing is a new project source; ETTm2 is a later period of a previously used source, and electricity uses previously unused source-order channels 4–7. Foundation-model pretraining overlap is not excluded.',
            'Training scales use the common training interval. This changes optimization windows, not strict label availability. Dense targets overlap; sparse targets do not. All arms use identical steps and within-budget sampling. The optional validation policy compares two arms and therefore has more selection budget; it is not a new PEFT method.',
            'ETTm2 forecasts 12 hours; hourly sources forecast 48 hours. Missing Beijing targets are masked, not imputed. No evaluation score sets hyperparameters.','',
            'Next action: '+('Use the sparse/dense interaction to design a distinct preservation mechanism, then audit novelty and validate on additional untouched sources/seeds. Do not declare the uniform anchor itself novel.' if support else 'Do not launch more uniform-anchor variants on these same E targets. Inspect train/V forgetting and gradient interference to identify a mechanism before preregistering a different intervention; retain these negative and mixed results.'),'',
            f"Integrity: {len(seen)} prediction caches independently replayed; maximum metric discrepancy {max(errors):.3g}; {verified['historical_files_unchanged']} historical files unchanged."]
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
        write_json(OUT/'next_action.json',dict(decision=summary['decision'],automatic_new_training=False,
            reason='A scientifically distinct next mechanism requires reasoning and a new frozen protocol; this controller only runs the authorized bounded study.'))
    print(json.dumps(verified,indent=2)); return summary


def archive():
    check_contract()
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR='/tmp/tsfm-window-mpl')
    with (OUT/'archive_checks.txt').open('x') as log:
        subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests','automation/overnight_followup/test_watch.py'],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    subprocess.run(['git','diff','--exit-code'],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--exit-code'],cwd=ROOT,check=True)
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
    # Live queue heartbeat stays ignored; only final scientific artifacts are archived.
    for p in OUT.rglob('*'):
        if p.is_file(): assert p.suffix in ('.json','.md','.txt','.csv','.png') and p.stat().st_size<10*2**20,p
    relative = str(OUT.relative_to(ROOT))
    subprocess.run(['git','add','--',relative],cwd=ROOT,check=True)
    staged = subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    assert staged and all(p.startswith(relative+'/') for p in staged)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--only','-m','Archive sealed heldout anchoring window-budget study','--',relative],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True,timeout=120)


def queue():
    lock = open(ROOT/'.cache/gpu.lock','a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    initialize(); start=time.monotonic(); child=None
    state=dict(status='RUNNING',pid=os.getpid(),started_unix=time.time(),jobs=[])
    def emit(**kw):
        state.update(kw); state.update(heartbeat_unix=time.time(),wall_seconds=time.monotonic()-start)
        write_json(CACHE/'queue_status.json',state)
    def stop(signum,frame): raise KeyboardInterrupt(f'Queue signal {signum}')
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    try:
        for phase in ('train','seal','evaluate','finalize','archive'):
            env=dict(os.environ)
            if phase in ('seal','finalize','archive'): env['CUDA_VISIBLE_DEVICES']=''
            with (CACHE/(phase+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,__file__,phase],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                record=dict(phase=phase,pid=child.pid,started_unix=time.time()); emit(current_job=record)
                while child.poll() is None:
                    if time.monotonic()-start>CFG['max_queue_wall_seconds']: raise TimeoutError('Queue wall cap')
                    emit(); time.sleep(5)
                record.update(exit_code=child.returncode,finished_unix=time.time());state['jobs'].append(record)
                write_json(CACHE/(phase+'_exit.json'),record)
                if child.returncode: raise RuntimeError(f'{phase} exited {child.returncode}; preserve attempt, no automatic retry')
                child=None
        emit(status='COMPLETE',finished_unix=time.time(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=10)
        emit(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'INCONCLUSIVE_EXECUTION',error=str(exc))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['stage','smoke','start','queue','train','seal','evaluate','finalize','archive','status'])
    parser.add_argument('--verify-only',action='store_true');args=parser.parse_args();wire()
    if args.command=='start':
        assert not CACHE.exists() and not OUT.exists()
        log=ROOT/'.cache'/(CFG['run_id']+'_controller.log')
        with log.open('x') as f:
            p=subprocess.Popen([sys.executable,__file__,'queue'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,log=str(log))))
        return
    if args.command=='status':
        for p in (CACHE/'queue_status.json',OUT/'anchor/status.json'):
            if p.exists(): print(p, p.read_text())
        return
    if args.command=='train':
        w=B.Worker('anchor')
        try: return w.train()
        except BaseException as exc:
            w.emit(status='INCONCLUSIVE_EXECUTION',error=str(exc));(w.folder/'traceback.txt').write_text(traceback.format_exc());raise
    if args.command=='finalize': return finalize(args.verify_only)
    return dict(stage=stage,smoke=smoke,queue=queue,seal=seal_evaluation,evaluate=evaluate,archive=archive)[args.command]()


if __name__=='__main__': main()
