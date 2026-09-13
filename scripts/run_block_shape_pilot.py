"""Fixed 48-fit pilot. A sealed choice precedes all heldout feature extraction."""
import copy
import fcntl
import gc
import json
import subprocess
import time
import traceback
import numpy as np
import torch
from tsfm_peft_screen.reproducibility import ROOT, sha, write_json, digest, source_hashes, seed_all, guard
from tsfm_peft_screen.backbone import load_base, native_loss
from tsfm_peft_screen.forecast_query.model import ForecastModel
from tsfm_peft_screen.forecast_query.gpu import GPUWatch
from tsfm_peft_screen.block_shape.method import SplitAdapter, per_origin_loss, acceptance, snapshot_transaction, rollback_transaction
from tsfm_peft_screen.metrics import score

CFG = json.loads((ROOT / 'configs/block_shape_pilot.json').read_text())
OUT = ROOT / 'results/block_shape_pilot'
CACHE = ROOT / '.cache/block_shape_pilot'
TRAIN = [CFG['train_block_start'] + b * 512 + o for b in range(8) for o in CFG['train_within_block_origins']]
BLOCKS = torch.arange(8).repeat_interleave(15)
V = CFG['validation_origins']
E = CFG['evaluation_origins']

def save_state(m):
    return {n: p.detach().cpu().clone() for n, p in m.named_parameters() if p.requires_grad}

def restore(m, state):
    params = {n: p for n, p in m.named_parameters() if p.requires_grad}
    assert params.keys() == state.keys()
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(state[n].to(p))

def batch(values, origins):
    x = np.concatenate([values[o-4096:o, :4].T for o in origins]).astype(np.float32)
    y = np.concatenate([values[o:o+48, :4].T for o in origins]).astype(np.float32)
    g = torch.arange(len(origins), device='cuda').repeat_interleave(4)
    return torch.tensor(x, device='cuda'), torch.tensor(y, device='cuda'), g

def extract(base, values, origins, tag):
    rows = {k: [] for k in ['h', 'f0', 'target', 'raw_target', 'loc', 'scale']}
    tick = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for start in range(0, len(origins), 2):
            watch.check('cache_' + tag)
            x, y, g = batch(values, origins[start:start+2])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                enc, (loc, scale), _, n = base.encode(context=x, group_ids=g, num_output_patches=3)
                assert n == 256
                hidden = enc.last_hidden_state[:, -3:]
                z = base.output_patch_embedding(hidden).reshape(len(x), 3, 21, 16).permute(0, 2, 1, 3).reshape(len(x), 21, 48).float()
            entry = dict(h=torch.nn.functional.layer_norm(hidden.float(), (768,)),
                         f0=z.sinh().sort(1).values, target=(y-loc)/scale, raw_target=y,
                         loc=loc.float(), scale=scale.float())
            for k in rows:
                rows[k].append(entry[k].detach())
    result = {k: torch.cat(v) for k, v in rows.items()}
    assert all(not t.requires_grad for t in result.values())
    path = CACHE / (tag + '_features.pt')
    torch.save({k: v.cpu() for k, v in result.items()}, path)
    cache_receipts.append(dict(tag=tag, origins=origins, sha256=sha(path),
                               seconds=time.monotonic()-tick, peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    write_json(OUT / 'feature_cache.json', cache_receipts)
    return result

def subset(features, index):
    idx = torch.tensor([4*i+c for i in index for c in range(4)], device='cuda')
    return {k: v[idx] for k, v in features.items()}

def raw_prediction(p, f):
    return (p * f['scale'][:, None] + f['loc'][:, None]).detach().cpu().numpy().reshape(-1, 4, 21, 48)

def target_array(f):
    return f['raw_target'].cpu().numpy().reshape(-1, 4, 48)

def metric(p, f, scale):
    return score(p, target_array(f), scale)

def store_predictions(tag, p, f, scale):
    path = CACHE / (tag + '.npz')
    np.savez_compressed(path, prediction=p, target=target_array(f), scale=scale)
    return sha(path)

def predict(model, arm, features, values, origins):
    with torch.no_grad():
        if arm != 'lora':
            return raw_prediction(model(features['h'], features['f0'])[0], features)
        arrays = []
        for start in range(0, len(origins), 2):
            watch.check('lora_evaluation')
            x, _, g = batch(values, origins[start:start+2])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, p, _, _ = model(x, g)
            arrays.append(p.cpu().numpy().reshape(-1, 4, 21, 48))
        return np.concatenate(arrays)

def split_step(m, center_opt, shape_opt, sampled, full, arm):
    center_opt.zero_grad(set_to_none=True)
    if shape_opt:
        shape_opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    tick = time.monotonic()
    p, log_gap = m(sampled['h'], sampled['f0'])
    task_loss = per_origin_loss(p, sampled['target']).mean()
    loss = task_loss + (CFG['anchor_lambda'] * log_gap.square().mean() if arm == 'anchor' else 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    norm = torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.)
    center_opt.step()
    gate = None
    if shape_opt:
        snapshot = None
        if arm in ['pooled', 'block']:
            snapshot = snapshot_transaction(m.shape, shape_opt)
            with torch.no_grad():
                before = per_origin_loss(m(full['h'], full['f0'])[0], full['target'])
        shape_opt.step()
        if snapshot:
            with torch.no_grad():
                after = per_origin_loss(m(full['h'], full['f0'])[0], full['target'])
                gate = acceptance(after-before, BLOCKS.to('cuda'), CFG['block_margin'] if arm == 'block' else 0.)
            if not gate['accept']:
                rollback_transaction(m.shape, shape_opt, snapshot)
    torch.cuda.synchronize()
    return dict(loss=float(task_loss.detach()), objective=float(loss.detach()), gradient_norm=float(norm),
                gate=gate, seconds=time.monotonic()-tick, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())

def lora_step(m, opt, values, origins):
    x, y, g = batch(values, origins)
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    tick = time.monotonic()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z, _, loc, scale = m(x, g)
        loss = native_loss(z, y, loc, scale)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    norm = torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.)
    opt.step()
    torch.cuda.synchronize()
    return dict(loss=float(loss.detach()), objective=float(loss.detach()), gradient_norm=float(norm), gate=None,
                seconds=time.monotonic()-tick, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())

def prepare_splits():
    """Mechanical data staging only. No heldout statistics, plots or scoring."""
    receipts = {}
    for name in CFG['datasets']:
        src = ROOT / 'data/processed' / name
        with np.load(src / 'fit.npz') as f:
            prefix = f['values'][:, :4]
        with np.load(src / 'evaluation.npz') as f:
            values = np.concatenate([prefix, f['tail'][:, :4]])
        assert max(TRAIN)+48 < min(V) and max(V)+48 < min(E) and max(E)+48 <= len(values)
        scale = np.maximum(np.nanstd(values[10240:14336], axis=0, dtype=np.float64), 1e-6)
        # The new training loader can only see through the end of V.
        np.savez_compressed(CACHE / (name+'_development.npz'), values=values[:max(V)+48], scale=scale)
        np.savez_compressed(CACHE / (name+'_heldout.npz'), values=values[:max(E)+48], scale=scale)
        receipts[name] = dict(source_fit_sha256=sha(src/'fit.npz'), source_tail_sha256=sha(src/'evaluation.npz'),
                              development_sha256=sha(CACHE/(name+'_development.npz')),
                              heldout_sha256=sha(CACHE/(name+'_heldout.npz')), scale=scale.tolist())
    return receipts

def main():
    global watch
    lock = open(ROOT / '.cache/gpu.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not OUT.exists(), 'Immutable run directory'
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip(), 'Commit execution code first'
    OUT.mkdir()
    CACHE.mkdir()
    contract = dict(execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    source_hashes=source_hashes(), config_sha256=sha(ROOT/'configs/block_shape_pilot.json'),
                    config=CFG, data=prepare_splits(), historical_results_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'results').rglob('*')) if p.is_file() and OUT not in p.parents})
    write_json(OUT/'contract.json', contract)
    write_json(OUT/'status.json', dict(status='WAITING_FOR_GPU', completed_fits=0))
    watch = GPUWatch(OUT/'gpu_monitor.json')
    idle_since = None
    wait_start = time.monotonic()
    while True:
        row = watch.read('startup_wait')
        good = not row['external_pids'] and row['free_mib'] >= 4096 and row['utilization_percent'] < 90
        idle_since = (idle_since or time.monotonic()) if good else None
        if idle_since and time.monotonic()-idle_since >= 30:
            break
        if time.monotonic()-wait_start > CFG['gpu_startup_timeout_seconds']:
            raise TimeoutError('GPU remained busy for the 12-hour startup limit')
        time.sleep(5)
    started = time.monotonic()
    feature_sets, data = {}, {}
    seed_all(30000)
    base = load_base()
    for name in CFG['datasets']:
        with np.load(CACHE/(name+'_development.npz')) as f:
            values, scale = f['values'], f['scale']
        data[name] = (values, scale)
        feature_sets[name] = (extract(base, values, TRAIN, name+'_train'), extract(base, values, V, name+'_V'))
    del base
    gc.collect()
    torch.cuda.empty_cache()
    # Real features: identity, gradient flow, and preserved ordered outputs.
    for name, (tr, va) in feature_sets.items():
        m = SplitAdapter().cuda()
        p, _ = m(tr['h'][:8], tr['f0'][:8])
        error = float((p-tr['f0'][:8]).abs().max())
        assert torch.allclose(p, tr['f0'][:8], atol=2e-5, rtol=1e-5)
        per_origin_loss(p, tr['target'][:8]).mean().backward()
        assert m.center.up.weight.grad.norm() > 0 and m.shape.up.weight.grad.norm() > 0
        smokes.append(dict(dataset=name, initial_normalized_max_abs=error,
                           center_gradient_norm=float(m.center.up.weight.grad.norm()),
                           shape_gradient_norm=float(m.shape.up.weight.grad.norm()),
                           trainable_parameters=sum(p.numel() for p in m.parameters()), optimizer_updates=0))
        del m, p
    write_json(OUT/'smoke.json', smokes)
    schedules = {str(seed): np.random.default_rng(seed).integers(0, len(TRAIN), (CFG['steps'], 2)).tolist() for seed in CFG['seeds']}
    write_json(OUT/'schedules.json', schedules)
    for name in CFG['datasets']:
        values, scale = data[name]
        tr, va = feature_sets[name]
        f0_v = raw_prediction(va['f0'], va)
        for seed in CFG['seeds']:
            for arm in CFG['arms']:
                for recipe, lr in enumerate(CFG['learning_rates'][arm]):
                    watch.check('fit_start', True)
                    seed_all(seed)
                    m = ForecastModel('standard', seed) if arm == 'lora' else SplitAdapter(seed, arm == 'center').cuda()
                    trainable = [p for p in m.parameters() if p.requires_grad]
                    if arm == 'lora':
                        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.)
                        shape_opt = None
                    else:
                        opt = torch.optim.AdamW(m.center.parameters(), lr=lr, weight_decay=0.)
                        shape_opt = None if arm == 'center' else torch.optim.AdamW(m.shape.parameters(), lr=lr, weight_decay=0.)
                    fid = f'{name}_{seed}_{arm}_r{recipe}'
                    tick = time.monotonic()
                    best, best_blend, steps = None, None, []
                    for step in range(CFG['steps']+1):
                        if step:
                            watch.check(fid)
                            idx = schedules[str(seed)][step-1]
                            r = lora_step(m, opt, values, [TRAIN[i] for i in idx]) if arm == 'lora' else split_step(m, opt, shape_opt, subset(tr, idx), tr, arm)
                            steps.append(r)
                            guard(tick)
                        if step in CFG['checkpoints']:
                            p = predict(m, arm, va, values, V)
                            met = metric(p, va, scale)
                            pred_tag = f'V_{fid}_{step}'
                            pred_sha = store_predictions(pred_tag, p, va, scale)
                            row = dict(fit=fid, dataset=name, seed=seed, arm=arm, recipe=recipe, step=step,
                                       metrics=met, prediction_tag=pred_tag, prediction_sha256=pred_sha)
                            trajectories.append(row)
                            if best is None or (met['scaled_2pinball'], step) < (best['metrics']['scaled_2pinball'], best['step']):
                                best = row
                                torch.save(save_state(m), CACHE/(fid+'_best.pt'))
                            if arm != 'block':
                                # Stronger baseline budget is deliberate: all checkpoints and three blend weights.
                                for alpha in CFG['blend_alpha']:
                                    bm = metric(alpha*np.sort(p, axis=2)+(1-alpha)*f0_v, va, scale)
                                    br = dict(**{k: row[k] for k in ['fit', 'dataset', 'seed', 'arm', 'recipe', 'step']},
                                              alpha=alpha, metrics=bm)
                                    blend_trajectories.append(br)
                                    if best_blend is None or (bm['scaled_2pinball'], step, alpha) < (best_blend['metrics']['scaled_2pinball'], best_blend['step'], best_blend['alpha']):
                                        best_blend = br
                                        torch.save(save_state(m), CACHE/(fid+'_blend.pt'))
                            write_json(OUT/'trajectories.json', trajectories)
                            write_json(OUT/'blend_trajectories.json', blend_trajectories)
                    record = dict(fit=fid, dataset=name, seed=seed, arm=arm, recipe=recipe, learning_rate=lr,
                                  best=best, best_blend=best_blend, steps=CFG['steps'], step_resources=steps,
                                  trainable_parameters=sum(p.numel() for p in trainable),
                                  wall_seconds=time.monotonic()-tick, median_step_seconds=float(np.median([r['seconds'] for r in steps])),
                                  peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in steps),
                                  checkpoint_sha256=sha(CACHE/(fid+'_best.pt')),
                                  blend_checkpoint_sha256=sha(CACHE/(fid+'_blend.pt')) if best_blend else None,
                                  accepted_shape_steps=sum(r['gate']['accept'] for r in steps if r['gate'] is not None))
                    fits.append(record)
                    write_json(OUT/'fits.json', fits)
                    write_json(OUT/'status.json', dict(status='TRAINING', completed_fits=len(fits), total_fits=48))
                    print('FIT COMPLETE', len(fits), '/', 48, fid, 'best', best['step'], 'V', best['metrics']['scaled_2pinball'], 'shape_accept', record['accepted_shape_steps'], flush=True)
                    del m, opt, shape_opt, trainable
                    gc.collect()
                    torch.cuda.empty_cache()
    assert len(fits) == CFG['fit_cap'] and source_hashes() == contract['source_hashes']
    selections, calibrations, blends = [], [], []
    for name in CFG['datasets']:
        va = feature_sets[name][1]
        scale = data[name][1]
        choices = []
        for shift in CFG['calibration_shift']:
            for width in CFG['calibration_width']:
                center = va['f0'][:, 10:11]
                pred = raw_prediction(center+shift+width*(va['f0']-center), va)
                choices.append(dict(dataset=name, shift=shift, width=width, metrics=metric(pred, va, scale)))
        calibrations.append(min(choices, key=lambda r: (r['metrics']['scaled_2pinball'], abs(r['shift']), abs(r['width']-1))))
        for seed in CFG['seeds']:
            pool = [f for f in fits if f['dataset']==name and f['seed']==seed]
            for arm in CFG['arms']:
                r = min([f for f in pool if f['arm']==arm], key=lambda f: (f['best']['metrics']['scaled_2pinball'], f['best']['step'], f['recipe']))
                selections.append(dict(**r['best'], checkpoint_sha256=r['checkpoint_sha256']))
            r = min([f for f in pool if f['best_blend']], key=lambda f: (f['best_blend']['metrics']['scaled_2pinball'], f['best_blend']['step'], f['best_blend']['alpha'], f['arm'], f['recipe']))
            blends.append(dict(**r['best_blend'], checkpoint_sha256=r['blend_checkpoint_sha256']))
    seal = dict(selections=selections, calibrations=calibrations, blends=blends,
                contract_sha256=sha(OUT/'contract.json'), source_hashes=source_hashes())
    seal['seal_sha256'] = digest(seal)
    write_json(OUT/'selection_seal.json', seal)
    write_json(OUT/'status.json', dict(status='SEALED_EVALUATION', completed_fits=48))
    del feature_sets, data
    gc.collect()
    torch.cuda.empty_cache()
    for name in CFG['datasets']:
        assert sha(CACHE/(name+'_heldout.npz')) == contract['data'][name]['heldout_sha256']
        with np.load(CACHE/(name+'_heldout.npz')) as f:
            values, scale = f['values'], f['scale']
        base = load_base()
        ev = extract(base, values, E, name+'_E')
        del base
        gc.collect()
        torch.cuda.empty_cache()
        f0 = raw_prediction(ev['f0'], ev)
        def add(arm, seed, pred, **details):
            tag = f'E_{name}_{seed}_{arm}'
            evaluation.append(dict(dataset=name, seed=seed, arm=arm, metrics=metric(pred, ev, scale),
                                   prediction_tag=tag, prediction_sha256=store_predictions(tag, pred, ev, scale), **details))
            write_json(OUT/'evaluation.json', evaluation)
        add('F0', None, f0)
        c = next(r for r in calibrations if r['dataset']==name)
        med = ev['f0'][:, 10:11]
        add('calibration', None, raw_prediction(med+c['shift']+c['width']*(ev['f0']-med), ev), shift=c['shift'], width=c['width'])
        for seed in CFG['seeds']:
            for arm in CFG['arms']+['blend']:
                choice = next(r for r in (blends if arm=='blend' else selections) if r['dataset']==name and r['seed']==seed and (arm=='blend' or r['arm']==arm))
                actual_arm = choice['arm']
                m = ForecastModel('standard', seed) if actual_arm=='lora' else SplitAdapter(seed, actual_arm=='center').cuda()
                path = CACHE/(choice['fit']+('_blend.pt' if arm=='blend' else '_best.pt'))
                assert sha(path)==choice['checkpoint_sha256']
                restore(m, torch.load(path, weights_only=True))
                pred = predict(m, actual_arm, ev, values, E)
                if arm=='blend':
                    pred = choice['alpha']*np.sort(pred, axis=2)+(1-choice['alpha'])*f0
                shape_change = None
                if actual_arm!='lora':
                    with torch.no_grad():
                        shape_change = float(m(ev['h'], ev['f0'])[1].abs().mean())
                add(arm, seed, pred, fit=choice['fit'], step=choice['step'], shape_log_multiplier_abs_mean=shape_change)
                del m
                gc.collect()
                torch.cuda.empty_cache()
        del ev
    watch.check('complete', True)
    write_json(OUT/'status.json', dict(status='COMPLETE', completed_fits=48, proposed_training_iterations=5760,
                                      wall_seconds=time.monotonic()-started, expanded=False))
    print('ALL 48 FITS AND SEALED EVALUATION COMPLETE', flush=True)

fits, trajectories, blend_trajectories, evaluation, cache_receipts, smokes = [], [], [], [], [], []
if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            write_json(OUT/'status.json', dict(status='BLOCKED', completed_fits=len(fits), error=str(exc), traceback=traceback.format_exc()))
        raise
