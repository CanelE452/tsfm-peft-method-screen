"""Preregistered 24-step 2x2 resource diagnostic; never opens V/E."""
import copy
import fcntl
import gc
import hashlib
import json
import random
import signal
import subprocess
import time
import traceback
from contextlib import nullcontext

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from tsfm_peft_screen.backbone import native_loss, REVISION
from tsfm_peft_screen.forecast_query.model import ForecastModel
from tsfm_peft_screen.forecast_query.checkpoint_diagnostic import DiagnosticModel, StorageObserver
from tsfm_peft_screen.forecast_query.gpu import GPUWatch
from tsfm_peft_screen.memory.common import batch
from tsfm_peft_screen.reproducibility import ROOT, sha, write_json, source_hashes, seed_all, guard

CFG = json.loads((ROOT / 'configs/forecast_query_checkpoint_diagnostic.json').read_text())
OUT = ROOT / 'results/forecast_query_checkpoint_diagnostic'
CACHE = ROOT / '.cache/forecast_query_checkpoint_diagnostic'


def cpu_tree(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(cpu_tree(v) for v in obj)
    return copy.deepcopy(obj)


def tree_hash(obj):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            h.update(str((tuple(v.shape), str(v.dtype))).encode())
            h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(str((v.shape, v.dtype)).encode())
            h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode())
                visit(v[k])
        elif isinstance(v, (tuple, list)):
            for x in v:
                visit(x)
        else:
            h.update(repr(v).encode())
    visit(obj)
    return h.hexdigest()


def params(m):
    return {n: p for n, p in m.named_parameters() if p.requires_grad}


def rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def snapshot(m, opt):
    return dict(parameters=cpu_tree(params(m)), optimizer=cpu_tree(opt.state_dict()), rng=cpu_tree(rng()))


def restore(m, opt, state):
    ps = params(m)
    assert ps.keys() == state['parameters'].keys()
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for n, p in ps.items():
            p.copy_(state['parameters'][n])
    opt.load_state_dict(copy.deepcopy(state['optimizer']))
    random.setstate(state['rng']['python'])
    np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['cpu'])
    torch.cuda.set_rng_state_all(state['rng']['cuda'])
    gc.collect()
    torch.cuda.empty_cache()


def flatten(ps):
    return torch.cat([p.reshape(-1) for _, p in sorted(ps.items())])


def delta(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return dict(max_absolute=float((a-b).abs().max()),
                relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-30)))


def compare(a, b, precision):
    tol = CFG['tolerances'][precision]
    rows = {k: delta(a[k], b[k]) for k in ('z', 'raw', 'loss', 'gradient', 'update', 'adam')}
    passed = all(r['max_absolute'] <= tol['max_absolute'] and r['relative_l2'] <= tol['relative_l2'] for r in rows.values())
    norm_difference = abs(a['gradient_norm'] - b['gradient_norm'])
    rng_equal = a['rng_after_sha256'] == b['rng_after_sha256']
    return dict(metrics=rows, gradient_norm_abs_difference=norm_difference, rng_equal=rng_equal,
                passed=passed and rng_equal and norm_difference <= tol['max_absolute'])


def step(m, opt, data, precision, start_state, export=True, observer=None, update=True):
    x, y, g = data
    ps = params(m)
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_allocated = torch.cuda.memory_allocated()
    tick = time.perf_counter()
    m.observer = observer
    ctx = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
    with ctx:
        z, raw, loc, scale = m(x, g)
        with observer.phase('loss') if observer else nullcontext():
            loss = native_loss(z, y, loc, scale)
    with observer.phase('backward') if observer else nullcontext():
        loss.backward()
    with observer.phase('clip') if observer else nullcontext():
        norm = torch.nn.utils.clip_grad_norm_(list(ps.values()), 1.)
    if update:
        opt.step()
    torch.cuda.synchronize()
    resource = dict(seconds=time.perf_counter()-tick, start_allocated_bytes=start_allocated,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), loss=float(loss.detach()),
                    gradient_norm=float(norm))
    # CPU exports and finite checks are outside the timed/peak measurement.
    assert torch.isfinite(loss) and torch.isfinite(raw).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps.values())
    assert all(torch.isfinite(p).all() for p in ps.values())
    m.observer = None
    if not export:
        return resource, None
    now = cpu_tree(ps)
    adam = torch.cat([v.detach().cpu().reshape(-1) for _, s in sorted(opt.state_dict()['state'].items())
                      for key, v in sorted(s.items()) if isinstance(v, torch.Tensor)])
    artifact = dict(z=z.detach().cpu(), raw=raw.detach().cpu(), loss=loss.detach().cpu().reshape(1),
                    gradient=flatten({n: p.grad.detach().cpu() for n, p in ps.items()}),
                    update=flatten({n: now[n] - start_state['parameters'][n] for n in now}),
                    adam=adam, gradient_norm=float(norm), rng_after_sha256=tree_hash(rng()))
    assert artifact['update'].abs().sum() > 0
    return resource, artifact


def old_forward_parity(m, data, precision):
    x, _, g = data
    ctx = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
    # Historical standard has CP on, query CP off. No backward in this check.
    with torch.no_grad(), ctx:
        new = m(x, g)
        if m.arm == 'query':
            saved = m.query_hidden
            m.query_hidden = lambda c: ForecastModel.query_hidden(m, c)
            try:
                old = ForecastModel.forward(m, x, g)
            finally:
                m.query_hidden = saved
        else:
            old = ForecastModel.forward(m, x, g)
    rows = [delta(a, b) for a, b in zip(old, new)]
    tol = CFG['tolerances'][precision]
    assert all(r['relative_l2'] <= tol['relative_l2'] and r['max_absolute'] <= tol['max_absolute'] for r in rows)
    return rows


def main():
    assert not OUT.exists() and not CACHE.exists(), 'Immutable diagnostic: output already exists'
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip(), 'Commit before GPU run'
    lock = open(ROOT / '.cache/gpu.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def interrupted(signum, frame):
        raise TimeoutError(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    OUT.mkdir()
    CACHE.mkdir()
    counts = dict(warmup_optimizer_updates=0, measured_optimizer_updates=0,
                  fp32_equivalence_optimizer_updates=0, instrumented_backward_only=0, fits=0, evaluation_accesses=0)
    records, equivalence, compatibility, profiles = [], [], [], []
    start = time.monotonic()
    old_hashes = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/'results').rglob('*')
                  if p.is_file() and OUT not in p.parents}
    watch = GPUWatch(OUT/'gpu_monitor.json')
    contract = dict(execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    source_hashes=source_hashes(), runner_sha256=sha(__file__),
                    protocol_sha256=sha(ROOT/'docs/FORECAST_QUERY_CHECKPOINT_DIAGNOSTIC.md'),
                    config=CFG, model_revision=REVISION, torch_version=torch.__version__,
                    fit_data_hashes={n:sha(ROOT/'data/processed'/n/'fit.npz') for n in CFG['datasets']},
                    historical_result_hashes=old_hashes, new_fits=0, evaluation_accesses=0)
    model_files = {}
    expected = json.loads((ROOT/'results/screening_summary/common_integrity.json').read_text())['model_files']
    for filename, expected_hash in expected.items():
        path = hf_hub_download('amazon/chronos-2', filename, revision=REVISION, local_files_only=True)
        model_files[filename] = sha(path)
        assert model_files[filename] == expected_hash
    contract['model_files'] = model_files
    write_json(OUT/'contract.json', contract)

    def checkpoint_status(status, **extra):
        write_json(OUT/'status.json', dict(status=status, counts=counts, **extra))

    def boundary(phase):
        if time.monotonic()-start > CFG['total_timeout_seconds']:
            raise TimeoutError('Total diagnostic timeout')
        watch.check(phase, force=True)
        guard(start)

    try:
        idle = None
        while True:
            row = watch.read('startup_wait')
            good = not row['external_pids'] and row['free_mib'] >= 4096 and row['utilization_percent'] < 90
            if good:
                idle = idle or time.monotonic()
                if time.monotonic()-idle >= 30:
                    break
            else:
                idle = None
            if time.monotonic()-start > CFG['startup_timeout_seconds']:
                raise TimeoutError('GPU startup wait expired')
            time.sleep(5)
        seed_all(CFG['seed'])
        shared = None
        for arm in CFG['arms']:
            boundary('load_'+arm)
            m = DiagnosticModel(arm, CFG['seed'])
            opt = torch.optim.AdamW(list(params(m).values()), lr=CFG['lr'], weight_decay=0.)
            frozen_before = tree_hash({n:p for n,p in m.named_parameters() if not p.requires_grad})
            contract.setdefault('frozen_parameter_sha256', {})[arm] = frozen_before
            if shared is None:
                data = batch('ettm2', CFG['context'], CFG['origins'])
                for _ in range(CFG['shared_state_warmup_updates']):
                    boundary('shared_warmup')
                    step(m, opt, data, 'bf16', None, export=False)
                    counts['warmup_optimizer_updates'] += 1
                shared = snapshot(m, opt)
                torch.save(shared, CACHE/'shared_state.pt')
                contract['shared_state_sha256'] = sha(CACHE/'shared_state.pt')
                contract['shared_state_content_sha256'] = tree_hash(shared)
                del data
            write_json(OUT/'contract.json', contract)
            for cp in CFG['checkpoint']:
                m.checkpoint_enabled = cp
                restore(m, opt, shared)
                data = batch('ettm2', CFG['context'], CFG['origins'])
                boundary('kernel_warmup')
                step(m, opt, data, 'bf16', shared, export=False)
                counts['warmup_optimizer_updates'] += 1
                del data
            for dataset in CFG['datasets']:
                data = batch(dataset, CFG['context'], CFG['origins'])
                # Precision-specific compatibility and CP equality; FP32 separate from timing.
                for precision in ('fp32', 'bf16'):
                    restore(m, opt, shared)
                    m.checkpoint_enabled = False
                    boundary('historical_forward_parity')
                    compatibility.append(dict(arm=arm, dataset=dataset, precision=precision,
                                              differences=old_forward_parity(m, data, precision)))
                for precision, repeats in (('fp32', 1), ('bf16', CFG['repeats'])):
                    for repeat in range(repeats):
                        pair = {}
                        order = [False, True] if repeat % 2 == 0 else [True, False]
                        for cp in order:
                            m.checkpoint_enabled = cp
                            restore(m, opt, shared)
                            initial_hash = tree_hash(snapshot(m, opt))
                            assert initial_hash == contract['shared_state_content_sha256']
                            boundary('trial_'+arm)
                            resource, artifact = step(m, opt, data, precision, shared)
                            key = 'measured_optimizer_updates' if precision == 'bf16' else 'fp32_equivalence_optimizer_updates'
                            counts[key] += 1
                            tag = f'{arm}_{int(cp)}_{dataset}_{precision}_{repeat}'
                            path = CACHE/(tag+'.pt')
                            torch.save(artifact, path)
                            pair[cp] = artifact
                            records.append(dict(tag=tag, arm=arm, checkpoint=cp, dataset=dataset,
                                                precision=precision, repeat=repeat, initial_state_sha256=initial_hash,
                                                cache_sha256=sha(path), **resource))
                            write_json(OUT/'measurements.json', records)
                            checkpoint_status('RUNNING')
                            boundary('after_trial')
                            print('TRIAL', tag, 'MiB', round(resource['peak_allocated_bytes']/2**20,2),
                                  'seconds', round(resource['seconds'],4), flush=True)
                        eq = dict(arm=arm, dataset=dataset, precision=precision, repeat=repeat,
                                  **compare(pair[False], pair[True], precision))
                        equivalence.append(eq)
                        write_json(OUT/'equivalence.json', equivalence)
                        assert eq['passed'], ('CP numerical equivalence failed', eq)
                        del pair, artifact
                del data
            for cp in CFG['checkpoint']:
                m.checkpoint_enabled = cp
                restore(m, opt, shared)
                data = batch('ettm2', CFG['context'], CFG['origins'])
                boundary('storage_inventory')
                obs = StorageObserver(m)
                with torch.autograd.graph.saved_tensors_hooks(obs.pack, lambda t: t):
                    step(m, opt, data, 'bf16', shared, export=False, observer=obs, update=False)
                counts['instrumented_backward_only'] += 1
                profiles.append(dict(arm=arm, checkpoint=cp, **obs.result()))
                write_json(OUT/'storage_profiles.json', profiles)
                del obs, data
            assert tree_hash({n:p for n,p in m.named_parameters() if not p.requires_grad}) == frozen_before
            del m, opt
            gc.collect()
            torch.cuda.empty_cache()
        assert counts == dict(warmup_optimizer_updates=6, measured_optimizer_updates=24,
                              fp32_equivalence_optimizer_updates=8, instrumented_backward_only=4, fits=0, evaluation_accesses=0)
        assert source_hashes() == contract['source_hashes']
        assert all(sha(ROOT/p) == h for p,h in old_hashes.items())
        write_json(OUT/'historical_forward_parity.json', compatibility)
        boundary('complete')
        checkpoint_status('COMPLETE', wall_seconds=time.monotonic()-start,
                          historical_results_unchanged=True, numerical_equivalence_passed=True,
                          verdict='DIAGNOSTIC_ONLY', expanded=False)
    except Exception as e:
        checkpoint_status('IMPLEMENTATION_BLOCKED', error=str(e), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
