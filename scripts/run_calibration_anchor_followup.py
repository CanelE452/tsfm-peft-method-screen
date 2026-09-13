"""Isolated train/evaluate worker. All topics seal V selections before any E."""
import argparse
import gc
import gzip
import json
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tsfm_peft_screen.backbone import native_loss
from tsfm_peft_screen.forecast_query.gpu import GPUWatch
from tsfm_peft_screen.forecast_query.equal_time import preserve_rng
from tsfm_peft_screen.overnight.methods import raw_loss, mixture_quantiles, teacher_weights
from tsfm_peft_screen.calibration_anchor.method import (
    TOPICS, PROPOSED, PilotModel, calibration_weights, shuffled_weights,
    weighted_pinball, verdict_from_rows,
)
from tsfm_peft_screen.reproducibility import ROOT, sha, digest, write_json, seed_all, guard
from tsfm_peft_screen.metrics import score

CFG = json.loads((ROOT/'configs/calibration_anchor_20260914.json').read_text())
OUT = ROOT/'results'/CFG['run_id']
CACHE = ROOT/'.cache'/CFG['run_id']


def oo(split):
    start, stop, stride = CFG[split+'_origins']
    return list(range(start, stop+1, stride))


def data(dataset, heldout=False):
    p = CACHE/'data'/(dataset+('_evaluation.npz' if heldout else '_development.npz'))
    receipt = json.loads((OUT/'contract.json').read_text())['staged_data'][dataset]
    assert sha(p) == receipt['evaluation_sha256' if heldout else 'development_sha256']
    with np.load(p, allow_pickle=False) as f:
        return f['values'], f['scale']


def batch(values, origins, context=1024):
    assert min(origins) >= context and max(origins)+48 <= len(values)
    x = np.concatenate([values[o-context:o].T for o in origins]).astype(np.float32)
    y = np.concatenate([values[o:o+48].T for o in origins]).astype(np.float32)
    return (torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda(),
            torch.arange(len(origins), device='cuda').repeat_interleave(4))


def trainable(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def parameters(model):
    return {n: p.detach().cpu().clone() for n, p in trainable(model).items()}


def restore(model, weights):
    ps = trainable(model)
    assert ps.keys() == weights.keys()
    with torch.no_grad():
        for name, value in weights.items():
            ps[name].copy_(value)


def tensor_hash(items):
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(items.items()):
        v = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((v.shape, v.dtype)).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def frozen_hash(model):
    return tensor_hash({n: p for n, p in model.named_parameters() if not p.requires_grad})



def objective(arm, z, p, y, loc, model_scale, channel_scale, ps, initial_gpu, teacher, indices):
    loss = native_loss(z,y,loc,model_scale)/21 if arm=='native' else raw_loss(p,y,channel_scale)
    regularizer = torch.zeros((),device=p.device)
    if arm=='l2_anchor':
        regularizer = CFG['l2_anchor_coefficient']*sum((v-initial_gpu[n]).square().sum() for n,v in ps.items())
    if arm in ('weighted_loss','shuffled_anchor','calibration_anchor'):
        weight = teacher['train']['calibration_weights']
        if arm=='shuffled_anchor':
            weight = shuffled_weights(weight)
        weight = torch.tensor(np.tile(weight,(len(indices),1,1)),device=p.device)
        if arm=='weighted_loss':
            loss = weighted_pinball(p,y,channel_scale,weight)
    else:
        weight = torch.ones_like(p)
    if arm in ('full_anchor','shuffled_anchor','calibration_anchor'):
        target = torch.tensor(teacher['train']['prediction'][indices].reshape(len(p),21,48),device=p.device)
        regularizer = CFG['prediction_anchor_coefficient']*(
            (p.sort(dim=1).values-target.sort(dim=1).values).abs()/channel_scale[:,None,None]*weight).mean()
    return loss, regularizer


def check_contract():
    c = json.loads((OUT/'contract.json').read_text())
    for f, value in c['source_hashes'].items():
        assert sha(ROOT/f) == value, f
    for f, value in c['historical_result_hashes'].items():
        assert sha(ROOT/f) == value, f
    for f, value in c['model_files'].items():
        from huggingface_hub import hf_hub_download
        from tsfm_peft_screen.backbone import MODEL_ID, REVISION
        assert sha(hf_hub_download(MODEL_ID, f, revision=REVISION, local_files_only=True)) == value
    return c


class Worker:
    def __init__(self, topic, phase):
        self.topic, self.phase = topic, phase
        self.out, self.cache = OUT/topic, CACHE/topic
        self.out.mkdir(exist_ok=True)
        self.cache.mkdir(exist_ok=True)
        self.start = time.monotonic()
        self.watch = GPUWatch(self.out/(phase+'_gpu.json'))
        self.state = dict(topic=topic, phase=phase, status='WAITING_FOR_GPU',
                          fits_attempted=0, fits_completed=0, training_updates=0)
        self.emit()

    def emit(self, **kw):
        self.state.update(kw)
        self.state['wall_seconds'] = time.monotonic()-self.start
        write_json(self.out/(self.phase+'_status.json'), self.state)

    def check(self, phase, force=False):
        if time.monotonic()-self.start > CFG['max_phase_wall_seconds']:
            raise TimeoutError('Phase wall cap')
        self.watch.check(phase, force)
        guard()

    def wait(self):
        idle = None
        while True:
            if time.monotonic()-self.start > CFG['max_phase_wall_seconds']:
                raise TimeoutError('GPU wait exhausted phase cap')
            r = self.watch.read('startup_wait')
            good = (not r['external_pids'] and r['free_mib'] >= CFG['startup_free_mib']
                    and r['utilization_percent'] < CFG['startup_utilization_below_percent'])
            idle = (idle or time.monotonic()) if good else None
            self.emit(status='WAITING_FOR_GPU', external_pids=r['external_pids'], free_mib=r['free_mib'])
            if idle is not None and time.monotonic()-idle >= CFG['startup_idle_seconds']:
                return
            time.sleep(5)

    def evaluate(self, model, values, scale, origins, tag, frozen=False, context=1024):
        predictions, targets = [], []
        start = time.monotonic()
        with preserve_rng(), torch.no_grad():
            for j in range(0, len(origins), 2):
                self.check('evaluate_'+tag)
                x, y, g = batch(values, origins[j:j+2], context)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    _, p, _, _ = model(x, g, frozen=frozen)
                assert torch.isfinite(p).all()
                predictions.append(p.cpu().numpy().reshape(-1, 4, 21, 48))
                targets.append(y.cpu().numpy().reshape(-1, 4, 48))
        p, y = np.concatenate(predictions), np.concatenate(targets)
        path = self.cache/(tag+'.npz')
        assert not path.exists(), path
        np.savez_compressed(path, prediction=p, target=y, scale=scale)
        return dict(metrics=score(p, y, scale), prediction_file=str(path.relative_to(CACHE)),
                    prediction_sha256=sha(path), inference_wall_seconds=time.monotonic()-start,
                    evaluated_origins=len(origins))

    def cached_prediction(self, row):
        with np.load(CACHE/row['prediction_file'], allow_pickle=False) as z:
            return z['prediction']

    def teacher_setup(self, dataset, model, values, scale):
        start = time.monotonic()
        lengths = CFG['teacher_contexts'] if self.topic == 'distill' else [CFG['context']]
        records, arrays = {}, {}
        for split in ('train', 'validation'):
            records[split], parts = [], []
            for context in lengths:
                row = self.evaluate(model, values, scale, oo(split),
                                    f'{dataset}_teacher_{split}_{context}', True, context)
                row['context'] = context
                records[split].append(row)
                parts.append(self.cached_prediction(row))
            components = np.stack(parts)
            if self.topic == 'distill':
                prediction = mixture_quantiles(components, CFG['teacher_samples_per_component'])
                weights = teacher_weights(components, scale)
            else:
                prediction = components[0]
                weights = np.ones((*prediction.shape[:2], 48), dtype=np.float32)
            arrays[split] = dict(prediction=prediction, weights=weights)
            if self.topic == 'distill':
                y = np.stack([values[o:o+48].T for o in oo(split)])
                path = self.cache/f'{dataset}_teacher_{split}_mixture.npz'
                np.savez_compressed(path, prediction=prediction, target=y, scale=scale, weights=weights)
                records[split].append(dict(context='mixture', metrics=score(prediction, y, scale),
                    prediction_file=str(path.relative_to(CACHE)), prediction_sha256=sha(path)))
        targets = np.stack([values[o:o+48].T for o in oo('train')])
        weights, calibration_receipt = calibration_weights(arrays['train']['prediction'],targets)
        arrays['train']['calibration_weights'] = weights
        weight_path = self.cache/f'{dataset}_calibration_weights.npz'
        np.savez_compressed(weight_path,weights=weights)
        records['calibration'] = dict(**calibration_receipt,
            weights_file=str(weight_path.relative_to(CACHE)),weights_sha256=sha(weight_path))
        records['wall_seconds'] = time.monotonic()-start
        return records, arrays

    def fit(self, dataset, arm, seed, lr, recipe, values, scale, teacher):
        fid = f'{dataset}_{seed}_{arm}_{recipe}'
        self.state['fits_attempted'] += 1
        assert self.state['fits_attempted'] <= CFG['max_fits_per_topic']
        self.emit(status='TRAINING', dataset=dataset, fit_id=fid, step=0)
        write_json(self.out/(fid+'_attempt.json'), dict(fit_id=fid, started_unix=time.time()))
        self.check('model_load', True)
        seed_all(seed)
        model = PilotModel(arm, seed)
        ps = trainable(model)
        init = parameters(model)
        initial_hash = tensor_hash(init)
        original_frozen = frozen_hash(model)
        optimizer = torch.optim.AdamW(ps.values(), lr=lr, weight_decay=0.)
        scale_tensor = torch.tensor(np.tile(scale, 2), device='cuda', dtype=torch.float32)
        initial_gpu = {n: v.cuda() for n, v in init.items()} if arm == 'l2_anchor' else None
        rng = np.random.default_rng(seed)
        train_origins = oo('train')
        trajectories, resources = [], []
        active_seconds = 0.
        fit_start = time.monotonic()
        sample_sequence = []
        for step in range(CFG['checkpoints'][-1]+1):
            if step in CFG['checkpoints']:
                tag = fid+f'_V_{step}'
                row = self.evaluate(model, values, scale, oo('validation'), tag)
                path = self.cache/(tag+'.pt')
                torch.save(parameters(model), path)
                row.update(dataset=dataset, arm=arm, seed=seed, lr=lr, recipe=recipe, step=step,
                    fit_id=fid, checkpoint_file=str(path.relative_to(CACHE)), checkpoint_sha256=sha(path))
                trajectories.append(row)
                write_json(self.out/(fid+'_trajectory.json'), trajectories)
                print('V', fid, step, row['metrics']['scaled_2pinball'], flush=True)
            if step == CFG['checkpoints'][-1]:
                break
            self.check('training_'+fid)
            if time.monotonic()-fit_start > CFG['max_fit_wall_seconds']:
                raise TimeoutError('Fit wall cap, no implicit retry')
            indices = rng.integers(len(train_origins), size=2)
            selected = [train_origins[i] for i in indices]
            sample_sequence.append(selected)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            x, y, g = batch(values, selected)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z, p, loc, s = model(x, g)
            loss, regularizer = objective(arm, z, p, y, loc, s, scale_tensor,
                                          ps, initial_gpu, teacher, indices)
            total = loss+regularizer
            assert torch.isfinite(total) and torch.isfinite(p).all()
            total.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in ps.values())
            norm = torch.nn.utils.clip_grad_norm_(list(ps.values()), 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            active_seconds += elapsed
            resources.append(dict(step=step+1, seconds=elapsed, loss=float(loss.detach()),
                regularizer=float(regularizer.detach()), gradient_norm=float(norm),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            self.state['training_updates'] += 1
            if (step+1) % 25 == 0:
                self.emit(step=step+1)
        assert original_frozen == frozen_hash(model)
        assert initial_hash != tensor_hash(parameters(model)), 'Parameters did not change'
        winner = min(trajectories, key=lambda r: (r['metrics']['scaled_2pinball'], r['step'], r['lr']))
        result = dict(fit_id=fid, dataset=dataset, arm=arm, seed=seed, lr=lr, recipe=recipe,
            updates=len(resources), active_seconds=active_seconds, wall_seconds=time.monotonic()-fit_start,
            trainable_parameters=sum(p.numel() for p in ps.values()), frozen_sha256=original_frozen,
            initial_parameters_sha256=initial_hash, final_parameters_sha256=tensor_hash(parameters(model)),
            sample_sequence_sha256=digest(sample_sequence), best=winner, resources=resources)
        write_json(self.out/(fid+'_fit.json'), result)
        self.state['fits_completed'] += 1
        self.emit()
        del model, optimizer, ps, init, initial_gpu
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def train(self):
        assert not (self.out/'selection_seal.json').exists(), 'No repeat of a sealed topic'
        check_contract()
        self.wait()
        fits, teacher_records, teacher_arrays = [], {}, {}
        for dataset in CFG['datasets']:
            values, scale = data(dataset)
            if self.topic == 'calibration':
                seed_all(CFG['seeds'][0])
                model = PilotModel('raw', CFG['seeds'][0])
                self.emit(status='TEACHER_CACHE', dataset=dataset)
                teacher_records[dataset], teacher_arrays[dataset] = self.teacher_setup(dataset, model, values, scale)
                write_json(self.out/'teachers.json', teacher_records)
                del model
                gc.collect()
                torch.cuda.empty_cache()
        # Distillation first fits its ordinary raw-loss baseline on both datasets.
        # Teacher gate depends on these V scores, never on another topic's E.
        arms_first = ['raw'] if self.topic == 'distill' else TOPICS[self.topic]
        for dataset in CFG['datasets']:
            values, scale = data(dataset)
            for seed in CFG['seeds']:
                for arm in arms_first:
                    for recipe, lr in enumerate(CFG['learning_rates']):
                        fits.append(self.fit(dataset, arm, seed, lr, recipe, values, scale, teacher_arrays.get(dataset)))
        headroom = None
        if self.topic == 'distill':
            headroom = []
            for dataset in CFG['datasets']:
                records = teacher_records[dataset]['validation']
                mixture = next(r['metrics']['scaled_2pinball'] for r in records if r['context'] == 'mixture')
                best_component = min(r['metrics']['scaled_2pinball'] for r in records if r['context'] != 'mixture')
                raw_mean = float(np.mean([min(f['best']['metrics']['scaled_2pinball'] for f in fits
                    if f['dataset'] == dataset and f['seed'] == seed) for seed in CFG['seeds']]))
                ref = min(best_component, raw_mean)
                headroom.append(dict(dataset=dataset, teacher=mixture, best_component=best_component,
                    raw_lora_mean=raw_mean, gain=1-mixture/ref,
                    passed=mixture <= (1-CFG['teacher_headroom_gain'])*ref))
            write_json(self.out/'teacher_headroom.json', headroom)
            if all(r['passed'] for r in headroom):
                for dataset in CFG['datasets']:
                    values, scale = data(dataset)
                    for seed in CFG['seeds']:
                        for arm in TOPICS[self.topic][1:]:
                            for recipe, lr in enumerate(CFG['learning_rates']):
                                fits.append(self.fit(dataset, arm, seed, lr, recipe, values, scale, teacher_arrays[dataset]))
        selections = []
        for key in sorted({(f['dataset'], f['seed'], f['arm']) for f in fits}):
            pool = [f['best'] for f in fits if (f['dataset'], f['seed'], f['arm']) == key]
            selections.append(min(pool, key=lambda r: (r['metrics']['scaled_2pinball'], r['step'], r['lr'])))
        # Keep the indexed manifest small; complete step ledgers remain in
        # individual immutable receipts with a hash in this manifest.
        manifest = []
        for fit in fits:
            row = {k:v for k,v in fit.items() if k != 'resources'}
            row['resource_receipt_sha256'] = sha(self.out/(fit['fit_id']+'_fit.json'))
            manifest.append(row)
        write_json(self.out/'fits.json', manifest)
        stopped = headroom is not None and not all(r['passed'] for r in headroom)
        seal = dict(topic=self.topic, selections=selections, contract_sha256=sha(OUT/'contract.json'),
                    sealed_unix=time.time(), decision='STOP_NO_TEACHER_HEADROOM' if stopped else 'READY_FOR_E',
                    fits_sha256=sha(self.out/'fits.json'))
        seal['seal_sha256'] = digest(seal)
        write_json(self.out/'selection_seal.json', seal)
        check_contract()
        self.emit(status=seal['decision'])
        return seal

    def evaluate_selected(self):
        check_contract()
        # The supervisor seals every topic's terminal training outcome first.
        barrier = json.loads((OUT/'evaluation_barrier.json').read_text())
        checksum = barrier.pop('sha256')
        assert checksum == digest(barrier)
        assert barrier['contract_sha256'] == sha(OUT/'contract.json')
        assert set(barrier['topics']) == set(CFG['topics'])
        for topic, entry in barrier['topics'].items():
            for f, h in entry['receipts'].items():
                assert sha(OUT/topic/f) == h
        seal = json.loads((self.out/'selection_seal.json').read_text())
        checksum = seal.pop('seal_sha256')
        assert checksum == digest(seal)
        assert barrier['sealed_unix'] >= seal['sealed_unix']
        assert seal['contract_sha256'] == sha(OUT/'contract.json')
        assert seal['decision'] == 'READY_FOR_E'
        assert sha(self.out/'fits.json') == seal['fits_sha256']
        self.wait()
        rows, reloads = [], []
        for dataset in CFG['datasets']:
            # This is the first target read/scoring for this topic's E.
            values, scale = data(dataset, heldout=True)
            self.emit(status='EVALUATING', dataset=dataset, evaluation_open_unix=time.time())
            seed_all(CFG['seeds'][0])
            model = PilotModel('raw', CFG['seeds'][0])
            row = self.evaluate(model, values, scale, oo('evaluation'), dataset+'_F0_E', True)
            rows.append(dict(dataset=dataset, arm='F0', seed=None, **row))
            if self.topic == 'distill':
                component_rows = []
                for context in CFG['teacher_contexts']:
                    r = self.evaluate(model, values, scale, oo('evaluation'), f'{dataset}_F0_{context}_E', True, context)
                    rows.append(dict(dataset=dataset, arm=f'F0_context_{context}', seed=None, **r))
                    component_rows.append(r)
                mixture = mixture_quantiles(np.stack([self.cached_prediction(r) for r in component_rows]))
                target = np.stack([values[o:o+48].T for o in oo('evaluation')])
                path = self.cache/f'{dataset}_teacher_mixture_E.npz'
                np.savez_compressed(path, prediction=mixture, target=target, scale=scale)
                rows.append(dict(dataset=dataset, arm='teacher_mixture', seed=None,
                    metrics=score(mixture, target, scale), prediction_file=str(path.relative_to(CACHE)),
                    prediction_sha256=sha(path)))
            del model
            gc.collect()
            torch.cuda.empty_cache()
            for selected in [s for s in seal['selections'] if s['dataset'] == dataset]:
                self.check('selected_model_load', True)
                seed_all(selected['seed'])
                model = PilotModel(selected['arm'], selected['seed'])
                path = CACHE/selected['checkpoint_file']
                assert sha(path) == selected['checkpoint_sha256']
                restore(model, torch.load(path, map_location='cpu', weights_only=True))
                dev, _ = data(dataset)
                reloaded = self.evaluate(model, dev, scale, oo('validation'), selected['fit_id']+'_selected_V_reload')
                a, b = self.cached_prediction(selected), self.cached_prediction(reloaded)
                assert np.array_equal(a, b), 'Selected checkpoint V replay not bit-exact'
                reloads.append(dict(selected_prediction=selected['prediction_file'], replay=reloaded,
                                    max_absolute_error=float(np.max(np.abs(a-b)))))
                row = self.evaluate(model, values, scale, oo('evaluation'), selected['fit_id']+'_E')
                rows.append(dict(dataset=dataset, arm=selected['arm'], seed=selected['seed'],
                                 selected_step=selected['step'], selected_checkpoint_sha256=selected['checkpoint_sha256'], **row))
                write_json(self.out/'evaluation.json', rows)
                del model
                gc.collect()
                torch.cuda.empty_cache()
        write_json(self.out/'selected_reloads.json', reloads)
        # Multi-pass teacher is a quality reference, not a same-cost baseline.
        # Conservative gate includes it anyway; no inaccurate efficiency claim.
        report = verdict_from_rows(rows, seal['selections'], self.topic, CFG)
        report.update(fits_attempted=json.loads((self.out/'train_status.json').read_text())['fits_attempted'])
        write_json(self.out/'summary.json', report)
        check_contract()
        self.emit(status='COMPLETE', verdict=report['verdict'])
        return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--topic', choices=TOPICS, required=True)
    p.add_argument('--phase', choices=['train', 'evaluate'], required=True)
    args = p.parse_args()
    def terminate(signum, frame):
        raise TimeoutError(f'Worker terminated by signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    w = Worker(args.topic, args.phase)
    try:
        w.train() if args.phase == 'train' else w.evaluate_selected()
    except BaseException as exc:
        w.emit(status='INCONCLUSIVE_EXECUTION', error=f'{type(exc).__name__}: {exc}')
        (w.out/(args.phase+'_traceback.txt')).write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
