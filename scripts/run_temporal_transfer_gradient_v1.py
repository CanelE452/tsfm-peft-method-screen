"""One-shot fixed-state, train/V-only gradient diagnostic; no fitting or optimizer."""
import csv
import gc
import hashlib
import json
import os
import pickle
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from tsfm_peft_screen.backbone import MODEL_ID, REVISION, QUANTILES, native_loss
from tsfm_peft_screen.metrics import score, independent
from tsfm_peft_screen.reassessment import DiagnosticModel
from tsfm_peft_screen.reproducibility import ROOT, sha, write_json, seed_all, guard

PARENT = ROOT / 'results/temporal_transfer_diagnostic_v1'
OUT = ROOT / 'results/temporal_transfer_diagnostic_v1_D'
CACHE = ROOT / '.cache/temporal_transfer_diagnostic_v1_D'
NAMES = ['native_T', 'eval_T', 'eval_S', 'eval_D', 'anchor_T']


def read(p):
    return json.loads(Path(p).read_text())


def eval_loss(p, y, scale):
    """Exact metric reduction, float64; sanitize missing targets before arithmetic."""
    p = p.double().sort(dim=2).values
    y, scale = y.double(), scale.double()
    valid = torch.isfinite(y)
    count = valid.sum((0, 2))
    active = count > 0
    if not active.any():
        raise ValueError('No targets')
    safe = torch.where(valid, y, torch.zeros_like(y))
    err = safe[:, :, None, :] - p
    q = p.new_tensor(QUANTILES)[None, None, :, None]
    pin = torch.where(valid[:, :, None, :], 2 * torch.maximum(q * err, (q - 1) * err), 0.)
    return (pin.sum((0, 3))[active] / count[active, None] / scale[active, None]).mean()


def tensor_hash(items):
    h = hashlib.sha256()
    for n, p in sorted(items.items()):
        a = p.detach().cpu().contiguous().numpy()
        h.update(n.encode()); h.update(str((a.shape, str(a.dtype))).encode()); h.update(a.tobytes())
    return h.hexdigest()


def restore(params, saved):
    assert params.keys() == saved.keys()
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(saved[n])


def flatten(items):
    return torch.cat([p.detach().cpu().double().reshape(-1) for p in items])


def put_vector(params, vector):
    offset = 0
    with torch.no_grad():
        for p in params.values():
            p.copy_(vector[offset:offset+p.numel()].reshape(p.shape).to(p))
            offset += p.numel()
    assert offset == len(vector)


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def set_rng(s):
    random.setstate(s[0]); np.random.set_state(s[1]); torch.set_rng_state(s[2]); torch.cuda.set_rng_state_all(s[3])


def rng_hash(s):
    # Tensor pickle storage IDs are not stable; hash explicit contents instead.
    return hashlib.sha256(pickle.dumps(s[:2]) + s[2].numpy().tobytes() + b''.join(t.numpy().tobytes() for t in s[3])).hexdigest()


def geometry(a, b):
    na, nb, dot = float(a.norm()), float(b.norm()), float(a @ b)
    return na, nb, dot, dot / (na * nb) if na and nb else 'UNDEFINED_ZERO_GRADIENT'


def csv_write(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


class Budget:
    def __init__(self):
        self.start = time.monotonic()
        self.count = dict(forward=0, backward=0, states_started=0, states_completed=0,
                          perturbations=0, discarded_perturbations=0, optimizer_updates=0, new_fits=0,
                          E_array_reads=0, permanent_weight_changes=0)
        self.telemetry = []

    def save(self, status='RUNNING', **extra):
        write_json(OUT/'execution_receipt.json', dict(status=status, **self.count,
                   elapsed_seconds=time.monotonic()-self.start, **extra))

    def resource(self, phase, startup=False):
        g = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(',')
        lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], text=True).splitlines()
        apps = [dict(pid=int(p[0]), name=p[1].strip(), memory_mib=int(p[2])) for p in (line.split(',') for line in lines)]
        allowed = lambda a: a['pid'] == os.getpid() or (a['name'] == '/usr/share/rustdesk/rustdesk' and a['memory_mib'] <= 512)
        r = dict(phase=phase, seconds=time.monotonic()-self.start, free_mib=int(g[0]), util=int(g[1]), apps=apps)
        self.telemetry.append(r); write_json(OUT/'gpu_telemetry.json', self.telemetry)
        if any(not allowed(a) for a in apps):
            raise RuntimeError('RESOURCE_BUSY: unexpected compute process')
        if r['free_mib'] < (4096 if startup else 1024) or (startup and r['util'] > 35):
            raise RuntimeError('RESOURCE_BUSY: insufficient GPU headroom')
        guard(self.start)

    def call(self, kind, label):
        self.resource(label)
        cap = dict(forward=256, backward=60)[kind]
        assert self.count[kind] < cap
        self.count[kind] += 1
        self.save(last_attempt=label)  # Failed attempts count too; never retry automatically.


def load_development(state, contract):
    inp = contract['inputs'][state['source']]
    path = ROOT/state['development_file']
    assert str(path.relative_to(ROOT)) == inp['development_path']
    assert path.name.endswith('_development.npz') and sha(path) == inp['development_sha256']
    with np.load(path, allow_pickle=False) as f:
        assert set(f.files) == {'values', 'scale'}
        values, scale = f['values'].copy(), f['scale'].copy()
    assert np.array_equal(scale, state['scale']) and values.shape[1] == 4
    cfg=contract['config']['data'][state['source']]
    train=list(range(cfg['train'][0],cfg['train'][1]+1,cfg['train'][2]))
    validation=list(range(cfg['validation'][0],cfg['validation'][1]+1,cfg['validation'][2]))
    assert state['train_origins']==train[-2:]
    assert state['S_probe_origins']==validation[6:8] and state['D_probe_origins']==validation[14:16]
    assert train[-1]+48<=validation[0] and validation[7]+48<=validation[8]
    keys = dict(T='train_origins', S='S_probe_origins', D='D_probe_origins')
    batches = {}
    for region, key in keys.items():
        origins = state[key]
        assert len(origins) == 2 and max(origins)+48 <= len(values)
        x = np.concatenate([values[o-1024:o].T for o in origins]).astype('float32')
        y = np.concatenate([values[o:o+48].T for o in origins]).astype('float32')
        batches[region] = (torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda(), torch.arange(2, device='cuda').repeat_interleave(4))
    return batches, torch.tensor(scale, device='cuda', dtype=torch.float64)


def validated_loss(p, y, sc):
    pp, yy = p.reshape(2, 4, 21, 48), y.reshape(2, 4, 48)
    loss = eval_loss(pp, yy, sc)
    a, b, s = pp.detach().cpu().numpy(), yy.cpu().numpy(), sc.cpu().numpy()
    official = score(a, b, s)['scaled_2pinball']
    scalar = independent(a, b, s)
    assert np.isfinite(float(loss.detach())) and abs(float(loss.detach())-official) <= 1e-10 and abs(official-scalar) <= 1e-10
    sorted_p = pp.detach().sort(dim=2).values
    warnings = dict(sort_ties=int((sorted_p[:, :, 1:] == sorted_p[:, :, :-1]).sum()),
                    target_ties=int(((sorted_p == yy[:, :, None, :]) & torch.isfinite(yy[:, :, None, :])).sum()),
                    missing_targets=int((~torch.isfinite(yy)).sum()), metric_error=abs(float(loss.detach())-official))
    return loss, warnings


def snapshot_history():
    return {str(p.relative_to(ROOT)): sha(p) for folder in ['src','scripts','configs','tests','research','results','automation']
            for p in sorted((ROOT/folder).rglob('*')) if p.is_file() and '__pycache__' not in str(p) and OUT not in p.parents}


def main():
    assert not OUT.exists() and not CACHE.exists(), 'No overwrite or implicit retry'
    manifest = read(PARENT/'gradient_probe_manifest.json')
    assert len(manifest['states']) == 12
    assert read(PARENT/'artifact_verification.json')['status'] == 'PASS'
    contract = read(ROOT/'results/anchor_window_study_20260914/contract.json')
    from huggingface_hub import hf_hub_download
    assert contract['model_revision'] == REVISION
    for name, h in contract['model_files'].items():
        assert sha(hf_hub_download(MODEL_ID, name, revision=REVISION, local_files_only=True)) == h
    history = snapshot_history()
    OUT.mkdir(); CACHE.mkdir()
    write_json(OUT/'source_and_history_hashes.json', history)
    manifest.update(status='RUNNING', forward_plan=180, backward_plan=60,
        gpu_policy='No external compute except exact RustDesk display executable <=512 MiB; startup free>=4 GiB/util<=35%, step free>=1 GiB; existing RAM/time guards unchanged.',
        precision='BF16 baseline versus FP32; all perturbations FP32, float64 metric reduction',
        approximation_rule='Both epsilon signs must agree, actual/predicted signs agree, abs(actual-predicted)<=0.5*abs(predicted)+floor; floor=8*float32_eps*max(1,abs(base_loss)). Both deltas must exceed floor. Computational reliability filter, not a performance PASS threshold.',
        budget_semantics='Every attempted model forward/autograd.grad counted including failures. No numerical retry, fit, optimizer, E access, push, notification or follow-up.')
    write_json(OUT/'gradient_probe_manifest.json', manifest)
    budget = Budget(); geoms=[]; effects=[]; baselines=[]; restores=[]
    try:
        for k in range(4):
            budget.resource(f'startup_{k}', startup=True)
            if k < 3:
                time.sleep(10)
        for state in manifest['states']:
            t0 = time.monotonic()
            sid = f"{state['dataset']}_{state['seed']}_{state['arm']}"
            assert state['budget']==233 and state['lr']==3e-5 and state['step']==450
            cp = ROOT/state['checkpoint_file']
            assert sha(cp) == state['checkpoint_sha256']
            budget.count['states_started'] += 1; budget.save(last_state=sid)
            seed_all(state['seed'])
            batches, sc = load_development(state, contract)
            model = DiagnosticModel('anchor', state['arm'], state['seed'])
            params = {n:p for n,p in model.named_parameters() if p.requires_grad}
            saved = torch.load(cp, map_location='cpu', weights_only=True)
            restore(params, saved)
            assert len(params)==192 and sum(p.numel() for p in params.values())==1179648
            original_hash = tensor_hash(params)
            assert original_hash == tensor_hash(saved)
            frozen = lambda: tensor_hash({n:p for n,p in model.named_parameters() if not p.requires_grad})
            frozen_before = frozen(); theta=flatten(saved.values()); scale_phi=max(1.,float(theta.norm()))
            # saved dict order must match the model's coordinate order.
            assert list(saved) == list(params)
            rng = rng_state(); rh = rng_hash(rng)
            grads={}; bases={}; fp32pred={}; details={}
            direction=None; actual_step=None

            def forward(region, bf16=False, frozen_base=False):
                budget.call('forward', f'{sid}:{region}:{"BF16" if bf16 else "FP32"}:{frozen_base}')
                set_rng(rng)
                x, _, groups = batches[region]
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                    out = model(x, groups, frozen=frozen_base)
                assert all(torch.isfinite(v).all() for v in out)
                return out

            def gradient(loss, name, retain=False):
                budget.call('backward', f'{sid}:{name}')
                g = torch.autograd.grad(loss, tuple(params.values()), retain_graph=retain)
                vector=flatten(g)
                assert torch.isfinite(vector).all()
                grads[name]=vector

            try:
                with torch.no_grad():
                    teacher=forward('T', frozen_base=True)[1].detach()
                for region in ['T','S','D']:
                    z,p,loc,sc_model = forward(region)
                    y = batches[region][1]
                    loss, warnings = validated_loss(p,y,sc)
                    bases[region]=float(loss.detach()); fp32pred[region]=p.detach().cpu().numpy(); details[region]=warnings
                    if region=='T':
                        task = native_loss(z,y,loc,sc_model)/21
                        anchor = .1*((p.sort(dim=1).values-teacher.sort(dim=1).values).abs()/sc.float().repeat(2)[:,None,None]).mean()
                        gradient(task, 'native_T', True); gradient(loss, 'eval_T', True); gradient(anchor, 'anchor_T')
                        details['native_loss']=float(task.detach()); details['anchor_loss']=float(anchor.detach())
                        del task, anchor, teacher
                    else:
                        gradient(loss, 'eval_'+region)
                    del z,p,loc,sc_model,loss
                for a in NAMES:
                    for b in NAMES[NAMES.index(a):]:
                        na,nb,dot,cos=geometry(grads[a],grads[b])
                        geoms.append(dict(state=sid,a=a,b=b,norm_a=na,norm_b=nb,dot=dot,cosine=cos))
                trajectory=read(ROOT/f"results/anchor_window_study_20260914/anchor/{sid}_0_trajectory.json")
                row=next(r for r in trajectory if r['step']==450)
                assert sha(ROOT/row['prediction_file'])==row['prediction_sha256']
                with np.load(ROOT/row['prediction_file'],allow_pickle=False) as f:
                    cached=f['prediction'].copy()
                with torch.no_grad():
                    for region in ['T','S','D']:
                        p=forward(region,bf16=True)[1]
                        loss,w=validated_loss(p,batches[region][1],sc)
                        pred=p.cpu().numpy(); diff=pred-fp32pred[region]
                        parity=None
                        if region in ('S','D'):
                            sl=slice(6,8) if region=='S' else slice(14,16)
                            parity=float(np.max(np.abs(pred.reshape(2,4,21,48)-cached[sl])))
                        baselines.append(dict(state=sid,region=region,fp32_loss=bases[region],bf16_loss=float(loss),
                            bf16_minus_fp32=float(loss)-bases[region],prediction_max_abs=float(np.max(abs(diff))),
                            cached_bf16_prediction_max_abs=parity,fp32_sort_ties=details[region]['sort_ties'],
                            fp32_target_ties=details[region]['target_ties'],missing_targets=details[region]['missing_targets']))
                        del p,loss
                csv_write(OUT/'precision_baselines.csv',baselines)
                for name in ['native_T','eval_T']:
                    norm=float(grads[name].norm())
                    if norm==0:
                        details[name+'_direction']='UNDEFINED_ZERO_GRADIENT'
                        continue
                    direction=-grads[name]/norm
                    for factor in [1e-4,1e-3]:
                        restore(params,saved); set_rng(rng)
                        assert tensor_hash(params)==original_hash and rng_hash(rng_state())==rh
                        eps=factor*scale_phi
                        budget.count['perturbations']+=1; budget.save()
                        try:
                            put_vector(params,theta+eps*direction)
                            actual_step=flatten(params.values())-theta
                            with torch.no_grad():
                                for region in ['S','D']:
                                    p=forward(region)[1]
                                    loss,warnings=validated_loss(p,batches[region][1],sc)
                                    delta=float(loss)-bases[region]
                                    pred=eps*float(grads['eval_'+region]@direction)
                                    floor=8*np.finfo(np.float32).eps*max(1.,abs(bases[region]))
                                    reliable=abs(delta)>floor and abs(pred)>floor and delta*pred>0 and abs(delta-pred)<=.5*abs(pred)+floor
                                    effects.append(dict(state=sid,direction=name,region=region,factor=factor,epsilon=eps,
                                        phi_norm=float(theta.norm()),base_loss=bases[region],perturbed_loss=float(loss),
                                        actual_delta=delta,first_order_delta=pred,actual_step_first_order=float(grads['eval_'+region]@actual_step),
                                        actual_step_norm=float(actual_step.norm()),step_relative_rounding=float((actual_step-eps*direction).norm())/eps,
                                        approximation_error=abs(delta-pred),numerical_floor=float(floor),local_reliable=bool(reliable),
                                        sort_ties=warnings['sort_ties'],target_ties=warnings['target_ties']))
                                    del p,loss
                        finally:
                            restore(params,saved); set_rng(rng)
                            check=dict(state=sid,direction=name,factor=factor,
                                trainable_restored=tensor_hash(params)==original_hash,
                                frozen_unchanged=frozen()==frozen_before,
                                checkpoint_unchanged=sha(cp)==state['checkpoint_sha256'],rng_restored=rng_hash(rng_state())==rh)
                            restores.append(check); budget.count['discarded_perturbations']+=1
                            write_json(OUT/'restoration_checks.json',restores)
                            assert all(check[k] for k in ['trainable_restored','frozen_unchanged','checkpoint_unchanged','rng_restored'])
                np.savez_compressed(CACHE/f'{sid}_gradients.npz',**{k:v.numpy() for k,v in grads.items()})
                write_json(OUT/f'{sid}_state.json', dict(state=state,diagnostics=details,
                    coordinate_shapes={n:list(p.shape) for n,p in params.items()},coordinate_order=list(params),
                    trainable_hash=original_hash,frozen_hash=frozen_before,seconds=time.monotonic()-t0,
                    gradient_file=str((CACHE/f'{sid}_gradients.npz').relative_to(ROOT)),gradient_sha256=sha(CACHE/f'{sid}_gradients.npz')))
            finally:
                restore(params,saved); set_rng(rng)
                assert tensor_hash(params)==original_hash and frozen()==frozen_before and sha(cp)==state['checkpoint_sha256']
            budget.count['states_completed']+=1
            csv_write(OUT/'gradient_geometry.csv',geoms);csv_write(OUT/'local_perturbation_effects.csv',effects)
            budget.save(last_state=sid,last_state_seconds=time.monotonic()-t0,resources=guard(budget.start))
            print(json.dumps(dict(state=sid,completed=budget.count['states_completed'],seconds=time.monotonic()-t0,counts=budget.count)),flush=True)
            del model,params,saved,theta,grads,batches,sc,fp32pred,cached,direction,actual_step
            gc.collect();torch.cuda.empty_cache()
        changed=[p for p,h in history.items() if sha(ROOT/p)!=h]
        assert not changed
        budget.save('COMPLETED',historical_files_verified=len(history),historical_changes=changed,resources=guard(budget.start),
                    git_push=False,automatic_followup=False)
        manifest['status']='COMPLETED';write_json(OUT/'gradient_probe_manifest.json',manifest)
    except BaseException as e:
        budget.save('INTERRUPTED',error=repr(e),historical_changes=[p for p,h in history.items() if sha(ROOT/p)!=h])
        raise


if __name__ == '__main__':
    main()
