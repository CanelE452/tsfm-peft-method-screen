"""Bounded TRAIN-only, zero-update feasibility probe on a real frozen TSFM."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from priority12.common import save, read, sha, tensor_hash, Watch as BaseWatch
from subspace_core import moments, top_basis, project, self_test

OUT = ROOT/'results/block_gradient_feasibility_20260919'
CACHE = ROOT/'.cache/block_gradient_feasibility_20260919'
DATA = ROOT/'.cache/additive_persistence_validation_v1_20260917/data'
SOURCES = ['electricity', 'ettm1']
GRAD_CAP = 48


class Watch(BaseWatch):
    def sample(self):
        r = super().sample()
        for a in r['apps']:
            a['allowed_desktop'] = a['name'] == '/usr/share/rustdesk/rustdesk'
        r['busy'] = any(not a['own'] and not a['allowed_desktop'] for a in r['apps']) or r['free_mib'] < 1024
        return r


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if (OUT/'SEAL.json').exists():
        check_seal()
        return
    cpu = self_test()
    hashes = {}
    for p in [Path(__file__), HERE/'subspace_core.py', HERE/'PROTOCOL.md',
              ROOT/'experiments/outlier_signal_followup_v2_20260917/common.py',
              ROOT/'experiments/outlier_signal_followup_v2_20260917/model.py',
              ROOT/'experiments/outlier_signal_peft_v1_20260917/model.py',
              ROOT/'scripts/priority12/common.py']:
        hashes[str(p.relative_to(ROOT))] = sha(p)
    receipt_path = ROOT/'results/outlier_signal_followup_v2_20260917/download_receipts.json'
    receipt = read(receipt_path)['amazon/chronos-bolt-small']
    hashes[str(receipt_path.relative_to(ROOT))] = sha(receipt_path)
    for p,h in receipt['files'].items():
        assert sha(ROOT/p) == h, p
        hashes[p] = h
    old = read(ROOT/'results/outlier_signal_followup_v2_20260917/DATA_MANIFEST.json')
    blocks, audits = [], []
    for source in SOURCES:
        packet = DATA/source/'TRAIN_inputs.npz'
        labels = DATA/source/'TRAIN_labels.npz'
        for p in [packet,labels]:
            expected = next(h for k,h in old[source]['packet_hashes']['TRAIN'].items() if Path(k).name == p.name)
            assert sha(p) == expected, str(p)
            hashes[str(p.relative_to(ROOT))] = expected
        d = np.load(packet)
        origins = d['origins']
        assert len(origins) == 256 and (np.diff(origins)>0).all()
        assert d['x'].shape == (256,4,512) and d['sigma'].shape == (4,)
        assert (origins>=512).all() and (origins+64<=old[source]['bounds'][1]).all()
        gap = int(origins[160]-512-(origins[95]+64))
        assert gap >= 0, ('CALIBRATION_HOLDOUT_OVERLAP', source)
        for role,indices in [('calibration', np.arange(96)), ('later_train', np.arange(160,256))]:
            o = origins[indices]
            days = o//old[source]['period']
            assert len(np.unique(days)) == 96
            target = o[:,None]+np.arange(64)
            audits.append(dict(source=source, role=role, origins=96, distinct_days=96,
                               first_origin=int(o.min()), last_origin=int(o.max()),
                               embargo_slots=gap, input_target_cross_role_overlap=False,
                               unique_target_slots=int(len(np.unique(target))),
                               total_target_slots=int(target.size),
                               internal_target_overlap=int(target.size-len(np.unique(target))),
                               grid=old[source]['grid']['type']))
            for k,idx in enumerate(indices.reshape(12,8)):
                blocks.append(dict(source=source,role=role,block=k,indices=idx.tolist(),origins=origins[idx].tolist()))
    assert len(blocks) == GRAD_CAP
    save(OUT/'BLOCKS.json',blocks)
    pd.DataFrame(audits).to_csv(OUT/'ORIGIN_AUDIT.csv', index=False)
    save(OUT/'CPU_CHECKS.json',cpu)
    hashes[str((OUT/'BLOCKS.json').relative_to(ROOT))] = sha(OUT/'BLOCKS.json')
    save(OUT/'SEAL.json',dict(at=time.time(),hashes=hashes,gradient_cap=GRAD_CAP,
                             optimizer_updates=0,new_fits=0,rank=8,
                             allowed_data_role='TRAIN only',model_revision=receipt['revision']))
    print('PREPARED: 48 gradient calls, zero updates', flush=True)


def check_seal():
    for p,h in read(OUT/'SEAL.json')['hashes'].items():
        assert sha(ROOT/p) == h, ('SEALED_CHANGED', p)


def append(row):
    with open(OUT/'GRADIENT_LEDGER.jsonl', 'a') as f:
        f.write(json.dumps(row,allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())


def run():
    check_seal()
    if (OUT/'GRADIENTS_COMPLETE.json').exists():
        print('Existing complete probe: no duplicate calls', flush=True)
        return
    assert not (OUT/'GRADIENT_LEDGER.jsonl').exists(), 'Partial probe requires exact-state audit; do not replay automatically'
    from experiments.outlier_signal_followup_v2_20260917.common import build, setup
    from experiments.outlier_signal_peft_v1_20260917.model import LoRA, loss_2pinball
    import shutil
    assert shutil.disk_usage(ROOT).free > 8*2**30
    setup()
    watch = Watch(OUT,'gradient_probe',wall_cap=7200)
    completed = []
    call_count = 0
    try:
        watch.boundary(startup=True)
        for source in SOURCES:
            m = build('B0',91932,device='cuda').eval().requires_grad_(False)
            adapters = [(n,v) for n,v in m.base.named_modules() if isinstance(v,LoRA)]
            assert len(adapters) == 36
            assert all(torch.count_nonzero(v.b).item()==0 for _,v in adapters)
            params = [v.base.weight for _,v in adapters]
            for p in params:
                p.requires_grad_(True)
            before_hash = tensor_hash(m.state_dict())
            inp = np.load(DATA/source/'TRAIN_inputs.npz')
            target = np.load(DATA/source/'TRAIN_labels.npz')['y']
            assert target.shape==(256,4,64) and np.isfinite(target).all()
            path = CACHE/f'{source}_gradients.npy'
            result = np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(24,36,512,512))
            source_blocks = [b for b in read(OUT/'BLOCKS.json') if b['source']==source]
            torch.cuda.reset_peak_memory_stats()
            t0 = time.monotonic()
            for row_index, block in enumerate(source_blocks):
                start = watch.before()
                idx = block['indices']
                x = torch.tensor(inp['x'][idx].reshape(32,512),device='cuda')
                y = torch.tensor(target[idx].reshape(32,64),dtype=torch.float32,device='cuda')
                sigma = torch.tensor(np.tile(inp['sigma'],8),dtype=torch.float32,device='cuda')
                pred = m(x,sigma)
                if row_index == 0:
                    with torch.no_grad():
                        native = m.base(context=x).quantile_preds
                    assert torch.equal(pred,native), 'WRAPPER_NATIVE_OUTPUT_MISMATCH'
                loss = loss_2pinball(pred,y,sigma,m.base.quantiles)
                assert torch.isfinite(loss)
                call_count += 1
                assert call_count<=GRAD_CAP
                append(dict(status='INTENT',call=call_count,source=source,block=row_index,at=time.time()))
                grads = torch.autograd.grad(loss,params)
                assert all(torch.isfinite(g).all() for g in grads)
                assert all(p.grad is None for p in m.parameters())
                result[row_index] = torch.stack(grads).detach().cpu().numpy()
                result.flush()
                after,contaminated = watch.after(start)
                assert not contaminated, 'EXTERNAL_COMPUTE_CONTAMINATION'
                append(dict(status='COMPLETE',call=call_count,source=source,block=row_index,
                            role=block['role'],loss=float(loss),at=time.time(),optimizer_updates=0))
                del grads,pred,loss,x,y,sigma
                if row_index % 6 == 5:
                    print(f'{source}: gradient {row_index+1}/24, total {call_count}/48',flush=True)
            after_hash = tensor_hash(m.state_dict())
            assert before_hash==after_hash, 'FROZEN_MODEL_CHANGED'
            result.flush()
            del result
            completed.append(dict(source=source,path=str(path.relative_to(ROOT)),sha256=sha(path),
                                  layers=[n for n,_ in adapters],gradient_calls=24,
                                  native_forward_equal=True,model_hash_before=before_hash,model_hash_after=after_hash,
                                  wall_seconds=time.monotonic()-t0,
                                  peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            del m,params,adapters
            torch.cuda.empty_cache()
        assert call_count==GRAD_CAP
        save(OUT/'GRADIENTS_COMPLETE.json',dict(at=time.time(),gradient_calls=call_count,
               new_fits=0,optimizer_updates=0,V_E_files_read=False,models=completed))
    finally:
        watch.close()


def analyse():
    check_seal()
    done = read(OUT/'GRADIENTS_COMPLETE.json')
    rows, aggregate = [], []
    for source in SOURCES:
        record = next(r for r in done['models'] if r['source']==source)
        assert sha(ROOT/record['path'])==record['sha256']
        g = np.load(ROOT/record['path'],mmap_mode='r')
        totals = {a:dict(inner=0.,update_sq=0.,hold_sq=0.,calib_energy=0.)
                  for a in ['RANDOM_ORTHO','MEAN_SVD','SECOND_MOMENT','CROSS_BLOCK','FULL_GRADIENT']}
        for layer_idx,layer in enumerate(record['layers']):
            mean,square,second,cross = moments(g[:12,layer_idx])
            later = g[12:,layer_idx].astype(float).mean(0)
            random,_ = np.linalg.qr(np.random.default_rng(91932+layer_idx).normal(size=(512,8)))
            bases = {'RANDOM_ORTHO':random}
            eig_stats = {}
            for arm,matrix in [('MEAN_SVD',square),('SECOND_MOMENT',second),('CROSS_BLOCK',cross)]:
                basis,eig = top_basis(matrix,8)
                bases[arm]=basis
                eig_stats[arm] = dict(min_eigenvalue=float(eig[0]),eighth_eigenvalue=float(eig[-8]),positive_eigenvalues=int((eig>0).sum()))
            for arm in totals:
                d = mean if arm=='FULL_GRADIENT' else project(mean,bases[arm])
                inner=float((later*d).sum())
                update_sq=float((d*d).sum())
                hold_sq=float((later*later).sum())
                denom=np.sqrt(update_sq*hold_sq)
                assert denom>0 and np.isfinite(denom)
                energy=float((mean*d).sum())
                rows.append(dict(source=source,layer=layer,arm=arm,rank=512 if arm=='FULL_GRADIENT' else 8,
                                 holdout_direction_inner=inner,holdout_direction_cosine=inner/denom,
                                 update_squared_norm=update_sq,holdout_gradient_squared_norm=hold_sq,
                                 calibration_projection_energy=energy,**eig_stats.get(arm,{})))
                for key,val in [('inner',inner),('update_sq',update_sq),('hold_sq',hold_sq),('calib_energy',energy)]:
                    totals[arm][key]+=val
            if layer_idx % 12==11:
                print(f'{source}: CPU subspaces {layer_idx+1}/36',flush=True)
        for arm,t in totals.items():
            aggregate.append(dict(source=source,arm=arm,holdout_direction_inner=t['inner'],
                                  holdout_direction_cosine=t['inner']/np.sqrt(t['update_sq']*t['hold_sq']),
                                  update_norm=np.sqrt(t['update_sq']),calibration_projection_energy=t['calib_energy']))
    f=pd.DataFrame(rows);summary=pd.DataFrame(aggregate)
    assert len(f)==360 and len(summary)==10
    f.to_csv(OUT/'LAYER_RESULTS.csv',index=False)
    summary.to_csv(OUT/'AGGREGATE_RESULTS.csv',index=False)
    findings=[]
    for source in SOURCES:
        a=summary[(summary.source==source)&(summary.arm=='CROSS_BLOCK')].iloc[0]
        b=summary[(summary.source==source)&(summary.arm=='MEAN_SVD')].iloc[0]
        findings.append(dict(source=source,cross_inner=float(a.holdout_direction_inner),mean_inner=float(b.holdout_direction_inner),
                             cross_cosine=float(a.holdout_direction_cosine),mean_cosine=float(b.holdout_direction_cosine),
                             cross_better_on_both=bool(a.holdout_direction_inner>b.holdout_direction_inner and
                                                       a.holdout_direction_cosine>b.holdout_direction_cosine and
                                                       a.holdout_direction_inner>0)))
    save(OUT/'VERIFICATION.json',dict(status='COMPLETE_DIAGNOSTIC',new_fits=0,optimizer_updates=0,
         gradient_calls=48,primary_results=findings,automatic_training=False,novelty_established=False,
         goal_achieved=False,independent_test=False))
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','analyse'])
    args=parser.parse_args()
    {'prepare':prepare,'run':run,'analyse':analyse}[args.command]()
