"""Forward-only fixed checkpoint ablation; 16 validation predictions, zero fits."""
import argparse
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/peft_rank12_20260915'))
from common import read, save, sha, csvwrite, cpu_state, tensor_hash, frozen_hash, restore, cleanup, preserve_rng, Watch
from rank2_model import make
from rank2_data import load, batch, metrics, independent
from run_rank2 import configure
OUT = ROOT / 'results/lora_projection_diagnostic_20260916'
EXP = ROOT / 'experiments/lora_projection_diagnostic_20260916'
CACHE = ROOT / '.cache/lora_projection_diagnostic_20260916'
BAL = ROOT / 'results/channel_phase_balance_20260916'
DATA = ROOT / '.cache/channel_sharing_screen_v1_20260915'
ARMS = ['FULL', 'QK_ONLY', 'V_ONLY', 'NONE']


def hashes(mapping):
    for p, h in mapping.items():
        assert sha(ROOT / p) == h, p


def predict(model, values, std, origins, path, watch):
    assert not path.exists()
    pp, yy = [], []
    start = watch.before()
    tick = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with preserve_rng(), torch.no_grad():
        model.eval()
        for i in range(0, len(origins), 8):
            watch.boundary()
            x, y = batch(values, origins[i:i+8])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p = model(x).forecast
            assert torch.isfinite(p).all()
            pp.append(p.float().cpu().numpy()); yy.append(y.cpu().numpy())
    end, contaminated = watch.after(start)
    assert not contaminated, 'EXTERNAL_COMPUTE_DURING_PREDICTION'
    p, y = np.concatenate(pp), np.concatenate(yy)
    met = metrics(p, y, std)
    scalar_mse = independent(p, y)
    scalar_mae = math.fsum(math.fsum(abs(float(a)-float(b)) for a,b in zip(p[:,c].flat,y[:,c].flat) if np.isfinite(b))/int(np.isfinite(y[:,c]).sum()) for c in range(32))/32
    assert math.isclose(scalar_mse, met['mse'], abs_tol=1e-12, rel_tol=1e-12)
    assert math.isclose(scalar_mae, met['mae'], abs_tol=1e-12, rel_tol=1e-12)
    np.savez_compressed(path, prediction=p, target=y, std=std, origins=np.array(origins))
    return dict(metrics=met, scalar_mse=scalar_mse, scalar_mae=scalar_mae,
                prediction_path=str(path.relative_to(ROOT)), prediction_sha256=sha(path),
                forward_wall_seconds=time.monotonic()-tick,
                forward_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20), p


def run():
    assert not CACHE.exists() and not (OUT/'seal.json').exists(), 'NO_DUPLICATE_RUN'
    baseline = read(BAL/'seal.json'); hashes(baseline['source_hashes']); hashes(baseline['model_files'])
    for d in baseline['data'].values(): hashes(d['staged'])
    fits = read(BAL/'fits.json'); assert len(fits) == 4
    history = {str(p.relative_to(ROOT)):sha(p) for folder in ['results','research'] for p in (ROOT/folder).rglob('*') if p.is_file() and OUT not in p.parents}
    save(OUT/'historical_hashes.json', history)
    sources = dict(baseline['source_hashes'])
    for p in [Path(__file__), OUT/'PROTOCOL.md', BAL/'seal.json', BAL/'fits.json']:
        sources[str(p.relative_to(ROOT))] = sha(p)
    refs = {f['best'][k]:f['best'][k.replace('_path','_sha256')] for f in fits for k in ['checkpoint_path','prediction_path']}
    hashes(refs)
    save(OUT/'seal.json', dict(at=time.time(), head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), source_hashes=sources, model_files=baseline['model_files'], data=baseline['data'], references=refs, cells=[dict(dataset=f['dataset'],seed=f['seed'],checkpoint=f['best']) for f in fits], arms=ARMS, predictions_cap=16, fit_cap=0, updates_cap=0, exposure='REUSED_VALIDATION', wall_cap=1800))
    CACHE.mkdir(); configure(); watch=None; records=[]; checks=[]; model=None
    status=dict(status='PREPARED',fits=0,optimizer_updates=0,predictions_completed=0)
    save(OUT/'status.json',status)
    try:
        watch=Watch(OUT,'projection',1800);watch.boundary(startup=True)
        status['status']='RUNNING';save(OUT/'status.json',status)
        for fit in fits:
            d, seed = fit['dataset'], fit['seed']; cp=fit['best']
            model=make('LH',baseline['data'][d]['channel_ids'],seed,'cuda').eval()
            state=torch.load(ROOT/cp['checkpoint_path'],map_location='cpu',weights_only=True)
            restore(model,state); fixed=frozen_hash(model); buffers=tensor_hash(dict(model.named_buffers()))
            groups={g:[n for n in state if f'.SelfAttention.{g}.lora_B.' in n] for g in ['q','k','v']}
            assert all(len(names)==8 for names in groups.values())
            values,std=load(DATA,d,'development'); origins=baseline['data'][d]['origins']['validation']
            with np.load(ROOT/cp['prediction_path']) as z:
                reference=z['prediction'].copy(); reference_y=z['target'].copy(); reference_o=z['origins'].copy()
            for arm in ARMS:
                disabled={'FULL':[], 'QK_ONLY':['v'], 'V_ONLY':['q','k'], 'NONE':['q','k','v']}[arm]
                edited={n:v.clone() for n,v in state.items()}
                changed=[n for g in disabled for n in groups[g]]
                for n in changed: edited[n].zero_()
                assert all(torch.equal(v,edited[n]) for n,v in state.items() if n not in changed)
                assert all(torch.count_nonzero(edited[n])==0 for n in changed)
                restore(model,edited); assert tensor_hash(cpu_state(model))==tensor_hash(edited)
                before=tensor_hash(cpu_state(model))
                r,p=predict(model,values,std,origins,CACHE/f'{d}_{seed}_{arm}.npz',watch)
                assert tensor_hash(cpu_state(model))==before and frozen_hash(model)==fixed and tensor_hash(dict(model.named_buffers()))==buffers
                with np.load(ROOT/r['prediction_path']) as z:
                    assert np.array_equal(z['target'],reference_y,equal_nan=True) and np.array_equal(z['origins'],reference_o)
                if arm=='FULL':assert np.array_equal(p,reference), 'HISTORICAL_FULL_REPLAY_NOT_EXACT'
                records.append(dict(dataset=d,seed=seed,arm=arm,checkpoint_path=cp['checkpoint_path'],checkpoint_sha256=cp['checkpoint_sha256'],selected_epoch=cp['epoch'],zeroed_B_names=changed,**r))
                checks.append(dict(dataset=d,seed=seed,arm=arm,unchanged_during_forward=True,only_declared_B_changed=True,full_exact=arm=='FULL'))
                save(OUT/'predictions.json',records);save(OUT/'checks.json',checks)
                status['predictions_completed']=len(records);save(OUT/'status.json',status)
                print(d,seed,arm,r['metrics']['mse'],flush=True)
            restore(model,state);assert tensor_hash(cpu_state(model))==tensor_hash(state)
            model=None;cleanup()
        hashes(history);hashes(sources);hashes(refs)
        assert len(records)==16
        status.update(status='COMPLETE',historical_files_preserved=len(history));save(OUT/'status.json',status)
    except BaseException as exc:
        status.update(status='INCONCLUSIVE_EXECUTION',error=repr(exc));save(OUT/'status.json',status)
        save(OUT/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        model=None;cleanup()
        if watch:watch.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['run'])
    parser.parse_args()
    run()
