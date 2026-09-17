"""Inference only; no optimizer construction, backward or parameter updates."""
import argparse,traceback
import pandas as pd
from .common import *
from .prepare import prepare,seal

def run():
    old.setup();prepare();seal();check_seal();watch=Watch();manifest=read(OUT/'PREDICTIONS.json') if (OUT/'PREDICTIONS.json').exists() else {}
    checks=read(OUT/'MODEL_CHECKS.json') if (OUT/'MODEL_CHECKS.json').exists() else {};oldmanifest=read(PRIOR/'PREDICTIONS_MANIFEST.json')
    try:
        watch.boundary(startup=True)
        for j in read(OUT/'INFERENCE_PLAN.json'):
            key=j['id']
            if key in manifest:assert sha(ROOT/manifest[key]['path'])==manifest[key]['sha256'];continue
            watch.boundary();row=j['row'];model=load_model(row);model.requires_grad_(False);assert not any(p.requires_grad for p in model.parameters())
            before=tensor_hash(model.state_dict());source=row['source'];x0,s0=inputs(source,'standard');refkey=f"{source}__standard__selected__{row['arm']}__{row['seed']}"
            # Test the exact trained forward before applying any diagnostic gate change.
            replay=predict(model,x0[:32],s0[:32],32,watch);ref=np.load(ROOT/oldmanifest[refkey]['path'],mmap_mode='r')[:32]
            assert np.array_equal(replay,ref),'OLD_CHECKPOINT_FORWARD_MISMATCH'
            model.arm=j['gate_arm'];x,s=inputs(j['panel'],j['kind']);torch.cuda.reset_peak_memory_stats();old.sync();t=time.perf_counter()
            alias=None
            if j['panel']=='neso_2025' and j['arm']=='C1' and row['step']==0:
                alias=key.replace('__C1__','__C0__');a=manifest[alias];path=ROOT/a['path'];p=np.load(path,mmap_mode='r');assert np.array_equal(predict(model,x[:32],s[:32],32,watch),p[:32])
            else:
                p=predict(model,x,s,32,watch);path=CACHE/'predictions'/f'{key}.npy';path.parent.mkdir(exist_ok=True);np.save(path,p)
            old.sync();elapsed=time.perf_counter()-t;assert before==tensor_hash(model.state_dict()),'FROZEN_WEIGHTS_CHANGED'
            restored=load_model(row);restored.requires_grad_(False);restored.arm=j['gate_arm'];repeat=predict(restored,x[:32],s[:32],32,watch)
            np.testing.assert_allclose(repeat,p[:32],rtol=1e-4,atol=0)
            after=tensor_hash(restored.state_dict());assert before==after
            checks[key]=dict(original_forward_exact=True,restored_fixed_gate_forward=True,all_weights_frozen=True,state_unchanged=True,weight_sha256=before,checkpoint_sha256=row['sha256'],optimizer_updates=0,gate_intervention=j['arm'] in SWAPS)
            manifest[key]=dict(panel=j['panel'],kind=j['kind'],arm=j['arm'],trained_arm=j['trained_arm'],gate_arm=j['gate_arm'],seed=j['seed'],path=str(path.relative_to(ROOT)),sha256=sha(path),shape=list(p.shape),alias_of=alias,inference_seconds=elapsed,peak_allocated=torch.cuda.max_memory_allocated(),checkpoint_sha256=row['sha256'])
            save(OUT/'PREDICTIONS.json',manifest);save(OUT/'MODEL_CHECKS.json',checks);save(OUT/'status.json',dict(execution='RUNNING',new_prediction_views=len(manifest),cap=111,optimizer_updates=0));print('SAVED',len(manifest),key,flush=True)
            del model,restored,p;old.cleanup()
        # All six output controls inherit source V values, with no NESO selection.
        cal=read(PRIOR/'CALIBRATION_SELECTION.json');sig=np.load(CACHE/'conditions/neso_2025/E_DISCOVERY_sigma.npy')
        for seed in [81551,81552,81553]:
            key=f'neso_2025__standard__C0__{seed}';a=np.load(ROOT/manifest[key]['path']).astype(float);b=np.load(ROOT/manifest[key.replace('__C0__','__C2__')]['path']).astype(float);c=cal[f'electricity_{seed}']
            for arm in ['C2_SHRINK','C0_BIAS']:
                newkey=key.replace('__C0__','__'+arm+'__')
                if newkey in manifest:assert sha(ROOT/manifest[newkey]['path'])==manifest[newkey]['sha256'];continue
                p=(a if c['alpha']==0 else b if c['alpha']==1 else a+c['alpha']*(b-a)) if arm=='C2_SHRINK' else a+c['beta']*sig[:,None,None]
                path=CACHE/'predictions'/f'{newkey}.npy';np.save(path,p);manifest[newkey]=dict(panel='neso_2025',kind='standard',arm=arm,seed=seed,path=str(path.relative_to(ROOT)),sha256=sha(path),shape=list(p.shape),calibration_from='original Electricity V only',calibration=c,forward_models=2 if arm=='C2_SHRINK' else 1)
        assert len(manifest)==111;save(OUT/'PREDICTIONS.json',manifest);check_seal()
        save(OUT/'ALL_PREDICTIONS_SAVED.json',dict(at=time.time(),new_prediction_views=111,manifest_sha256=sha(OUT/'PREDICTIONS.json'),optimizer_updates=0,new_scores_computed=False))
        from .score import score
        score();save(OUT/'status.json',dict(execution='COMPLETE_COMPUTE',new_prediction_views=111,optimizer_updates=0,automatic_successor=False))
    except BaseException as e:
        save(OUT/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),at=time.time()));save(OUT/'status.json',dict(execution='PARTIAL_RESOURCE' if isinstance(e,ResourceError) else 'INVALID_IMPLEMENTATION',error=str(e),optimizer_updates=0));raise
    finally:watch.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','seal','run']);a=p.parse_args();old.setup()
    if a.action=='prepare':prepare()
    elif a.action=='seal':prepare();seal()
    else:run()
