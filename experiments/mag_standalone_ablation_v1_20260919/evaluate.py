from .common import *

def load(r):
    m=build(r['arm'],r['seed'],r['source'])
    if r['arm']!='F0':restore(m,torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=True))
    return m

def evaluate(watch):
    check_seal();assert sum(1 for _ in open(OUT/'UPDATE_LEDGER.jsonl'))==MAIN_CAP
    choices=read(OUT/'LR_SELECTION.json');models=[]
    for source in SOURCES:
        for arm in ARMS:
            for seed in SEEDS:
                r=read(OUT/'fits'/fit_id(source,arm,seed,choices[source][arm]['lr'])/'receipt.json')
                for stage,c in [('selected',r['selected']),('fixed1024',r['checkpoints'][-1])]:models.append(dict(source=source,arm=arm,seed=seed,lr=r['lr'],stage=stage,**c))
        for seed in SEEDS:
            for stage in ['selected','fixed1024']:
                p=ROOT/snapshot()['snapshot']/'model.safetensors'
                models.append(dict(source=source,arm='F0',seed=seed,lr=None,stage=stage,step=0,checkpoint=str(p.relative_to(ROOT)),sha256=sha(p)))
    assert len(models)==24
    for r in models:assert sha(ROOT/r['checkpoint'])==r['sha256']
    allmodels=models+read(OUT/'BASELINE_MANIFEST.json')['reused_models'];assert len(allmodels)==48
    save(OUT/'MODEL_SELECTION.json',allmodels)
    if not (OUT/'EVALUATION_SEAL.json').exists():save(OUT/'EVALUATION_SEAL.json',dict(at=time.time(),selection_sha256=sha(OUT/'MODEL_SELECTION.json'),main_updates=MAIN_CAP))
    else:assert read(OUT/'EVALUATION_SEAL.json')['selection_sha256']==sha(OUT/'MODEL_SELECTION.json')
    done=read(OUT/'PREDICTIONS_MANIFEST.json') if (OUT/'PREDICTIONS_MANIFEST.json').exists() else read(OUT/'REUSED_PREDICTIONS.json')
    for r in models:
        for panel in [r['source']]+(['electricity_transfer',NEW] if r['source']=='electricity' else []):
            for kind in ['standard','shape']:
                key=f"{panel}__{kind}__{r['arm']}__s{r['seed']}__{r['stage']}"
                if key in done:assert sha(ROOT/done[key]['path'])==done[key]['sha256'];continue
                meta={k:r[k] for k in ['source','arm','seed','lr','stage','step']};meta.update(panel=panel,kind=kind,checkpoint_sha256=r['sha256'])
                matches=[v for v in done.values() if all(v.get(k)==meta[k] for k in ['source','arm','panel','kind','checkpoint_sha256']) and (r['arm']=='F0' or v['seed']==r['seed'])]
                if matches:
                    v=matches[0];assert sha(ROOT/v['path'])==v['sha256'];record=dict(**meta,path=v['path'],sha256=v['sha256'],shape=v['shape'],reused=True,inference_seconds=0.)
                else:
                    watch.boundary();m=load(r);before=tensor_hash(m.state_dict());x,s=inputs(panel,kind)
                    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();p=predict(m,x,s,32,watch);elapsed=time.perf_counter()-start
                    assert before==tensor_hash(m.state_dict());peak=torch.cuda.max_memory_allocated();del m;cleanup()
                    fresh=load(r);replay=predict(fresh,x[:32],s[:32],32,watch);assert np.array_equal(replay,p[:32]);del fresh;cleanup()
                    path=CACHE/'predictions'/f'{key}.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,p)
                    record=dict(**meta,path=str(path.relative_to(ROOT)),sha256=sha(path),shape=list(p.shape),reused=False,inference_seconds=elapsed,peak_allocated=peak,restore_exact=True,frozen_unchanged=True)
                done[key]=record;save(OUT/'PREDICTIONS_MANIFEST.json',done);status(execution='PREDICTING_E',prediction_views=len(done));print('E_PREDICTION',len(done),key,flush=True)
    assert len(done)==192
    save(OUT/'ALL_PREDICTIONS_SAVED.json',dict(at=time.time(),manifest_sha256=sha(OUT/'PREDICTIONS_MANIFEST.json'),views=192,new_arm_views=96,reused_baseline_views=96,new_E_scored=False))
