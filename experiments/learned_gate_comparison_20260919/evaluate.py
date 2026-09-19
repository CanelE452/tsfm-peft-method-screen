from .common import *


def load(r):
    if r['arm'] in ARMS:
        m=build(r['arm'],r['seed'],r['source']);restore(m,torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=True));return m
    from experiments.temporal_response_peft_20260919.evaluate import load as old_load
    return old_load(r)

def evaluate(watch):
    check_seal();assert sum(1 for _ in open(OUT/'UPDATE_LEDGER.jsonl'))==MAIN_CAP
    choices=read(OUT/'LR_SELECTION.json');models=[]
    for source in SOURCES:
        for arm in ARMS:
            for seed in SEEDS:
                r=read(OUT/'fits'/fit_id(source,arm,seed,choices[source][arm]['lr'])/'receipt.json')
                for stage,c in [('selected',r['selected']),('fixed1024',r['checkpoints'][-1])]:
                    models.append(dict(source=source,arm=arm,seed=seed,lr=r['lr'],stage=stage,B0_sha256=parent.baseline_row(source,seed)['sha256'],**c))
    for r in models:assert sha(ROOT/r['checkpoint'])==r['sha256']
    for row in read(OUT/'REUSED_MODELS.json'):
        row=dict(row);row['temporal_only']=True;row['B0_sha256']=parent.baseline_row(row['source'],row['seed'])['sha256'];models.append(row)
    save(OUT/'MODEL_SELECTION.json',models)
    if not (OUT/'EVALUATION_SEAL.json').exists():save(OUT/'EVALUATION_SEAL.json',dict(at=time.time(),selection_sha256=sha(OUT/'MODEL_SELECTION.json'),main_updates=MAIN_CAP))
    else:assert read(OUT/'EVALUATION_SEAL.json')['selection_sha256']==sha(OUT/'MODEL_SELECTION.json')
    done=read(OUT/'PREDICTIONS.json') if (OUT/'PREDICTIONS.json').exists() else read(OUT/'REUSED_PREDICTIONS.json')
    for r in models:
        for panel in ([NEW] if r.get('temporal_only') else [r['source']]+(['electricity_transfer',NEW] if r['source']=='electricity' else [])):
            for kind in ['standard','shape']:
                key=f"{panel}__{kind}__{r['arm']}__s{r['seed']}__{r['stage']}"
                if key in done:assert sha(ROOT/done[key]['path'])==done[key]['sha256'];continue
                meta={k:r[k] for k in ['source','arm','seed','lr','stage','step','B0_sha256']};meta.update(panel=panel,kind=kind,checkpoint_sha256=r['sha256'])
                matches=[v for v in done.values() if all(v.get(k)==meta[k] for k in ['source','arm','seed','B0_sha256','panel','kind','checkpoint_sha256'])]
                if matches:
                    v=matches[0];assert sha(ROOT/v['path'])==v['sha256'];record=dict(**meta,path=v['path'],sha256=v['sha256'],shape=v['shape'],reused=True,inference_seconds=0.)
                else:
                    watch.boundary();m=load(r);before=tensor_hash(m.state_dict())
                    x,s=inputs(panel,kind);torch.cuda.reset_peak_memory_stats();start=time.perf_counter();p=predict(m,x,s,32,watch);elapsed=time.perf_counter()-start
                    assert before==tensor_hash(m.state_dict());peak=torch.cuda.max_memory_allocated();del m;cleanup()
                    fresh=load(r);replay=predict(fresh,x[:32],s[:32],32,watch)
                    assert np.array_equal(replay,p[:32]);del fresh;cleanup()
                    path=CACHE/'predictions'/f'{key}.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,p)
                    record=dict(**meta,path=str(path.relative_to(ROOT)),sha256=sha(path),shape=list(p.shape),reused=False,inference_seconds=elapsed,peak_allocated=peak,restore_exact=True,frozen_unchanged=True)
                done[key]=record;save(OUT/'PREDICTIONS.json',done);status(execution='PREDICTING_E',prediction_views=len(done));print('E_PREDICTION',len(done),key,flush=True)
    assert len(done)==192
    save(OUT/'ALL_PREDICTIONS_SAVED.json',dict(at=time.time(),manifest_sha256=sha(OUT/'PREDICTIONS.json'),views=192,new_views=96,old_reference_views=96,new_E_scored=False))
