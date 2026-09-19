from .common import *
from experiments.persistence_evidence_extension_20260918 import common as ext

def selection():
    models=[]
    for source in SOURCES:
        for seed in SEEDS:
            for arm in ARMS:
                r=read(OUT/'fits'/fit_id(source,arm,seed)/'receipt.json')
                for stage,c in [('selected',r['selected']),('fixed1024',next(c for c in r['checkpoints'] if c['step']==1024))]:models.append(dict(source=source,seed=seed,arm=arm,stage=stage,lr=r['lr'],**c))
            for arm in ['C3','MAG_ONLY']:
                files=list((ROOT/'results/c3_training_factorial_20260918/fits').glob(f'{source}_{arm}_b{seed}_i{seed}_o{seed}_lr*/receipt.json'));assert len(files)==1
                r=read(files[0])
                for stage,c in [('selected',r['selected']),('fixed1024',next(c for c in r['checkpoints'] if c['step']==1024))]:models.append(dict(source=source,seed=seed,arm=arm,stage=stage,lr=r['lr'],context_only=True,**c))
            b=old.baseline_row(source,seed)
            for stage in ['selected','fixed1024']:models.append(dict(source=source,seed=seed,arm='B0',stage=stage,lr=None,context_only=True,checkpoint=b['checkpoint'],sha256=b['sha256'],step=b['step']))
    assert len(models)==64
    for m in models:assert sha(ROOT/m['checkpoint'])==m['sha256']
    return models

def load(r):
    m=build(r['arm'],r['seed'],r['source'])
    if r['arm']!='B0':restore(m,torch.load(ROOT/r['checkpoint'],weights_only=True,map_location='cpu'))
    return m

def evaluate(watch):
    check_seal();assert sum(1 for _ in open(OUT/'UPDATE_LEDGER.jsonl'))==MAIN_CAP
    rows=selection();save(OUT/'MODEL_SELECTION.json',rows)
    seal=dict(at=time.time(),selection_sha256=sha(OUT/'MODEL_SELECTION.json'),updates=MAIN_CAP)
    if (OUT/'EVALUATION_SEAL.json').exists():assert read(OUT/'EVALUATION_SEAL.json')['selection_sha256']==seal['selection_sha256']
    else:save(OUT/'EVALUATION_SEAL.json',seal)
    prior=[]
    for path in [old.OUT/'PREDICTIONS_MANIFEST.json',ROOT/'results/c3_training_factorial_20260918/PREDICTIONS.json',ROOT/'results/persistence_evidence_extension_20260918/PREDICTIONS.json']:
        items=read(path);prior+=list(items.values()) if isinstance(items,dict) else items
    done=read(OUT/'PREDICTIONS.json') if (OUT/'PREDICTIONS.json').exists() else {}
    for row in rows:
        for panel in [row['source']]+(['electricity_transfer'] if row['source']=='electricity' else []):
            for kind in ['standard','shape']:
                key=f"{panel}__{kind}__{row['arm']}__s{row['seed']}__{row['stage']}"
                if key in done:assert sha(ROOT/done[key]['path'])==done[key]['sha256'];continue
                meta=dict(panel=panel,kind=kind,arm=row['arm'],source=row['source'],seed=row['seed'],stage=row['stage'],step=row['step'],lr=row['lr'],checkpoint_sha256=row['sha256'])
                matches=[r for r in list(done.values())+prior if r.get('panel')==panel and r.get('kind')==kind and r.get('checkpoint_sha256')==row['sha256'] and (r.get('arm')==row['arm'] or (row['arm']=='PLAIN' and r.get('arm')=='C2') or (row['arm']=='B0' and r.get('arm')=='C0'))]
                if matches:
                    r=matches[0];assert sha(ROOT/r['path'])==r['sha256'];done[key]=dict(**meta,path=r['path'],sha256=r['sha256'],shape=r['shape'],reused=True,inference_seconds=0.)
                else:
                    watch.boundary();m=load(row);before=tensor_hash(m.state_dict());x,s=ext.inputs(panel,kind);torch.cuda.reset_peak_memory_stats();t=time.perf_counter();p=predict(m,x,s,32,watch);elapsed=time.perf_counter()-t
                    assert before==tensor_hash(m.state_dict());cost=dict(inference_seconds=elapsed,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
                    del m;cleanup();fresh=load(row);replay=predict(fresh,x[:32],s[:32],32,watch);assert np.array_equal(replay,p[:32]),'RESTORE_MISMATCH'
                    path=CACHE/'predictions'/f'{key}.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,p)
                    done[key]=dict(**meta,path=str(path.relative_to(ROOT)),sha256=sha(path),shape=list(p.shape),reused=False,restore_exact=True,state_unchanged=True,**cost)
                    del fresh,p;cleanup()
                save(OUT/'PREDICTIONS.json',done);status(execution='E_PREDICTION',prediction_views=len(done));print('PREDICTION',len(done),key,flush=True)
    assert len(done)==192
    check_seal();save(OUT/'ALL_PREDICTIONS_SAVED.json',dict(at=time.time(),views=len(done),manifest_sha256=sha(OUT/'PREDICTIONS.json'),new_E_scored=False))
