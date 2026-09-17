"""Seal selections, save every E prediction, then permit scorer label access."""
import pandas as pd
from .common import *
from .reference_core import STATES,rng,pinball_scalar
from .train import predict

def evaluation_seal():
    check_seal()
    selected=read(OUT/'MODEL_SELECTION.json');assert len(selected)==12
    receipts=[read(p) for p in (OUT/'fits').glob('*/receipt.json')]
    assert len(receipts)==24 and all(r['updates']==1024 and r['status']=='COMPLETE' for r in receipts)
    journal=[json.loads(l) for l in open(OUT/'optimizer.jsonl')]
    assert len(journal)==24576
    for row in selected:assert sha(ROOT/row['checkpoint'])==row['sha256']
    if not (OUT/'GLOBAL_EVALUATION_SEAL.json').exists():
        save(OUT/'GLOBAL_EVALUATION_SEAL.json',dict(selected=selected,LR_SELECTION_sha256=sha(OUT/'LR_SELECTION.json'),
             MODEL_SELECTION_sha256=sha(OUT/'MODEL_SELECTION.json'),execution_seal_sha256=sha(OUT/'EXECUTION_SEAL.json'),
             main_updates=24576,smoke_updates=12,postprocessing='none; native quantile .5, no sorting',
             reference_label='reused development period, not independent test',sealed_at=time.time()))
    else:assert read(OUT/'GLOBAL_EVALUATION_SEAL.json')['selected']==selected

@torch.no_grad()
def diagnostics(model,source,micro,watch,mode):
    f=CACHE/'conditions'/source;x=np.load(f/'V_SELECT_x.npy',mmap_mode='r');s=np.load(f/'V_SELECT_sigma.npy',mmap_mode='r')
    y=np.load(f/'V_SELECT_y.npy',mmap_mode='r');pred=[]
    for lo in range(0,len(x),micro):
        watch.boundary();xx=torch.tensor(np.array(x[lo:lo+micro]),device='cuda');ss=torch.tensor(np.array(s[lo:lo+micro]),device='cuda')
        pred.append(model(xx,ss,persistence_mode=mode).cpu().numpy())
    p=np.concatenate(pred);return np.mean(np.abs(p[:,4].astype(float)-y)/np.tile(read(OUT/'DATA_MANIFEST.json')[source]['sigma_train_population'],len(x)//4)[:,None],axis=-1).reshape(10,2,64,4).mean((1,2,3)).tolist()

def evaluate(watch):
    evaluation_seal()
    selected=read(OUT/'MODEL_SELECTION.json')
    jobs=list(selected)
    manifest=read(OUT/'predictions_manifest.json') if (OUT/'predictions_manifest.json').exists() else {}
    for key,r in read(OUT/'BASELINE_MANIFEST.json')['predictions'].items():
        assert sha(ROOT/r['path'])==r['sha256']
        manifest[key]=dict(r,reused=True,arm='C0')
    save(OUT/'predictions_manifest.json',manifest)
    resources=read(OUT/'inference_resources.json') if (OUT/'inference_resources.json').exists() else {}
    diag=read(OUT/'C3_validation_diagnostics.json') if (OUT/'C3_validation_diagnostics.json').exists() else {}
    for row in jobs:
        source,arm,seed=row['source'],row['arm'],row['seed'];key=f'{source}_{arm}_{seed}'
        if key in manifest:assert sha(ROOT/manifest[key]['path'])==manifest[key]['sha256'];continue
        micro=read(OUT/'microbatch.json')[source]['microbatch'];f=CACHE/'conditions'/source
        x=np.load(f/'E_DISCOVERY_x.npy',mmap_mode='r');s=np.load(f/'E_DISCOVERY_sigma.npy',mmap_mode='r')
        if arm=='SEASONAL_DAY':
            t=time.perf_counter();period=24 if source=='electricity' else 96
            median=x[:,-period:][:,np.arange(64)%period];p=np.repeat(median[:,None,:],9,axis=1)
            cost=dict(inference_seconds=time.perf_counter()-t,peak_allocated=0,peak_reserved=0,device='cpu',series=len(x))
        else:
            model=build(arm,seed,source=source)
            if seed:restore(model,torch.load(ROOT/row['checkpoint'],weights_only=True,map_location='cpu'))
            watch.boundary();torch.cuda.reset_peak_memory_stats();sync();t=time.perf_counter()
            p=predict(model,x,s,micro,watch);sync()
            cost=dict(inference_seconds=time.perf_counter()-t,peak_allocated=torch.cuda.max_memory_allocated(),
                      peak_reserved=torch.cuda.max_memory_reserved(),device='cuda',series=len(x),microbatch=micro)
            # All ten V conditions are reported for selected models; selection used only four.
            if seed:
                vx=np.load(f/'V_SELECT_x.npy',mmap_mode='r');vs=np.load(f/'V_SELECT_sigma.npy',mmap_mode='r')
                vp=predict(model,vx,vs,micro,watch);vpath=CACHE/'predictions'/f'{key}_V.npy';vpath.parent.mkdir(exist_ok=True)
                np.save(vpath,vp)
            if arm=='C3':
                diag[key]={mode:diagnostics(model,source,micro,watch,mode) for mode in ['zero','permute']}
                save(OUT/'C3_validation_diagnostics.json',diag)
            del model;cleanup()
        path=CACHE/'predictions'/f'{key}_E.npy';path.parent.mkdir(exist_ok=True);np.save(path,p)
        manifest[key]=dict(source=source,arm=arm,seed=seed,path=str(path.relative_to(ROOT)),sha256=sha(path),
                           shape=list(p.shape),quantile_interpretation='degenerate repeated point' if arm=='SEASONAL_DAY' else 'native unsorted .1-.9')
        resources[key]=cost;save(OUT/'predictions_manifest.json',manifest);save(OUT/'inference_resources.json',resources)
        print('E_PREDICTIONS_SAVED',key,flush=True)
    assert len(manifest)==16 # 12 new selected models + 4 reused B0 baselines.
    save(OUT/'ALL_E_PREDICTIONS_SAVED.json',dict(count=len(manifest),manifest_sha256=sha(OUT/'predictions_manifest.json'),timestamp=time.time(),labels_scored=False))
    score()

def labels(source):
    assert (OUT/'ALL_E_PREDICTIONS_SAVED.json').exists(),'E_PREDICTIONS_NOT_COMPLETE'
    y0=np.load(CACHE/'data'/source/'E_DISCOVERY_labels.npz')['y'].reshape(512,64)
    offset=np.load(CACHE/'conditions'/source/'E_DISCOVERY_offset.npy')
    return np.tile(y0,(20,1))+offset[:,None]

def score():
    assert read(OUT/'ALL_E_PREDICTIONS_SAVED.json')['manifest_sha256']==sha(OUT/'predictions_manifest.json')
    rows=[];physical={};quantiles=np.arange(1,10)/10
    for key,receipt in read(OUT/'predictions_manifest.json').items():
        source,arm,seed=receipt['source'],receipt['arm'],receipt['seed'];p=np.load(ROOT/receipt['path'],mmap_mode='r').astype(np.float64)
        assert sha(ROOT/receipt['path'])==receipt['sha256'];y=labels(source)
        sigma=np.tile(read(OUT/'DATA_MANIFEST.json')[source]['sigma_train_population'],20*128)
        mae=np.mean(np.abs(p[:,4]-y),axis=-1);mse=np.mean((p[:,4]-y)**2,axis=-1)
        error=y[:,None,:]-p;q=quantiles[None,:,None]
        pin=(2*np.maximum(q*error,(q-1)*error)/sigma[:,None,None]).mean((1,2))
        cross=(np.diff(p,axis=1)<0).mean((1,2))
        metrics={k:v.reshape(10,2,128,4).mean(axis=1) for k,v in
                 dict(nmae=mae/sigma,mae=mae,normalized_mse=mse/sigma**2,pinball=pin,crossing=cross).items()}
        origins=np.load(CACHE/'data'/source/'E_DISCOVERY_inputs.npz')['origins'];tag=np.load(CACHE/'conditions'/source/'E_history_subset.npy').reshape(128,4)
        physical[source]=dict(negative_transformed_target_values=int((y<0).sum()),negative_target_series=int((y<0).any(1).sum()),
                              meaning='synthetic mathematical stress, not certified physical events')
        for j,condition in enumerate(STATES):
            for i,origin in enumerate(origins):
                for channel in range(4):
                    rows.append(dict(source=source,arm=arm,seed=seed,condition=condition,origin=int(origin),channel=channel,
                                     history_subset=bool(tag[i,channel]),**{k:float(v[j,i,channel]) for k,v in metrics.items()}))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'scores_by_origin.csv',index=False)
    grouped=frame.groupby(['source','arm','seed','condition'],sort=True)[['nmae','mae','normalized_mse','pinball','crossing']].mean().reset_index()
    # Correct channel-level NRMSE then equal channel averaging, not global sqrt(mean MSE).
    rmse=frame.groupby(['source','arm','seed','condition','channel']).normalized_mse.mean().pow(.5).groupby(level=[0,1,2,3]).mean()
    grouped=grouped.merge(rmse.rename('nrmse').reset_index(),on=['source','arm','seed','condition'])
    grouped.to_csv(OUT/'scores_by_condition.csv',index=False);save(OUT/'physical_stress_counts.json',physical)
    paired(frame)

def paired(frame):
    contrasts=[('C3','C0'),('C3','C1'),('C3','C2'),('C2','C0'),('C2','C1'),('C1','C0')]
    rows=[]
    for source in SOURCES:
        sf=frame[frame.source==source];period=24 if source=='electricity' else 96
        panels={'FAULT':[x for x in STATES if x.startswith('POINT') or x.startswith('BURST')],
                'REFERENCE':['REFERENCE'],'SHIFT4':['SHIFT4'],'SHIFT8':['SHIFT8'],'SHIFT':['SHIFT4','SHIFT8'],'SHIFT_POINT':['SHIFT_POINT'],'HISTORY_SUBSET':['REFERENCE']}
        for panel,states in panels.items():
            f=sf[sf.condition.isin(states)]
            if panel=='HISTORY_SUBSET':f=f[f.history_subset]
            # Conditions/generator draws/channels/optimizer seeds are not independent days.
            table=f.groupby(['arm','origin']).nmae.mean().unstack(0)
            blocks=table.index.to_numpy()//(7*period);unique=np.unique(blocks)
            r=rng(84500,source,panel);weights=np.empty((2000,len(table)),np.float64)
            for k in range(2000):
                chosen=r.choice(unique,len(unique),replace=True)
                weights[k]=np.array([(chosen==b).sum() for b in blocks])
            weights/=weights.sum(axis=1,keepdims=True)
            for new,old in contrasts:
                a=table[new].to_numpy();b=table[old].to_numpy();ba=weights@a;bb=weights@b
                gain=100*(1-a.mean()/b.mean());draws=100*(1-ba/bb)
                lo,hi=np.quantile(draws,[.025,.975])
                seed_gains={f'seed_{seed}_gain_pct':float(100*(1-f[(f.arm==new)&(f.seed==seed)].groupby('origin').nmae.mean().mean()/f[(f.arm==old)&(f.seed==seed)].groupby('origin').nmae.mean().mean())) for seed in [81551,81552]}
                rows.append(dict(**seed_gains,source=source,panel=panel,new=new,baseline=old,new_nmae=float(a.mean()),baseline_nmae=float(b.mean()),
                                 gain_pct=float(gain),ci_low_pct=float(lo),ci_high_pct=float(hi),paired_difference=float((a-b).mean()),
                                 origins=len(table),time_blocks=len(unique),bootstrap_draws=2000))
    csvwrite(OUT/'paired_effects.csv',rows)

def verify(watch):
    frame=pd.read_csv(OUT/'scores_by_origin.csv');scalar=[];restore_checks=[]
    for key,r in read(OUT/'predictions_manifest.json').items():
        source=r['source'];p=np.load(ROOT/r['path'],mmap_mode='r');y=labels(source);sig=read(OUT/'DATA_MANIFEST.json')[source]['sigma_train_population']
        # Independently verify first and last origins, all channels/states, both generators.
        for state_index,condition in enumerate(STATES):
            for origin_index in [0,127]:
                for channel in range(4):
                    indices=[((state_index*2+g)*128+origin_index)*4+channel for g in range(2)]
                    maes=[sum(abs(float(p[i,4,h])-float(y[i,h])) for h in range(64))/64 for i in indices]
                    pin=pinball_scalar(p[indices].astype(float),y[indices],np.array([sig[channel]]*2),np.arange(1,10)/10)
                    origin=int(np.load(CACHE/'data'/source/'E_DISCOVERY_inputs.npz')['origins'][origin_index])
                    row=frame[(frame.source==source)&(frame.arm==r['arm'])&(frame.seed==r['seed'])&(frame.condition==condition)&(frame.origin==origin)&(frame.channel==channel)].iloc[0]
                    np.testing.assert_allclose(row.nmae,np.mean(maes)/sig[channel],rtol=1e-10,atol=1e-12)
                    np.testing.assert_allclose(row.pinball,pin,rtol=1e-10,atol=1e-12)
                    scalar.append(dict(key=key,condition=condition,origin=origin,channel=channel))
    for row in read(OUT/'MODEL_SELECTION.json'):
        key=f"{row['source']}_{row['arm']}_{row['seed']}";source=row['source'];model=build(row['arm'],row['seed'],source=source)
        restore(model,torch.load(ROOT/row['checkpoint'],map_location='cpu',weights_only=True))
        f=CACHE/'conditions'/source;x=np.load(f/'E_DISCOVERY_x.npy',mmap_mode='r');s=np.load(f/'E_DISCOVERY_sigma.npy',mmap_mode='r')
        micro=read(OUT/'microbatch.json')[source]['microbatch'];p=predict(model,x[:micro],s[:micro],micro,watch)
        original=np.load(ROOT/read(OUT/'predictions_manifest.json')[key]['path'],mmap_mode='r')[:micro]
        diff=float(np.max(np.abs(p-original)/s[:micro,None,None]));assert diff<=1e-5 or np.allclose(p,original,rtol=1e-4,atol=0)
        recovered=None
        if row['arm'] in ['C2','C3']:
            with torch.no_grad():
                off=model(torch.tensor(np.array(x[:micro]),device='cuda'),torch.tensor(np.array(s[:micro]),device='cuda'),residual_mode='off').cpu().numpy()
            baseline=read(OUT/'BASELINE_MANIFEST.json')['predictions'][f"{source}_C0_{row['seed']}"]
            original_b0=np.load(ROOT/baseline['path'],mmap_mode='r')[:micro]
            recovered=float(np.max(np.abs(off-original_b0)/s[:micro,None,None]));assert recovered==0.
        restore_checks.append(dict(key=key,max_normalized_error=diff,adapter_off_B0_max_normalized_error=recovered));del model;cleanup()
    journal=[json.loads(l) for l in open(OUT/'optimizer.jsonl')]
    assert len(journal)==24576 and len({(r['fit'],r['step']) for r in journal})==24576
    fits=list((OUT/'fits').glob('*/receipt.json'));assert len(fits)==24
    for path in fits:
        r=read(path);assert r['updates']==1024 and r['frozen_unchanged']
        steps=[j['step'] for j in journal if j['fit']==r['fit']];assert steps==list(range(1,1025))
        selected=min(r['checkpoints'],key=lambda v:(v['objective'],v['step']));assert selected==r['selected']
    check_seal()
    save(OUT/'verification.json',dict(status='VERIFIED',independent_scalar_rows=len(scalar),metrics_per_row=2,
                                    selected_model_restore=restore_checks,main_updates=24576,smoke_updates=12,
                                    unique_journal_updates=True,all_frozen_weights_preserved=True,
                                    all_E_predictions_saved_before_label_scoring=True,
                                    CPU_metric_tolerance=dict(rtol=1e-10,atol=1e-12),
                                    GPU_tolerance='normalized max error 1e-5 OR rtol 1e-4'))
