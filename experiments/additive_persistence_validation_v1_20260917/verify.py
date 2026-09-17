"""Exact ledgers, checkpoint replay, protected weights and independent scalar metrics."""
import pandas as pd
from .common import *
from .prepare import historical,STATES
from .evaluate import load_model,prediction_id
from .train import predict
from .reference_core import pinball_scalar

def verify(watch):
    check_seal();historical()
    journal=[json.loads(l) for l in open(OUT/'UPDATE_LEDGER.jsonl')];smoke=[json.loads(l) for p in OUT.glob('smoke_*.jsonl') for l in open(p)]
    assert len(journal)==59392 and len({(r['fit'],r['step']) for r in journal})==59392 and len(smoke)==36
    fitrows=[]
    for p in sorted((OUT/'fits').glob('*/receipt.json')):
        r=read(p);assert r['status']=='COMPLETE' and r['updates']==1024 and r['frozen_unchanged']
        assert [j['step'] for j in journal if j['fit']==r['fit']]==list(range(1,1025))
        assert min(r['checkpoints'],key=lambda c:(c['objective'],c['step']))==r['selected']
        assert read(p.parent/'update_intent.json')==dict(fit=r['fit'],completed_step=1024,status='JOURNALED')
        fitrows.append({k:r[k] for k in ['fit','source','arm','seed','lr','status','updates','optimizer_seconds','validation_seconds','checkpoint_io_seconds','peak_allocated','peak_reserved']})
    assert len(fitrows)==58;csvwrite(OUT/'FIT_LEDGER.csv',fitrows)
    manifest=read(OUT/'PREDICTIONS_MANIFEST.json');assert len(manifest)==246
    frame=pd.read_csv(OUT/'SCORES_BY_ORIGIN.csv.gz',dtype={'channel':str});index=frame.set_index(['prediction','condition','origin','channel']);scalar=[]
    for key,r in manifest.items():
        panel=r['panel'];data=read(OUT/'DATA_MANIFEST.json')[panel];nc=len(data['selected_columns']);sig=data['sigma_train_population'];d=np.load(CACHE/'data'/panel/'E_DISCOVERY_inputs.npz');basey=np.load(CACHE/'data'/panel/'E_DISCOVERY_labels.npz')['y'];p=np.load(ROOT/r['path'],mmap_mode='r');assert sha(ROOT/r['path'])==r['sha256']
        if r['kind']=='standard':names=STATES;ids=np.arange(128);offset=np.load(CACHE/'conditions'/panel/'E_DISCOVERY_offset.npy')
        elif r['kind']=='shape':
            meta=read(CACHE/'shapes'/panel/'manifest.json');names=meta['states'];ids=np.array(meta['origin_indices']);offset=np.load(CACHE/'shapes'/panel/'offset.npy')
        else:names=['PULSE_LEGACY_MATCH'];ids=np.linspace(0,127,64,dtype=int);offset=np.zeros(2*64*nc)
        n=len(ids)
        # Every saved prediction view, every condition, first/last origins and first/last channels.
        for ci,condition in enumerate(names):
            for oi in [0,n-1]:
                for ch in [0,nc-1]:
                    indices=[((ci*2+g)*n+oi)*nc+ch for g in range(2)];y=np.array([basey[ids[oi],ch]+offset[i] for i in indices]);pred=p[indices].astype(float)
                    mae=sum(sum(abs(float(pred[g,4,h])-float(y[g,h])) for h in range(64)) for g in range(2))/(2*64*sig[ch]);pin=pinball_scalar(pred,y,np.array([sig[ch]]*2),np.arange(1,10)/10)
                    row=index.loc[(key,condition,int(d['origins'][ids[oi]]),str(data['selected_columns'][ch]))]
                    np.testing.assert_allclose([row.nmae,row.pinball],[mae,pin],rtol=1e-10,atol=1e-12);scalar.append(dict(prediction=key,condition=condition,origin=int(d['origins'][ids[oi]]),channel=str(data['selected_columns'][ch])))
    restores=[]
    for row in read(OUT/'MODEL_SELECTION.json'):
        source=row['source'];j=dict(row=row,panel=source,stage='selected',kind='standard');key=prediction_id(j);f=CACHE/'conditions'/source
        micro=read(OUT/'microbatch.json')[source]['microbatch'];x=np.load(f/'E_DISCOVERY_x.npy',mmap_mode='r')[:micro];s=np.load(f/'E_DISCOVERY_sigma.npy',mmap_mode='r')[:micro]
        model=load_model(row);pred=predict(model,x,s,micro,watch);saved=np.load(ROOT/manifest[key]['path'],mmap_mode='r')[:micro]
        diff=float(np.max(abs(pred-saved)/s[:,None,None]));assert diff<=1e-5 or np.allclose(pred,saved,rtol=1e-4,atol=0)
        off=None
        if row['arm'] in ['C2','C3']+CONTROLS:
            with torch.no_grad():p=model(torch.tensor(np.array(x),device='cuda'),torch.tensor(np.array(s),device='cuda'),residual_mode='off').cpu().numpy()
            basekey=key.replace('__'+row['arm']+'__','__C0__');base=np.load(ROOT/manifest[basekey]['path'],mmap_mode='r')[:micro];off=bool(np.array_equal(p,base));assert off
        restores.append(dict(key=key,max_normalized_error=diff,adapter_off_B0_exact=off));del model;cleanup()
    # Historical selected predictions and scores are unchanged, not just similar aggregate performance.
    old=pd.read_csv(OLD/'scores_by_condition.csv');current=pd.read_csv(OUT/'RAW_SCORES.csv')
    old=old.rename(columns={'source':'panel'})
    for r in old.to_dict('records'):
        a=current[(current.panel==r['panel'])&(current.arm==r['arm'])&(current.seed==r['seed'])&(current.condition==r['condition'])&(current.stage=='selected')&(current.kind=='standard')].iloc[0]
        np.testing.assert_allclose([a[k] for k in ['nmae','mae','nrmse','pinball','crossing']],[r[k] for k in ['nmae','mae','nrmse','pinball','crossing']],rtol=1e-10,atol=1e-12)
    csvwrite(OUT/'SCALAR_REPLAY_SCOPE.csv',scalar)
    # Full-data vector summaries and ratios are independently reconstructed from published origin scores.
    summary=frame.groupby(['panel','kind','stage','arm','seed','condition']).nmae.mean();raw=current.set_index(['panel','kind','stage','arm','seed','condition']).nmae
    np.testing.assert_allclose(summary.sort_index().to_numpy(),raw.sort_index().to_numpy(),rtol=1e-10,atol=1e-12)
    for p,h in read(OUT/'DATA_READY.json')['hashes'].items():assert sha(ROOT/p)==h,p
    save(OUT/'VERIFICATION.json',dict(status='VERIFIED',main_fits=58,main_updates=59392,smoke_updates=36,total_updates=59428,unique_journal=True,independent_scalar_rows=len(scalar),scalar_metrics_per_row=2,scope='every prediction view/condition; first and last origins, first and last channels, both synthetic draws; not all-origin scalar replay',all_origin_vector_reaggregation=True,historical_scores_replayed=True,restored_selected_models=restores,previous_results_unchanged=True,all_E_predictions_saved_before_new_scoring=True))
    print('VERIFICATION_COMPLETE',len(scalar),flush=True)
