import pandas as pd
from .common import *
from experiments.persistence_evidence_extension_20260918.score import metric_arrays
from experiments.outlier_signal_peft_v1_20260917.reference_core import pinball_scalar
METRICS=['nmae','mae','normalized_mse','pinball','crossing']
CONTRASTS=[('F0_MAG','F0_PLAIN'),('F0_MAG','F0'),('B0_MAG','B0_PLAIN'),('B0_MAG','B0'),('F0_PLAIN','F0'),('B0_PLAIN','B0'),('F0_MAG','B0_MAG')]

def target_data(panel,kind):
    d=np.load(data_path(panel)/'E_DISCOVERY_inputs.npz');yy=np.load(data_path(panel)/'E_DISCOVERY_labels.npz')['y'];nc=yy.shape[1]
    if kind=='standard':names=STATES;ids=np.arange(len(d['origins']));offset=np.load(panel_path(panel,kind)/'E_DISCOVERY_offset.npy')
    else:
        meta=read(panel_path(panel,kind)/'manifest.json');names=meta['states'];ids=np.asarray(meta['origin_indices']);offset=np.load(panel_path(panel,kind)/'offset.npy')
    y=np.tile(yy[ids].reshape(-1,64),(2*len(names),1))+offset[:,None];s=np.tile(d['sigma'],len(y)//nc)
    return names,ids,d['origins'][ids],nc,y,s

def weights(origins,panel):
    period=96 if panel=='ettm1' else 24;block=np.asarray(origins)//(7*period);weeks=np.unique(block);rng=np.random.default_rng(91942)
    counts=np.array([np.bincount(rng.integers(0,len(weeks),len(weeks)),minlength=len(weeks)) for _ in range(2000)])
    w=counts[:,np.searchsorted(weeks,block)].astype(float);w/=w.sum(1,keepdims=True)
    return w,len(weeks)

def score():
    check_seal();marker=read(OUT/'ALL_PREDICTIONS_SAVED.json');assert marker['manifest_sha256']==sha(OUT/'PREDICTIONS_MANIFEST.json')
    assert marker['at']>read(OUT/'EVALUATION_SEAL.json')['at'];rows=[];scalar_count=0
    for key,r in read(OUT/'PREDICTIONS_MANIFEST.json').items():
        panel,kind=r['panel'],r['kind'];names,ids,origins,nc,y,s=target_data(panel,kind);n=len(ids)
        assert sha(ROOT/r['path'])==r['sha256'];p=np.load(ROOT/r['path'],mmap_mode='r');assert p.shape==(len(y),9,64)
        val=np.concatenate([metric_arrays(p[k:k+256],y[k:k+256],s[k:k+256]) for k in range(0,len(p),256)]).reshape(len(names),2,n,nc,5).mean(1)
        for ci,condition in enumerate(names):
            for oi,origin in enumerate(origins):
                for ch in range(nc):rows.append(dict(panel=panel,kind=kind,arm=r['arm'],seed=r['seed'],stage=r['stage'],step=r['step'],condition=condition,origin=int(origin),channel=ch,**dict(zip(METRICS,map(float,val[ci,oi,ch])))))
            for oi in [0,n-1]:
                for ch in sorted({0,nc-1}):
                    ii=[((ci*2+z)*n+oi)*nc+ch for z in range(2)];pp=p[ii].astype(float);target=y[ii];sigma=s[ii]
                    a=sum(abs(float(pp[z,4,h])-float(target[z,h]))/float(sigma[z]) for z in range(2) for h in range(64))/128
                    b=pinball_scalar(pp,target,sigma,np.arange(1,10)/10)
                    np.testing.assert_allclose([a,b],val[ci,oi,ch,[0,3]],rtol=1e-10,atol=1e-11);scalar_count+=2
    f=pd.DataFrame(rows);keys=['panel','kind','arm','seed','stage','step','origin','channel'];fault=f[f.condition.str.startswith(('POINT','BURST'))].groupby(keys,as_index=False)[METRICS].mean();fault['condition']='FAULT';f=pd.concat([f,fault],ignore_index=True)
    f.to_csv(OUT/'ORIGIN_CHANNEL_SCORES.csv.gz',index=False,compression='gzip')
    origin=f.groupby(['panel','kind','arm','seed','stage','step','condition','origin'],as_index=False)[METRICS].mean();origin.to_csv(OUT/'ORIGIN_SCORES.csv.gz',index=False,compression='gzip')
    keys=['panel','kind','arm','seed','stage','step','condition']
    raw=f.groupby(keys)[METRICS].mean();rmse=f.groupby(keys+['channel']).normalized_mse.mean().pow(.5).groupby(level=list(range(len(keys)))).mean();raw.join(rmse.rename('nrmse')).reset_index().to_csv(OUT/'RAW_SCORES.csv',index=False)
    effects=[];seedrows=[];interactions=[]
    for (panel,kind,stage,condition),g in origin.groupby(['panel','kind','stage','condition']):
        table=g.groupby(['origin','arm']).nmae.mean().unstack();assert set(table.columns)==set(ALL_ARMS)
        w,nblocks=weights(table.index.to_numpy(),panel)
        metadata=dict(panel=panel,kind=kind,stage=stage,condition=condition,block_count=nblocks,origin_count=len(table),seed_count=2)
        for proposed,baseline in CONTRASTS:
            a=table[proposed].to_numpy();b=table[baseline].to_numpy();boot=100*(1-(w@a)/(w@b));lo,hi=np.quantile(boot,[.025,.975]);ab=w@(b-a);al,ah=np.quantile(ab,[.025,.975])
            seeds={}
            for seed,z in g.groupby('seed'):
                v=z.groupby('arm').nmae.mean();gain=float(100*(1-v[proposed]/v[baseline]));seeds[str(seed)]=gain
                seedrows.append(dict(**metadata,proposed=proposed,baseline=baseline,seed=int(seed),proposed_nmae=float(v[proposed]),baseline_nmae=float(v[baseline]),gain_pct=gain,absolute_gain=float(v[baseline]-v[proposed])))
            effects.append(dict(**metadata,proposed=proposed,baseline=baseline,proposed_nmae=float(a.mean()),baseline_nmae=float(b.mean()),gain_pct=float(100*(1-a.mean()/b.mean())),ci_low=float(lo),ci_high=float(hi),absolute_gain=float((b-a).mean()),absolute_ci_low=float(al),absolute_ci_high=float(ah),seed_gains=json.dumps(seeds),main_table=kind=='standard' and condition in ['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT']))
        v={k:table[k].to_numpy() for k in ALL_ARMS}
        terms=dict(plain_gain_F0=v['F0']-v['F0_PLAIN'],plain_gain_B0=v['B0']-v['B0_PLAIN'],mag_gain_F0=v['F0']-v['F0_MAG'],mag_gain_B0=v['B0']-v['B0_MAG'],mag_specific_F0=v['F0_PLAIN']-v['F0_MAG'],mag_specific_B0=v['B0_PLAIN']-v['B0_MAG'])
        terms['second_stage_interaction_mag']=terms['mag_gain_B0']-terms['mag_gain_F0'];terms['mag_rule_interaction']=terms['mag_specific_B0']-terms['mag_specific_F0']
        for term,a in terms.items():
            lo,hi=np.quantile(w@a,[.025,.975]);interactions.append(dict(**metadata,term=term,nmae_difference=float(a.mean()),ci_low=float(lo),ci_high=float(hi)))
    pd.DataFrame(effects).to_csv(OUT/'EFFECTS.csv',index=False);pd.DataFrame(seedrows).to_csv(OUT/'SEED_EFFECTS.csv',index=False);pd.DataFrame(interactions).to_csv(OUT/'INTERACTION_EFFECTS.csv',index=False)
    save(OUT/'VERIFICATION.json',dict(status='SCORED_SCALAR_VERIFIED',scalar_metrics=scalar_count,prediction_views=192,all_predictions_saved_before_scoring=True,raw_score_rows=len(raw),effect_rows=len(effects),paper_pass=False))
    print('SCORED',len(raw),len(effects),flush=True)
