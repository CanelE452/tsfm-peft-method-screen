import pandas as pd
from .common import *
from experiments.persistence_evidence_extension_20260918 import common as ext
from experiments.persistence_evidence_extension_20260918.score import metric_arrays
from experiments.outlier_signal_peft_v1_20260917.reference_core import pinball_scalar

def score():
    check_seal();marker=read(OUT/'ALL_PREDICTIONS_SAVED.json');assert marker['manifest_sha256']==sha(OUT/'PREDICTIONS.json')
    assert marker['at']>read(OUT/'EVALUATION_SEAL.json')['at']
    rows=[];channel_rows=[];scalar_count=0
    for key,r in read(OUT/'PREDICTIONS.json').items():
        panel,kind=r['panel'],r['kind'];d=np.load(ext.data_path(panel)/'E_DISCOVERY_inputs.npz');yy=np.load(ext.data_path(panel)/'E_DISCOVERY_labels.npz')['y'];nc=yy.shape[1]
        if kind=='standard':names=ext.STATES;ids=np.arange(128);offset=np.load(ext.panel_path(panel,kind)/'E_DISCOVERY_offset.npy')
        else:
            meta=read(ext.panel_path(panel,kind)/'manifest.json');names=meta['states'];ids=np.asarray(meta['origin_indices']);offset=np.load(ext.panel_path(panel,kind)/'offset.npy')
        assert sha(ROOT/r['path'])==r['sha256'];p=np.load(ROOT/r['path'],mmap_mode='r');n=len(ids)
        y=np.tile(yy[ids].reshape(-1,64),(2*len(names),1))+offset[:,None];s=np.tile(d['sigma'],len(y)//nc)
        values=np.concatenate([metric_arrays(p[k:k+256],y[k:k+256],s[k:k+256]) for k in range(0,len(p),256)]).reshape(len(names),2,n,nc,5).mean(1)
        for ci,condition in enumerate(names):
            for oi,origin in enumerate(d['origins'][ids]):
                v=values[ci,oi].mean(0);rows.append(dict(panel=panel,kind=kind,arm=r['arm'],seed=r['seed'],stage=r['stage'],step=r['step'],condition=condition,origin=int(origin),nmae=float(v[0]),mae=float(v[1]),pinball=float(v[3])))
            for ch in range(nc):
                v=values[ci,:,ch].mean(0);channel_rows.append(dict(panel=panel,kind=kind,arm=r['arm'],seed=r['seed'],stage=r['stage'],condition=condition,channel=ch,nmae=float(v[0]),mae=float(v[1]),pinball=float(v[3])))
            for oi,ch in [(0,0),(n-1,nc-1)]:
                ii=[((ci*2+z)*n+oi)*nc+ch for z in range(2)];pp=p[ii].astype(float);target=y[ii];sigma=s[ii]
                a=sum(abs(float(pp[z,4,h])-float(target[z,h]))/float(sigma[z]) for z in range(2) for h in range(64))/128
                b=pinball_scalar(pp,target,sigma,np.arange(1,10)/10)
                np.testing.assert_allclose([a,b],values[ci,oi,ch,[0,3]],rtol=1e-10,atol=1e-11);scalar_count+=2
    f=pd.DataFrame(rows);keys=['panel','kind','arm','seed','stage','step','origin'];fault=f[f.condition.str.startswith(('POINT','BURST'))].groupby(keys,as_index=False)[['nmae','mae','pinball']].mean();fault['condition']='FAULT';f=pd.concat([f,fault],ignore_index=True)
    f.to_csv(OUT/'ORIGIN_SCORES.csv.gz',index=False,compression='gzip');pd.DataFrame(channel_rows).to_csv(OUT/'CHANNEL_SCORES.csv',index=False)
    raw=f.groupby(['panel','kind','arm','seed','stage','step','condition'],as_index=False)[['nmae','mae','pinball']].mean();raw.to_csv(OUT/'RAW_SCORES.csv',index=False)
    effects=[]
    for (panel,kind,stage,condition),g in f.groupby(['panel','kind','stage','condition']):
        table=g.groupby(['origin','arm']).nmae.mean().unstack();origins=table.index.to_numpy();period=24 if panel.startswith('electricity') else 96;weeks=np.unique(origins//(7*period))
        random=np.random.default_rng(91942);counts=np.array([np.bincount(random.integers(0,len(weeks),len(weeks)),minlength=len(weeks)) for _ in range(2000)]);w=counts[:,np.searchsorted(weeks,origins//(7*period))].astype(float);w/=w.sum(1,keepdims=True)
        for proposed in ['C3','MAG_ONLY']:
            for baseline in ARMS+['B0','PLAIN']:
                a=table[proposed].to_numpy();b=table[baseline].to_numpy();gain=100*(1-a.mean()/b.mean());boot=100*(1-(w@a)/(w@b));lo,hi=np.quantile(boot,[.025,.975]);bl,bh=np.quantile(boot,[.05/8,1-.05/8])
                seeds={}
                for seed,z in g.groupby('seed'):
                    v=z.groupby('arm').nmae.mean();seeds[str(seed)]=float(100*(1-v[proposed]/v[baseline]))
                effects.append(dict(panel=panel,kind=kind,stage=stage,condition=condition,proposed=proposed,baseline=baseline,proposed_nmae=float(a.mean()),baseline_nmae=float(b.mean()),gain_pct=float(gain),ci_low=float(lo),ci_high=float(hi),bonf4_low=float(bl),bonf4_high=float(bh),seed_gains=json.dumps(seeds),primary_family=panel=='electricity_transfer' and stage=='selected' and kind=='standard' and condition=='SHIFT8' and baseline in ARMS))
    e=pd.DataFrame(effects);e.to_csv(OUT/'EFFECTS.csv',index=False)
    save(OUT/'VERIFICATION.json',dict(status='SCORED_AND_SCALAR_VERIFIED',scalar_metrics=scalar_count,prediction_views=144,raw_score_rows=len(raw),effect_rows=len(e),all_predictions_saved_before_new_scoring=True,new_fits=16,main_updates=MAIN_CAP,smoke_updates=SMOKE_CAP,paper_pass=False,goal_achieved=False))
    print(e[e.primary_family].to_string(index=False),flush=True)
