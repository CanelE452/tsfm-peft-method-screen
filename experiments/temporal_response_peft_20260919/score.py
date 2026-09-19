import pandas as pd
from .common import *
from experiments.persistence_evidence_extension_20260918 import common as ext
from experiments.persistence_evidence_extension_20260918.score import metric_arrays
from experiments.outlier_signal_peft_v1_20260917.reference_core import pinball_scalar
PANELS=['electricity','electricity_transfer','ettm1']

def score():
    check_seal();done=read(OUT/'PREDICTIONS.json');marker=read(OUT/'ALL_PREDICTIONS_SAVED.json');assert marker['manifest_sha256']==sha(OUT/'PREDICTIONS.json')
    assert marker['at']>read(OUT/'EVALUATION_SEAL.json')['at'];rows=[];scalar=0
    for key,r in done.items():
        panel,kind=r['panel'],r['kind'];d=np.load(ext.data_path(panel)/'E_DISCOVERY_inputs.npz');y0=np.load(ext.data_path(panel)/'E_DISCOVERY_labels.npz')['y'];nc=y0.shape[1]
        if kind=='standard':names=ext.STATES;ids=np.arange(128);offset=np.load(ext.panel_path(panel,kind)/'E_DISCOVERY_offset.npy')
        else:
            meta=read(ext.panel_path(panel,kind)/'manifest.json');names=meta['states'];ids=np.asarray(meta['origin_indices']);offset=np.load(ext.panel_path(panel,kind)/'offset.npy')
        assert sha(ROOT/r['path'])==r['sha256'];p=np.load(ROOT/r['path'],mmap_mode='r');n=len(ids)
        y=np.tile(y0[ids].reshape(-1,64),(2*len(names),1))+offset[:,None];s=np.tile(d['sigma'],len(y)//nc)
        m=np.concatenate([metric_arrays(p[k:k+256],y[k:k+256],s[k:k+256]) for k in range(0,len(y),256)]).reshape(len(names),2,n,nc,5).mean(1)
        for ci,c in enumerate(names):
            for oi,origin in enumerate(d['origins'][ids]):
                for ch in range(nc):rows.append(dict(panel=panel,kind=kind,stage=r['stage'],arm=r['arm'],seed=r['seed'],step=r['step'],condition=c,origin=int(origin),channel=ch,nmae=float(m[ci,oi,ch,0]),mae=float(m[ci,oi,ch,1]),pinball=float(m[ci,oi,ch,3])))
            for oi in [0,n-1]:
                for ch in [0,nc-1]:
                    ii=[((ci*2+z)*n+oi)*nc+ch for z in range(2)];pp=p[ii].astype(float);yy=y[ii];ss=s[ii]
                    a=sum(abs(float(pp[z,4,h])-float(yy[z,h]))/float(ss[z]) for z in range(2) for h in range(64))/128;b=pinball_scalar(pp,yy,ss,np.arange(1,10)/10)
                    np.testing.assert_allclose(m[ci,oi,ch,[0,3]],[a,b],rtol=1e-10,atol=1e-12);scalar+=2
    f=pd.DataFrame(rows);f.to_csv(OUT/'ORIGIN_SCORES.csv.gz',index=False,compression='gzip')
    keys=['panel','kind','stage','arm','seed','step','origin','channel'];fault=f[f.condition.str.startswith(('POINT','BURST'))].groupby(keys,as_index=False)[['nmae','mae','pinball']].mean();fault['condition']='FAULT';f=pd.concat([f,fault],ignore_index=True)
    raw=f.groupby(['panel','kind','stage','arm','seed','step','condition'],as_index=False)[['nmae','mae','pinball']].mean();raw.to_csv(OUT/'RAW_SCORES.csv',index=False)
    effects=[];seed_effects=[]
    for (panel,kind,stage,cond),g in f.groupby(['panel','kind','stage','condition']):
        table=g.groupby(['origin','arm']).nmae.mean().unstack();assert set(table.columns)==set(ARMS+['B0','C3','MAG_ONLY'])
        origins=table.index.to_numpy();period=24 if panel.startswith('electricity') else 96;weeks=np.unique(origins//(7*period));random=np.random.default_rng(91919+PANELS.index(panel));counts=np.array([np.bincount(random.integers(0,len(weeks),len(weeks)),minlength=len(weeks)) for _ in range(2000)]);w=counts[:,np.searchsorted(weeks,origins//(7*period))].astype(float);assert (w.sum(1)>0).all();w/=w.sum(1,keepdims=True)
        for baseline in ['PLAIN','ANCHOR','SHUFFLE','IDEAL','B0','C3','MAG_ONLY']:
            a=table.TRP.to_numpy();b=table[baseline].to_numpy();gain=100*(1-a.mean()/b.mean());boots=100*(1-(w@a)/(w@b));lo,hi=np.quantile(boots,[.025,.975]);bl,bh=np.quantile(boots,[.05/8,1-.05/8]);seed_gains=[]
            for seed,z in g.groupby('seed'):
                means=z.groupby('arm').nmae.mean();sg=100*(1-means.TRP/means[baseline]);seed_gains.append(sg);seed_effects.append(dict(panel=panel,kind=kind,stage=stage,condition=cond,baseline=baseline,seed=int(seed),TRP_nmae=means.TRP,baseline_nmae=means[baseline],gain_pct=sg))
            # Independent replay from raw per-seed source scores, not the bootstrap table.
            z=raw[(raw.panel==panel)&(raw.kind==kind)&(raw.stage==stage)&(raw.condition==cond)]
            ra=z[z.arm=='TRP'].nmae.mean();rb=z[z.arm==baseline].nmae.mean();np.testing.assert_allclose([a.mean(),b.mean(),gain],[ra,rb,100*(1-ra/rb)],rtol=1e-10,atol=1e-10)
            effects.append(dict(panel=panel,kind=kind,stage=stage,condition=cond,baseline=baseline,TRP_nmae=a.mean(),baseline_nmae=b.mean(),gain_pct=gain,ci_low=lo,ci_high=hi,bonferroni4_low=bl,bonferroni4_high=bh,both_seeds_positive=all(v>0 for v in seed_gains),primary_family=panel=='electricity_transfer' and kind=='standard' and stage=='selected' and cond=='SHIFT8' and baseline in ['PLAIN','ANCHOR','SHUFFLE','IDEAL']))
    e=pd.DataFrame(effects);e.to_csv(OUT/'EFFECTS.csv',index=False);pd.DataFrame(seed_effects).to_csv(OUT/'SEED_EFFECTS.csv',index=False)
    primary=e[e.primary_family];assert len(primary)==4
    protection=e[(e.panel=='electricity_transfer')&(e.stage=='selected')&(e.condition.isin(['REFERENCE','FAULT']))&(e.baseline=='PLAIN')];assert len(protection)==2
    signal=bool((primary.gain_pct>=1).all() and (primary.bonferroni4_low>0).all() and primary.both_seeds_positive.all() and (protection.gain_pct>=-1).all())
    save(OUT/'VERIFICATION.json',dict(status='VERIFIED',views=len(done),scalar_metrics=scalar,effect_rows=len(e),all_predictions_saved_before_scoring=True,four_comparisons_meet_predeclared_signal=signal,paper_pass=False,independent_source=False,automatic_successor=False))
    from .report import report
    report()
