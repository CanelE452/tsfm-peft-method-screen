"""Evaluation after complete prediction freeze; independent scalar and point replay."""
import pandas as pd
from .common import *
from experiments.persistence_evidence_extension_20260918.score import metric_arrays

def score():
    check_seal();manifest=read(OUT/'PREDICTIONS.json');assert sha(OUT/'PREDICTIONS.json')==read(OUT/'ALL_PREDICTIONS_SAVED.json')['manifest_sha256']
    rows=[];scalar=0;data=read(ext.OUT/'DATA_MANIFEST.json')
    for key,r in manifest.items():
        panel,kind=r['panel'],r['kind'];d=np.load(data_path(panel)/'E_DISCOVERY_inputs.npz');y0=np.load(data_path(panel)/'E_DISCOVERY_labels.npz')['y'];nc=y0.shape[1]
        assert sha(ROOT/r['path'])==r['sha256'];p=np.load(ROOT/r['path'],mmap_mode='r')
        if kind=='standard':names=STATES;ids=np.arange(128);offset=np.load(panel_path(panel,kind)/'E_DISCOVERY_offset.npy')
        else:
            meta=read(panel_path(panel,kind)/'manifest.json');names=meta['states'];ids=np.array(meta['origin_indices']);offset=np.load(panel_path(panel,kind)/'offset.npy')
        n=len(ids);y=np.tile(y0[ids].reshape(-1,64),(2*len(names),1))+offset[:,None];sigma=np.tile(d['sigma'],len(y)//nc)
        m=np.concatenate([metric_arrays(p[lo:lo+256],y[lo:lo+256],sigma[lo:lo+256]) for lo in range(0,len(y),256)]).reshape(len(names),2,n,nc,5).mean(1)
        for ci,state in enumerate(names):
            for oi,origin in enumerate(d['origins'][ids]):
                for ch,cid in enumerate(data[panel]['selected_columns']):rows.append(dict(panel=panel,kind=kind,arm=r['arm'],seed=r['seed'],condition=state,origin=int(origin),channel=str(cid),nmae=m[ci,oi,ch,0],mae=m[ci,oi,ch,1],normalized_mse=m[ci,oi,ch,2],pinball=m[ci,oi,ch,3],crossing=m[ci,oi,ch,4]))
            for oi in [0,n-1]:
                for ch in sorted({0,nc-1}):
                    ii=[((ci*2+z)*n+oi)*nc+ch for z in range(2)];pp=p[ii].astype(float);yy=y[ii];ss=sigma[ii]
                    a=sum(abs(float(pp[z,4,h])-float(yy[z,h]))/float(ss[z]) for z in range(2) for h in range(64))/128
                    b=pinball_scalar(pp,yy,ss,np.arange(1,10)/10)
                    np.testing.assert_allclose(m[ci,oi,ch,[0,3]],[a,b],rtol=1e-10,atol=1e-12);scalar+=1
    new=pd.DataFrame(rows);new.to_csv(OUT/'NEW_ORIGIN_SCORES.csv.gz',index=False,compression='gzip')
    prior=pd.read_csv(PRIOR/'SCORES_BY_ORIGIN.csv.gz',dtype={'channel':str});prior=prior[(prior.stage=='selected')&prior.kind.isin(['standard','shape'])]
    newer=pd.read_csv(ext.OUT/'NEW_ORIGIN_SCORES.csv.gz',dtype={'channel':str})
    frame=pd.concat([prior[new.columns],newer[new.columns],new],ignore_index=True);assert not frame.duplicated(['panel','kind','arm','seed','condition','origin','channel']).any()
    f=frame.assign(condition=frame.condition.map(lambda c:'FAULT' if c.startswith(('POINT','BURST')) else c))
    raw=f.groupby(['panel','kind','condition','arm','seed'])[['nmae','mae','normalized_mse','pinball','crossing']].mean().reset_index();raw.to_csv(OUT/'RAW_SCORES.csv',index=False)
    effects=[];factor=[];perseed=[]
    comparisons=[('C0',b) for b in ['F0','PERSISTENCE','SEASONAL']]+[('C3',b) for b in ['C0','C2','M_RECENCY','F0','PERSISTENCE','SEASONAL']]+[('FIX_C3_GC3',b) for b in ['FIX_C2_GC2','FIX_C3_GC2','FIX_C2_GC3']]
    for (panel,kind,condition),g in f.groupby(['panel','kind','condition']):
        means=g.groupby('arm').nmae.mean();available=[(a,b) for a,b in comparisons if a in means and b in means]
        grid=g.groupby(['origin','arm']).nmae.mean().unstack();origins=grid.index.to_numpy();period=data[panel]['period'];full=np.load(data_path(panel)/'E_DISCOVERY_inputs.npz')['origins'];blocks=np.unique(full//(7*period));rng=np.random.default_rng(89700+1000*PANELS.index(panel));cnt=np.stack([np.bincount(rng.integers(0,len(blocks),len(blocks)),minlength=len(blocks)) for _ in range(2000)]);drawhash=hashlib.sha256(cnt.tobytes()).hexdigest();w=cnt[:,np.searchsorted(blocks,origins//(7*period))].astype(float);assert (w.sum(1)>0).all();w/=w.sum(1,keepdims=True)
        draws=w@grid.to_numpy();lookup={a:i for i,a in enumerate(grid.columns)}
        for a,b in available:
            z=100*(1-draws[:,lookup[a]]/draws[:,lookup[b]]);lo,hi=np.quantile(z,[.025,.975]);A=float(means[a]);B=float(means[b])
            effects.append(dict(panel=panel,kind=kind,condition=condition,new=a,baseline=b,new_nmae=A,baseline_nmae=B,gain_pct=100*(1-A/B),ci_low_pct=lo,ci_high_pct=hi,common_draw_sha256=drawhash,new_seed_count=g[g.arm==a].seed.nunique(),baseline_seed_count=g[g.arm==b].seed.nunique(),origin_count=len(origins),exploratory=True))
            for seed in sorted(g[g.arm==a].seed.unique()):
                aa=g[(g.arm==a)&(g.seed==seed)].nmae.mean();bb=g[(g.arm==b)&(g.seed==seed)].nmae.mean()
                if pd.isna(bb):bb=g[(g.arm==b)&(g.seed==0)].nmae.mean()
                assert not pd.isna(bb)
                perseed.append(dict(panel=panel,kind=kind,condition=condition,new=a,baseline=b,seed=int(seed),gain_pct=100*(1-aa/bb)))
        for label,arms in [('C3_vs_C2_fixed1024',['FIX_C3_GC3','FIX_C3_GC2','FIX_C2_GC3','FIX_C2_GC2']),('C3_vs_RECENCY_selected',['C3','SEL_C3_GM_RECENCY','SEL_M_RECENCY_GC3','M_RECENCY'])]:
            if not all(a in means for a in arms):continue
            A,B,C,D=[float(means[a]) for a in arms];gate=.5*((B-A)+(D-C));weight=.5*((C-A)+(D-B));assert abs(gate+weight-(D-A))<1e-12
            da,db,dc,dd=[draws[:,lookup[a]] for a in arms]
            rec=dict(panel=panel,kind=kind,condition=condition,comparison=label,A=A,B=B,C=C,D=D,total=D-A,gate=gate,weights=weight,interaction=A-B-C+D)
            for name,z in [('total',dd-da),('gate',.5*((db-da)+(dd-dc))),('weights',.5*((dc-da)+(dd-db)))]:rec[name+'_low'],rec[name+'_high']=np.quantile(z,[.025,.975])
            factor.append(rec)
    e=pd.DataFrame(effects);e.to_csv(OUT/'EFFECTS.csv',index=False);pd.DataFrame(factor).to_csv(OUT/'FACTORIAL.csv',index=False);pd.DataFrame(perseed).to_csv(OUT/'SEED_EFFECTS.csv',index=False)
    for r in effects:
        x=raw[(raw.panel==r['panel'])&(raw.kind==r['kind'])&(raw.condition==r['condition'])];a=x[x.arm==r['new']].nmae.mean();b=x[x.arm==r['baseline']].nmae.mean();np.testing.assert_allclose([a,b,100*(1-a/b)],[r['new_nmae'],r['baseline_nmae'],r['gain_pct']],rtol=1e-10,atol=1e-10)
    check_seal();save(OUT/'VERIFICATION.json',dict(status='VERIFIED',new_origin_rows=len(new),scalar_rows=scalar,scalar_metrics_per_row=2,logical_views=114,effect_rows=len(effects),factorial_rows=len(factor),all_effects_replayed=True,new_fits=0,optimizer_updates=0,parent_results_unchanged=True,all_predictions_saved_before_scoring=True,deterministic_baselines_not_seed_replicated=True))
    print('VERIFIED',len(effects),len(factor),flush=True)
