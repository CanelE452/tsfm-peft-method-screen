"""New predictions scored after all selections and predictions have been sealed."""
import pandas as pd
from .common import *
from experiments.persistence_evidence_extension_20260918 import common as ext
from experiments.paper_readiness_20260918 import common as ready
from experiments.persistence_evidence_extension_20260918.score import metric_arrays
from experiments.outlier_signal_peft_v1_20260917.reference_core import pinball_scalar
PANELS=['electricity','electricity_transfer','ettm1']

def score():
    check_seal();manifest=read(OUT/'PREDICTIONS.json');assert sha(OUT/'PREDICTIONS.json')==read(OUT/'ALL_PREDICTIONS_SAVED.json')['manifest_sha256'];assert read(OUT/'EVALUATION_SEAL.json')['at']<read(OUT/'ALL_PREDICTIONS_SAVED.json')['at']
    rows=[];scalar=0;data=read(PRIOR/'DATA_MANIFEST.json')
    for key,r in manifest.items():
        panel,kind=r['panel'],r['kind'];d=np.load(ext.data_path(panel)/'E_DISCOVERY_inputs.npz');y0=np.load(ext.data_path(panel)/'E_DISCOVERY_labels.npz')['y'];nc=y0.shape[1]
        if kind=='standard':names=ext.STATES;ids=np.arange(128);offset=np.load(ext.panel_path(panel,kind)/'E_DISCOVERY_offset.npy')
        else:
            meta=read(ext.panel_path(panel,kind)/'manifest.json');names=meta['states'];ids=np.array(meta['origin_indices']);offset=np.load(ext.panel_path(panel,kind)/'offset.npy')
        assert sha(ROOT/r['path'])==r['sha256'];p=np.load(ROOT/r['path'],mmap_mode='r');n=len(ids);y=np.tile(y0[ids].reshape(-1,64),(2*len(names),1))+offset[:,None];s=np.tile(d['sigma'],len(y)//nc)
        m=np.concatenate([metric_arrays(p[i:i+256],y[i:i+256],s[i:i+256]) for i in range(0,len(y),256)]).reshape(len(names),2,n,nc,5).mean(1)
        for ci,cond in enumerate(names):
            for oi,origin in enumerate(d['origins'][ids]):
                for ch,cid in enumerate(data[panel]['selected_columns']):rows.append(dict(panel=panel,kind=kind,condition=cond,arm=r['arm'],seed=r['seed'],origin=int(origin),channel=str(cid),nmae=m[ci,oi,ch,0],mae=m[ci,oi,ch,1],normalized_mse=m[ci,oi,ch,2],pinball=m[ci,oi,ch,3],crossing=m[ci,oi,ch,4]))
            for oi in [0,n-1]:
                for ch in [0,nc-1]:
                    ii=[((ci*2+z)*n+oi)*nc+ch for z in range(2)];pp=p[ii].astype(float);yy=y[ii];ss=s[ii]
                    a=sum(abs(float(pp[z,4,h])-float(yy[z,h]))/float(ss[z]) for z in range(2) for h in range(64))/128;b=pinball_scalar(pp,yy,ss,np.arange(1,10)/10)
                    np.testing.assert_allclose(m[ci,oi,ch,[0,3]],[a,b],rtol=1e-10,atol=1e-12);scalar+=1
    new=pd.DataFrame(rows);new.to_csv(OUT/'NEW_ORIGIN_SCORES.csv.gz',index=False,compression='gzip')
    oldframe=pd.read_csv(PRIOR/'SCORES_BY_ORIGIN.csv.gz',dtype={'channel':str});oldframe=oldframe[(oldframe.stage=='selected')&oldframe.kind.isin(['standard','shape'])&oldframe.panel.isin(PANELS)]
    extension=pd.read_csv(ext.OUT/'NEW_ORIGIN_SCORES.csv.gz',dtype={'channel':str});extension=extension[extension.panel.isin(PANELS)&extension.kind.eq('shape')&extension.arm.isin(old.CONTROLS)]
    f0=pd.read_csv(ready.OUT/'NEW_ORIGIN_SCORES.csv.gz',dtype={'channel':str});f0=f0[f0.panel.isin(PANELS)&f0.arm.eq('F0')]
    full=pd.concat([oldframe[new.columns],extension[new.columns],f0[new.columns],new],ignore_index=True);assert not full.duplicated(['panel','kind','condition','arm','seed','origin','channel']).any()
    full.to_csv(OUT/'ALL_ORIGIN_SCORES.csv.gz',index=False,compression='gzip')
    fault=full[full.condition.str.startswith(('POINT','BURST'))].assign(condition='FAULT');f=pd.concat([full,fault]);keys=['panel','kind','condition','arm','seed'];metrics=['nmae','mae','normalized_mse','pinball','crossing'];raw=f.groupby(keys)[metrics].mean();nr=f.groupby(keys+['channel']).normalized_mse.mean().pow(.5).groupby(level=list(range(5))).mean();raw=raw.join(nr.rename('nrmse')).reset_index();raw.to_csv(OUT/'RAW_SCORES.csv',index=False)
    effects=[];perseed=[];paired=[]
    contrasts=[('C3',b) for b in ARMS+['C0','C2','M_RECENCY','M_MEAN','M_ROTATE16','C2_SHRINK','C0_BIAS','F0']]+[(a,b) for a in ARMS for b in ['C0','C2']]
    for (panel,kind,cond),g in f.groupby(['panel','kind','condition']):
        arms=sorted(g.arm.unique());origins=sorted(g.origin.unique());channels=sorted(g.channel.unique());nc=len(channels);grid=g.groupby(['origin','channel','arm']).nmae.mean().unstack().reindex(pd.MultiIndex.from_product([origins,channels],names=['origin','channel']))[arms].to_numpy().reshape(len(origins),nc,len(arms));assert np.isfinite(grid).all()
        period=data[panel]['period'];allorig=np.load(ext.data_path(panel)/'E_DISCOVERY_inputs.npz')['origins'];blocks=np.unique(allorig//(7*period));rng=np.random.default_rng(86700+1000*PANELS.index(panel));cnt=np.stack([np.bincount(rng.integers(0,len(blocks),len(blocks)),minlength=len(blocks)) for _ in range(2000)]);w=cnt[:,np.searchsorted(blocks,np.array(origins)//(7*period))].astype(float);assert (w.sum(1)>0).all();w/=w.sum(1,keepdims=True);draws={'time':w@grid.mean(1)}
        if panel=='electricity_transfer':
            cw=np.array([np.bincount(rng.integers(0,nc,nc),minlength=nc)/nc for _ in range(2000)]);draws['series_and_time']=np.einsum('bi,bc,icm->bm',w,cw,grid,optimize=True)
        point=grid.mean((0,1))
        for a,b in contrasts:
            if a not in arms or b not in arms:continue
            ai,bi=arms.index(a),arms.index(b)
            for label,z in draws.items():
                gain=100*(1-z[:,ai]/z[:,bi]);lo,hi=np.quantile(gain,[.025,.975]);blo,bhi=np.quantile(gain,[.05/6,1-.05/6]);effects.append(dict(panel=panel,kind=kind,condition=cond,new=a,baseline=b,new_nmae=point[ai],baseline_nmae=point[bi],gain_pct=100*(1-point[ai]/point[bi]),ci_type=label,ci_low_pct=lo,ci_high_pct=hi,bonferroni3_low_pct=blo if a=='C3' and b in ARMS else np.nan,bonferroni3_high_pct=bhi if a=='C3' and b in ARMS else np.nan,conditional_on_fixed_seeds=True,development_E_reused=True))
            for seed in [81551,81552,81553]:
                aa=g[(g.arm==a)&(g.seed==seed)].nmae.mean();bb=g[(g.arm==b)&(g.seed==(0 if b=='F0' else seed))].nmae.mean();assert np.isfinite(aa+bb);perseed.append(dict(panel=panel,kind=kind,condition=cond,new=a,baseline=b,seed=seed,new_nmae=aa,baseline_nmae=bb,gain_pct=100*(1-aa/bb)))
            if a=='C3' and b in ARMS:
                for oi,o in enumerate(origins):paired.append(dict(panel=panel,kind=kind,condition=cond,new=a,baseline=b,origin=o,new_nmae=grid[oi,:,ai].mean(),baseline_nmae=grid[oi,:,bi].mean(),absolute_error_benefit=(grid[oi,:,bi]-grid[oi,:,ai]).mean()))
    pd.DataFrame(effects).to_csv(OUT/'MATCHED_CONTRASTS.csv',index=False);pd.DataFrame(perseed).to_csv(OUT/'SEED_EFFECTS.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'PAIRED_ORIGIN_DIFFERENCES.csv',index=False)
    for r in effects:
        g=raw[(raw.panel==r['panel'])&(raw.kind==r['kind'])&(raw.condition==r['condition'])];a=g[g.arm==r['new']].nmae.mean();b=g[g.arm==r['baseline']].nmae.mean();np.testing.assert_allclose([a,b,100*(1-a/b)],[r['new_nmae'],r['baseline_nmae'],r['gain_pct']],rtol=1e-10,atol=1e-10)
    check_seal();save(OUT/'SCORE_VERIFICATION.json',dict(status='VERIFIED',new_origin_rows=len(new),all_origin_rows=len(full),scalar_rows=scalar,scalar_metrics_per_row=2,effect_rows_replayed=len(effects),all_predictions_saved_before_scoring=True,all_negative_conditions_preserved=True));print('SCORE VERIFIED',len(effects),flush=True)
