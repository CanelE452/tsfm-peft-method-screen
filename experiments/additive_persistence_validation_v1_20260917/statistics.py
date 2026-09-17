"""Paired time and series/time intervals; goals are never pooled into a winner."""
import pandas as pd
from .common import *
from .prepare import STATES,SHAPES
PANELS={'electricity':0,'ettm1':1,'ettm2':2,'electricity_transfer':3}
CONDITIONS={'HISTORY_SUBSET':15,'FAULT':0,'REFERENCE':1,'SHIFT4':2,'SHIFT8':3,'SHIFT_POINT':4,**{n:i+5 for i,n in enumerate(SHAPES+['PAIRED_SHIFT8_D32','PULSE_LEGACY_MATCH'])}}

def analyze_effects(frame):
    scores=frame.copy();scores['metric_panel']=scores.condition.map(lambda c:'FAULT' if c.startswith(('POINT','BURST')) else c)
    group=['panel','kind','stage','arm','seed','metric_panel']
    scores.groupby(group).nmae.mean().reset_index().to_csv(OUT/'PANEL_SCORES.csv',index=False)
    selected=scores[scores.stage=='selected'];history=selected[(selected.condition=='REFERENCE') & selected.history_subset].copy();history['metric_panel']='HISTORY_SUBSET';selected=pd.concat([selected,history],ignore_index=True);rows=[];seedrows=[]
    exposure=read(OUT/'SERIES_PANEL_MANIFEST.json')['prior_exposure_by_channel']
    exp=selected[selected.panel=='electricity_transfer'].copy();exp['exposure']=exp.channel.astype(str).map(exposure)
    exp.groupby(['kind','arm','seed','metric_panel','exposure']).nmae.mean().reset_index().to_csv(OUT/'EXPOSURE_SCORES.csv',index=False)
    for (panel,kind,condition),f in selected.groupby(['panel','kind','metric_panel'],sort=True):
        available=set(f.arm);contrasts=[(a,b) for a,b in [('C2','C0'),('C3','C0'),('C3','C2'),('C1','C0'),('C3','M_MEAN'),('C3','M_ROTATE16'),('C3','M_RECENCY'),('C3','C2_SHRINK'),('C3','C0_BIAS')] if a in available and b in available]
        seeds=sorted(f.seed.unique());period=24 if panel.startswith('electricity') else 96
        for seed in seeds:
            g=f[f.seed==seed].groupby('arm').nmae.mean()
            for a,b in contrasts:seedrows.append(dict(panel=panel,kind=kind,condition=condition,seed=int(seed),new=a,baseline=b,new_nmae=float(g[a]),baseline_nmae=float(g[b]),gain_pct=float(100*(1-g[a]/g[b]))))
        scopes=[('all3',seeds)]
        if panel!='ettm2':scopes.insert(0,('original2',[81551,81552]))
        for scope,seedset in scopes:
            sub=f[f.seed.isin(seedset)];arms=sorted(available);origins=np.sort(sub.origin.unique());channels=sorted(sub.channel.astype(str).unique());n=len(origins);nc=len(channels)
            g=sub.assign(channel=sub.channel.astype(str)).groupby(['origin','channel','arm']).nmae.mean().unstack('arm').reindex(pd.MultiIndex.from_product([origins,channels],names=['origin','channel']))[arms]
            values=g.to_numpy().reshape(n,nc,len(arms))
            if condition=='HISTORY_SUBSET':
                # Pair all methods on the same observed trigger rows, average channels within each retained origin.
                assert np.array_equal(np.isfinite(values),np.broadcast_to(np.isfinite(values[:,:,:1]),values.shape))
                values=np.nanmean(values,axis=1,keepdims=True);nc=1
            assert np.isfinite(values).all()
            blocks=origins//(period*7);unique=np.unique(blocks);seed=86700+1000*PANELS[panel]+10*CONDITIONS[condition]+(0 if scope=='original2' else 1)
            r=np.random.default_rng(seed);weights=np.zeros((2000,n))
            for k in range(2000):
                sampled=r.choice(unique,len(unique));weights[k]=[(sampled==b).sum() for b in blocks]
            weights/=weights.sum(1,keepdims=True)
            draws={'time':weights@values.mean(1)}
            if panel=='electricity_transfer':
                cw=np.array([np.bincount(r.integers(0,nc,nc),minlength=nc)/nc for _ in range(2000)])
                draws['series_and_time']=np.einsum('bi,bc,icm->bm',weights,cw,values,optimize=True)
            point=values.mean((0,1))
            for a,b in contrasts:
                ai,bi=arms.index(a),arms.index(b);mechanism=b.startswith('M_');gain=100*(1-point[ai]/point[bi])
                for ci,draw in draws.items():
                    effect=100*(1-draw[:,ai]/draw[:,bi]);lo,hi=np.quantile(effect,[.025,.975]);clo,chi=np.quantile(effect,[.05/6,1-.05/6]) if mechanism else (None,None)
                    rows.append(dict(panel=panel,kind=kind,condition=condition,seed_scope=scope,seed_count=len(seedset),new=a,baseline=b,new_nmae=float(point[ai]),baseline_nmae=float(point[bi]),absolute_error_difference=float(point[ai]-point[bi]),gain_pct=float(gain),ci_type=ci,ci_low_pct=float(lo),ci_high_pct=float(hi),bonferroni3_low_pct=None if clo is None else float(clo),bonferroni3_high_pct=None if chi is None else float(chi),origins=n,channels=nc,time_blocks=len(unique),draws=2000,bootstrap_seed=seed,conditional_on_observed_seeds=True))
    allrows=pd.DataFrame(rows);allrows.to_csv(OUT/'UNCERTAINTY.csv',index=False)
    allrows[(allrows.kind=='standard')&allrows.baseline.isin(['C0','C2'])].to_csv(OUT/'PRIMARY_CONTRASTS.csv',index=False)
    allrows[(allrows.kind=='standard')&allrows.baseline.isin(CONTROLS+['C2_SHRINK','C0_BIAS'])].to_csv(OUT/'MECHANISM_CONTRASTS.csv',index=False)
    pd.DataFrame(seedrows).to_csv(OUT/'SEED_EFFECTS.csv',index=False)
