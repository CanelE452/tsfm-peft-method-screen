"""Paired frozen-weight evidence, all registered conditions retained."""
import pandas as pd
from .common import *

def metric_arrays(p,y,sigma):
    p=np.asarray(p,dtype=float);y=np.asarray(y,dtype=float);s=np.asarray(sigma,dtype=float);err=y-p[:,4];qe=y[:,None,:]-p;q=np.arange(1,10)[None,:,None]/10
    assert np.isfinite(p).all() and np.isfinite(y).all()
    return np.stack([(abs(err)/s[:,None]).mean(1),abs(err).mean(1),(err**2/s[:,None]**2).mean(1),(2*np.maximum(q*qe,(q-1)*qe)/s[:,None,None]).mean((1,2)),(np.diff(p,axis=1)<0).mean((1,2))],-1)

def score():
    check_seal();marker=read(OUT/'ALL_PREDICTIONS_SAVED.json');manifest=read(OUT/'PREDICTIONS.json');assert sha(OUT/'PREDICTIONS.json')==marker['manifest_sha256']
    rows=[];scalar=[];data=read(OUT/'DATA_MANIFEST.json')
    for key,r in manifest.items():
        panel=r['panel'];kind=r['kind'];d=np.load(data_path(panel)/'E_DISCOVERY_inputs.npz');y0=np.load(data_path(panel)/'E_DISCOVERY_labels.npz')['y'];nc=y0.shape[1];p=np.load(ROOT/r['path'],mmap_mode='r');assert sha(ROOT/r['path'])==r['sha256']
        if kind=='standard':names=STATES;ids=np.arange(128);offset=np.load(panel_path(panel,kind)/'E_DISCOVERY_offset.npy')
        else:
            meta=read(panel_path(panel,kind)/'manifest.json');names=meta['states'];ids=np.array(meta['origin_indices']);offset=np.load(panel_path(panel,kind)/'offset.npy')
        n=len(ids);y=np.tile(y0[ids].reshape(-1,64),(2*len(names),1))+offset[:,None];sigma=np.tile(d['sigma'],len(y)//nc)
        m=np.concatenate([metric_arrays(p[lo:lo+256],y[lo:lo+256],sigma[lo:lo+256]) for lo in range(0,len(y),256)]).reshape(len(names),2,n,nc,5).mean(1)
        for ci,state in enumerate(names):
            for oi,origin in enumerate(d['origins'][ids]):
                for ch,cid in enumerate(data[panel]['selected_columns']):rows.append(dict(prediction=key,panel=panel,kind=kind,arm=r['arm'],seed=r['seed'],condition=state,origin=int(origin),channel=str(cid),nmae=float(m[ci,oi,ch,0]),mae=float(m[ci,oi,ch,1]),normalized_mse=float(m[ci,oi,ch,2]),pinball=float(m[ci,oi,ch,3]),crossing=float(m[ci,oi,ch,4])))
            for oi in [0,n-1]:
                for ch in sorted({0,nc-1}):
                    ii=[((ci*2+z)*n+oi)*nc+ch for z in range(2)];pp=p[ii].astype(float);yy=y[ii];ss=sigma[ii]
                    independent=sum(abs(float(pp[z,4,h])-float(yy[z,h]))/float(ss[z]) for z in range(2) for h in range(64))/(2*64)
                    independent_pin=pinball_scalar(pp,yy,ss,np.arange(1,10)/10)
                    np.testing.assert_allclose(m[ci,oi,ch,[0,3]],[independent,independent_pin],rtol=1e-10,atol=1e-12)
                    scalar.append(dict(prediction=key,condition=state,origin_index=oi,channel=str(data[panel]['selected_columns'][ch])))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'NEW_ORIGIN_SCORES.csv.gz',index=False,compression='gzip');csvwrite(OUT/'SCALAR_REPLAY_SCOPE.csv',scalar)
    prior=pd.read_csv(PRIOR/'SCORES_BY_ORIGIN.csv.gz',dtype={'channel':str});prior=prior[(prior.stage=='selected')&prior.kind.isin(['standard','shape'])]
    both=pd.concat([prior[frame.columns],frame],ignore_index=True);assert not both.duplicated(['panel','kind','arm','seed','condition','origin','channel']).any()
    group=['panel','kind','arm','seed','condition'];raw=both.groupby(group)[['nmae','mae','pinball','crossing']].mean();rmse=both.groupby(group+['channel']).normalized_mse.mean().pow(.5).groupby(level=list(range(5))).mean();raw.join(rmse.rename('nrmse')).reset_index().to_csv(OUT/'RAW_SCORES.csv',index=False)
    analyze(both)
    # Independently reconstruct all non-FAULT primary effects from per-condition raw scores.
    raw=raw.reset_index();effect=pd.read_csv(OUT/'EFFECTS.csv');checks=0
    for r in effect.to_dict('records'):
        f=raw[(raw.panel==r['panel'])&(raw.kind==r['kind'])]
        f=f[f.condition.str.startswith(('POINT','BURST'))] if r['condition']=='FAULT' else f[f.condition==r['condition']]
        a=float(f[f.arm==r['new']].nmae.mean());b=float(f[f.arm==r['baseline']].nmae.mean())
        np.testing.assert_allclose([r['new_nmae'],r['baseline_nmae'],r['gain_pct']],[a,b,100*(1-a/b)],rtol=1e-10,atol=1e-10);checks+=1
    check_seal();save(OUT/'VERIFICATION.json',dict(status='VERIFIED',optimizer_updates=0,new_prediction_views=111,new_origin_rows=len(frame),scalar_rows=len(scalar),scalar_metrics_per_row=2,scalar_scope='all new view/condition; first/last origins; first/last channel; both draws',effect_rows_independently_replayed=checks,model_checks=len(read(OUT/'MODEL_CHECKS.json')),all_weights_frozen_and_unchanged=True,old_results_unchanged=True,all_new_predictions_saved_before_scoring=True,common_day_draws=True,automatic_successor=False));print('VERIFIED',checks,flush=True)

def analyze(frame):
    f=frame.copy();f['condition']=f.condition.map(lambda c:'FAULT' if c.startswith(('POINT','BURST')) else c)
    effects=[];seeds=[];factor=[]
    contrasts=[('C2','C0'),('C3','C0'),('C3','C2'),('C3','M_MEAN'),('C3','M_ROTATE16'),('C3','M_RECENCY'),('C3','C2_SHRINK'),('C3','C0_BIAS'),('C1','C0'),('C3_W_RECENCY_G','C3'),('RECENCY_W_C3_G','M_RECENCY'),('C3','RECENCY_W_C3_G'),('C3_W_RECENCY_G','M_RECENCY')]
    for (panel,kind,condition),a in f.groupby(['panel','kind','condition'],sort=True):
        arms=sorted(a.arm.unique());origins=sorted(a.origin.unique());channels=sorted(a.channel.astype(str).unique());nr=len(origins);nc=len(channels);available=[(x,y) for x,y in contrasts if x in arms and y in arms]
        for seed,ss in a.groupby('seed'):
            e=ss.groupby('arm').nmae.mean()
            for x,y in available:seeds.append(dict(panel=panel,kind=kind,condition=condition,seed=int(seed),new=x,baseline=y,new_nmae=float(e[x]),baseline_nmae=float(e[y]),gain_pct=float(100*(1-e[x]/e[y]))))
        grid=a.assign(channel=a.channel.astype(str)).groupby(['origin','channel','arm']).nmae.mean().unstack('arm').reindex(pd.MultiIndex.from_product([origins,channels],names=['origin','channel']))[arms].to_numpy().reshape(nr,nc,len(arms));assert np.isfinite(grid).all()
        period=read(OUT/'DATA_MANIFEST.json')[panel]['period'];full=np.load(data_path(panel)/'E_DISCOVERY_inputs.npz')['origins'];blocks=np.unique(full//(7*period));bootseed=88700+1000*PANELS.index(panel);rng=np.random.default_rng(bootseed)
        counts=np.zeros((2000,len(blocks)),dtype=np.int64)
        for k in range(2000):counts[k]=np.bincount(rng.integers(0,len(blocks),len(blocks)),minlength=len(blocks))
        drawhash=hashlib.sha256(counts.tobytes()).hexdigest();indices=np.searchsorted(blocks,np.array(origins)//(7*period));w=counts[:,indices].astype(float);assert (w.sum(1)>0).all();w/=w.sum(1,keepdims=True)
        draws={'time':w@grid.mean(1)}
        if panel=='electricity_transfer':
            cw=np.array([np.bincount(rng.integers(0,nc,nc),minlength=nc)/nc for _ in range(2000)]);draws['series_and_time']=np.einsum('bi,bc,icm->bm',w,cw,grid,optimize=True)
        point=grid.mean((0,1))
        for x,y in available:
            ai,bi=arms.index(x),arms.index(y)
            for label,z in draws.items():
                g=100*(1-z[:,ai]/z[:,bi]);lo,hi=np.quantile(g,[.025,.975]);blo,bhi=np.quantile(g,[.05/6,1-.05/6]) if y in old.CONTROLS else (np.nan,np.nan)
                effects.append(dict(panel=panel,kind=kind,condition=condition,new=x,baseline=y,new_nmae=float(point[ai]),baseline_nmae=float(point[bi]),absolute_error_difference=float(point[ai]-point[bi]),gain_pct=float(100*(1-point[ai]/point[bi])),ci_type=label,ci_low_pct=float(lo),ci_high_pct=float(hi),bonferroni3_low_pct=blo,bonferroni3_high_pct=bhi,seed_count=3,origin_count=nr,channel_count=nc,bootstrap_seed=bootseed,draws=2000,common_draw_sha256=drawhash,conditional_on_fixed_seeds=True,exploratory_shapes_and_interventions=True))
        if all(x in arms for x in ['C3','M_RECENCY',*SWAPS]):
            A,B,C,D=(point[arms.index(x)] for x in ['C3','C3_W_RECENCY_G','RECENCY_W_C3_G','M_RECENCY'])
            gate=.5*((B-A)+(D-C));weight=.5*((C-A)+(D-B));assert abs(gate+weight-(D-A))<1e-12
            factor.append(dict(panel=panel,kind=kind,condition=condition,C3weights_C3gate=A,C3weights_RECENCYgate=B,RECENCYweights_C3gate=C,RECENCYweights_RECENCYgate=D,total_nmae_benefit=D-A,gate_nmae_benefit=gate,weights_nmae_benefit=weight,interaction=A-B-C+D,meaning='fixed learned-functions decomposition; not training-causal identification or selectable new method'))
    pd.DataFrame(effects).to_csv(OUT/'EFFECTS.csv',index=False);pd.DataFrame(seeds).to_csv(OUT/'SEED_EFFECTS.csv',index=False);pd.DataFrame(factor).to_csv(OUT/'FACTORIAL_DECOMPOSITION.csv',index=False)
