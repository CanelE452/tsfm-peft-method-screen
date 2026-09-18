"""Every draw retained until input-only strata are defined; all conditions reported."""
import pandas as pd
from .common import *
from experiments.persistence_evidence_extension_20260918.score import metric_arrays
METRICS=['nmae','mae','normalized_mse','pinball','crossing']

def load_scores(row,panel,kind):
    d,names,ids,nc,offset=metadata(panel,kind);n=len(ids);y0=np.load(data_path(panel)/'E_DISCOVERY_labels.npz')['y'];y=np.tile(y0[ids].reshape(-1,64),(2*len(names),1))+offset[:,None];s=np.tile(d['sigma'],len(y)//nc)
    assert sha(ROOT/row['path'])==row['sha256'];p=np.load(ROOT/row['path'],mmap_mode='r');m=np.concatenate([metric_arrays(p[lo:lo+256],y[lo:lo+256],s[lo:lo+256]) for lo in range(0,len(y),256)])
    # Independent scalar replay on every condition and draw, first/last origin and channel.
    checks=0
    for ci in range(len(names)):
        for draw in range(2):
            for oi in [0,n-1]:
                for ch in sorted({0,nc-1}):
                    i=((ci*2+draw)*n+oi)*nc+ch
                    a=sum(abs(float(p[i,4,h])-float(y[i,h]))/float(s[i]) for h in range(64))/64
                    b=pinball_scalar(p[i:i+1],y[i:i+1],s[i:i+1],np.arange(1,10)/10)
                    np.testing.assert_allclose(m[i,[0,3]],[a,b],rtol=1e-10,atol=1e-12);checks+=1
    return m.reshape(len(names),2,n,nc,5),checks

def diagnostic(include_new=False):
    check_seal();old.setup();records=[r for r in read(PRIOR/'PREDICTIONS_MANIFEST.json').values() if r.get('stage')=='selected']+list(read(ext.OUT/'PREDICTIONS.json').values())
    for r in read(ready.OUT/'PREDICTIONS.json').values():
        if r['arm'] in ['SEL_C3_GM_RECENCY','SEL_M_RECENCY_GC3']:records.append(dict(r,arm={'SEL_C3_GM_RECENCY':'C3_W_RECENCY_G','SEL_M_RECENCY_GC3':'RECENCY_W_C3_G'}[r['arm']]))
    panels=DIAG_PANELS if include_new else DIAG_PANELS[:-1]
    if include_new:
        assert sha(OUT/'PREDICTIONS.json')==read(OUT/'ALL_PREDICTIONS_SAVED.json')['manifest_sha256'];records+=list(read(OUT/'PREDICTIONS.json').values())
    arms=['C3','C3_W_RECENCY_G','RECENCY_W_C3_G','M_RECENCY'];rows=[];seedrows=[];countsrows=[];originrows=[];scalar=0;equalchecks=0
    for panel in panels:
        for kind in ['standard','shape']:
            d,names,ids,nc,_=metadata(panel,kind);n=len(ids);x,s=inputs(panel,kind);gates=[]
            for lo in range(0,len(x),256):
                xx=torch.tensor(np.array(x[lo:lo+256]),dtype=torch.float32);ss=torch.tensor(np.array(s[lo:lo+256]),dtype=torch.float32);a=mechanism_gate(xx,ss,'C3');b=mechanism_gate(xx,ss,'M_RECENCY');gates.append((a==b).all(-1).numpy())
            equal=np.concatenate(gates).reshape(len(names),2,n,nc);scores=[]
            for seed in [81551,81552,81553]:
                aa=[]
                for arm in arms:
                    matched=[r for r in records if (r['panel'],r['kind'],r['arm'],r['seed'])==(panel,kind,arm,seed)];assert len(matched)==1,(panel,kind,arm,seed,len(matched))
                    m,c=load_scores(matched[0],panel,kind);scalar+=c;aa.append(m[...,0])
                scores.append(np.stack(aa,-1))
            scores=np.stack(scores,0) # seed, condition, draw, origin, channel, arm
            for pair in [(0,1),(2,3)]:
                np.testing.assert_allclose(scores[...,pair[0]][:,equal],scores[...,pair[1]][:,equal],rtol=1e-10,atol=1e-12);equalchecks+=int(equal.sum())*3
            period=24 if panel.startswith(('electricity','neso')) else 96;origins=d['origins'][ids];boot=bootstrap_counts(origins,period)
            groups=[(c,[i]) for i,c in enumerate(names)]
            if kind=='standard':groups.append(('FAULT',[i for i,c in enumerate(names) if c.startswith(('POINT','BURST'))]))
            for cond,ci in groups:
                eq=equal[ci];sc=scores[:,ci];total_count=eq.size;parts=[]
                for label,mask in [('ALL',np.ones_like(eq,bool)),('EQUAL',eq),('DIFFERENT',~eq)]:
                    count=int(mask.sum());cnt=mask.sum(axis=(0,1,3)).astype(float);countsrows.append(dict(panel=panel,kind=kind,condition=cond,stratum=label,contexts=count,fraction=count/total_count))
                    if not count:
                        rows.append(dict(panel=panel,kind=kind,condition=cond,stratum=label,contexts=0,fraction=0,status='EMPTY_INPUT_STRATUM'));continue
                    sums=(sc*mask[None,...,None]).sum(axis=(1,2,4)) # seed,origin,arm
                    vals=sums.sum(1)/count;mean=vals.mean(0);t,g,w=decompose(mean);np.testing.assert_allclose(g+w,t,rtol=0,atol=1e-12)
                    denom=boot@cnt;valid=denom>0;draws=(boot[valid]@sums.mean(0))/denom[valid,None];bt,bg,bw=decompose(draws);gain=100*(1-draws[:,0]/draws[:,3]);lo,hi=np.quantile(gain,[.025,.975]);r=dict(panel=panel,kind=kind,condition=cond,stratum=label,contexts=count,fraction=count/total_count,status='SCORED',C3_nmae=mean[0],RECENCY_nmae=mean[3],gain_pct=100*(1-mean[0]/mean[3]),ci_low_pct=lo,ci_high_pct=hi,total=t,gate=g,weights=w,weighted_total=t*count/total_count,weighted_gate=g*count/total_count,weighted_weights=w*count/total_count,bootstrap_nonempty=len(draws),posthoc=True)
                    for k,z in [('gate',bg),('weights',bw),('total',bt)]:r[k+'_low'],r[k+'_high']=np.quantile(z,[.025,.975])
                    rows.append(r)
                    if label=='ALL':whole=np.array([t,g,w])
                    else:parts.append(np.array([t,g,w])*count/total_count)
                    for si,seed in enumerate([81551,81552,81553]):
                        st,sg,sw=decompose(vals[si]);seedrows.append(dict(panel=panel,kind=kind,condition=cond,stratum=label,seed=seed,contexts=count,C3_nmae=vals[si,0],RECENCY_nmae=vals[si,3],gain_pct=100*(1-vals[si,0]/vals[si,3]),total=st,gate=sg,weights=sw))
                        for oi,origin in enumerate(origins):originrows.append(dict(panel=panel,kind=kind,condition=cond,stratum=label,seed=seed,origin=int(origin),contexts=int(cnt[oi]),sum_A=sums[si,oi,0],sum_B=sums[si,oi,1],sum_C=sums[si,oi,2],sum_D=sums[si,oi,3]))
                np.testing.assert_allclose(np.sum(parts,axis=0),whole,rtol=1e-9,atol=1e-12)
            print('DIAGNOSED',panel,kind,flush=True)
    suffix='' if include_new else '_OLD'
    pd.DataFrame(rows).to_csv(OUT/f'CONDITIONAL_EFFECTS{suffix}.csv',index=False);pd.DataFrame(seedrows).to_csv(OUT/f'CONDITIONAL_SEEDS{suffix}.csv',index=False);pd.DataFrame(countsrows).to_csv(OUT/f'CONTEXT_COUNTS{suffix}.csv',index=False);pd.DataFrame(originrows).to_csv(OUT/f'CONDITIONAL_ORIGIN_SUMS{suffix}.csv.gz',index=False,compression='gzip')
    save(OUT/f'DIAGNOSTIC_VERIFICATION{suffix}.json',dict(status='VERIFIED',panels=panels,conditional_rows=len(rows),scalar_draw_rows=scalar,metrics_per_scalar_row=2,identical_mask_same_weight_scalar_checks=equalchecks,strata_reconstruct_full_effect=True,posthoc=True,new_fits=0,updates=0,unavailable='ETTm2 RECENCY was not trained; no reconstruction by retraining'))

def score_new():
    check_seal();marker=read(OUT/'ALL_PREDICTIONS_SAVED.json');assert marker['manifest_sha256']==sha(OUT/'PREDICTIONS.json');manifest=read(OUT/'PREDICTIONS.json');rows=[];scalar=0
    for key,r in manifest.items():
        kind=r['kind'];d,names,ids,nc,_=metadata(NEW,kind);m,c=load_scores(r,NEW,kind);scalar+=c;m=m.mean(1)
        for ci,cond in enumerate(names):
            for oi,origin in enumerate(d['origins'][ids]):
                for ch in range(nc):rows.append(dict(panel=NEW,kind=kind,condition=cond,arm=r['arm'],seed=r['seed'],origin=int(origin),channel='ND',**{k:m[ci,oi,ch,i] for i,k in enumerate(METRICS)}))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'NEW_ORIGIN_SCORES.csv.gz',index=False,compression='gzip');fault=frame[frame.condition.str.startswith(('POINT','BURST'))].assign(condition='FAULT');f=pd.concat([frame,fault]);keys=['panel','kind','condition','arm','seed'];raw=f.groupby(keys)[METRICS].mean();nr=f.groupby(keys+['channel']).normalized_mse.mean().pow(.5).groupby(level=list(range(len(keys)))).mean();raw=raw.join(nr.rename('nrmse')).reset_index();raw.to_csv(OUT/'RAW_SCORES.csv',index=False)
    effects=[];seeds=[];contrasts=[('C3',a) for a in ['C0','C1','C2','M_MEAN','M_ROTATE16','M_RECENCY','F0','PERSISTENCE','SEASONAL']]+[('C0',a) for a in ['F0','PERSISTENCE','SEASONAL']]
    for (kind,cond),g in f.groupby(['kind','condition']):
        grid=g.groupby(['origin','arm']).nmae.mean().unstack();means=grid.mean();boot=bootstrap_counts(grid.index.to_numpy(),24);z=boot@grid.to_numpy()/boot.sum(1,keepdims=True);lookup={a:i for i,a in enumerate(grid.columns)}
        for a,b in contrasts:
            diff=100*(1-z[:,lookup[a]]/z[:,lookup[b]]);lo,hi=np.quantile(diff,[.025,.975]);blo,bhi=np.quantile(diff,[.05/6,1-.05/6]);effects.append(dict(panel=NEW,kind=kind,condition=cond,new=a,baseline=b,new_nmae=means[a],baseline_nmae=means[b],gain_pct=100*(1-means[a]/means[b]),ci_low_pct=lo,ci_high_pct=hi,bonferroni3_low_pct=blo if kind=='standard' and cond=='SHIFT8' and a=='C3' and b in ['C0','C2','M_RECENCY'] else np.nan,bonferroni3_high_pct=bhi if kind=='standard' and cond=='SHIFT8' and a=='C3' and b in ['C0','C2','M_RECENCY'] else np.nan,conditional_on_fixed_seeds=True,bootstrap_draws=2000,bootstrap_seed=90401))
            for seed in [81551,81552,81553]:
                aa=g[(g.arm==a)&(g.seed==seed)].nmae.mean();bb=g[(g.arm==b)&(g.seed==(0 if b in ['F0','PERSISTENCE','SEASONAL'] else seed))].nmae.mean();assert np.isfinite(aa+bb);seeds.append(dict(kind=kind,condition=cond,new=a,baseline=b,seed=seed,new_nmae=aa,baseline_nmae=bb,gain_pct=100*(1-aa/bb)))
    pd.DataFrame(effects).to_csv(OUT/'EFFECTS.csv',index=False);pd.DataFrame(seeds).to_csv(OUT/'SEED_EFFECTS.csv',index=False)
    for r in effects:
        g=raw[(raw.kind==r['kind'])&(raw.condition==r['condition'])];a=g[g.arm==r['new']].nmae.mean();b=g[g.arm==r['baseline']].nmae.mean();np.testing.assert_allclose([a,b,100*(1-a/b)],[r['new_nmae'],r['baseline_nmae'],r['gain_pct']],rtol=1e-10,atol=1e-10)
    check_seal();save(OUT/'SCORE_VERIFICATION.json',dict(status='VERIFIED',new_origin_rows=len(frame),scalar_draw_rows=scalar,metrics_per_scalar_row=2,all_effects_replayed=len(effects),all_predictions_saved_before_scoring=True,new_fits=0,updates=0));print('NEW SCORE VERIFIED',len(effects),flush=True)
if __name__=='__main__':
    import sys
    if sys.argv[-1]=='old':diagnostic(False)
    elif sys.argv[-1]=='score':score_new()
    else:score_new();diagnostic(True)
