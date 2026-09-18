"""Frozen-function decomposition and descriptive, observed-only strata."""
import pandas as pd
from .common import *
from experiments.c3_identifiability_temporal_20260918.analyze import load_scores
ARMS=['C3','C3_W_MAG_G','MAG_W_C3_G','MAG_ONLY']

def masks(f):
    peak=f.peak_abs_d.to_numpy();run=f.longest_same_sign_run.to_numpy();last=f.last_extreme_position.to_numpy();eq=f.gate_equal.to_numpy(bool)
    return {'ALL':{'ALL':np.ones(len(f),bool)},'gate':{'EQUAL':eq,'DIFFERENT':~eq},'magnitude':{'LE3':peak<=3,'GT3_LE6':(peak>3)&(peak<=6),'GT6_LE12':(peak>6)&(peak<=12),'GT12':peak>12},'run':{'NONE':run==0,'LEN1_7':(run>=1)&(run<8),'LEN8_31':(run>=8)&(run<32),'LEN32_PLUS':run>=32},'position':{'NONE':last<0,'OLD':(last>=0)&(last<384),'RECENT128':last>=384}}

def analyze():
    check_seal();assert read(OUT/'ALL_PREDICTIONS_SAVED.json')['manifest_sha256']==sha(OUT/'PREDICTIONS.json')
    rec=records()+list(read(OUT/'PREDICTIONS.json').values());features=pd.read_csv(OUT/'INPUT_FEATURES.csv.gz');summary=[];seedrows=[];originrows=[];channelrows=[];rawrows=[];scalar=0;equalchecks=0;replayrows=0
    for panel in PANELS:
        for kind in ['standard','shape']:
            d,names,ids,nc,_=metadata(panel,kind);n=len(ids);origins=d['origins'][ids];period=24 if panel.startswith('electricity') else 96;boot=bootstrap(origins,period,panel)
            ff=features[(features.panel==panel)&(features.kind==kind)];group_masks=masks(ff);sc=[]
            for seed in SEEDS:
                mm=[]
                for arm in ARMS:
                    row=next(r for r in rec if (r['panel'],r['kind'],r['arm'],r['seed'])==(panel,kind,arm,seed));m,c=load_scores(row,panel,kind);scalar+=c;mm.append(m)
                sc.append(np.stack(mm,axis=-2))
            sc=np.stack(sc) # seed,condition,draw,origin,channel,arm,metric
            assert sc.shape==(3,len(names),2,n,nc,4,5)
            eq=ff.gate_equal.to_numpy(bool).reshape(len(names),2,n,nc)
            for a,b in [(0,1),(2,3)]:
                np.testing.assert_allclose(sc[...,a,:][:,eq],sc[...,b,:][:,eq],rtol=1e-5,atol=1e-6);equalchecks+=3*int(eq.sum())
            # Diagonal nMAE/MAE/pinball reaggregation must match previous published results.
            prior=pd.read_csv(parent.OUT/'RAW_SCORES.csv')
            for ci,name in enumerate(names):
                for si,seed in enumerate(SEEDS):
                    for ai,arm in enumerate(ARMS):
                        v=sc[si,ci,:,:, :,ai,:];means=v.mean((0,1,2));rawrows.append(dict(panel=panel,kind=kind,condition=name,seed=seed,arm=arm,nmae=means[0],mae=means[1],normalized_mse=means[2],pinball=means[3],crossing=means[4]))
                        if arm in ['C3','MAG_ONLY']:
                            pr=prior[(prior.panel==panel)&(prior.kind==kind)&(prior.condition==name)&(prior.seed==seed)&(prior.arm==arm)].iloc[0]
                            np.testing.assert_allclose(means[[0,1,3]],pr[['nmae','mae','pinball']].to_numpy(float),rtol=1e-10,atol=1e-12);replayrows+=1
            groups=[(c,[i]) for i,c in enumerate(names)]
            if kind=='standard':groups.append(('FAULT',[i for i,c in enumerate(names) if c.startswith(('POINT','BURST'))]))
            for cond,ci in groups:
                scores=sc[:,ci,...,0];totalcontexts=scores.shape[1]*2*n*nc
                for family,subs in group_masks.items():
                    assert np.all(np.sum(np.stack(list(subs.values())),axis=0)==1)
                    pieces=[]
                    for label,flatmask in subs.items():
                        mask=flatmask.reshape(len(names),2,n,nc)[ci];count=int(mask.sum());cnt=mask.sum((0,1,3)).astype(float)
                        if count==0:
                            summary.append(dict(panel=panel,kind=kind,condition=cond,family=family,stratum=label,contexts=0,status='EMPTY'));continue
                        sums=(scores*mask[None,...,None]).sum((1,2,4));vals=sums.sum(1)/count;mean=vals.mean(0);t,g,w,i=decomposition(mean);np.testing.assert_allclose(t,g+w,atol=1e-12,rtol=0)
                        denom=boot@cnt;valid=denom>0;z=boot[valid]@sums.mean(0)/denom[valid,None];bt,bg,bw,bi=decomposition(z);gain=100*bt/z[:,0]
                        row=dict(panel=panel,kind=kind,condition=cond,family=family,stratum=label,contexts=count,fraction=count/totalcontexts,status='SCORED',A=mean[0],B=mean[1],C=mean[2],D=mean[3],total=t,gate=g,weights=w,interaction=i,MAG_gain_pct=100*t/mean[0],weighted_total=t*count/totalcontexts,weighted_gate=g*count/totalcontexts,weighted_weights=w*count/totalcontexts,bootstrap_nonempty=len(z),posthoc=True)
                        for name,values in [('total',bt),('gate',bg),('weights',bw),('interaction',bi),('MAG_gain_pct',gain)]:row[name+'_low'],row[name+'_high']=np.quantile(values,[.025,.975])
                        summary.append(row);pieces.append(np.array([t,g,w])*count/totalcontexts)
                        if family=='ALL':whole=np.array([t,g,w])
                        for si,seed in enumerate(SEEDS):
                            st,sg,sw,sint=decomposition(vals[si]);seedrows.append(dict(panel=panel,kind=kind,condition=cond,family=family,stratum=label,seed=seed,contexts=count,A=vals[si,0],B=vals[si,1],C=vals[si,2],D=vals[si,3],total=st,gate=sg,weights=sw,interaction=sint))
                    np.testing.assert_allclose(np.sum(pieces,axis=0),whole,rtol=1e-9,atol=1e-12)
                v=scores.mean((1,2,4));cv=scores.mean((0,1,2,3)) # seed,origin,4; channel,4
                for si,seed in enumerate(SEEDS):
                    for oi,o in enumerate(origins):originrows.append(dict(panel=panel,kind=kind,condition=cond,seed=seed,origin=int(o),A=v[si,oi,0],B=v[si,oi,1],C=v[si,oi,2],D=v[si,oi,3]))
                columns=read(old.OUT/'DATA_MANIFEST.json')[panel]['selected_columns']
                for ch,cid in enumerate(columns):
                    t,g,w,i=decomposition(cv[ch]);loo=np.delete(cv,ch,axis=0).mean(0);lt,lg,lw,li=decomposition(loo);channelrows.append(dict(panel=panel,kind=kind,condition=cond,channel=str(cid),A=cv[ch,0],D=cv[ch,3],total=t,gate=g,weights=w,interaction=i,contribution=t/nc,leave_out_MAG_gain_pct=100*lt/loo[0]))
            print('ANALYZED',panel,kind,flush=True)
    pd.DataFrame(summary).to_csv(OUT/'DECOMPOSITION.csv',index=False);pd.DataFrame(seedrows).to_csv(OUT/'SEED_DECOMPOSITION.csv',index=False);pd.DataFrame(originrows).to_csv(OUT/'ORIGIN_ERRORS.csv.gz',index=False,compression='gzip');pd.DataFrame(channelrows).to_csv(OUT/'CHANNEL_CONTRIBUTIONS.csv',index=False);pd.DataFrame(rawrows).to_csv(OUT/'RAW_SCORES.csv',index=False)
    checks=read(OUT/'MODEL_CHECKS.json');assert len(checks)==36 and all(all(r.values()) for r in checks.values());check_seal()
    save(OUT/'VERIFICATION.json',dict(status='VERIFIED',new_fits=0,optimizer_updates=0,new_prediction_views=36,reused_prediction_views=36,selected_models=12,scalar_metric_rows=scalar,metrics_per_scalar_row=2,diagonal_score_rows_replayed=replayrows,equal_gate_weight_checks=equalchecks,partitions_reconstruct_all=True,decomposition_identity=True,all_model_checks=True,parents_unchanged=True))
    save(OUT/'status.json',dict(execution='VERIFIED',new_fits=0,optimizer_updates=0,completed_views=36,automatic_successor=False));print('DIAGNOSIS_VERIFIED',flush=True)
if __name__=='__main__':analyze()
