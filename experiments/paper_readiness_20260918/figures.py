"""Paper figures and tables from locked results; no model inference or selection."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/paper_readiness_mpl')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator,ScalarFormatter
from .common import *
PAPER=ROOT/'papers/persistence_adaptation'
LABELS={'electricity':'Electricity (4)','electricity_transfer':'Electricity transfer (16)','ettm1':'ETTm1','ettm2':'ETTm2','neso_2025':'NESO 2025'}

def build():
    assert read(OUT/'VERIFICATION.json')['status']=='VERIFIED';check_seal();PAPER.mkdir(parents=True,exist_ok=True);figdir=PAPER/'figures';figdir.mkdir(exist_ok=True);tabdir=PAPER/'tables';tabdir.mkdir(exist_ok=True)
    effect=pd.read_csv(OUT/'EFFECTS.csv');fac=pd.read_csv(OUT/'FACTORIAL.csv');raw=pd.read_csv(OUT/'RAW_SCORES.csv');seeds=pd.read_csv(OUT/'SEED_EFFECTS.csv');old_e=pd.read_csv(ext.OUT/'EFFECTS.csv');old_e=old_e[old_e.ci_type=='time']
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    def finish(name):
        plt.tight_layout()
        for extn in ['png','pdf','svg']:plt.savefig(figdir/(name+'.'+extn),dpi=180,bbox_inches='tight')
        svg=figdir/(name+".svg");svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
        plt.close()
    # One exact table per caption, with complete data also exported.
    main=effect[(effect.kind=='standard')&(effect.condition=='SHIFT8')&(effect.new=='C3')&effect.baseline.isin(['C0','C2','M_RECENCY'])];main.to_csv(tabdir/'T1_main_SHIFT8.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,base in zip(axes,['C0','C2','M_RECENCY']):
        for i,p in enumerate(PANELS):
            s=main[(main.panel==p)&(main.baseline==base)]
            if s.empty:ax.text(0,i,'not run',color='grey');continue
            r=s.iloc[0];ax.hlines(i,r.ci_low_pct,r.ci_high_pct,color='black');ax.scatter(r.gain_pct,i,s=40,color="tab:blue")
            z=seeds[(seeds.panel==p)&(seeds.kind=='standard')&(seeds.condition=='SHIFT8')&(seeds.new=='C3')&(seeds.baseline==base)];ax.scatter(z.gain_pct,np.full(len(z),i)+.15,marker='x',s=22,color='tab:orange')
        ax.axvline(0,color='grey',lw=.7);ax.set_yticks(range(5),[LABELS[p] for p in PANELS]);ax.invert_yaxis();ax.set_title('C3 vs '+base);ax.set_xlabel('nMAE reduction (%)')
    fig.suptitle('SHIFT8: means, conditional time-block 95% intervals, and individual seeds');finish('F1_main_seed_effects')
    baseline=effect[(effect.kind=='standard')&effect.new.isin(['C0','C3'])&effect.baseline.isin(['F0','PERSISTENCE','SEASONAL'])];baseline.to_csv(tabdir/'T2_baseline_strength.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,cond in zip(axes,['REFERENCE','FAULT','SHIFT8']):
        x=raw[(raw.kind=='standard')&(raw.condition==cond)&raw.arm.isin(['F0','PERSISTENCE','SEASONAL','C0','C2','C3','M_RECENCY'])].groupby(['panel','arm']).nmae.mean().unstack().reindex(PANELS)
        x.index=[LABELS[p] for p in x.index];x.plot.bar(ax=ax,logy=True,rot=65,width=.85);ax.set_title(cond);ax.set_ylabel('nMAE (log scale)');ax.set_xlabel('');ax.legend(fontsize=6)
    finish('F2_foundation_simple_baselines')
    fac.to_csv(tabdir/'T3_factorial_all.csv',index=False)
    fixed=fac[(fac.comparison=='C3_vs_C2_fixed1024')&(fac.kind=='standard')];fig,axes=plt.subplots(1,3,figsize=(14,4));conditions=['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT']
    for ax,p in zip(axes,['electricity','electricity_transfer','neso_2025']):
        x=fixed[fixed.panel==p].set_index('condition').loc[conditions];y=np.arange(5)
        for col,offset,color in [('gate',-.15,'tab:blue'),('weights',.15,'tab:orange')]:
            ax.barh(y+offset,x[col],height=.3,label=col,color=color);ax.hlines(y+offset,x[col+'_low'],x[col+'_high'],lw=.8,color='black')
        ax.plot(x.total,y,'kx',label='total');ax.axvline(0,color='grey');ax.set_yticks(y,conditions);ax.set_title(LABELS[p]);ax.legend(fontsize=7);ax.set_xlabel('absolute nMAE benefit');ax.xaxis.set_major_locator(MaxNLocator(4));fmt=ScalarFormatter(useMathText=True);fmt.set_powerlimits((-3,-3));ax.xaxis.set_major_formatter(fmt)
    fig.suptitle('C3 vs C2: matched LR and 1024 updates; inference gate and learned-weight contrasts');finish('F3_matched_factorial')
    shape=old_e[(old_e.kind=='shape')&(old_e.new=='C3')&old_e.baseline.isin(['C2','M_RECENCY'])];shape.to_csv(tabdir/'T4_shapes.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,7));names=SHAPES+['PAIRED_SHIFT8_D32'];lim=max(1,shape.gain_pct.abs().max())
    for ax,base in zip(axes,['C2','M_RECENCY']):
        z=shape[shape.baseline==base].pivot(index='condition',columns='panel',values='gain_pct').reindex(index=names,columns=PANELS)
        ax.imshow(z,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto');ax.set_xticks(range(5),[LABELS[p] for p in PANELS],rotation=70);ax.set_yticks(range(9),names);ax.set_title('C3 vs '+base+' (%)')
        for i in range(9):
            for j in range(5):
                a=z.iloc[i,j];ax.text(j,i,'N/A' if pd.isna(a) else f'{a:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(a)>.6*lim else 'black')
    finish('F4_all_shapes')
    mix=[];thresholds=[]
    for p in PANELS:
        m=raw[(raw.panel==p)&(raw.kind=='standard')].groupby(['arm','condition']).nmae.mean()
        for base in ['C0','C2','M_RECENCY']:
            if base not in m.index.get_level_values(0):continue
            dr=m[base,'REFERENCE']-m['C3','REFERENCE'];df=m[base,'FAULT']-m['C3','FAULT'];ds=m[base,'SHIFT8']-m['C3','SHIFT8']
            for q in [0.,.1,.5,1.]:
                d0=(1-q)*dr+q*df;den=ds-d0;root=-d0/den if den else float('nan');thresholds.append(dict(panel=p,baseline=base,fault_share_among_nonshift=q,benefit_at_w0=d0,benefit_at_w1=ds,crossover_shift_share=root if 0<=root<=1 else float('nan'),synthetic_sensitivity_not_prevalence_estimate=True))
                for w in np.linspace(0,1,101):mix.append(dict(panel=p,baseline=base,q=q,w=w,absolute_nmae_benefit=(1-w)*d0+w*ds))
    mix=pd.DataFrame(mix);mix.to_csv(tabdir/'T5_mixture_curves.csv',index=False);pd.DataFrame(thresholds).to_csv(tabdir/'T5_mixture_crossovers.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,p in zip(axes.flat,PANELS):
        for base,color in [('C0','tab:blue'),('C2','tab:orange'),('M_RECENCY','tab:green')]:
            for q,style in [(0,'-'),(.1,'--'),(.5,':'),(1,'-.')]:
                z=mix[(mix.panel==p)&(mix.baseline==base)&(mix.q==q)];ax.plot(z.w,z.absolute_nmae_benefit,style,color=color,lw=1,label=f'{base}, q={q}')
        ax.axhline(0,color='grey',lw=.7);ax.set_title(LABELS[p]);ax.set_xlabel('hypothetical SHIFT8 share');ax.set_ylabel('absolute nMAE benefit')
    axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='center',fontsize=8);finish('F5_operating_mix_sensitivity')
    # Validation curves are original observed trajectories, not new training.
    curves=[]
    for row in read(PRIOR/'MODEL_SELECTION.json'):
        if row['arm'] not in ['C2','C3','M_RECENCY'] or row['source']=='ettm2':continue
        base=old.OLD if row.get('reused') else PRIOR;receipt=read(base/'fits'/row['fit']/'receipt.json')
        for r in receipt['checkpoints']:curves.append(dict(source=row['source'],arm=row['arm'],seed=row['seed'],lr=row['lr'],step=r['step'],objective=r['objective'],selected=r['step']==row['step']))
    curves=pd.DataFrame(curves);curves.to_csv(tabdir/'T6_validation_trajectories.csv',index=False);fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,p in zip(axes,['electricity','ettm1']):
        for arm,color in [('C2','tab:blue'),('C3','tab:orange'),('M_RECENCY','tab:green')]:
            for i,(seed,z) in enumerate(curves[(curves.source==p)&(curves.arm==arm)].groupby('seed')):
                ax.plot(z.step,z.objective,color=color,ls=['-','--',':'][i],label=f'{arm}/{seed}');zz=z[z.selected];ax.scatter(zz.step,zz.objective,color=color,s=20)
        ax.set_title(LABELS[p]);ax.set_xlabel('additional optimizer updates');ax.set_ylabel('original V equal-condition nMAE');ax.legend(fontsize=6)
    finish('F6_validation_trajectories')
    resource=pd.read_csv(PRIOR/'RESOURCE_REPORT.csv');resource=resource[(resource.kind=='inference_profile')&resource.arm.isin(['C0','C1','C2','C3','M_RECENCY'])];resource.to_csv(tabdir/'T7_resource_profiles.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    p='electricity_transfer';time=resource[resource.panel==p].groupby('arm').median_seconds.mean();means=raw[(raw.panel==p)&(raw.kind=='standard')].groupby(['arm','condition']).nmae.mean()
    for arm in ['C0','C1','C2','C3','M_RECENCY']:
        for ax,cond in zip(axes,['REFERENCE','SHIFT8']):
            ax.scatter(time[arm]*1000,means[arm,cond]);ax.annotate(arm,(time[arm]*1000,means[arm,cond]),xytext=(3,3),textcoords='offset points');ax.set_title(cond);ax.set_xlabel('mean of per-seed median time (ms / 128 inputs)');ax.set_ylabel('nMAE')
    fig.suptitle('Electricity transfer (16): RTX 3080 FP32 profiles; shared B0 training excluded');finish('F7_cost_accuracy')
    # Deterministic example selection locked before scoring: first E origin/channel/draw/seed.
    oldpred=list(read(PRIOR/'PREDICTIONS_MANIFEST.json').values())+list(read(ext.OUT/'PREDICTIONS.json').values());fig,axes=plt.subplots(2,3,figsize=(14,7))
    example_rows=[]
    for ri,p in enumerate(['electricity_transfer','neso_2025']):
        d=np.load(data_path(p)/'E_DISCOVERY_inputs.npz');yy=np.load(data_path(p)/'E_DISCOVERY_labels.npz')['y'];nc=yy.shape[1];offset=np.load(panel_path(p,'standard')/'E_DISCOVERY_offset.npy');xx,ss=inputs(p,'standard')
        for ci,cond in enumerate(['REFERENCE','POINT8','SHIFT8']):
            ix=STATES.index(cond)*2*128*nc;x=xx[ix];y=yy[0,0]+offset[ix];ax=axes[ri,ci];ax.plot(np.arange(-80,0),x[-80:],color='grey',label='observed');ax.plot(np.arange(64),y,color='black',label='target')
            for arm,color in [('C0','tab:blue'),('C2','tab:green'),('C3','tab:orange')]:
                rr=next(r for r in oldpred if r['panel']==p and r['kind']=='standard' and r['arm']==arm and r['seed']==81551 and r.get('stage','selected')=='selected');pred=np.load(ROOT/rr['path'],mmap_mode='r')[ix,4];ax.plot(np.arange(64),pred,color=color,label=arm)
                for h,val in enumerate(pred):example_rows.append(dict(panel=p,condition=cond,arm=arm,seed=81551,origin=int(d['origins'][0]),channel=0,draw=0,horizon=h,prediction=float(val),target=float(y[h])))
            ax.axvline(0,color='grey',ls=':');ax.set_title(LABELS[p]+' / '+cond);ax.set_xlabel('relative index');ax.legend(fontsize=6)
    fig.suptitle('Fixed first-origin examples, not selected for model advantage');finish('F8_fixed_examples');pd.DataFrame(example_rows).to_csv(tabdir/'T8_fixed_examples.csv',index=False)
    # Export all scored conditions, seed effects and both interval definitions.
    (pd.read_csv(OUT/'RAW_SCORES_WITH_NRMSE.csv') if (OUT/'RAW_SCORES_WITH_NRMSE.csv').exists() else raw).to_csv(tabdir/'ALL_RAW_SCORES.csv',index=False);seeds.to_csv(tabdir/'ALL_SEED_EFFECTS.csv',index=False);effect.to_csv(tabdir/'ALL_EFFECTS.csv',index=False)
    save(PAPER/'FIGURE_MANIFEST.json',dict(figures=sorted(p.name for p in figdir.glob('*.png')),tables=sorted(p.name for p in tabdir.glob('*.csv')),new_training=0,example_selection='first E origin/channel/draw, seed81551; frozen protocol',intervals='exploratory paired 7-day blocks; conditional on fixed seeds',prior_results_unchanged=True))
    print('PAPER_FIGURES_COMPLETE',flush=True)
if __name__=='__main__':build()
