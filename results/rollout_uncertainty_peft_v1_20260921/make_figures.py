"""Presentation only: reads finalized score/resource CSVs, never models or targets."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
SOURCES=['Electricity','ETTh1']
METHODS=['F0_MEDIAN','F0_NATIVE','R_ROLLOUT_LORA','S_STATE_ADAPTER','U_UNCERTAINTY_ADAPTER','R_NATIVE','R_MC16','CHRONOS2_DIRECT']
SHORT=['F0-med','F0-native','R','S','U','R-native','R-MC16','C2-direct']
COLORS=['#999999','#000000','#0072B2','#E69F00','#D55E00','#009E73','#CC79A7','#56B4E9']
MARKERS=['o','s','^','v','D','P','X','*']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

def read(name):
    p=HERE/name
    return pd.read_csv(p if p.exists() else p.with_suffix(p.suffix+'.gz'))

def save(fig,name,caption):
    fig.savefig(HERE/(name+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(HERE/(name+'.pdf'),bbox_inches='tight')
    # Grayscale review artifact stays in the local cache, not an extra figure type.
    plt.close(fig)
    return {'name':name,'caption':caption,'files':[{ 'path':name+ext,'sha256':hashlib.sha256((HERE/(name+ext)).read_bytes()).hexdigest()} for ext in ['.png','.pdf']]}

def main():
    verification=json.loads((HERE/'VERIFICATION.json').read_text(encoding='utf-8'))
    assert verification['status']=='PASS'
    scores=read('SCORES_SUMMARY.csv')
    assert not scores.seed.eq(92120).any()
    resources=read('RESOURCES.csv')
    artifacts=[]
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for row,source in enumerate(SOURCES):
        for col,variant in enumerate(['ordered','affine']):
            ax=axes[row,col]
            for method,label,color,marker in zip(METHODS,SHORT,COLORS,MARKERS):
                values=[]; ranges=[]
                for block in range(1,5):
                    v=scores[(scores.source==source)&(scores.method==method)&(scores.variant==variant)&(scores.window==f'block_{block}')].scaled_pinball.to_numpy()
                    assert len(v) in [1,2]
                    values.append(v.mean()); ranges.append((v.max()-v.min())/2)
                ax.errorbar([1,2,3,4],values,yerr=ranges,label=label,color=color,marker=marker,markersize=4,linewidth=1.1,capsize=2)
            ax.set(title=f'{chr(65+row*2+col)}  {source} / {variant}',xlabel='Forecast block (64 hours each)',ylabel='TRAIN-scaled twice-pinball (lower is better)',xticks=[1,2,3,4])
            ax.grid(axis='y',alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    artifacts.append(save(fig,'FIGURE_1_BLOCK_SCORES','Each point is the mean of repeat seeds 92121/92122; error bars show their range, not a confidence interval. Fixed F0/C2 curves have one model. Electricity: 32 series, 209 TEST origins; ETTh1: 7 series, 135 origins. Paired 7-origin bootstrap CIs are in EFFECTS.csv.'))

    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    metrics=[('coverage80','Pointwise 80% coverage'),('scaled_width80','80% width / TRAIN std'),('pce','PCE (lower is better)')]
    for row,source in enumerate(SOURCES):
        for col,(metric,label) in enumerate(metrics):
            ax=axes[row,col]
            for i,(method,color) in enumerate(zip(METHODS,COLORS)):
                for variant,offset,marker in [('ordered',-.13,'o'),('affine',.13,'s')]:
                    v=scores[(scores.source==source)&(scores.method==method)&(scores.variant==variant)&(scores.window=='full_256')][metric].to_numpy()
                    ax.errorbar(i+offset,v.mean(),yerr=(v.max()-v.min())/2,color=color,marker=marker,markerfacecolor=color if variant=='ordered' else 'white',capsize=2,markersize=5)
            ax.set(title=f'{chr(65+row*3+col)}  {source}',ylabel=label,xticks=range(8),xticklabels=SHORT)
            ax.tick_params(axis='x',rotation=45)
            ax.set_ylim(bottom=0)
            if metric=='coverage80': ax.axhline(.8,color='black',linestyle='--',linewidth=.8); ax.set_ylim(0,1)
            ax.grid(axis='y',alpha=.15)
    fig.suptitle('Filled circle: ordered output; open square: CAL-only affine output')
    artifacts.append(save(fig,'FIGURE_2_CALIBRATION_WIDTH','Coverage, width, and PCE are separate quantities. Error bars show the range of the same two repeat seeds (not CI); fixed baselines have one model. Coverage is pointwise, not simultaneous trajectory coverage.'))

    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for row,source in enumerate(SOURCES):
        for col,variant in enumerate(['ordered','affine']):
            ax=axes[row,col]
            for method,label,color,marker in zip(METHODS,SHORT,COLORS,MARKERS):
                v=scores[(scores.source==source)&(scores.method==method)&(scores.variant==variant)&(scores.window=='full_256')].scaled_pinball.to_numpy()
                r=resources[(resources.source==source)&(resources.method==method)&(resources.variant==variant)&(resources.batch_size==8)]
                x=r.median_wall_seconds.mean()*1000/8
                low=r.min_wall_seconds.mean()*1000/8; high=r.max_wall_seconds.mean()*1000/8
                ax.errorbar(x,v.mean(),xerr=np.array([[x-low],[high-x]]),yerr=(v.max()-v.min())/2,label=label,color=color,marker=marker,markersize=7,capsize=2,linestyle='none')
            ax.set_xscale('log')
            ax.set(title=f'{chr(65+row*2+col)}  {source} / {variant}',xlabel='Wall time per example at batch 8 (ms, log scale)',ylabel='TRAIN-scaled twice-pinball (lower is better)')
            ax.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    artifacts.append(save(fig,'FIGURE_3_QUALITY_LATENCY','RTX 4070, FP32, TF32 off. x is mean across model seeds of median wall time from three timed repetitions at batch 8; horizontal bars span mean minima/maxima, vertical bars span two seed scores. These are ranges, not CIs. Loading excluded; GPU/CPU transfer, metadata, sorting and applicable affine included. ETTh1 batch 8 repeats one of its seven independent series. Chronos-2 differs in architecture/size/pretraining.'))
    (HERE/'FIGURES_MANIFEST.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'figure_types':3,'artifacts':artifacts},indent=2),encoding='utf-8')

if __name__=='__main__': main()
