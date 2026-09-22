from common import *
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS=['#4477AA','#66CCEE','#228833','#CCBB44','#AA3377','#EE7733']


def save(fig,name):
    directory=RESULTS/'figures';directory.mkdir(exist_ok=True)
    fig.savefig(directory/f'{name}.png',dpi=220,facecolor='white')
    fig.savefig(directory/f'{name}.svg',facecolor='white')
    svg=directory/f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


def metric_points(ax,summaries,arms,title):
    for index,arm in enumerate(arms):
        seeds=[0] if arm in ['A0','N0'] else [92201,92202]
        values=[summaries[f'{arm}_{seed}']['macro']['pinball'] for seed in seeds]
        ax.scatter(np.full(len(values),index),values,color=COLORS[index%len(COLORS)],s=38,zorder=3)
        ax.plot([index-.18,index+.18],[np.mean(values)]*2,color='black',lw=2)
    ax.set_xticks(range(len(arms)),arms,rotation=25,ha='right')
    ax.set_ylabel('TEST normalized twice-pinball (lower is better)')
    ax.set_title(title+'\nDots: individual seeds; black line: score mean',loc='left',fontsize=11)
    ax.grid(axis='y',alpha=.2)


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none'})
    summaries=read(RESULTS/'TEST_SCORES.json')['summaries']
    effects=read(RESULTS/'EFFECTS.json')
    fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
    for i,cat in enumerate(['Q','T','F']):
        rows=[r for r in effects['comparisons'] if r['category']==cat and r['primary_V_selected']]
        mean=next(r for r in rows if r['seed']=='score_mean')
        lo,hi=mean['ci_gain_pct']
        ax.hlines(i,lo,hi,color=COLORS[i],lw=4)
        ax.scatter(mean['gain_pct'],i,s=95,color=COLORS[i],zorder=4)
        for r in rows:
            if r['seed']!='score_mean':ax.scatter(r['gain_pct'],i+(.12 if r['seed']==92201 else -.12),marker='x',s=55,color='black')
        ax.text(1.01,i,f"{mean['gain_pct']:+.3f}%",transform=ax.get_yaxis_transform(),va='center')
    ax.axvline(0,color='black',lw=1);ax.axvline(1,color='grey',ls='--',lw=1)
    ax.set_yticks(range(3),['Q / forecast rank allocation','T / adaptation delta transfer','F / periodic personalization'])
    ax.set_xlabel('Gain over validation-selected baseline (%) — positive favors candidate')
    ax.set_title('Electricity development screen: extra value of each candidate\nBars: paired 14-day block 95% CI; x: two fixed seeds; dashed: +1% project criterion',loc='left')
    ax.grid(axis='x',alpha=.2)
    save(fig,'triage_effects')

    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    qarms=['Q_FP','Q_STD','Q_LOFTQ','Q_QERA','Q_IO16','Q_FORECAST']
    metric_points(axes[0,0],summaries,qarms,'Q: all predefined controls')
    for i,arm in enumerate(qarms):
        resources=[read(RESULTS/'resources'/f'{arm}_{s}.json') for s in [92201,92202]]
        x=np.mean([r['storage_ratio']*100 for r in resources])
        y=np.mean([summaries[f'{arm}_{s}']['macro']['pinball'] for s in [92201,92202]])
        axes[0,1].scatter(x,y,s=80,color=COLORS[i]);axes[0,1].annotate(arm,(x,y),xytext=(4,4),textcoords='offset points',fontsize=8)
    axes[0,1].axvspan(0,60,color='#228833',alpha=.07)
    axes[0,1].axvline(60,ls='--',color='grey');axes[0,1].set_xlim(35,110)
    axes[0,1].set_xlabel('Serialized base + adapter / BF16 base (%)')
    axes[0,1].set_ylabel('TEST primary score');axes[0,1].set_title('Actual packed deployment storage; cap = 60%',loc='left')
    alloc=read(RESULTS/'Q_ALLOCATION.json');groups=alloc['groups']
    labels=['Input','Enc attn','Enc FFN','Dec self','Dec cross','Dec FFN','Output']
    axes[1,0].bar(labels,[alloc['ranks'][g] for g in groups],color=COLORS[-1])
    axes[1,0].axhline(4,ls='--',color='black',label='uniform rank4')
    axes[1,0].set_ylabel('Frozen rank');axes[1,0].legend();axes[1,0].set_title('Allocation from TRAIN contexts only',loc='left')
    axes[1,1].bar(labels,[alloc['sensitivities'][g]['sensitivity'] for g in groups],color=COLORS[0])
    axes[1,1].set_ylabel('Decrease in normalized forecast discrepancy')
    axes[1,1].set_title('32 contexts; 288 example forwards; no future target',loc='left')
    save(fig,'Q_results')

    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    metric_points(axes[0,0],summaries,['A0','T_A1','N0','T_RECENT','T_KD','T_BLEND','T_DELTA'],'T: old/new baselines and all students')
    for i,arm in enumerate(['T_RECENT','T_KD','T_BLEND','T_DELTA']):
        for seed in [92201,92202]:
            fit=read(RESULTS/'fits'/f'{arm}_{seed}'/'FIT.json')
            axes[0,1].plot([r['step'] for r in fit['validation']],[r['score']['macro']['pinball'] for r in fit['validation']],
                           color=COLORS[i],ls='-' if seed==92201 else '--',marker='.',label=arm if seed==92201 else None)
    axes[0,1].legend(fontsize=8);axes[0,1].set_xlabel('Optimizer updates');axes[0,1].set_ylabel('BRIDGE validation true-target score')
    axes[0,1].set_title('Solid seed92201 / dashed seed92202',loc='left')
    for i,arm in enumerate(['T_KD','T_BLEND','T_DELTA']):
        frames=[pd.read_json(RESULTS/'fits'/f'{arm}_{s}'/'training.jsonl',lines=True) for s in [92201,92202]]
        true=np.mean([f['true_loss'].to_numpy() for f in frames],axis=0)
        pseudo=.5*np.mean([f['pseudo_loss'].to_numpy() for f in frames],axis=0)
        ratio=np.mean([.5*f['pseudo_gradient_norm'].to_numpy()/np.maximum(f['true_gradient_norm'].to_numpy(),1e-12) for f in frames],axis=0)
        axes[1,0].plot(np.arange(16,257,16),pseudo.reshape(16,16).mean(1),color=COLORS[i],label=arm+' 0.5*pseudo')
        axes[1,0].plot(np.arange(16,257,16),true.reshape(16,16).mean(1),color=COLORS[i],ls='--',alpha=.6)
        axes[1,1].plot(np.arange(16,257,16),ratio.reshape(16,16).mean(1),color=COLORS[i],label=arm)
    axes[1,0].legend(fontsize=8);axes[1,0].set_title('Actual training losses: dashed=true, solid=weighted pseudo',loc='left',fontsize=10)
    axes[1,1].axhline(1,color='grey',ls='--');axes[1,1].legend(fontsize=8)
    axes[1,1].set_ylabel('0.5 * pseudo gradient norm / true gradient norm')
    axes[1,1].set_title('Both terms audited; no lambda retuning',loc='left')
    for a in axes[1]:a.set_xlabel('Optimizer updates (16-step means)')
    save(fig,'T_results')

    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    farms=['F_LOCAL','F_SHARED','F_AFFINE','F_HEAD','F_PERIODIC']
    metric_points(axes[0,0],summaries,farms,'F: client-macro score')
    local=np.mean([summaries[f'F_LOCAL_{s}']['by_series']['pinball'] for s in [92201,92202]],axis=0)
    for i,arm in enumerate(farms[1:]):
        values=np.mean([summaries[f'{arm}_{s}']['by_series']['pinball'] for s in [92201,92202]],axis=0)
        axes[0,1].plot(range(4),100*(local-values)/local,marker='o',color=COLORS[i],label=arm)
    axes[0,1].axhline(0,color='black',lw=1);axes[0,1].axhline(-5,color='#AA3377',ls='--')
    axes[0,1].set_xticks(range(4),['col148','col41','col176','col135']);axes[0,1].legend(fontsize=8)
    axes[0,1].set_ylabel('Seed-score mean gain over independent LOCAL (%)')
    axes[0,1].set_title('All four clients; dashed = -5% harm criterion',loc='left')
    for i,arm in enumerate(farms):
        vals=[read(RESULTS/'fits'/f'{arm}_{s}'/'FIT.json') for s in [92201,92202]]
        axes[1,0].bar(i,np.mean([v['uploaded_bytes']+v['downloaded_bytes'] for v in vals])/2**20,color=COLORS[i])
        axes[1,1].bar(i,np.mean([v['seconds'] for v in vals]),color=COLORS[i])
    for a in axes[1]:a.set_xticks(range(5),farms,rotation=20,ha='right')
    axes[1,0].set_ylabel('Round communication MiB / workflow');axes[1,0].set_title('B only; common backbone/A provisioning excluded',loc='left',fontsize=10)
    axes[1,1].set_ylabel('Training + validation seconds / workflow');axes[1,1].set_title('One GPU; 16 rounds x 4 clients x 4 updates',loc='left')
    save(fig,'F_results')


if __name__=='__main__':main()
