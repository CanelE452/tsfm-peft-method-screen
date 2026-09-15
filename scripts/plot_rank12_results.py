"""Plot completed, independently verified R2 scores; no inference or fitting."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/channel_sharing_screen_v1_20260915'
def rows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))
def main():
    assert json.loads((OUT/'status.json').read_text())['status']=='COMPLETE'
    assert json.loads((OUT/'independent_verification.json').read_text())['E_scope']=='MEASURED'
    metrics=[r for r in rows('metrics.csv') if r['role']=='selected'];counts={r['arm']:int(r['total_trainable']) for r in rows('PARAMETER_BUDGET.csv')};arms=list(counts);datasets=['electricity','traffic'];colors=dict(zip(arms,plt.get_cmap('tab10').colors))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,d in zip(axes,datasets):
        for a in arms:
            vals=[float(r['mse']) for r in metrics if r['dataset']==d and r['arm']==a];assert len(vals)==2
            ax.scatter([counts[a]/1e6]*2,vals,color=colors[a],alpha=.4,s=20);ax.scatter(counts[a]/1e6,np.mean(vals),color=colors[a],marker='D',label=a)
        ax.set(title=d+' / reused development E',xlabel='Total trainable parameters (millions)',ylabel='Selected standardized MSE');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(OUT/'parameters_vs_error.png',dpi=170);plt.close(fig)
    comp=[r for r in rows('comparisons.csv') if r['seed']=='mean'];fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,d in zip(axes,datasets):
        r=[r for r in comp if r['dataset']==d];x=np.arange(len(r));g=[float(v['gain_percent']) for v in r];lo=[float(v['CI95_lower']) for v in r];hi=[float(v['CI95_upper']) for v in r]
        ax.hlines(x,lo,hi,color='gray');ax.scatter(g,x,c=['tab:blue' if v>0 else 'tab:orange' for v in g]);ax.axvline(0,color='black',linewidth=.7);ax.set_yticks(x,[v['baseline'] for v in r]);ax.set(title=d,xlabel='BASIS_BUDGET MSE reduction (%)\nPaired time-block bootstrap 95% interval');ax.grid(axis='x',alpha=.2)
    fig.savefig(OUT/'basis_gains.png',dpi=170);plt.close(fig)
    tr=rows('trajectories.csv');fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,d in enumerate(datasets):
        for j,z in enumerate(['41000','41001']):
            ax=axes[i,j]
            for a in arms:
                r=[r for r in tr if r['dataset']==d and r['seed']==z and r['arm']==a];ax.plot([int(v['epoch']) for v in r],[float(v['mse']) for v in r],'.-',linewidth=1,color=colors[a],label=a)
            ax.set(title=d+' / seed '+z,xlabel='Epoch (0 = INIT)',ylabel='Validation standardized MSE');ax.legend(fontsize=6,ncol=2);ax.grid(alpha=.2)
    fig.savefig(OUT/'validation_curves.png',dpi=170);plt.close(fig);print('Verified R2 figures generated; no GPU/model calls')
if __name__=='__main__':main()
