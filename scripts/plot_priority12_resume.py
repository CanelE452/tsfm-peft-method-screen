"""Plot verified completed C results; no GPU inference or parameter updates."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/channel_basis_pilot_resume_20260915'
def rows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))
def main():
    state=json.loads((OUT/'status.json').read_text());assert state['status'] in ['COMPLETE','ONE_SOURCE_PILOT']
    verification=json.loads((OUT/'independent_verification.json').read_text());assert verification['forecasting_evaluation']=='MEASURED'
    metrics=[r for r in rows('metrics.csv') if r['role']=='selected'];comparisons=[r for r in rows('comparisons.csv') if r['seed']=='mean'];trajectory=rows('trajectories.csv');params={r['arm']:int(r['trainable']) for r in rows('parameter_counts.csv')};arms=list(params);datasets=sorted({r['dataset'] for r in metrics});colors=dict(zip(arms,plt.get_cmap('tab10').colors))
    fig,axes=plt.subplots(len(datasets),2,figsize=(12,4*len(datasets)),squeeze=False,layout='constrained')
    for i,d in enumerate(datasets):
        for j,seed in enumerate(['40000','40001']):
            ax=axes[i,j]
            for a in arms:
                r=sorted([r for r in trajectory if r['dataset']==d and r['seed']==seed and r['arm']==a],key=lambda v:int(v['step']))
                ax.plot([int(v['step']) for v in r],[float(v['mse']) for v in r],'o-',color=colors[a],label=a,linewidth=1.3,markersize=3)
            ax.set(title=f'{d} / seed {seed}',xlabel='Effective optimizer updates',ylabel='Validation standardized MSE');ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
    fig.savefig(OUT/'validation_trajectories.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,len(datasets),figsize=(6*len(datasets),4.5),squeeze=False,layout='constrained')
    for ax,d in zip(axes[0],datasets):
        for j,a in enumerate(arms):
            values=[float(r['mse']) for r in metrics if r['dataset']==d and r['arm']==a];ax.scatter([params[a]/1e6]*len(values),values,s=18,color=colors[a],alpha=.45)
            ax.scatter(params[a]/1e6,np.mean(values),marker='D',s=45,color=colors[a],label=a)
        ax.set(title=d+' / reused development E',xlabel='Total trainable parameters (millions)',ylabel='Selected standardized MSE');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(OUT/'parameters_vs_error.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,len(datasets),figsize=(6*len(datasets),4.5),squeeze=False,layout='constrained')
    for ax,d in zip(axes[0],datasets):
        r=[r for r in comparisons if r['dataset']==d];x=np.arange(len(r));g=np.array([float(v['gain_percent']) for v in r]);lo=np.array([float(v['CI95_lower']) for v in r]);hi=np.array([float(v['CI95_upper']) for v in r])
        ax.hlines(x,lo,hi,color='gray');ax.scatter(g,x,c=['tab:blue' if v>0 else 'tab:orange' for v in g]);ax.axvline(0,color='black',linewidth=.8)
        ax.set_yticks(x,[v['baseline'] for v in r]);ax.set(title=d,xlabel='BASIS4 MSE gain over baseline (%)\nPaired time-block bootstrap 95% interval');ax.grid(axis='x',alpha=.2)
    fig.savefig(OUT/'basis_gains.png',dpi=170);plt.close(fig)
    print('Plotted verified metrics only; no new GPU calls')
if __name__=='__main__':main()
