"""Presentation-only figures from the completed score tables; no new evaluation."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *


def main():
    decisions=read(RESULTS/'DECISIONS.json')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    names={'h1':['SHARED','RIDGE','UNSHRUNK','FIXED_SHRINK','PERMUTED_G'],
           'h2':['F0_NATIVE','QV_LORA','CENTROID_BIAS','KEY_BIAS','GENERIC_BIAS']}
    for ax,c in zip(axes,['h1','h2']):
        y=np.arange(len(names[c]));comparisons=decisions[c]['comparisons']
        for j in range(2):ax.scatter([comparisons[a]['seed_effects'][j] for a in names[c]],y+(-.1 if j==0 else .1),label=f'Seed {j+1}')
        ax.axvline(0,color='black',lw=1);ax.set_yticks(y,names[c]);ax.invert_yaxis()
        ax.set_title(c.upper()+': mechanism and simple controls (zoom)')
        ax.set_xlabel('Candidate improvement over control (%)');ax.legend(loc='lower left');ax.grid(axis='x',alpha=.2)
    fig.savefig(RESULTS/'mechanism_effects.png',dpi=180);plt.close(fig)
    frame=pd.read_csv(RESULTS/'seed_scores.csv')
    subset=frame[(frame.candidate=='h2')&frame.selected]
    v=subset.groupby(['arm','condition']).scaled_pinball.mean().unstack('condition')
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    im=ax.imshow(v.to_numpy(),cmap='viridis',aspect='auto')
    labels={'ALIGNED48_LAST':'Aligned last 48','BLOCK48':'Block 48 (primary)','CLEAN':'Clean','IID48':'IID 48'}
    ax.set_xticks(range(len(v.columns)),[labels[c] for c in v.columns]);ax.set_yticks(range(len(v)),v.index)
    for i in range(len(v)):
        for j in range(len(v.columns)):
            ax.text(j,i,f'{v.iloc[i,j]:.4f}',ha='center',va='center',color='white' if v.iloc[i,j]<(v.max().max()+v.min().min())/2 else 'black')
    ax.set_title('H2 selected-mode score; mask families kept separate')
    fig.colorbar(im,ax=ax,label='Scaled twice-pinball (lower better)')
    fig.savefig(RESULTS/'mask_conditions.png',dpi=180);plt.close(fig)
    save(RESULTS/'PUBLICATION_NOTES.json',dict(status='COMPLETE',changes=['Report opening clarifies strong primary effects do not identify structural benefit',
        'No GO means no MECHANISM_SIGNAL_ONLY success label','Mechanism zoom added without removing full-scale effects','Mask axis labels enlarged'],
        score_tables_changed=False,decisions_changed=False,additional_forward=0,additional_optimizer_updates=0,
        replay='finalize.py creates numerical artifacts and draft prose; publication_figures.py improves plots. Published REPORT_KO.md contains reviewed editorial explanations.'))
    manifest=read(RESULTS/'MANIFEST.json');manifest['source_hashes']=source_hashes()
    manifest['results']={str(p.relative_to(RESULTS)):sha(p) for p in sorted(RESULTS.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'}
    save(RESULTS/'MANIFEST.json',manifest)


if __name__=='__main__':main()
