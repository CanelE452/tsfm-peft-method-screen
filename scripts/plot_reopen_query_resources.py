import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tsfm_peft_screen.reproducibility import ROOT
P=ROOT/'results/reopen_query_resource_20260914'
f=json.loads((P/'frontier.json').read_text());m=json.loads((P/'measurements.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
for ax,ds in zip(axes,['ettm2','electricity']):
    rr=[r for r in f['points'] if r['dataset']==ds]
    pp=ax.scatter([r['memory']/2**20 for r in rr],[r['time']*1000 for r in rr],c=[r['loss'] for r in rr],cmap='viridis',s=65,zorder=3)
    for r in rr:
        mm=next(a for a in m if a['dataset']==ds and a['arm']==r['arm'] and a['checkpoint_blocks']==r['cp'])
        times=np.array([a['seconds']*1000 for a in mm['measured']]);median=r['time']*1000
        ax.errorbar(r['memory']/2**20,median,yerr=[[median-times.min()],[times.max()-median]],fmt='none',color='gray',capsize=2,alpha=.6)
        label=('S'+str(r['cp'])) if r['arm']=='standard' else ('Q on' if r['cp'] else 'Q off') if r['arm']=='query' else r['arm'].title()
        ax.annotate(label,(r['memory']/2**20,median),xytext=(5,6),textcoords='offset points',fontsize=8)
    ax.set(title=ds,xlabel='Peak allocated GPU memory (MiB)',ylabel='Median optimizer step (ms)')
    ax.grid(alpha=.2);ax.margins(x=.13,y=.14)
    fig.colorbar(pp,ax=ax,label='Historical mean primary loss (lower is better)',shrink=.82)
fig.suptitle('Storage-only resource measurements; quality reused from historical equal-time runs\nS: Standard checkpointed blocks; Q: Query; bars: range of 3 steps',fontsize=11)
fig.savefig(P/'quality_memory_time.png',dpi=180);fig.savefig(P/'quality_memory_time.svg');plt.close(fig)

svg=P/'quality_memory_time.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
