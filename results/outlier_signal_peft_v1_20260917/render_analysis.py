"""Post-verification figures and V-only diagnostic accounting; no learning."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/outlier_signal_peft_v1_20260917'
CACHE=ROOT/'.cache/outlier_signal_peft_v1_20260917'
assert json.loads((OUT/'verification.json').read_text()).get('status')=='VERIFIED'
scores=pd.read_csv(OUT/'scores_by_condition.csv')
arms=['A0','A1','A2','A3','A4','A5'];sources=['electricity','ettm1']
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,source in zip(axes,sources):
    s=scores[(scores.source==source)&scores.arm.isin(arms)&scores.seed.isin([81501,81502])].copy()
    s=s[s.condition.str.startswith(('POINT','BURST'))]
    s['amplitude']=s.condition.str.extract(r'(\d+)$').astype(int)
    for arm in arms:
        values=s[s.arm==arm].groupby('amplitude').nmae.mean()
        ax.plot(values.index,values.values,marker='o',label=arm)
    ax.set(title=source,xlabel='Fault amplitude (multiples of r0)',ylabel='nMAE (lower is better)',xticks=[4,8,16])
    ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8)
fig.savefig(OUT/'fault_magnitude.png',dpi=180);fig.savefig(OUT/'fault_magnitude.pdf');plt.close(fig)
effects=pd.read_csv(OUT/'paired_effects.csv');panels=['FAULT','REFERENCE','SHIFT','SHIFT_POINT','HISTORY_SUBSET']
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,source in zip(axes,sources):
    f=effects[(effects.source==source)&effects.new.eq('A5')&effects.baseline.eq('A4')].set_index('panel').loc[panels]
    for i,row in enumerate(f.itertuples()):
        ax.hlines(i,row.ci_low_pct,row.ci_high_pct,color='tab:blue');ax.plot(row.gain_pct,i,'o',color='tab:blue')
    ax.axvline(0,color='black',linewidth=.8);ax.set(title=source,yticks=range(len(panels)),yticklabels=panels,xlabel='A5 gain over A4 (%) with paired block 95% interval')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
fig.savefig(OUT/'residual_added_value.png',dpi=180);fig.savefig(OUT/'residual_added_value.pdf');plt.close(fig)
manifest={};diagnostics=[];normal_rows=[]
states=['REFERENCE','POINT4','POINT8','POINT16','BURST4','BURST8','BURST16','SHIFT4','SHIFT8','SHIFT_POINT']
diag=json.loads((OUT/'A5_validation_diagnostics.json').read_text())
for row in json.loads((OUT/'MODEL_SELECTION.json').read_text()):
    source,arm,seed=row['source'],row['arm'],row['seed'];key=f'{source}_{arm}_{seed}'
    path=CACHE/'predictions'/f'{key}_V.npy';p=np.load(path,mmap_mode='r');f=CACHE/'conditions'/source
    y=np.load(f/'V_SELECT_y.npy');s=np.load(f/'V_SELECT_sigma.npy')
    errors=(np.abs(p[:,4].astype(float)-y)/s[:,None]).mean(1).reshape(10,2,64,4).mean((1,2,3))
    for condition,value in zip(states,errors):normal_rows.append(dict(source=source,arm=arm,seed=seed,condition=condition,nmae=float(value)))
    if arm=='A5':
        for mode in ['zero','permute']:
            for condition,normal,altered in zip(states,errors,diag[key][mode]):
                diagnostics.append(dict(source=source,seed=seed,condition=condition,mode=mode,normal_nmae=float(normal),altered_nmae=altered,altered_error_change_pct=100*(altered/normal-1)))
    manifest[key]=dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint=row['checkpoint'],checkpoint_sha256=row['sha256'])
pd.DataFrame(normal_rows).to_csv(OUT/'selected_validation_scores.csv',index=False)
pd.DataFrame(diagnostics).to_csv(OUT/'A5_validation_diagnostic_effects.csv',index=False)
(OUT/'validation_predictions_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Post-verification figures and V-only diagnostic baseline comparison complete')
