"""CPU-only retrospective audit. Does not fit models or change old results."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from tsfm_peft_screen.reproducibility import ROOT, sha, write_json
from tsfm_peft_screen.backbone import QUANTILES
OUT=ROOT/'research/peft_rethink_2026_09_13'
CACHE=ROOT/'.cache/block_shape_pilot'
SOURCE=ROOT/'results/block_shape_pilot'
OUT.mkdir(exist_ok=True)
ev=json.loads((SOURCE/'evaluation.json').read_text())
seal=json.loads((SOURCE/'selection_seal.json').read_text())
traj=json.loads((SOURCE/'trajectories.json').read_text())
audit={'scope':'Retrospective CPU calculations on existing predictions; zero new fits or GPU work.',
       'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [SOURCE/'evaluation.json',SOURCE/'selection_seal.json',SOURCE/'trajectories.json']},
       'datasets':{},'native_vs_primary_V_selection':[]}
q=np.array(QUANTILES)[None,:,None]
comparison=[]
for name in ['ettm2','electricity']:
    rows=[r for r in ev if r['dataset']==name]
    f0=next(r['metrics']['scaled_2pinball'] for r in rows if r['arm']=='F0')
    means={a:float(np.mean([r['metrics']['scaled_2pinball'] for r in rows if r['arm']==a])) for a in ['lora','split','center','anchor','pooled','block']}
    audit['datasets'][name]={'F0':f0,'mean_primary':means,'improvement_vs_F0_percent':{a:100*(f0-v)/f0 for a,v in means.items()},'block_vs_pooled_percent':100*(means['block']/means['pooled']-1)}
    for arm,value in means.items():
        comparison.append(dict(dataset=name,arm=arm,seed_mean_primary=value,F0_primary=f0,gain_vs_F0_percent=100*(f0-value)/f0,source='results/block_shape_pilot/evaluation.json'))
    path=CACHE/f'{name}_V_features.pt'
    audit['input_hashes'][str(path.relative_to(ROOT))]=sha(path)
    f=torch.load(path,weights_only=True)
    loc=f['loc'].numpy()[:,None];scale=f['scale'].numpy()[:,None];target=f['target'].numpy()
    for seed in [30000,30001]:
        for arm in means:
            rs=[r.copy() for r in traj if r['dataset']==name and r['seed']==seed and r['arm']==arm]
            for r in rs:
                path=CACHE/(r['prediction_tag']+'.npz')
                assert sha(path)==r['prediction_sha256']
                with np.load(path) as z:p=z['prediction'].reshape(-1,21,48)
                pred=np.arcsinh((np.sort(p,axis=1)-loc)/scale)
                e=np.arcsinh(target)[:,None]-pred
                r['native']=float((2*np.maximum(q*e,(q-1)*e)).mean(-1).sum(-1).mean())
            raw=min(rs,key=lambda r:(r['metrics']['scaled_2pinball'],r['step'],r['recipe']))
            native=min(rs,key=lambda r:(r['native'],r['step'],r['recipe']))
            audit['native_vs_primary_V_selection'].append(dict(dataset=name,seed=seed,arm=arm,raw_choice=[raw['recipe'],raw['step']],native_choice=[native['recipe'],native['step']],different=(raw['recipe'],raw['step'])!=(native['recipe'],native['step']),raw_V_penalty_if_native_selected_percent=100*(native['metrics']['scaled_2pinball']/raw['metrics']['scaled_2pinball']-1)))
audit['selected_steps']={str(s):sum(r['step']==s for r in seal['selections']) for s in [0,30,60,120]}
audit['native_selection_mismatch_count']=sum(r['different'] for r in audit['native_vs_primary_V_selection'])
audit['native_objective_caveat']='Reconstructed from sorted raw quantiles and cached context normalization; not unsorted LoRA training loss. Cannot attribute training failure causally to objective mismatch.'
write_json(OUT/'audit.json',audit)
for filename,rows in [('comparison.csv',comparison),('validation_objective_choices.csv',audit['native_vs_primary_V_selection'])]:
    with open(OUT/filename,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
construct={'scope':'Exact toy counterexamples, not empirical Chronos performance or novelty proof.','correlation_counterexample':{}}
for name,x in [('same_direction',np.array([[1.,1.],[-1.,-1.]])),('opposite_direction',np.array([[1.,-1.],[-1.,1.]]))]:
    construct['correlation_counterexample'][name]=dict(mean=x.mean(0).tolist(),marginal_variance=x.var(0).tolist(),cross_covariance=(x.T@x/len(x)).tolist(),nonlinear_response_samples=np.square(x.sum(1)).tolist(),response_mean=float(np.square(x.sum(1)).mean()))
levels=np.array([.1,.5,.9])
construct['quantile_mixture_counterexample']=dict(components='Uniform[0,1] and Uniform[10,11], equal weights',quantile_levels=levels.tolist(),average_component_quantiles=(5+levels).tolist(),mixture_quantiles_generalized_inverse=np.where(levels<=.5,2*levels,9+2*levels).tolist())
assert construct['correlation_counterexample']['same_direction']['response_mean']==4
assert construct['correlation_counterexample']['opposite_direction']['response_mean']==0
write_json(OUT/'construct_checks.json',construct)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,name in zip(axes,['ettm2','electricity']):
    rows=[r for r in comparison if r['dataset']==name]
    ax.bar([r['arm'] for r in rows],[r['gain_vs_F0_percent'] for r in rows],color=['#28658c' if r['arm']=='lora' else '#9aa2a8' for r in rows])
    ax.axhline(0,color='black',linewidth=.6)
    ax.set_title(name);ax.set_ylabel('Loss reduction vs F0 (%)')
    ax.tick_params(axis='x',rotation=35)
fig.tight_layout();fig.savefig(OUT/'observed_gains.png',dpi=160);plt.close(fig)
print('RETHINK AUDIT COMPLETE: 24 selections; native/raw mismatch',audit['native_selection_mismatch_count'],'; zero new fits')
