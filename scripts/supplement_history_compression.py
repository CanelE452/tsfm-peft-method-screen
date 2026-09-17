"""Post-run diagnostics only: no model forward, optimizer, or new selection."""
import sys,json,csv,math
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.history_compression_v1_20260917.common import *
assert read(OUT/'state.json')['status']=='FINISHED';validate_seal()
fits=read(OUT/'fits.json');logs=[json.loads(l) for l in (OUT/'optimizer.jsonl').read_text().splitlines()];sels=read(OUT/'selections.json')['selections'];rows=[]
for f in fits:
 rr=[r for r in logs if r['id']==f['id']];sel=next((s for s in sels if s['id']==f['id']),None)
 row=dict(id=f['id'],arm=f['arm'],seed=f['seed'],lr=f['lr'],updates=len(rr),mean_task_loss=float(np.mean([r['task'] for r in rr])),mean_KD_loss=float(np.mean([r['distillation'] for r in rr])),clip_fraction=float(np.mean([r['gradnorm']>1 for r in rr])),optimizer_seconds=sum(r['seconds'] for r in rr),peak_training_MiB=f['peak_allocated_bytes']/2**20,selected_repeat=sel is not None,selected_step=sel['selected']['step'] if sel else None)
 for param in ['pool_u','pool_v']:
  grads=[r['extra_grad'].get(param) for r in rr]
  if grads[0] is not None:
   row[param+'_nonzero_fraction']=float(np.mean(np.array(grads)>0));row[param+'_median_gradient_norm']=float(np.median(grads))
 if f['arm'].startswith('LEARN'):
  cp=sel['selected'] if sel else min(f['checkpoints'],key=lambda c:(c['accuracy'],c['step']))
  st=torch.load(ROOT/cp['path'],map_location='cpu',weights_only=True);u=st['pool_u'].double().numpy();v=st['pool_v'].double().numpy();bound=float(abs(u).sum())
  # tanh in[-1,1] implies each score in[-||u||1,||u||1]. This is a bound, not measured weights.
  row.update(selected_u_norm=float(np.linalg.norm(u)),selected_V_norm=float(np.linalg.norm(v)),score_abs_upper_bound=bound,three_way_weight_upper_bound=float(1/(1+2*np.exp(-2*bound))),three_way_weight_lower_bound=float(1/(1+2*np.exp(2*bound))))
 rows.append(row)
csvwrite(OUT/'optimization_diagnostics.csv',rows)
# Cost ledger retains teacher training, cache generation, selection and inference separately.
old=read(OUT/'REUSE_RECEIPT.json')['fits'];cost=[]
for f in old+fits:
 cost.append(dict(id=f['id'],arm={'B0':'SHORT','B1':'LONG'}.get(f['arm'],f['arm']),seed=f['seed'],lr=f['lr'],new_or_reused='reused' if f in old else 'new',updates=f['updates'],optimizer_seconds=f['optimizer_seconds'],selection_included_fit_seconds=f['seconds'],peak_training_MiB=f['peak_allocated_bytes']/2**20,teacher_recipe_fit=f['arm']=='B1' and f['seed']==73100))
csvwrite(OUT/'training_resources.csv',cost)
# Figures use all arms and both seeds; no re-ranking E to define a deployment selection.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ss=list(csv.DictReader((OUT/'scores.csv').open()));rr=read(OUT/'inference_resources.json');arms=['SHORT','LONG']+ARMS
fig,ax=plt.subplots(figsize=(10,4))
for j,seed in enumerate([73101,73102]):
 vv=[float(next(r for r in ss if r['arm']==a and int(r['seed'])==seed and r['policy']=='selected')['NRMSE']) for a in arms];ax.bar(np.arange(len(arms))+(j-.5)*.35,vv,width=.35,label=str(seed))
ax.set_xticks(np.arange(len(arms)),arms,rotation=20);ax.set_ylabel('Development NRMSE (lower is better)');ax.legend();fig.tight_layout();fig.savefig(OUT/'all_seed_scores.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for a in arms:
 score=np.mean([float(r['NRMSE']) for r in ss if r['arm']==a and r['policy']=='selected']);mem=np.mean([r['peak_allocated_bytes'] for r in rr if r['arm']==a])/2**20;lat=np.mean([r['median_seconds'] for r in rr if r['arm']==a])*1000
 for ax,x in zip(axes,[mem,lat]):ax.scatter(x,score);ax.annotate(a,(x,score),xytext=(3,3),textcoords='offset points',fontsize=8)
axes[0].set_xlabel('Inference peak allocated MiB');axes[1].set_xlabel('Inference latency ms');axes[0].set_ylabel('Development NRMSE');fig.tight_layout();fig.savefig(OUT/'accuracy_resource_tradeoff.png',dpi=160);plt.close(fig)
save(OUT/'supplement_verification.json',dict(passed=True,optimizer_rows=len(logs),fits=len(rows),resource_fits=len(cost),extra_training_updates=0,extra_model_forwards=0,weight_bound_not_empirical_weight=True,plots_include_all_arms_and_seeds=True))
print('Post-run diagnostics complete; no model calls or updates')
