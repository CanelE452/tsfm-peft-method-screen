"""Independent completed-artifact audit; no training or new model forecasts."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];O=R/'results/censor_tail_controlled_v1'
def read(p):return json.loads(p.read_text())
def write(name,obj):(O/name).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
fits=read(O/'fits.json');state=read(O/'status.json');assert state['stage']=='REPORTED'
rows=list(csv.DictReader(open(O/'train_diagnostics.csv')));assert len(rows)==2160
counts={};detailed=[]
for f in fits:
    rr=[r for r in rows if int(r['seed'])==f['seed'] and r['arm']==f['arm']]
    assert [int(r['step']) for r in rr]==list(range(1,361))
    assert sum(r['nonzero_parameter_update']=='True' for r in rr)==f['nonzero_parameter_updates']==360
    assert sum(r['clipped']=='True' for r in rr)==f['clipping_count']
    assert all(np.isfinite(float(r[k])) for r in rr for k in ['loss','task','survival','gradient_norm'])
    assert all(abs(float(r['loss'])-(float(r['task'])+float(r['lambda_c'])*float(r['survival'])))<2e-6 for r in rr)
    counts[f"{f['seed']}_{f['arm']}"]=len(rr)
    if f['arm']=='TAIL':
        assert sum(int(r['censored']) for r in rr)==7957
        gg=[float(r['strong_shift_gradient_max']) for r in rr if r['strong_shift_gradient_max']]
        assert all(np.isfinite(g) and g<0 for g in gg)
        task=np.array([float(r['task']) for r in rr]);tail=np.array([float(r['survival']) for r in rr]);lc=float(rr[0]['lambda_c'])
        detailed.append(dict(seed=f['seed'],censored_occurrences=sum(int(r['censored']) for r in rr),
            upper_tail_occurrences=sum(int(r['right']) for r in rr),lower_tail_occurrences=sum(int(r['left']) for r in rr),
            strong_upper_occurrences=sum(int(r['strong_right']) for r in rr),strong_gradient_batches=len(gg),
            largest_common_shift_gradient=max(gg),floored_gap_fraction=sum(int(r['floored_gaps']) for r in rr)/sum(int(r['gaps']) for r in rr),
            maximum_proxy_knot_shift=max(float(r['max_knot_shift']) for r in rr),
            weighted_survival_mean=float(lc*tail.mean()),task_mean=float(task.mean()),
            median_weighted_survival_to_task=float(np.median(lc*tail/task)),
            clipping_fraction=f['clipping_count']/360))
for seed in [41000,41001]:
    checkpoints=[torch.load(R/f'.cache/censor_tail_controlled_v1/{seed}_{arm}/step_0.pt',weights_only=True,map_location='cpu') for arm in ['NAIVE','DROP','TAIL']]
    assert all(a.keys()==checkpoints[0].keys() for a in checkpoints)
    assert all(torch.equal(checkpoints[0][k],c[k]) for c in checkpoints[1:] for k in c)
    del checkpoints
res=read(O/'resources_run.json');assert res['model_forward']==5424 and res['model_backward']==2160
assert res['output_tensor_backward']==sum(d['strong_gradient_batches'] for d in detailed)
telemetry=read(O/'gpu_run.json');assert not any(not a['allowed'] for r in telemetry for a in r['apps'])
lam=read(O/'lambda.json');tasks=np.array(lam['tasks']);tails=np.array(lam['survival'])
assert np.mean(tasks)==lam['T0'] and np.mean(tails)==lam['C0']
assert float(np.clip(.1*tasks.mean()/tails.mean(),.001,100))==lam['lambda_c']
cal=dict(T0=lam['T0'],C0=lam['C0'],lambda_c=lam['lambda_c'],survival_batch_median=float(np.median(tails)),
         survival_batch_max=float(tails.max()),largest_batch_share=float(tails.max()/tails.sum()),clipped=lam['clipped'])
summary=dict(status='VERIFIED',fit_update_rows=counts,initial_tensors_equal_within_seed=True,all_updates_changed_parameters=True,
             main_model_forwards=5424,main_model_backwards=2160,output_only_backwards=res['output_tensor_backward'],
             lambda_recalculated=True,calibration=cal,TAIL=detailed,min_free_gpu_mib=min(r['free_mib'] for r in telemetry),
             disallowed_compute_samples=0,resources_total_seconds=read(O/'resources_preflight.json')['seconds']+res['seconds'])
write('completion_audit.json',summary)
print(json.dumps(summary,indent=2))
