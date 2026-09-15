"""User-fixed 2026-09-15 v2 policy. Historical v1 policy remains unchanged."""
import numpy as np
import torch

def difference(a,b):
    a=a.detach().cpu().double().reshape(-1);b=b.detach().cpu().double().reshape(-1)
    n=float(a.norm());return dict(max_absolute=float((a-b).abs().max()),relative_l2=float((a-b).norm()/max(n,1e-12)),reference_norm=n)

def relative_ok(d,tol):return d['max_absolute']<=1e-6 if d['reference_norm']<1e-12 else d['relative_l2']<=tol

def check_parity(reference,actual,precision,micro=False,scale=None):
    assert precision in ['fp32','bf16'] and scale is not None
    sc=torch.tensor(np.tile(scale,reference['raw'].shape[0]//len(scale)),dtype=torch.float64)[:,None,None]
    keys=['z','raw','loss','raw_gradient','clipped_gradient','update','adam']
    metrics={k:difference(reference[k],actual[k]) for k in keys}
    metrics['raw_scaled']=difference(reference['raw'].double()/sc,actual['raw'].double()/sc)
    if micro:
        outtol,gtol,abstol,lossabs=(1e-5,1e-4,1e-4,1e-7) if precision=='fp32' else (1e-2,1e-2,2e-2,1e-6)
        gates={k:relative_ok(metrics[k],outtol) for k in ['z','raw_scaled']}
        gates.update({k:relative_ok(metrics[k],gtol) for k in ['raw_gradient','clipped_gradient','update']})
        gates['scaled_absolute']=metrics['raw_scaled']['max_absolute']<=abstol
        gates['loss']=metrics['loss']['max_absolute']<=lossabs+outtol*abs(float(reference['loss'].double().item()))
    else:
        tol=1e-5 if precision=='fp32' else 1e-4
        gates={k:relative_ok(metrics[k],tol) for k in ['z','raw_scaled','loss','raw_gradient','clipped_gradient','update','adam']}
        gates['scaled_absolute']=metrics['raw_scaled']['max_absolute']<=tol
        gates['z_absolute']=metrics['z']['max_absolute']<=tol
    gates['rng']=reference['rng']==actual['rng']
    per_tensor={}
    if 'tensor_layout' in reference:
        for key in ['raw_gradient','clipped_gradient','update']:
            offset=0;per_tensor[key]={}
            for name,size in reference['tensor_layout']:
                per_tensor[key][name]=difference(reference[key][offset:offset+size],actual[key][offset:offset+size]);offset+=size
            assert offset==reference[key].numel()
    if 'adam_layout' in reference:
        offset=0;per_tensor['adam']={}
        for name,size in reference['adam_layout']:
            per_tensor['adam'][name]=difference(reference['adam'][offset:offset+size],actual['adam'][offset:offset+size]);offset+=size
        assert offset==reference['adam'].numel()
    kink=None
    if 'residual' in reference:
        a=reference['residual'].double();b=actual['residual'].double();flip=(a>0)!=(b>0)
        kink=dict(sign_flips=int(flip.sum()),total=a.numel(),reference_abs_max_at_flip=float(a[flip].abs().max()) if flip.any() else 0.,
            actual_abs_max_at_flip=float(b[flip].abs().max()) if flip.any() else 0.,loss_gradient_jump_L1_upper=float(flip.sum()*2/(a.shape[0]*a.shape[-1])),
            meaning='Pinball output-derivative jump before model Jacobian; not an attribution of parameter gradient difference')
    return dict(passed=all(gates.values()),precision=precision,microbatch_comparison=micro,metrics=metrics,gates=gates,rng_equal=gates['rng'],per_tensor=per_tensor,pinball_kink=kink)

def path_check(start,reference,actual,ref_raw,act_raw,scale,precision):
    ref=torch.cat([(reference[n]-start[n]).double().reshape(-1) for n in sorted(start)])
    act=torch.cat([(actual[n]-start[n]).double().reshape(-1) for n in sorted(start)])
    d=difference(ref,act);sc=torch.tensor(np.tile(scale,ref_raw.shape[0]//len(scale)),dtype=torch.float64)[:,None,None]
    out=difference(ref_raw.double()/sc,act_raw.double()/sc)
    tol,absolute=(1e-3,2e-4) if precision=='fp32' else (5e-2,5e-2)
    return dict(passed=relative_ok(d,tol) and out['max_absolute']<=absolute,total_delta=d,end_raw_scaled=out,
        per_tensor={n:difference(reference[n]-start[n],actual[n]-start[n]) for n in start})
