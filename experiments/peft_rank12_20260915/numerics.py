"""R1 user-fixed RMS contract; historical policy remains untouched."""
import numpy as np
import torch

def diff(a,b):
    a=a.detach().cpu().double().reshape(-1);b=b.detach().cpu().double().reshape(-1);assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    e=a-b;norm=float(a.norm());return dict(rms_error=float(e.square().mean().sqrt()),rms_reference=float(a.square().mean().sqrt()),max_absolute=float(e.abs().max()),reference_norm=norm,relative_l2=float(e.norm()/max(norm,1e-12)))
def parts(a):
    out={};offset=0
    for name,n in a['adam_layout']:
        key=name.rsplit('.',1)[-1];out.setdefault(key,[]).append(a['adam'][offset:offset+n]);offset+=n
    return {k:torch.cat(v) for k,v in out.items()}
def check_parity(ref,act,precision,micro=False,scale=None):
    assert ref['tensor_layout']==act['tensor_layout'] and ref['adam_layout']==act['adam_layout']
    sc=torch.as_tensor(np.tile(scale,ref['raw'].shape[0]//len(scale)),dtype=torch.float64)[:,None,None]
    keys=['z','raw','loss','raw_gradient','clipped_gradient','update'];m={k:diff(ref[k],act[k]) for k in keys};m['raw_scaled']=diff(ref['raw'].double()/sc,act['raw'].double()/sc)
    aa,bb=parts(ref),parts(act)
    for k in ['exp_avg','exp_avg_sq']:m[k]=diff(aa[k],bb[k])
    def rms(k,atol,rtol):return m[k]['rms_error']<=atol+rtol*m[k]['rms_reference']
    if not micro:
        gates={k:rms(k,1e-7,1e-5) for k in ['z','raw_scaled','loss','raw_gradient','clipped_gradient','update','exp_avg','exp_avg_sq']}
        gates.update({k+'_max':m[k]['max_absolute']<=1e-4 for k in ['z','raw_scaled']})
    elif precision=='fp32':
        gates={k:rms(k,1e-6,1e-4) for k in ['z','raw_scaled','loss']};gates.update({k:rms(k,1e-8,1e-4) for k in ['raw_gradient','clipped_gradient','update']});gates.update({k:rms(k,1e-10,2e-4) for k in ['exp_avg','exp_avg_sq']});gates.update({k+'_max':m[k]['max_absolute']<=1e-4 for k in ['z','raw_scaled']})
    else:
        gates={k:m[k]['max_absolute']<=1e-6 if m[k]['reference_norm']<1e-12 else m[k]['relative_l2']<=1e-2 for k in ['z','raw_scaled','raw_gradient','update']};gates['raw_scaled_max']=m['raw_scaled']['max_absolute']<=.02
    gates['rng']=ref['rng']==act['rng'];per={}
    for k in ['raw_gradient','clipped_gradient','update']:
        offset=0;per[k]={}
        for name,n in ref['tensor_layout']:per[k][name]=diff(ref[k][offset:offset+n],act[k][offset:offset+n]);offset+=n
    for key in ['exp_avg','exp_avg_sq']:
        per[key]={};offset=0
        for name,n in ref['adam_layout']:
            if name.endswith('.'+key):per[key][name]=diff(ref['adam'][offset:offset+n],act['adam'][offset:offset+n])
            offset+=n
    ra,rb=ref['residual'].double(),act['residual'].double();flip=(ra>0)!=(rb>0)
    if 'probe_z' in ref:
        for key in ['probe_z','probe_scaled_raw']:
            m[key]=diff(ref[key],act[key])
            if micro and precision=='fp32':gates[key]=rms(key,1e-6,1e-4) and m[key]['max_absolute']<=1e-4
    legacy={k:m[k]['relative_l2']<=1e-5 and m[k]['max_absolute']<=1e-5 for k in ['z','raw','raw_gradient','update']}
    return dict(passed=all(gates.values()),precision=precision,microbatch_comparison=micro,metrics=m,gates=gates,per_tensor=per,legacy_fp32_1e5_pass=all(legacy.values()) if precision=='fp32' else None,pinball_sign_flips=int(flip.sum()),pinball_elements=ra.numel(),pinball_near_zero_1e5=int((ra.abs()<1e-5).sum()))
