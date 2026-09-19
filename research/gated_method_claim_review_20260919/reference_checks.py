"""Check limits of existing MAG/adapter claims; CPU, no optimizer/model fit."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.c3_weakness_controls_20260918.model import ControlModel
from experiments.additive_b0_adapter_v1_20260917.model import ResidualAdapter
OUT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(2);torch.manual_seed(91961)
    x=torch.randn(4,512,dtype=torch.float64);x[:,480:]+=8
    sigma=torch.tensor([1.,1.2,1.4,1.6],dtype=torch.float64)
    gate=ControlModel.gate(SimpleNamespace(arm='MAG_ONLY'),x,sigma)
    xx=x.numpy();med=np.quantile(xx,.5,axis=-1,keepdims=True)
    scale=np.maximum(1.4826*np.quantile(np.abs(xx-med),.5,axis=-1,keepdims=True),.1*sigma.numpy()[:,None])
    reference=1-(np.abs((xx-med)/scale)>3).reshape(4,32,16).mean(-1)
    assert np.array_equal(reference,gate.numpy())
    transformed=ControlModel.gate(SimpleNamespace(arm='MAG_ONLY'),2*x+4,2*sigma)
    assert torch.equal(gate,transformed)
    # Existing adapter, manually specified nonzero weights; not a learned model.
    adapter=ResidualAdapter(91961).double()
    with torch.no_grad():adapter.up.weight.normal_(0,.1);adapter.up.bias.fill_(.05)
    h=torch.randn(4,32,512,dtype=torch.float64)
    h2=adapter(h,gate);cap=h.square().mean(-1).sqrt().quantile(.5,dim=-1,keepdim=True).clamp_min(1e-6)
    bound=gate[:,:,None]*cap[:,:,None]
    assert ((h2-h).abs()<=bound+1e-14).all()
    assert torch.equal(h2[gate==0],h[gate==0])
    # Local theta Jacobian identity tested with the SAME upstream cotangent.
    cotangent=torch.randn_like(h)
    ga=torch.autograd.grad((h2*cotangent).sum(),tuple(adapter.parameters()))
    ones=adapter(h,torch.ones_like(gate))
    gb=torch.autograd.grad((ones*(cotangent*gate[:,:,None])).sum(),tuple(adapter.parameters()))
    error=max(float((a-b).abs().max()) for a,b in zip(ga,gb));assert error<1e-11
    # A zero residual on one token is not a forecast invariance guarantee.
    z=torch.tensor([1.,2.]);g=torch.tensor([0.,1.]);d=torch.tensor([.5,.5])
    z2=z+g*d
    assert z2[0]==z[0]
    # Frozen readout mixes two tokens; no training takes place.
    f=lambda q:q[0]+q[1]
    assert f(z2)!=f(z)
    # Per-token scaling need not reduce the norm after summing signed gradients.
    contributions=np.array([1.,-1.]);weighted=contributions*np.array([1.,0.])
    assert abs(weighted.sum())>abs(contributions.sum())
    files=[Path(__file__),ROOT/'experiments/c3_weakness_controls_20260918/model.py',ROOT/'experiments/additive_b0_adapter_v1_20260917/model.py',ROOT/'experiments/outlier_signal_peft_v1_20260917/model.py']
    result=dict(status='CPU_REFERENCE_VERIFIED',optimizer_updates=0,real_forecast_inferences=0,new_candidate=False,
        numpy_MAG_gate_exact=True,positive_affine_gate_example_exact=True,coordinatewise_residual_bound=True,
        zero_gated_patch_unchanged=True,local_parameter_jacobian_max_abs_error=error,
        mixed_readout_base=float(f(z)),mixed_readout_adapted=float(f(z2)),
        summed_gradient_abs_ungated=float(abs(contributions.sum())),summed_gradient_abs_gated=float(abs(weighted.sum())),
        scope='Synthetic CPU checks of existing code and logical counterexamples; no Chronos forecast experiment, causal discovery or novel theorem.',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files})
    (OUT/'REFERENCE_CHECKS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
