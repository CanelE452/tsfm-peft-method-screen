import numpy as np
import torch
from torch import nn
from tsfm_peft_screen.candidates.freshness_v2 import corrupt,RULES,AffineLowRank

def test_async_masks_and_causal_state():
    x=np.arange(1344,dtype=np.float32).reshape(4,336)
    for kind in RULES:
        a,r=corrupt(x,kind,1000);b,s=corrupt(x,kind,1000)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(r,s)
        mask=np.isfinite(a)
        assert np.unique(mask,axis=0).shape[0]==4
        np.testing.assert_array_equal(a[mask],x[mask])
        assert np.all(r[:,:,1]>=0) and np.all(r[:,:,1]<=1)
        altered=x.copy();altered[:,250:]=np.nan
        aa,rr=corrupt(altered,kind,1000)
        np.testing.assert_array_equal(rr[:,:250],r[:,:250])
    a,r=corrupt(x,'clean',1000)
    np.testing.assert_array_equal(a,x)
    np.testing.assert_array_equal(r,np.broadcast_to([1.,0.,1.],r.shape))

def test_centered_affine_reduces_exactly_to_standard_at_clean_and_has_gradients():
    torch.manual_seed(1);base=nn.Linear(12,16);base.requires_grad_(False)
    module=AffineLowRank(base);x=torch.randn(4,25,12)
    with torch.no_grad():module.lora_B.normal_();module.condition.weight.normal_()
    module.state=x.new_tensor([1,0,1]).expand(4,25,3)
    standard=base(x)+2*nn.functional.linear(nn.functional.linear(x,module.lora_A),module.lora_B)
    torch.testing.assert_close(module(x),standard,rtol=0,atol=0)
    module.state=torch.rand(4,25,3)
    mult,shift=module.modulation(module.state)
    assert (mult>=0).all() and (mult<=2).all() and shift.abs().sum()>0
    module(x).square().mean().backward()
    grad=module.condition.weight.grad
    assert torch.isfinite(grad).all() and grad[:8].abs().sum()>0 and grad[8:].abs().sum()>0

def test_zero_initialized_output_and_group_attention_layout():
    base=nn.Linear(12,16);base.requires_grad_(False)
    module=AffineLowRank(base,group_attention=True)
    module.state=torch.rand(4,25,3);x=torch.randn(25,4,12)
    torch.testing.assert_close(module(x),base(x),rtol=0,atol=0)
