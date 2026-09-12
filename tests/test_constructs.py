import numpy as np
import pytest
import torch
from torch import nn
from tsfm_peft_screen.candidates.freshness import corrupt,token_state
from tsfm_peft_screen.candidates.dualclock import event_features,EventAdapter
from tsfm_peft_screen.candidates.patchphase import patch_phase,unpatch,PhaseAdapter
from tsfm_peft_screen.candidates.fr_lora import aligned,regularizer,objective
from tsfm_peft_screen.candidates.maturity import matured_mask,preservation
from tsfm_peft_screen.candidates.censor import cdf
from tsfm_peft_screen.data import Panel
from tsfm_peft_screen.lora import LowRank

def test_freshness_causality_and_same_information():
    x=np.arange(672,dtype=float).reshape(2,336)
    for kind in ['clean','block6','block12','block24','refresh2','refresh4','refresh8','stale']:
        a,r=corrupt(x,kind);assert a.shape==x.shape and r.shape==(2,336,3)
        assert np.all(r[:,:,1]>=0) and np.all(r[:,:,1]<=1)
        valid=r[:,:,0].astype(bool);assert np.array_equal(a[valid],x[valid])
        t=token_state(torch.tensor(r));assert t.shape==(2,25,3)
    a,r=corrupt(x,'block6');assert np.isnan(a[:,-6:]).all() and np.all(r[:,-1,1]==6/336)

def test_gate_is_active_and_distinct_from_additive_feature():
    base=nn.Linear(12,16);base.requires_grad_(False)
    gate=LowRank(base,'freshness');x=torch.randn(2,25,12);gate.state=torch.randn(2,25,3)
    with torch.no_grad():gate.lora_B.fill_(.1);gate.condition.weight.fill_(.2)
    gate(x).sum().backward();assert gate.condition.weight.grad.abs().sum()>0

def test_event_context_only_and_zero_handling():
    x=torch.zeros(3,336);x[1,10]=2;x[1,30]=4;x[2,335]=3
    seq,lens,summary=event_features(x);assert lens.tolist()==[1,2,1]
    assert seq[1,0,0]==10/336 and seq[1,1,1]==20/336
    for full in [False,True]:
        a=EventAdapter(16,full);h=torch.randn(3,3,16);assert torch.equal(a(h,(seq,lens,summary)),h)

@pytest.mark.parametrize('phase',[0,4,8,12])
def test_phase_roundtrip_values_masks_timestamps(phase):
    x=torch.arange(336).double()[None].repeat(3,1);x[1,12:18]=float('nan')
    p=patch_phase(x,phase);out=unpatch(p)
    torch.testing.assert_close(out,x,rtol=0,atol=0,equal_nan=True)
    masks=torch.isfinite(x).double();assert torch.equal(unpatch(patch_phase(masks,phase)),masks)
    times=torch.arange(-336,0)[None].double();assert torch.equal(unpatch(patch_phase(times,phase)),times)
    assert p[0].numel()>=x.numel()

def test_fr_constant_correction_and_alignment():
    a,b=aligned(57);assert list(range(57,105))[a]==list(range(81,129))[b]
    base=torch.randn(8,21,48);raw=base+2;scale=torch.ones(8)
    assert regularizer('FR_LORA',raw,base,scale)<1e-12
    assert regularizer('F0_ANCHOR_LORA',raw,base,scale)>0

def test_lambda_zero_loss_grad_first_update_identical():
    weights=[]
    for usefr in [False,True]:
        torch.manual_seed(42);p=nn.Parameter(torch.randn(8,21,48));opt=torch.optim.AdamW([p],lr=1e-4,weight_decay=0)
        task=(p-3).square().mean();reg=regularizer('FR_LORA',p,torch.zeros_like(p),torch.ones(8))
        loss=objective(task,reg,0) if usefr else task;loss.backward();grad=p.grad.clone();opt.step();weights.append((loss.detach(),grad,p.detach()))
    for x,y in zip(*weights):assert torch.equal(x,y)

def test_maturity_masks_and_nonzero_second_step_gradient():
    mask=matured_mask(100,124);assert mask.sum()==24 and not mask[24:].any()
    assert matured_mask(100,148).all()
    pre=torch.zeros(2,21,48);p=nn.Parameter(pre.clone());assert preservation(p,pre,mask,torch.ones(2))==0
    with torch.no_grad():p.add_(.1)
    preservation(p,pre,mask,torch.ones(2)).backward();assert p.grad[...,:24].abs().sum()==0 and p.grad[...,24:].abs().sum()>0

def test_monotone_cdf_knots_tails_and_gradients():
    from tsfm_peft_screen.backbone import QUANTILES
    q=torch.tensor(QUANTILES);p=q[None,:,None].expand(2,-1,48).clone().requires_grad_()
    y=torch.linspace(-.1,1.1,48).expand(2,-1);f=cdf(p,y)
    assert torch.all(torch.diff(f,dim=-1)>=0);assert torch.all((f>=0)&(f<=1))
    assert torch.allclose(cdf(p,torch.ones(2,48)*.5),torch.ones(2,48)*.5)
    f.sum().backward();assert torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    duplicate=torch.zeros(2,21,48);assert torch.isfinite(cdf(duplicate,y)).all()

def test_data_e_unavailable_before_seal(tmp_path,monkeypatch):
    import json
    import tsfm_peft_screen.data as data
    monkeypatch.setattr(data,'ROOT',tmp_path)
    d=tmp_path/'data/processed/jena';d.mkdir(parents=True)
    (d/'manifest.json').write_text(json.dumps({'origins':{'train':[336],'evaluation':[500]}}))
    np.savez(d/'fit.npz',values=np.ones((500,4)),scale=np.ones(4),rms_scale=np.ones(4),caps=np.ones(4),zero_fraction=np.zeros(4),channels=np.array(['a','b','c','d']))
    p=Panel('jena');x,y=p.window(336);assert x.shape==(4,336) and y.shape==(4,48)
    with pytest.raises(AssertionError):p.window(500)
