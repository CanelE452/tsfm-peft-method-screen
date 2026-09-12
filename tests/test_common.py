import numpy as np
import pytest
import torch
from torch import nn
from tsfm_peft_screen.lora import LowRank
from tsfm_peft_screen.metrics import score,independent
from tsfm_peft_screen.selection import choose,seal,require_seal

def test_metric_independent_missing_channel_scaling():
    rng=np.random.default_rng(41);p=rng.normal(size=(7,3,21,48));y=rng.normal(size=(7,3,48));y[0,1,3:]=np.nan;s=np.array([.1,2,10])
    assert abs(score(p,y,s)['scaled_2pinball']-independent(p,y,s))<1e-12
    assert score(p*2,y*2,s*2)['scaled_2pinball']==score(p,y,s)['scaled_2pinball']

def test_lora_identity_and_gradients():
    base=nn.Linear(12,16);base.requires_grad_(False);m=LowRank(base);x=torch.randn(2,3,12)
    assert torch.equal(m(x),base(x))
    m(x).square().mean().backward();assert m.lora_B.grad.abs().sum()>0 and base.weight.grad is None

def test_immutable_selection(tmp_path):
    records=[dict(step=4,lr=.1,validation_loss=1),dict(step=0,lr=.2,validation_loss=1)]
    assert choose(records)['step']==0
    p=tmp_path/'selection.json';seal(p,records,{'a':1});require_seal(p,{'a':1})
    with pytest.raises(FileExistsError):seal(p,records,{'a':1})
    with pytest.raises(AssertionError):require_seal(p,{'a':2})
