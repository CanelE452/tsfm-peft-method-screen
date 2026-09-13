import json
from pathlib import Path
import numpy as np
import torch
from tsfm_peft_screen.calibration_anchor.method import calibration_weights,shuffled_weights,weighted_pinball,TOPICS,PROPOSED
from tsfm_peft_screen.overnight.methods import raw_loss

def test_weights_only_use_selected_training_origins_and_keep_budget():
    rng=np.random.default_rng(5)
    p=np.sort(rng.normal(size=(9,4,21,48)),axis=2)
    y=rng.normal(size=(9,4,48))
    w,receipt=calibration_weights(p,y)
    assert w.shape==(4,21,48) and np.isfinite(w).all() and (w>0).all()
    assert np.allclose(w.mean((1,2)),1)
    assert receipt['origins_used']==5
    yy=y.copy();yy[1::2]=999
    assert np.array_equal(w,calibration_weights(p,yy)[0])
    assert np.allclose(w,calibration_weights(p*3+5,y*3+5)[0])
    sh=shuffled_weights(w)
    assert np.array_equal(np.sort(w.reshape(4,-1),axis=1),np.sort(sh.reshape(4,-1),axis=1))
    assert not np.array_equal(w,sh)

def test_bad_calibration_gets_less_preservation_without_changing_quantile_order():
    p=np.tile(np.linspace(-2,2,21)[None,None,:,None],(10,1,1,48))
    y=np.zeros((10,1,48))
    w,_=calibration_weights(p,y)
    assert w[0,0,0] > w[0,9,0]
    assert w[0,20,0] > w[0,10,0]
    assert (w>0).all()

def test_uniform_weighted_control_equals_raw_loss_and_gradients():
    p=torch.randn(8,21,48,requires_grad=True)
    y=torch.randn(8,48);s=torch.arange(1,9).float()
    a=raw_loss(p,y,s);g=torch.autograd.grad(a,p,retain_graph=True)[0]
    b=weighted_pinball(p,y,s,torch.ones_like(p))
    h=torch.autograd.grad(b,p)[0]
    assert torch.equal(a,b) and torch.equal(g,h)

def test_followup_budget_and_new_e_targets():
    root=Path(__file__).parents[1]
    c=json.loads((root/'configs/calibration_anchor_20260914.json').read_text())
    p=json.loads((root/'configs/overnight_20260913.json').read_text())
    assert len(TOPICS['calibration'])==7
    assert PROPOSED['calibration']=='calibration_anchor'
    assert c['max_total_fit_attempts']==7*2*2*2==56
    assert c['max_training_updates']==56*900
    assert c['evaluation_origins'][0]>=p['evaluation_origins'][1]+p['horizon']
    assert len(range(c['evaluation_origins'][0],c['evaluation_origins'][1]+1,c['evaluation_origins'][2]))==8
    assert c['gate']==p['gate']
