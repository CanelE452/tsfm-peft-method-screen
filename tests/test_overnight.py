import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from tsfm_peft_screen.backbone import QUANTILES
from tsfm_peft_screen.lora import LowRank
from tsfm_peft_screen.metrics import independent
from tsfm_peft_screen.overnight.methods import (
    raw_loss, context_features, ConditionedLowRank, mixture_quantiles, teacher_weights,
    verdict_from_rows, TOPICS,
)

def test_training_loss_matches_independent_report():
    rng=np.random.default_rng(123)
    p=rng.normal(size=(2,4,21,48)).astype(np.float32)
    y=rng.normal(size=(2,4,48)).astype(np.float32)
    s=np.array([1.,2.,3.,4.],dtype=np.float32)
    t=torch.tensor(p.reshape(8,21,48),requires_grad=True)
    loss=raw_loss(t,torch.tensor(y.reshape(8,48)),torch.tensor(np.tile(s,2)))
    assert float(loss.detach()) == pytest.approx(independent(p,y,s),rel=2e-7)
    loss.backward()
    assert torch.isfinite(t.grad).all() and t.grad.abs().sum()>0

@pytest.mark.parametrize('group',[False,True])
def test_conditioned_rank_initial_identity_and_live_gate(group):
    torch.manual_seed(1)
    base=torch.nn.Linear(16,16,bias=False)
    base.requires_grad_(False)
    old=LowRank(base,group_attention=group)
    old.lora_B.data.normal_(std=.05)
    new=ConditionedLowRank(copy.deepcopy(old))
    x=torch.randn(5,3,16) if group else torch.randn(3,5,16)
    new.features=torch.randn(3,6)
    assert torch.equal(old(x),new(x))
    new(x).square().mean().backward()
    assert new.condition.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in new.base.parameters())
    new.enabled=False
    assert torch.equal(new(x),new.base(x))

def test_features_finite_invariant_and_channel_local():
    x=torch.randn(4,1024)
    for kind in ['moment_gate','drift_gate']:
        a=context_features(x,kind)
        assert a.shape==(4,6) and torch.isfinite(a).all()
        assert torch.allclose(a,context_features(3*x+10,kind),atol=2e-5)
        assert torch.isfinite(context_features(torch.ones_like(x),kind)).all()
        changed=x.clone();changed[1]=100
        assert torch.equal(a[0],context_features(changed,kind)[0])

def test_mixture_is_distribution_mixture_not_quantile_average():
    # Mixture of two point masses has separated quantiles, not all fives.
    components=np.zeros((2,1,1,21,48),dtype=np.float32)
    components[1]=10
    q=mixture_quantiles(components)
    assert q.shape==(1,1,21,48)
    assert np.all(q[:,:,2]==0) and np.all(q[:,:,18]==10)
    assert not np.all(q==5)
    assert np.all(np.diff(q,axis=2)>=0)
    same=np.ones_like(components)*7
    assert np.all(mixture_quantiles(same)==7)

def test_teacher_weights_use_disagreement_not_targets():
    a=np.zeros((3,2,4,21,48),dtype=np.float32)
    a[0,:,:,:,-16:]=10
    w=teacher_weights(a,np.ones(4))
    assert np.allclose(w.mean(-1),1)
    assert (w[:,:,-16:]<w[:,:,:16]).all()
    assert np.all(teacher_weights(np.zeros_like(a),np.ones(4))==1)

def test_gate_requires_both_domains_each_seed_and_nonzero_step():
    cfg={'datasets':['a','b'],'seeds':[1,2],'gate':{'min_gain':.01,'max_seed_ratio':1.01}}
    rows=[];sels=[]
    for d in cfg['datasets']:
        rows.append(dict(dataset=d,seed=None,arm='F0',metrics={'scaled_2pinball':1.1}))
        for seed in cfg['seeds']:
            for arm in TOPICS['anchor']:
                value=.98 if arm=='prediction_anchor' else 1.
                rows.append(dict(dataset=d,seed=seed,arm=arm,metrics={'scaled_2pinball':value}))
                sels.append(dict(dataset=d,seed=seed,arm=arm,step=150))
    assert verdict_from_rows(rows,sels,'anchor',cfg)['verdict']=='PILOT_PASS'
    rows[-1]['metrics']['scaled_2pinball']=1.04
    assert verdict_from_rows(rows,sels,'anchor',cfg)['verdict']=='PILOT_STOP'
    rows[-1]['metrics']['scaled_2pinball']=.98
    sels[-1]['step']=0
    assert verdict_from_rows(rows,sels,'anchor',cfg)['verdict']=='PILOT_STOP'

def test_split_and_attempt_budget_contract():
    cfg=json.loads((Path(__file__).parents[1]/'configs/overnight_20260913.json').read_text())
    origin=lambda s:list(range(cfg[s+'_origins'][0],cfg[s+'_origins'][1]+1,cfg[s+'_origins'][2]))
    assert max(origin('train'))+48<=min(origin('validation'))
    assert max(origin('validation'))+48<=min(origin('evaluation'))
    assert len(origin('validation'))==16 and len(origin('evaluation'))==32
    assert min(origin('train'))>=max(cfg['teacher_contexts'])
    fits=sum(len(a)*len(cfg['datasets'])*len(cfg['seeds'])*len(cfg['learning_rates']) for a in TOPICS.values())
    assert fits==cfg['max_total_fit_attempts']==96
    assert fits*cfg['checkpoints'][-1]==cfg['max_training_updates']==86400
