"""Window intervention, missing targets and heldout-selection firewall tests."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('window_study_tests',ROOT/'scripts/run_anchor_window_study.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def test_window_budgets_are_nested_with_identical_endpoints_and_safe_splits():
    for source,c in M.CFG['data'].items():
        dense=M.train_origins(source,233);sparse=M.train_origins(source,32)
        assert len(set(sparse))==32 and set(sparse)<=set(dense)
        assert sparse[0]==dense[0] and sparse[-1]==dense[-1]
        assert dense[-1]+48<=c['validation'][0] and c['validation'][1]+48<=c['evaluation'][0]
        assert len(M.origins(c['evaluation']))==32
        assert M.train_origins(source,32)==sparse
    assert M.CFG['data']['ettm2_later']['train'][0]>26064
    assert M.CFG['data']['electricity_new']['channels']==[4,5,6,7]


def test_heldout_loader_refuses_before_reading_values_without_global_seal(tmp_path,monkeypatch):
    monkeypatch.setattr(M,'OUT',tmp_path)
    def forbidden(*args,**kwargs): raise AssertionError('Heldout values touched before barrier')
    monkeypatch.setattr(M.np,'load',forbidden)
    with pytest.raises(FileNotFoundError): M.load_heldout('beijing_32')


def test_validation_policy_is_fixed_from_v_including_plain_ties():
    selections=[]
    for ds in M.CFG['topics']['anchor']['datasets']:
        for seed in (34000,34001):
            for arm in ('native','native_anchor'):
                selections.append(dict(dataset=ds,seed=seed,arm=arm,
                    metrics={'scaled_2pinball':1 if arm=='native' or seed==34000 else .9}))
    policies=M.validation_policies(selections)
    assert len(policies)==12
    assert all(p['chosen_arm']==('native' if p['seed']==34000 else 'native_anchor') for p in policies)


def test_missing_native_targets_have_finite_gradients_and_independent_metric():
    from tsfm_peft_screen.reassessment import objective
    z=torch.zeros(8,21,48,requires_grad=True);p=z.sinh()
    y=torch.ones(8,48);y[0,:]=float('nan');y[2,::3]=float('nan')
    loss,reg=objective('anchor','native_anchor',z,p,y,torch.zeros(8,1),torch.ones(8,1),torch.ones(8),p.detach()+.1)
    (loss+reg).backward()
    assert torch.isfinite(loss+reg) and torch.isfinite(z.grad).all()
    yy=y.numpy().reshape(2,4,48);pp=p.detach().numpy().reshape(2,4,21,48)
    assert abs(M.score(pp,yy,np.ones(4))['scaled_2pinball']-M.independent(pp,yy,np.ones(4)))<1e-12


def test_block_statistic_matches_primary_with_missingness(tmp_path,monkeypatch):
    monkeypatch.setattr(M,'ROOT',tmp_path)
    rng=np.random.default_rng(51)
    y=rng.normal(size=(32,4,48));y[::2,0,:13]=np.nan
    p=rng.normal(size=(32,4,21,48));scale=np.array([1.,2.,3.,4.])
    np.savez(tmp_path/'sample.npz',prediction=p,target=y,scale=scale)
    numerator,count=M.channel_origin_loss({'prediction_file':'sample.npz'})
    assert abs((numerator.sum(0)/count.sum(0)).mean()-M.score(p,y,scale)['scaled_2pinball'])<1e-12
