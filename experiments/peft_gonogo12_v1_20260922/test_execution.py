import numpy as np
import pytest
import torch
import evaluation
import ledger
from test_data import dummy
from runner import coefficients,h1_states


def test_gate_estimator_and_permutation_preserve_each_client_mean():
    states={}
    for sid,a,b in [('a',1.,1.),('b',0.,2.)]:
        states[sid]={arm:{0:{'d':torch.zeros(8)},16:{'d':torch.full((8,),v/2)},32:{'d':torch.full((8,),v)}} for arm,v in [('BLOCK_A',a),('BLOCK_B',b)]}
    stats,tau=coefficients(states,2)
    assert tau==.5
    assert stats['a']['g']==1 and stats['b']['g']==pytest.approx(1/3)
    swapped=h1_states(states,2,'PERMUTED_G',tau)
    torch.testing.assert_close(swapped['a']['d'],torch.full((8,),1/3))
    torch.testing.assert_close(swapped['b']['d'],torch.ones(8))


def test_packet_weights_equal_clean_and_iid_and_test_needs_seal(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluation,'RESULTS',tmp_path)
    d=dummy();p=evaluation.packet(d,'h2','dev','cal')
    assert len(p['x'])==4*8*3
    clean=p['conditions']=='CLEAN';iid=p['conditions']=='IID48'
    assert p['weights'][clean].sum()==p['weights'][iid].sum()
    with pytest.raises(AssertionError):evaluation.packet(d,'h2','eval','test','BLOCK48')


def test_ledger_never_replays_reserved_or_completed_attempt(tmp_path,monkeypatch):
    monkeypatch.setattr(ledger,'RESULTS',tmp_path)
    monkeypatch.setattr(ledger,'event',lambda *a,**k:None)
    book=ledger.Ledger();book.start('test','h1','main',1);book.reserve('test')
    with pytest.raises(AssertionError):book.reserve('test')
    book.completed('test');book.finish('test')
    with pytest.raises(RuntimeError):book.start('test','h1','main',1)
    assert book.state['counts']['h1']['main_updates']==1
