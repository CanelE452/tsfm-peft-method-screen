import numpy as np
from common import CONFIG
from data import Data, ALL_IDS, GROUP_IDS, mask_for, interpolate_context


def dummy():
    values=np.arange(26304,dtype=np.float32)[:,None]+np.arange(20,dtype=np.float32)[None,:]
    sigmas={g:{sid:1. for sid in ids} for g,ids in GROUP_IDS.items()}
    return Data(values,list(ALL_IDS),{sid:j for j,sid in enumerate(ALL_IDS)},CONFIG,sigmas)


def test_split_supports_and_nonoverlap():
    d=dummy()
    assert len(set(ALL_IDS))==20
    for group in ['dev','eval']:
        a,b=d.origins(group,'adapt_a',6),d.origins(group,'adapt_b',6)
        assert len(a)==len(b)==29
        assert a[-1]+24<=b[0]-192
        assert len(d.origins(group,'cal',24))==8
        part='validation' if group=='dev' else 'test'
        assert len(d.origins(group,part,24))==14
    assert CONFIG['data']['ranges']['dev_validation'][1]<=CONFIG['data']['ranges']['eval_test'][0]


def test_mask_counts_alignment_and_repeat_determinism():
    for kind in ['IID48','BLOCK48','ALIGNED48_LAST']:
        m=mask_for('169',15000,'P0_H2',kind)
        assert m.sum()==144
        np.testing.assert_array_equal(m,mask_for('169',15000,'P0_H2',kind))
    aligned=mask_for('169',15000,'P0_H2','ALIGNED48_LAST').reshape(12,16).sum(1)
    assert set(aligned)=={0,16}


def test_hidden_context_values_and_future_never_change_interpolation():
    d=dummy();pairs=[('169',15000)]
    x,y,s,m=d.batch(pairs,'dev',condition='BLOCK48',role='TEST_INVARIANCE',interpolate=True)
    col=d.index['169'];idx=np.arange(15000-192,15000)[~m[0]]
    d.values[idx,col]=-1e8
    d.values[15000:,col]=9e8
    other,changed,_,_=d.batch(pairs,'dev',condition='BLOCK48',role='TEST_INVARIANCE',interpolate=True)
    np.testing.assert_array_equal(x,other)
    assert not np.array_equal(y,changed)


def test_interpolation_uses_observed_edges_and_interior_only():
    x=np.array([999,2,999,6,999],dtype=np.float32)
    observed=np.array([False,True,False,True,False])
    np.testing.assert_array_equal(interpolate_context(x,observed),[2,2,4,6,6])
