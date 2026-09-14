import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('temporal_v1',Path(__file__).resolve().parents[1]/'scripts/run_temporal_transfer_diagnostic_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_gain_units_and_invalid_references():
    assert m.gain(1.,1.)==0
    assert m.gain(.9,1.)==pytest.approx(10.)
    assert m.gain(1.1,1.)==pytest.approx(-10.)
    for a,b in [(1,0),(1,-1),(np.nan,1),(1,np.inf)]:
        with pytest.raises(ValueError):m.gain(a,b)


def fixture_rows():
    rows=[]
    for source in ['a','b','c']:
        for budget in [32,233]:
            for seed in [1,2]:
                p=float(seed);factor=.8 if budget==32 else .9
                for arm,loss in [('native',p),('native_anchor',p*factor)]:
                    rows.append(dict(dataset=f'{source}_{budget}',seed=seed,arm=arm,loss=loss))
    return rows


def test_identifier_pairs_permutation_macro_and_interaction():
    rows=fixture_rows();ds=sorted({r['dataset'] for r in rows})
    a=m.aggregate(rows,ds,[1,2]);b=m.aggregate(rows[::-1],ds,[1,2])
    assert a==b
    assert a['source_balanced_macro']==pytest.approx(15.)
    assert a['sparse_macro']==pytest.approx(20.)
    assert a['mean_interaction']==pytest.approx(10.)
    with pytest.raises(ValueError):m.aggregate(rows+[rows[0]],ds,[1,2])
    with pytest.raises(ValueError):m.aggregate(rows[:-1],ds,[1,2])


def test_ratio_of_means_is_not_mean_of_ratios():
    rows=fixture_rows()
    for r in rows:
        if r['arm']=='native_anchor' and r['seed']==2:r['loss']=2.
    a=m.aggregate(rows,sorted({r['dataset'] for r in rows}),[1,2])
    c=next(c for c in a['cells'] if c['dataset']=='a_32')
    assert c['gain_of_mean_losses']==pytest.approx(100*(3-2.8)/3)
    assert c['mean_seed_gains']==pytest.approx(10.)


def test_temporal_boundaries():
    oo=list(range(10752,12193,96))
    s=m.split_origins(oo,list(range(2048,9473,32)))
    assert s['RECENT4']==[4,5,6,7] and s['SPREAD4']==[0,2,4,7]
    with pytest.raises(ValueError):m.split_origins(oo,[10730])
    with pytest.raises(ValueError):m.split_origins(list(range(16)),[-100])
    with pytest.raises(ValueError):m.split_origins(oo[:-1],[2048])


def test_choice_ignores_diagnostic_and_E_fields():
    candidates=[dict(id='a',step=150,lr=3e-5,D_loss=0,E_loss=0),dict(id='b',step=450,lr=3e-5,D_loss=10,E_loss=10)]
    S={'a':2.,'b':1.}
    assert m.pick(candidates,S)['id']=='b'
    for c in candidates:c.update(D_loss=-999,E_loss=999)
    assert m.pick(candidates,S)['id']=='b'
    assert m.pick(candidates,{'a':1.,'b':1.})['id']=='a'


def test_missingness_statistics_recompose_not_origin_mean():
    rng=np.random.default_rng(51);p=rng.normal(size=(16,4,21,48));y=rng.normal(size=(16,4,48))
    y[::2,0,:47]=np.nan;sc=np.arange(1,5,dtype=float)
    n,c=m.components(p,y,sc)
    full=m.pooled(n,c,list(range(16)))
    assert full==pytest.approx(m.score(p,y,sc)['scaled_2pinball'],abs=1e-12)
    assert full==pytest.approx(m.independent(p,y,sc),abs=1e-12)
    assert full!=pytest.approx(np.mean([m.pooled(n,c,[i]) for i in range(16)]),abs=1e-8)
    assert ((n[:8].sum(0)+n[8:].sum(0))/(c[:8].sum(0)+c[8:].sum(0))).mean()==pytest.approx(full)


def test_S_prediction_selection_unchanged_by_future_targets():
    rng=np.random.default_rng(2);p=rng.normal(size=(16,4,21,48));y=rng.normal(size=(16,4,48));sc=np.ones(4)
    n,c=m.components(p,y,sc);before=m.pooled(n,c,range(8))
    y[8:]=10000;p[8:]=-10000
    n,c=m.components(p,y,sc)
    assert m.pooled(n,c,range(8))==before


def test_alpha_endpoints_sorted_and_global():
    rng=np.random.default_rng(31)
    f=np.sort(rng.normal(size=(16,4,21,48)),axis=2);a=np.sort(rng.normal(size=f.shape),axis=2)
    assert np.array_equal(m.blend(f,a,0),f)
    assert np.array_equal(m.blend(f,a,1),a)
    for alpha in [.25,.5,.75]:
        p=m.blend(f,a,alpha)
        assert np.all(np.diff(p,axis=2)>=0)
        np.testing.assert_allclose(p-f,alpha*(a-f),atol=1e-15)
