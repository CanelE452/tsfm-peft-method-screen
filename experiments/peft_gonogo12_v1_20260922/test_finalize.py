import numpy as np
import pandas as pd
import pytest
import finalize
import runner


def test_cube_aligns_dates_clients_and_broadcasts_single_reference():
    rows=[dict(arm='CHRONOS2_DIRECT',seed=0,series=str(i),origin=16104+j*24,scaled_pinball=i+j/10)
          for i in range(8) for j in range(14)]
    f=pd.DataFrame(rows).sample(frac=1,random_state=2)
    a=finalize.cube(f,'CHRONOS2_DIRECT',[92251,92252])
    assert a.shape==(2,8,14)
    np.testing.assert_array_equal(a[0],a[1])
    assert a[0,3,5]==3.5


def test_joint_date_bootstrap_preserves_constant_paired_effect():
    b=np.arange(1,225,dtype=float).reshape(2,8,14)
    r=finalize.compare(.98*b,b)
    np.testing.assert_allclose(r['seed_effects'],[2,2],atol=1e-12)
    np.testing.assert_allclose(r['conditional_ci95'],[2,2],atol=1e-12)
    assert r['winning_clients']==8


def test_existing_seal_is_never_overwritten_and_changed_source_rejected(tmp_path,monkeypatch):
    p=tmp_path/'ALL_SELECTIONS_SEALED.json';p.write_text('original')
    monkeypatch.setattr(runner,'RESULTS',tmp_path)
    checked=[];monkeypatch.setattr(runner,'check_seal',lambda:checked.append(True))
    runner.seal()
    assert checked==[True] and p.read_text()=='original'
    source=tmp_path/'source.py';source.write_text('old')
    old=runner.sha(source);source.write_text('new')
    monkeypatch.setattr(runner,'EXP',tmp_path)
    with pytest.raises(AssertionError):runner.verify_sources({'source.py':old})
