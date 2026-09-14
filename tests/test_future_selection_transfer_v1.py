import importlib.util
import sys
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(p))
import run_future_selection_transfer_v1 as m


def test_equal_budget_origins_and_disjoint_contexts():
    assert len(m.RULES['RECENT4'])==len(m.RULES['SPREAD4'])==4
    for source,r in m.RANGES.items():
        assert len(m.oo(r['S']))==16 and len(m.oo(r['D']))==32
        assert r['S'][1]+48<=r['D'][0]-1024


def test_tie_prefers_earlier_then_lower_lr():
    c=[dict(id='late',step=450,lr=3e-5),dict(id='high',step=0,lr=1e-4),dict(id='low',step=0,lr=3e-5)]
    assert m.selection(c,{x['id']:1. for x in c})['id']=='low'
    assert m.selection(c,{'late':.9,'high':1.,'low':1.})['id']=='late'


def test_future_loader_fails_before_raw_read_without_seal(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'OUT',tmp_path)
    with pytest.raises(FileNotFoundError):m.load_values('beijing','D',{})
