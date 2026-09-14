import importlib.util
from pathlib import Path
import json

spec = importlib.util.spec_from_file_location('followup_observe', Path(__file__).with_name('observe.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_completion_requires_followup_regardless_of_scientific_outcome():
    for decision in ['PASS', 'FAIL', 'MIXED', 'WINDOW_BUDGET_HYPOTHESIS_NOT_SUPPORTED']:
        assert m.outcome({'status':'COMPLETE'}, {'status':'VERIFIED_HELDOUT_STUDY','decision':decision}) == 'FOLLOWUP_REQUIRED'


def test_execution_error_and_unverified_are_not_silently_complete():
    assert m.outcome({'status':'INCONCLUSIVE_EXECUTION'}, None) == 'RECOVERY_REQUIRED'
    assert m.outcome({'status':'COMPLETE'}, None) == 'VERIFICATION_REQUIRED'
    assert m.outcome({'status':'RUNNING'}, None) == 'WAIT_QUEUE'


def test_real_files_transition_and_deduplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, 'STATE', tmp_path/'state')
    q = tmp_path/'queue.json'
    v = tmp_path/'verification.json'
    m.write(q, {'status':'RUNNING','started_unix':1})
    assert m.observe(q)['status'] == 'WAIT_QUEUE'
    m.write(q, {'status':'COMPLETE','started_unix':1,'finished_unix':2,'heartbeat_unix':2})
    m.write(v, {'status':'VERIFIED_HELDOUT_STUDY','completed_fits':48,'training_updates':43200})
    before = q.read_bytes(), v.read_bytes()
    first = m.observe(q,v)
    assert first['new_request'] and first['completed_fits']==48 and first['training_updates']==43200
    assert not first['automatic_continuation_enabled']
    assert (q.read_bytes(),v.read_bytes()) == before
    assert not m.observe(q,v)['new_request']
    m.write(q, {**m.read(q),'heartbeat_unix':900})
    assert not m.observe(q,v)['new_request']
    # A new registered run has a distinct completion event.
    q2 = tmp_path/'second.json'
    m.write(q2, {'status':'COMPLETE','started_unix':3,'finished_unix':4})
    assert m.observe(q2,v)['new_request']
    assert len(list((tmp_path/'state/requests').glob('*.json'))) == 2


def test_seed_improvement_is_not_loss_ratio():
    from math import isclose
    spec = importlib.util.spec_from_file_location('window_erratum', Path(__file__).with_name('review_window.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.gain(1.,1.)==0.
    assert isclose(module.gain(.9,1.),10.)
    assert isclose(module.gain(1.1,1.),-10.)
