import importlib.util
import json
import os
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('watch',Path(__file__).with_name('watch.py'))
w=importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

def q(status='COMPLETE',code=0):
    return dict(status=status,independent_finalization_exit_code=code)

def v(verdicts=('PILOT_STOP','PILOT_STOP','STOP_NO_TEACHER_HEADROOM')):
    return dict(status='VERIFIED',topics=[dict(topic=t,verdict=d) for t,d in zip(['anchor','drift','distill'],verdicts)])

def test_running_or_failed_execution_is_not_scientific_failure():
    assert w.eligibility(q('RUNNING'),v())=='WAIT_PARENT'
    assert w.eligibility(q('COMPLETE_WITH_ERRORS'),v())=='NO_TRIGGER_EXECUTION_ERROR'
    assert w.eligibility(q(code=1),v())=='NO_TRIGGER_EXECUTION_ERROR'
    assert w.eligibility(q(),None)=='NO_TRIGGER_UNVERIFIED'
    assert w.eligibility(q('STOPPED'),v())=='NO_TRIGGER_PARENT_INTERRUPTED'

def test_all_three_stops_only():
    assert w.eligibility(q(),v())=='ELIGIBLE_ALL_STOP'
    assert w.eligibility(q(),v(('PILOT_PASS','PILOT_STOP','PILOT_STOP')))=='NO_TRIGGER_PASS_FOUND'
    assert w.eligibility(q(),v(('PILOT_STOP','INCONCLUSIVE_EXECUTION','PILOT_STOP')))=='NO_TRIGGER_EXECUTION_ERROR'
    missing=v();missing['topics'].pop()
    assert w.eligibility(q(),missing)=='NO_TRIGGER_INCOMPLETE_VERDICTS'
    duplicate=v();duplicate['topics'][2]['topic']='anchor'
    assert w.eligibility(q(),duplicate)=='NO_TRIGGER_INCOMPLETE_VERDICTS'

def test_only_exact_session_resume_no_sandbox_bypass():
    sid='11111111-2222-4333-8444-555555555555'
    command=w.resume_command('/bin/codex',sid)
    assert command[command.index('resume')+1]==sid
    assert '--last' not in command and '--model' not in command
    assert '--sandbox' in command and 'workspace-write' in command
    assert not any('bypass' in s or 'ignore-' in s for s in command)

def test_idle_requires_ended_turn_and_quiet_period(tmp_path):
    p=tmp_path/'session.jsonl'
    def save(kinds,mtime=100):
        p.write_text('\n'.join(json.dumps(dict(type='event_msg',payload={'type':k})) for k in kinds))
        os.utime(p,(mtime,mtime))
    save(['task_started'])
    assert not w.session_idle(p,now=200)
    save(['task_started','task_complete'])
    assert w.session_idle(p,now=200)
    assert not w.session_idle(p,now=120)
    save(['task_complete','task_started'])
    assert not w.session_idle(p,now=200)
    save(['task_started','turn_aborted'])
    assert w.session_idle(p,now=200)
