"""Control-plane regression: a failed worker must not suppress later topics."""
import importlib
import json
import sys
from pathlib import Path
import pytest


@pytest.fixture
def queue_module(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    q=importlib.import_module('overnight_queue')
    monkeypatch.setattr(q,'ROOT',tmp_path)
    monkeypatch.setattr(q,'OUT',tmp_path/'results'/'run')
    monkeypatch.setattr(q,'CACHE',tmp_path/'.cache'/'run')
    monkeypatch.setattr(q.time,'sleep',lambda _:None)
    return q


def test_failed_topic_continues_and_every_training_job_precedes_evaluation(queue_module,monkeypatch):
    q=queue_module
    def initialize():
        q.OUT.mkdir(parents=True)
        q.CACHE.mkdir(parents=True)
        q.write_json(q.OUT/'contract.json',{'fixed':'contract'})
    monkeypatch.setattr(q,'initialize',initialize)
    events=[]
    class Child:
        def __init__(self,cmd,**kwargs):
            topic=cmd[cmd.index('--topic')+1]
            phase=cmd[cmd.index('--phase')+1]
            self.pid=100+len(events)
            events.append((topic,phase))
            self.returncode=1 if topic=='drift' else 0
            folder=q.OUT/topic
            folder.mkdir(exist_ok=True)
            q.write_json(folder/(phase+'_status.json'),{'status':'INCONCLUSIVE_EXECUTION' if self.returncode else 'OK'})
            if phase=='train' and self.returncode==0:
                decision='STOP_NO_TEACHER_HEADROOM' if topic=='distill' else 'READY_FOR_E'
                q.write_json(folder/'selection_seal.json',{'decision':decision})
            if phase=='evaluate':
                assert events[:3]==[('anchor','train'),('drift','train'),('distill','train')]
                assert (q.OUT/'evaluation_barrier.json').exists()
        def poll(self):
            return self.returncode
    monkeypatch.setattr(q.subprocess,'Popen',Child)
    class Done:
        returncode=0
    monkeypatch.setattr(q.subprocess,'run',lambda *a,**k:Done())
    q.run_queue()
    assert events==[('anchor','train'),('drift','train'),('distill','train'),('anchor','evaluate')]
    state=json.loads((q.OUT/'queue_status.json').read_text())
    assert state['status']=='COMPLETE_WITH_ERRORS'
    barrier=json.loads((q.OUT/'evaluation_barrier.json').read_text())
    assert barrier['topics']['drift']['decision']=='INCONCLUSIVE_EXECUTION'
    assert barrier['topics']['distill']['decision']=='STOP_NO_TEACHER_HEADROOM'
    assert state['jobs'][1]['exit_code']==1


def test_existing_run_cannot_be_overwritten(queue_module):
    q=queue_module
    q.OUT.mkdir(parents=True)
    with pytest.raises(AssertionError,match='Existing experiment'):
        q.initialize()
