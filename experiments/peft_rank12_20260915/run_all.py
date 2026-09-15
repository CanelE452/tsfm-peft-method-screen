"""One explicitly authorized R1 then R2 batch; no automatic research continuation."""
import json,subprocess,time
from common import ROOT,EXP,RESEARCH,R1,R2,read,save

def main():
    assert not (RESEARCH/'sequence.json').exists(),'This batch was already started; no duplicate run'
    sequence=[];save(RESEARCH/'sequence.json',sequence)
    def run(track,stage):
        env='.venv' if track==1 else '.venv-channel';cmd=[str(ROOT/'scripts/with_cuda.sh'),str(ROOT/env/'bin/python'),str(EXP/f'run_rank{track}.py'),stage];started=time.time();p=subprocess.run(cmd,cwd=ROOT);sequence.append(dict(track=track,stage=stage,exit_code=p.returncode,started_at=started,ended_at=time.time()));save(RESEARCH/'sequence.json',sequence);return p.returncode
    run(1,'numerical-check');q=read(R1/'status.json')
    if q['status']=='NUMERICS_BOUNDED':run(1,'profile');q=read(R1/'status.json')
    if q['status']=='RESOURCE_SIGNAL':run(1,'train-evaluate')
    run(1,'verify');q=read(R1/'status.json')
    fatal=any(word in q.get('error','').lower() for word in ['illegal memory','device-side assert','gpu_monitor_error'])
    if fatal:
        s=read(R2/'status.json');s.update(status='BLOCKED_COMMON_ENVIRONMENT',error='R1 shared GPU environment error');save(R2/'status.json',s)
    else:
        run(2,'smoke')
        if read(R2/'status.json')['status']=='SMOKE_PASSED':run(2,'train-evaluate')
    run(2,'verify')
    p=subprocess.run([str(ROOT/'scripts/with_cuda.sh'),str(ROOT/'.venv-channel/bin/python'),str(EXP/'report.py'),'combined'],cwd=ROOT);assert p.returncode==0
    print('RANK12 SEQUENCE FINISHED',read(R1/'status.json')['status'],read(R2/'status.json')['status'],flush=True)
if __name__=='__main__':main()
