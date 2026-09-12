"""Single-GPU serial supervisor. No Round 2 entry point."""
import json,subprocess,time
from pathlib import Path
from tsfm_peft_screen.reproducibility import ROOT,write_json
import fcntl
lock=open(ROOT/'.cache/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if all((ROOT/'results'/f'candidate_{i:02}'/'status.json').exists() for i in range(1,8)):
    print('All candidates have terminal verdicts. No rerun or Round2 launched.');raise SystemExit(0)
def wait_gpu():
    idle_since=None
    while True:
        status=subprocess.run([str(ROOT/'scripts/with_cuda.sh'),'nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        if not status.stdout.strip():
            if idle_since is None:idle_since=time.monotonic()
            if time.monotonic()-idle_since>=30:return
        else:
            idle_since=None
            print('WAIT external GPU process:',status.stdout.strip().replace('\n',','),flush=True)
        time.sleep(15)

wait_gpu()
# Repeat exact checkpoint audit after external-GPU contention caused preflight OOM.
subprocess.run([str(ROOT/'scripts/with_cuda.sh'),str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/audit_environment.py')],check=True,stdout=open(ROOT/'.cache/common_audit_final.log','w'))
for i in range(1,8):
    out=ROOT/'results'/f'candidate_{i:02}';out.mkdir(exist_ok=True)
    if (out/'status.json').exists():
        print('SKIP terminal candidate',i,flush=True);continue
    wait_gpu()
    log=open(ROOT/'.cache'/f'candidate_{i:02}.log','a');start=time.monotonic()
    try:
        p=subprocess.run([str(ROOT/'scripts/with_cuda.sh'),str(ROOT/'.venv/bin/python'),str(ROOT/f'scripts/run_candidate_{i:02}.py')],stdout=log,stderr=subprocess.STDOUT,timeout=14400)
        code=p.returncode
    except subprocess.TimeoutExpired:
        code=124
        write_json(out/'status.json',dict(verdict='IMPLEMENTATION_BLOCKED',error='Candidate supervisor timeout14400s',fit_count=None,stream_count=None))
    write_json(out/'job_exit.json',dict(exit_code=code,wall_seconds=time.monotonic()-start,log=str(ROOT/'.cache'/f'candidate_{i:02}.log')))
    log.close();print('CANDIDATE EXIT',i,code,flush=True)
print('ROUND 0 + ROUND 1 STOP. ROUND 2 NOT EXECUTED.',flush=True)
