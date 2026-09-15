"""Connect this existing worker to one audit; never launches or retries training."""
import argparse,json,os,subprocess,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/building_peft_topic_decision_20260916'
def read(p):return json.loads(p.read_text())
def write(status,**extra):
    p=OUT/'followup_connection.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(status=status,supervisor_pid=os.getpid(),worker_pid=args.worker_pid,worker_created_at=created,at=time.time(),**extra),indent=2)+'\n');t.replace(p)
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker-pid',type=int,required=True);args=parser.parse_args()
p=psutil.Process(args.worker_pid);created=p.create_time();cmd=p.cmdline()
assert any(v.endswith('experiments/building_peft_topic_decision_20260916/runner.py') for v in cmd) and cmd[-1]=='all'
write('WAITING_FOR_EXISTING_WORKER');print('WAITING_FOR_WORKER',args.worker_pid,flush=True)
while True:
    try:alive=p.is_running() and p.create_time()==created and p.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:alive=False
    if not alive:break
    if time.time()-read(OUT/'gpu_budget.json')['started_at']>14460:
        write('WORKER_STILL_ALIVE_AFTER_DEADLINE_NO_AUDIT_STARTED');raise SystemExit(2)
    time.sleep(1)
write('FINALIZING_EXISTING_RESULTS');print('WORKER_EXITED_STARTING_AUDIT',flush=True)
r=subprocess.run([str(ROOT/'scripts/with_cuda.sh'),str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/finalize_building_peft_topic_decision_20260916.py'),'all'],cwd=ROOT)
write('AUDIT_AND_REPORT_COMPLETE' if r.returncode==0 else 'AUDIT_ERROR_NO_AUTOMATIC_RETRY',returncode=r.returncode)
raise SystemExit(r.returncode)
