"""Wait for one verified runner process, then finalize once; never starts fits."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
import psutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/delta_adapter_comparison_20260919'
EXPECTED='experiments.delta_adapter_comparison_20260919.runner'

def save(value):
    p=OUT/'COMPLETION_HOOK.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True);parser.add_argument('--created',type=float,required=True);args=parser.parse_args()
    start=time.monotonic();base=dict(runner_pid=args.pid,runner_create_time=args.created,hook_pid=os.getpid(),automatic_training=False)
    p=psutil.Process(args.pid)
    assert p.create_time()==args.created and EXPECTED in p.cmdline(),'RUNNER_IDENTITY_MISMATCH'
    try:
        while True:
            try:live=p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:break
            if not live:break
            assert p.create_time()==args.created
            if time.monotonic()-start>10800:
                save(dict(**base,state='TIMEOUT_RUNNER_NOT_RESTARTED',at=time.time()));return
            save(dict(**base,state='WAITING_ON_VERIFIED_PROCESS',at=time.time(),elapsed_seconds=time.monotonic()-start))
            try:p.wait(timeout=10)
            except psutil.TimeoutExpired:pass
            except psutil.NoSuchProcess:break
        status=json.loads((OUT/'status.json').read_text())
        if status['execution']!='COMPLETE_COMPUTE':
            save(dict(**base,state='RUNNER_TERMINAL_WITHOUT_COMPLETION',at=time.time(),runner_status=status));return
        save(dict(**base,state='FINALIZING',at=time.time()))
        env=dict(os.environ,OPENBLAS_NUM_THREADS='4')
        subprocess.run([sys.executable,str(Path(__file__).with_name('finalize.py'))],cwd=ROOT,env=env,check=True)
        save(dict(**base,state='REPORTS_VERIFIED_PUBLICATION_PENDING',at=time.time(),runner_status=status))
    except BaseException:
        save(dict(**base,state='HOOK_ERROR',at=time.time(),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
