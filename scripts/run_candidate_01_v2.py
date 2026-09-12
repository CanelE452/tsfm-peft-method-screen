"""One authorized v2 execution; serial GPU lock, no automatic retry."""
import fcntl,json,subprocess,sys,time,traceback
from tsfm_peft_screen.reproducibility import ROOT,write_json
from tsfm_peft_screen.runners.freshness_v2 import OUT,CACHE
if '--worker' in sys.argv:
    from tsfm_peft_screen.runners.freshness_v2 import run
    try:run()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'failure.txt').write_text(traceback.format_exc())
        attempts=json.loads((OUT/'attempts.json').read_text()) if (OUT/'attempts.json').exists() else []
        write_json(OUT/'status.json',dict(verdict='IMPLEMENTATION_BLOCKED',error=str(exc),fit_count=len(attempts),completed_fits=sum(a['status']=='COMPLETE' for a in attempts),round2_executed=False))
        raise
    raise SystemExit(0)
lock=open(ROOT/'.cache/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if OUT.exists():raise FileExistsError('v2 results already exist; no retry/retuning')
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'Commit implementation before execution'
idle=None
while True:
    p=subprocess.run([str(ROOT/'scripts/with_cuda.sh'),'nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if p.stdout.strip():
        idle=None;print('WAIT external GPU process',p.stdout.strip(),flush=True)
    else:
        if idle is None:idle=time.monotonic()
        if time.monotonic()-idle>=30:break
    time.sleep(10)
CACHE.mkdir(parents=True);start=time.monotonic()
with open(CACHE/'run.log','w') as log:
    try:
        p=subprocess.run([str(ROOT/'scripts/with_cuda.sh'),sys.executable,__file__,'--worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=14400)
        code=p.returncode
    except subprocess.TimeoutExpired:code=124
OUT.mkdir(parents=True,exist_ok=True)
write_json(OUT/'job_exit.json',dict(exit_code=code,wall_seconds=time.monotonic()-start,log=str(CACHE/'run.log')))
print('V2 EXIT',code,flush=True)
raise SystemExit(code)
