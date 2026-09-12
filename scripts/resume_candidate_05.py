"""User-authorized repaired Round1; preserve the original screen verbatim."""
import fcntl,json,subprocess,sys,time,traceback
from pathlib import Path
from tsfm_peft_screen.reproducibility import ROOT,sha,write_json
OUT=ROOT/'results/candidate_05_repaired'
CACHE=ROOT/'.cache/candidate_05_repaired'
if '--worker' in sys.argv:
    from tsfm_peft_screen.runners.streaming_eval import run
    try:
        run(OUT,CACHE)
    except Exception as exc:
        (OUT/'failure.txt').write_text(traceback.format_exc())
        write_json(OUT/'status.json',dict(verdict='IMPLEMENTATION_BLOCKED',error=str(exc),fit_count=0,stream_count=len(json.loads((OUT/'attempts.json').read_text())) if (OUT/'attempts.json').exists() else 0))
        raise
    raise SystemExit(0)
lock=open(ROOT/'.cache/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if OUT.exists():raise FileExistsError('This recovery run already exists; do not overwrite or retune')
if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip():
    raise RuntimeError('Commit recovery implementation before executing')
idle_since=None
while True:
    p=subprocess.run([str(ROOT/'scripts/with_cuda.sh'),'nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if not p.stdout.strip():
        if idle_since is None:idle_since=time.monotonic()
        if time.monotonic()-idle_since>=30:break
    else:
        idle_since=None;print('WAIT external GPU process:',p.stdout.strip(),flush=True)
    time.sleep(10)
OUT.mkdir();CACHE.mkdir()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
write_json(OUT/'execution_source.json',dict(execution_commit=commit,authorization='2026-09-13 user: 나머지해줘; complete the stopped Candidate05 comparison',original_status_sha256=sha(ROOT/'results/candidate_05/status.json'),original_contract_sha256=sha(ROOT/'results/candidate_05/contract.json'),scope='five repaired Round1 streams; fixed original recipe; no Round2',evaluation_reuse='Same development E origins were partially observed in the original failed run; this is an implementation-recovery comparison, not a fresh holdout.',additional_stream_cap=5))
start=time.monotonic()
with open(CACHE/'run.log','w') as log:
    try:
        p=subprocess.run([str(ROOT/'scripts/with_cuda.sh'),sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=14400)
        code=p.returncode
    except subprocess.TimeoutExpired:
        code=124
        write_json(OUT/'status.json',dict(verdict='IMPLEMENTATION_BLOCKED',error='Recovery supervisor timeout14400s',fit_count=0,stream_count=None))
write_json(OUT/'job_exit.json',dict(exit_code=code,wall_seconds=time.monotonic()-start,log=str(CACHE/'run.log')))
print('Recovery exit',code,flush=True)
raise SystemExit(code)
