"""One-shot archival of the explicitly authorized diagnostics on existing main.

Waits for the verified queue and report. Never trains, retries, force-pushes,
changes another tracked file, resumes Codex, or sends a user notification.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/reassessment_diagnostics_20260914'
CACHE=ROOT/'.cache/reassessment_diagnostics_20260914'
STATUS=CACHE/'archive_status.json'


def emit(status,**extra):
    value=dict(status=status,pid=os.getpid(),updated_unix=time.time(),**extra)
    tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(STATUS)


def run(args,**kwargs):
    return subprocess.run(args,cwd=ROOT,check=True,timeout=600,**kwargs)


def main():
    assert not STATUS.exists(),'One shot only; retain prior attempts'
    emit('WAITING_FOR_REPORT')
    start=time.monotonic()
    while time.monotonic()-start<14400:
        q=json.loads((OUT/'queue_status.json').read_text())
        if q['status'] not in ('RUNNING','COMPLETE'):
            emit('QUEUE_INCOMPLETE',queue_status=q['status']);return
        p=CACHE/'report_watch.json'
        if p.exists():
            report=json.loads(p.read_text())
            if report['status']=='COMPLETE':break
            if report['status'] not in ('WAITING_FOR_VERIFIED_QUEUE','RENDERING'):
                emit('REPORT_INCOMPLETE',report_status=report['status']);return
        time.sleep(15)
    else:
        emit('WAIT_TIMEOUT');return
    assert q['status']=='COMPLETE'
    emit('VERIFYING_FOR_ARCHIVE')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'src'),MPLCONFIGDIR='/tmp/tsfm-diagnostic-mpl')
    with (OUT/'archive_checks.txt').open('x') as log:
        run([sys.executable,str(ROOT/'scripts/run_reassessment_diagnostics.py'),'finalize','--verify-only'],env=env,stdout=log,stderr=subprocess.STDOUT)
        run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests','automation/overnight_followup/test_watch.py'],env=env,stdout=log,stderr=subprocess.STDOUT)
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
    run(['git','diff','--exit-code'])
    run(['git','diff','--cached','--exit-code'])
    for p in OUT.rglob('*'):
        if p.is_file():
            assert p.suffix in ('.json','.csv','.md','.png','.txt'),p
            assert p.stat().st_size<10*2**20,p
    receipt=dict(status='VERIFIED_FOR_ARCHIVE',scope='Completed train/V diagnostics; no E results or publication PASS',
                 report_sha256=hashlib.sha256((OUT/'REPORT.md').read_bytes()).hexdigest(),
                 checks_sha256=hashlib.sha256((OUT/'archive_checks.txt').read_bytes()).hexdigest(),
                 new_fits=44,new_training_updates=46080,
                 render_script_sha256=hashlib.sha256((ROOT/'scripts/report_reassessment_diagnostics.py').read_bytes()).hexdigest())
    (OUT/'archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    relative=str(OUT.relative_to(ROOT))
    run(['git','add','--',relative])
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    assert staged and all(p.startswith(relative+'/') for p in staged)
    run(['git','diff','--cached','--check'])
    emit('COMMITTING_RESULTS')
    run(['git','commit','--only','-m','Archive completed anchoring and DualClock train-validation diagnostics','--',relative])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    emit('PUSHING_RESULTS',commit=commit)
    run(['git','push','origin','main'])
    emit('COMPLETE',commit=commit)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        emit('ARCHIVE_ERROR',error=f'{type(exc).__name__}: {exc}')
        raise
