"""Execute exactly one author-approved wind-only retry, without modifying v1."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
from resume_core import verify_package, execute, write_json, utc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--publish', action='store_true')
    ap.add_argument('--wind-csv', required=True)
    ap.add_argument('--wind-locations', required=True)
    args = ap.parse_args()
    pkg = Path(__file__).resolve().parent
    verify_package(pkg)
    repo = Path(args.repo).resolve()
    py = repo / '.venv' / 'bin' / 'python'
    if os.environ.get('WIND_RETRY_OWNED_VENV') != '1':
        if not py.is_file():
            raise RuntimeError('Existing .venv/bin/python required; no installs or fallback')
        env = os.environ.copy()
        env.update(WIND_RETRY_OWNED_VENV='1', PYTHONDONTWRITEBYTECODE='1')
        return subprocess.run([str(py), '-B', str(pkg/'run.py'), *sys.argv[1:]], env=env).returncode
    try:
        return execute(pkg, args)
    except Exception as e:
        # This is not a score, and a retry-lock conflict must not mutate the old run.
        receipt={'status':'RETRY_PRECONDITION_FAILED','reason':str(e),'utc':utc(),
                 'scientific_no_go':False,'worker_started':False,'do_not_delete_locks':True}
        print(json.dumps(receipt,ensure_ascii=False,indent=2),file=sys.stderr)
        if (repo/'.git').exists():
            write_json(repo/'.cache'/'wind_resume_v1_20260925'/'LAST_PRECONDITION_FAILURE.json',receipt)
        return 2
if __name__ == '__main__':
    raise SystemExit(main())
