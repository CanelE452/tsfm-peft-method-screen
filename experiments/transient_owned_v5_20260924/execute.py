#!/usr/bin/env python3
"""Install immutable source and run it once. Optional explicitly authorized allowlist push.

Usage from the target checkout:
  python3 /path/to/extracted/transient_owned_v5_20260924/execute.py --repo "$PWD" --publish
No editing/generation of experiment code is delegated to the CLI.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
sys.dont_write_bytecode=True
from transient_v5 import NAME,BASE_COMMIT
from transient_v5.util import require,ContractError,git,verify_manifest,write_json,read_json,file_hash,redact,now

EXPECTED_REMOTE=re.compile(r'^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)(?i:CanelE452/tsfm-peft-method-screen)(?:\.git)?/?$')


def remote_branch_sha(repo,branch):
    text=git(repo,'ls-remote','--heads','origin',f'refs/heads/{branch}',timeout=30)
    require(bool(text.strip()),'Current branch has no remote branch. No automatic branch creation/push.')
    return text.split()[0]


def publish(repo: Path,source: Path,result: Path,start_head: str,branch: str):
    """Publish ONLY immutable package and the one execution record, even if tests failed."""
    verify_manifest(source)
    require(git(repo,'rev-parse','HEAD')==start_head,'Local HEAD changed during execution; no automatic reconciliation')
    require(git(repo,'symbolic-ref','--short','HEAD')==branch,'Branch changed during execution')
    require(not git(repo,'diff','--cached','--name-only'),'Unrelated staged changes exist; no commit')
    require(remote_branch_sha(repo,branch)==start_head,'Remote moved/unpushed commits exist; no automatic pull/rebase/force')
    files=[source/rel for rel in read_json(source/'PACKAGE_MANIFEST.json')['files']]+[source/'PACKAGE_MANIFEST.json']
    allowed={'.json','.jsonl','.log','.csv','.md'}
    for p in result.rglob('*'):
        if p.is_file():
            require(not p.is_symlink() and p.suffix in allowed,'Unexpected result file; publication stopped')
            require(p.stat().st_size<=5*1024*1024,'Oversized result; no automatic upload')
            text=p.read_text(encoding='utf-8')
            require(redact(text)==text,'Potential secret in result; publication stopped')
            files.append(p)
    require(sum(p.stat().st_size for p in files)<=50*1024*1024,'Publication size limit exceeded')
    rels=[str(p.relative_to(repo)) for p in files]
    # Explicit allowlist. No weights, raw datasets, temporary attempts, or unrelated paths.
    git(repo,'diff','--check','--',*rels)
    git(repo,'add','-f','--',*rels)
    try:
        git(repo,'commit','-m',f'Run locked transient v5 readiness ({result.name}); preserve all outcomes','--only','--',*rels,timeout=60)
        commit=git(repo,'rev-parse','HEAD')
        verify_manifest(source)
        committed=set(git(repo,'diff-tree','--no-commit-id','--name-only','-r',commit).splitlines())
        require(committed<=set(rels),'Commit hook included paths outside the allowlist; push stopped')
        for fp in files:
            rel=str(fp.relative_to(repo))
            raw=subprocess.run(['git','-C',str(repo),'show',f'{commit}:{rel}'],capture_output=True,check=True).stdout
            require(hashlib.sha256(raw).hexdigest()==file_hash(fp),f'Committed content differs from tested artifact: {rel}')
        git(repo,'push','origin',f'HEAD:refs/heads/{branch}',timeout=120)
        require(remote_branch_sha(repo,branch)==commit,'Remote verification failed after push')
        return {'status':'PUSH_VERIFIED','commit':commit,'branch':branch,'published_files':len(rels),
                'result_relative_path':str(result.relative_to(repo))}
    except Exception:
        # Never reset, amend, unstash, or modify other index entries automatically.
        raise


def main():
    p=argparse.ArgumentParser(description='Run fixed v5 code once; --publish explicitly permits its allowlisted Git commit/push.')
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--publish',action='store_true')
    a=p.parse_args();repo=a.repo.expanduser().resolve();root=Path(__file__).resolve().parent
    receipt=root.parent/(NAME+'_EXECUTION_RECEIPT.json')
    require(not receipt.exists(),'Execution receipt already exists. Do not rerun; send it to ChatGPT.')
    phase='preflight';source=None;result=None
    try:
        manifest=verify_manifest(root)
        require(Path(git(repo,'rev-parse','--show-toplevel')).resolve()==repo,'--repo must be the actual repository root')
        remote=git(repo,'remote','get-url','origin')
        require(EXPECTED_REMOTE.fullmatch(remote) is not None,'Unexpected GitHub repository; stopped')
        head=git(repo,'rev-parse','HEAD');branch=git(repo,'symbolic-ref','--short','HEAD')
        require(not git(repo,'diff','--cached','--name-only'),'Staged changes already exist. CLI must not stash/commit them automatically.')
        git(repo,'merge-base','--is-ancestor',BASE_COMMIT,head)
        if a.publish:require(remote_branch_sha(repo,branch)==head,'Local/remote branch differ. Do not pull/rebase automatically.')
        interpreter=repo/'.venv'/'bin'/'python'
        require(interpreter.is_file(),'Expected existing .venv/bin/python not found; no installation or environment guessing')
        source=repo/'experiments'/NAME
        require(not source.exists(),'This package is already installed. No rerun/overwrite.')
        prior_results=repo/'results'/NAME
        require(not prior_results.exists(),'Previous v5 execution exists; send its logs, do not repeat')
        require(shutil.disk_usage(repo).free>2*1024**3,'Less than 2 GiB free disk')
        phase='install'
        source.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(root,source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        verify_manifest(source)
        run_id=datetime.datetime.now(datetime.timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
        result=repo/'results'/NAME/run_id
        env=os.environ.copy();env.update({'PYTHONPATH':str(source),'PYTHONDONTWRITEBYTECODE':'1',
              'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':''})
        cmd=[str(interpreter),'-m','transient_v5.suite','--repo',str(repo),'--run-id',run_id,'--backend','chronos']
        phase='run'
        r=subprocess.run(cmd,env=env,timeout=3700)
        result.mkdir(parents=True,exist_ok=True)
        if not (result/'SUMMARY.json').exists():
            write_json(result/'SUPERVISOR_FAILURE.json',{'status':'SUPERVISOR_FAILED','returncode':r.returncode,
                                                       'note':'No source edits or retries allowed.'})
        write_json(result/'INSTALL_RECEIPT.json',{'source_package_manifest':file_hash(root/'PACKAGE_MANIFEST.json'),
                    'base_at_install':head,'branch':branch,'installed_source':str(source.relative_to(repo)),
                    'command':cmd,'worker_exit_code':r.returncode,'utc':now(),'push_requested':a.publish})
        phase='publish'
        pushed=publish(repo,source,result,head,branch) if a.publish else {'status':'NOT_REQUESTED'}
        response={'status':'EXECUTED','worker_exit_code':r.returncode,'publication':pushed,
                  'summary':read_json(result/'SUMMARY.json') if (result/'SUMMARY.json').exists() else None,
                  'receipt_outside_repo':str(receipt),'cli_must_not_analyze_or_modify':True}
        write_json(receipt,response)
        print(json.dumps({'receipt':str(receipt),'publication':pushed,'worker_exit_code':r.returncode},ensure_ascii=False,indent=2))
        return r.returncode
    except Exception as e:
        err={'status':'STOPPED_NO_AUTOMATIC_REPAIR','phase':phase,'error':redact(str(e)),
             'traceback':redact(traceback.format_exc()),'source':None if source is None else str(source),
             'result':None if result is None else str(result),'utc':now()}
        # Preflight failures cannot always safely be committed. Return this small receipt to ChatGPT.
        if phase == 'run' and source is not None and result is not None:
            try:
                result.mkdir(parents=True,exist_ok=True)
                write_json(result/'EXECUTION_FAILURE.json',err)
                if a.publish:
                    err['publication']=publish(repo,source,result,head,branch)
            except Exception as pe:
                err['publication_error']=redact(str(pe))
        write_json(receipt,err)
        print(json.dumps(err,ensure_ascii=False,indent=2),file=sys.stderr)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
