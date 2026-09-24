#!/usr/bin/env python3
"""Run exactly once, without editing source, simulation, neural training or reserve access."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
sys.dont_write_bytecode=True


def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def verify_package(root):
    manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    allowed=set(manifest['files'])|{'PACKAGE_MANIFEST.json'}
    actual=set()
    for p in root.rglob('*'):
        if p.is_symlink():raise RuntimeError('Symlink in package')
        if p.is_file():
            rel=p.relative_to(root).as_posix()
            if '__pycache__' in p.parts or p.suffix=='.pyc':
                raise RuntimeError('Derived bytecode unexpectedly bundled; do not modify, return package')
            actual.add(rel)
    if actual != allowed:raise RuntimeError('Unexpected/missing package files: '+str(actual^allowed))
    for rel,sha in manifest['files'].items():
        if hash_file(root/rel)!=sha:raise RuntimeError('PACKAGE_HASH_MISMATCH '+rel)
    return manifest,hash_file(root/'PACKAGE_MANIFEST.json')


def bounded_process(cmd,cwd,env,log,timeout_s):
    start=time.monotonic()
    with Path(log).open('w',encoding='utf-8') as f:
        p=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            rc=p.wait(timeout=timeout_s);status='EXIT_OK' if rc==0 else 'EXIT_FAILED'
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=8)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            rc=124;status='TIMEOUT'
    return {'command':cmd,'returncode':rc,'status':status,'elapsed_seconds':time.monotonic()-start}


def main():
    parser=argparse.ArgumentParser(description='Fixed BOPTEST recheck: existing data only; CLI run/push only')
    parser.add_argument('--repo',required=True);parser.add_argument('--publish',action='store_true')
    a=parser.parse_args();root=Path(__file__).resolve().parent
    receipt_path=root.parent/(root.name+'_EXECUTION_RECEIPT.json')
    receipt={'status':'STARTED','code_was_not_edited_by_runner':True}
    output=None;repo=None;installed=None;git_info=None;config=None
    def dump_receipt():
        receipt_path.write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        print('RECEIPT',receipt_path,flush=True)
    try:
        manifest,fingerprint=verify_package(root)
        sys.path.insert(0,str(root))
        from recheck.common import read_json,write_json,require,file_hash,utc
        from recheck.gitops import validate_repository,publish_exact,git
        config=read_json(root/'RUN_CONFIG.json');repo=Path(a.repo).resolve()
        require(not root.is_relative_to(repo),'Extract outside repository, not inside tracked worktree')
        git_info=validate_repository(repo,config['repository'],config['source_commit'],a.publish)
        py=repo/'.venv/bin/python';require(py.is_file(),'Existing repository .venv/bin/python missing; no install/fallback')
        installed=repo/'experiments'/config['package_name']
        require(not installed.exists(),'Package destination already exists: refuse overwrite/re-execution')
        # Protect against new extraction paths being used to silently re-run the same study.
        lockdir=Path(git_info['git_dir'])/'run_only_packages'/config['package_name'];lockdir.mkdir(parents=True,exist_ok=True)
        run_id='run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        marker=lockdir/'EXECUTED_ONCE.json'
        with marker.open('x',encoding='utf-8') as f:
            json.dump({'run_id':run_id,'package_manifest_sha256':fingerprint,'utc':utc()},f)
        output=repo/'results'/config['package_name']/run_id;output.mkdir(parents=True,exist_ok=False)
        private=lockdir/run_id;private.mkdir(exist_ok=False)
        shutil.copytree(root,installed)
        verify_package(installed)
        receipt.update({'run_id':run_id,'package_manifest_sha256':fingerprint,'result_relative_path':str(output.relative_to(repo)),
                        'head_at_start':git_info['head']})
        write_json(output/'EXECUTION_START.json',{**receipt,'repository':config['repository'],'git':git_info,
                   'argv':sys.argv,'new_simulation_calls':0,'neural_runs':0,'reserve_access_allowed':False})
        env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONNOUSERSITE']='1'
        env['PYTHONPATH']=str(installed)
        for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
            env[k]=str(config['budgets']['threads'])
        # Unit tests cannot access the production data paths; they use temporary synthetic files.
        test_cmd=[str(py),'-B','-m','unittest','recheck.tests','-v']
        unit=bounded_process(test_cmd,installed,env,output/'unit_tests.log',180)
        write_json(output/'UNIT_TEST_PROCESS.json',unit)
        require(unit['returncode']==0,'Package self-tests failed; do not patch or rerun')
        work_cmd=[str(py),'-B','-m','recheck.worker','--repo',str(repo),'--package',str(installed),
                  '--public',str(output),'--private',str(private)]
        worker=bounded_process(work_cmd,installed,env,output/'worker.log',config['budgets']['worker_timeout_s'])
        write_json(output/'WORKER_PROCESS.json',worker)
        require(worker['returncode']==0,'Recheck failed; see worker.log and FAILURE.json; no retry')
        require((output/'FINAL_DECISION.json').is_file(),'Worker exited without final result')
        verify_package(installed);verify_package(root)
        receipt['status']='COMPLETED_BOUNDED_RECHECK'
    except Exception as exc:
        receipt.update({'status':'FAILED_OR_BLOCKED','error':str(exc),'error_type':type(exc).__name__})
        if output is not None:
            (output/'EXECUTION_FAILURE.json').write_text(json.dumps({'error':str(exc),'traceback':traceback.format_exc(),
                    'scientific_failure_claimed':False},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print('STOP:',str(exc),file=sys.stderr,flush=True)
    # Failures are also published, provided a safe same-HEAD/empty-index state remains.
    try:
        if output is not None and installed is not None and installed.exists():
            from recheck.common import write_json
            from recheck.gitops import publish_exact
            verify_package(installed)
            output_files=[p for p in sorted(output.rglob('*')) if p.is_file()]
            if any(p.is_symlink() for p in output.rglob('*')):raise RuntimeError('Symlink in result')
            if sum(p.stat().st_size for p in output_files)>config['budgets']['max_public_mib']*(1<<20):
                raise RuntimeError('Public results exceed declared size budget; do not broad-stage')
            write_json(output/'FILES_SHA256.json',{p.relative_to(output).as_posix():hash_file(p) for p in output_files})
            files=[str((installed/rel).relative_to(repo)) for rel in sorted(set(manifest['files'])|{'PACKAGE_MANIFEST.json'})]
            files += [str(p.relative_to(repo)) for p in sorted(output.rglob('*')) if p.is_file()]
            if a.publish:
                publication=publish_exact(repo,git_info['head'],git_info['branch'],files,
                        f'Run locked bounded transition-history recheck v2 ({receipt["run_id"]}); preserve all outcomes')
                receipt['publication']=publication
            else:receipt['publication']={'status':'NOT_REQUESTED','files_ready':len(files)}
    except Exception as exc:
        receipt['publication']={'status':'PUBLICATION_BLOCKED','error':str(exc)}
        print('PUBLICATION STOP:',exc,file=sys.stderr,flush=True)
    dump_receipt()
    print(json.dumps(receipt,indent=2,ensure_ascii=False),flush=True)
    ok=receipt['status']=='COMPLETED_BOUNDED_RECHECK'
    if a.publish:ok=ok and receipt.get('publication',{}).get('status')=='PUSH_VERIFIED'
    return 0 if ok else 1

if __name__=='__main__':
    raise SystemExit(main())
