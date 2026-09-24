"""RUN ONLY. This entry point needs only the Python standard library.
Source hashing excludes bytecode. Any per-topic failure is preserved and published
without changing code, retrying, adding methods, or silently using synthetic data.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
NAME='topic_axis_gonogo_v1_20260925'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--publish',action='store_true')
    ap.add_argument('--wind-csv');ap.add_argument('--wind-locations');ap.add_argument('--wind-archive')
    a=ap.parse_args();pkg=Path(__file__).resolve().parent;repo=Path(a.repo).resolve()
    mf=json.loads((pkg/'PACKAGE_MANIFEST.json').read_text());manifest_hash=sha(pkg/'PACKAGE_MANIFEST.json')
    for rel,expected in mf['files'].items():
        p=pkg/rel
        if not p.is_file() or p.is_symlink() or sha(p)!=expected:raise RuntimeError('PACKAGE_MISMATCH: '+rel)
    py=repo/'.venv'/'bin'/'python'
    if os.environ.get('TRIAGE_IN_REPO_VENV')!='1':
        if not py.is_file():raise RuntimeError('Existing repository .venv/bin/python required; no installation')
        env=os.environ.copy();env['TRIAGE_IN_REPO_VENV']='1';env['PYTHONDONTWRITEBYTECODE']='1'
        return subprocess.run([str(py),'-B',str(pkg/'run.py'),*sys.argv[1:]],env=env).returncode
    from triage.gitops import validate,publish_exact
    from triage.common import write_json,read_json,utc
    initial=validate(repo,a.publish)
    cache=repo/'.cache'/NAME;cache.mkdir(parents=True,exist_ok=True)
    lock=cache/(manifest_hash+'.started')
    with lock.open('x') as f:f.write(utc()+'\n')
    runid='run_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=repo/'results'/NAME/runid;out.mkdir(parents=True)
    py=repo/'.venv'/'bin'/'python'
    cfg=read_json(pkg/'RUN_CONFIG.json')
    source=repo/'experiments'/NAME
    if source.exists():raise RuntimeError('Package destination already exists; preserve it and return, no overwrite')
    source.mkdir(parents=True)
    for rel in [*mf['files'],'PACKAGE_MANIFEST.json']:
        dst=source/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(pkg/rel,dst)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(source),
          HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    start=time.monotonic();processes={}
    try:
        if not py.is_file():raise RuntimeError('Existing repository .venv/bin/python required; no install/fallback')
        with (out/'UNIT_TESTS.txt').open('w',encoding='utf-8',newline='\n') as f:
            p=subprocess.run([str(py),'-B','-m','unittest','discover','-s',str(source/'tests'),'-v'],env=env,cwd=source,stdout=f,stderr=subprocess.STDOUT,timeout=120)
        if p.returncode:raise RuntimeError('PACKAGE_UNIT_TEST_FAILURE')
        for topic in cfg['topics']:
            dest=out/topic;dest.mkdir()
            cmd=[str(py),'-B','-m','triage.worker','--topic',topic,'--repo',str(repo),'--cache',str(cache),
                  '--public',str(dest),'--config',str(source/'RUN_CONFIG.json')]
            for flag in ('wind_csv','wind_locations','wind_archive'):
                v=getattr(a,flag)
                if v:cmd.extend(['--'+flag.replace('_','-'),str(Path(v).resolve())])
            t=time.monotonic()
            with (dest/'CONSOLE.txt').open('w',encoding='utf-8',newline='\n') as f:
                try:
                    p=subprocess.run(cmd,cwd=source,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=cfg['topic_timeout_seconds'])
                    status={'returncode':p.returncode,'elapsed_s':time.monotonic()-t}
                except subprocess.TimeoutExpired:
                    status={'returncode':124,'elapsed_s':time.monotonic()-t,'timeout':True}
                    write_json(dest/'RESULT.json',{'axis_status':'BLOCKED_RESOURCE_TIMEOUT','scientific_no_go':False,'topic':topic})
            processes[topic]=status;write_json(dest/'PROCESS.json',status)
        summary={t:read_json(out/t/'RESULT.json') if (out/t/'RESULT.json').exists() else {'axis_status':'MISSING_RESULT'} for t in cfg['topics']}
        write_json(out/'ALL_TOPICS.json',summary)
    except Exception as e:
        write_json(out/'EXECUTION_FAILURE.json',{'reason':str(e),'scientific_no_go':False})
    finally:
        write_json(out/'RUN_MANIFEST.json',{'package_sha256':manifest_hash,'git_start':initial,'config':cfg,
            'processes':processes,'elapsed_s':time.monotonic()-start,'code_modified':False,
            'new_lora_fit':0,'new_peft_fit':0,'optimizer_updates':0,'model_downloads':0,
            'independence':'topics run independently; technical failure does not reject a hypothesis'})
    # Recheck copied sources AFTER execution; bytecode is neither listed nor published.
    for rel,expected in mf['files'].items():
        if sha(source/rel)!=expected:raise RuntimeError('SOURCE_CHANGED_DURING_EXECUTION: '+rel)
    public=[p for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.txt','.md')]
    sources=[source/r for r in [*mf['files'],'PACKAGE_MANIFEST.json']]
    receipt={'run':runid,'result':out.relative_to(repo).as_posix(),'status':'LOCAL_RESULTS_ONLY'}
    if a.publish:
        try:receipt.update(publish_exact(repo,initial,sources,public,f'Run locked three-topic difficulty triage ({runid}); preserve every outcome'))
        except Exception as e:receipt.update(status='PUBLISH_FAILED_NO_AUTORETRY',error=str(e))
    # Post-commit receipt stays in ignored cache; no recursive commit/amend loop.
    write_json(cache/(runid+'_RECEIPT.json'),receipt)
    print(json.dumps(receipt,indent=2,ensure_ascii=False))
    print('Per-topic results:',out/'ALL_TOPICS.json')
    return 0 if receipt['status'] in ('PUSH_VERIFIED','LOCAL_RESULTS_ONLY') else 2
if __name__=='__main__':raise SystemExit(main())
