"""Exact-file Git publication. Preserve unrelated changes; include failures.
Generated CSV uses LF. Whitespace checks apply to authored source, not diagnostic
stdout dumps. A git error records BOTH stdout and stderr. Never force/rebase/reset.
"""
from pathlib import Path
import subprocess
from .common import require,ContractError

REPOSITORY='CanelE452/tsfm-peft-method-screen'
def git(repo,*args,check=True,timeout=120):
    p=subprocess.run(['git','-C',str(repo),*args],capture_output=True,text=True,timeout=timeout)
    if check and p.returncode:raise ContractError('git '+args[0]+' failed: stdout='+p.stdout[-4000:]+' stderr='+p.stderr[-4000:])
    return p

def validate(repo,publish):
    require(Path(git(repo,'rev-parse','--show-toplevel').stdout.strip()).resolve()==repo,'Wrong repo root')
    remote=git(repo,'remote','get-url','origin').stdout.strip()
    normalized=remote.removesuffix('.git').replace('git@github.com:','https://github.com/').replace('ssh://git@github.com/','https://github.com/')
    require(normalized.lower()==('https://github.com/'+REPOSITORY).lower(),'Wrong repository origin')
    require(not git(repo,'diff','--cached','--name-only').stdout.strip(),'EXISTING_STAGED_CHANGES')
    head=git(repo,'rev-parse','HEAD').stdout.strip();branch=git(repo,'symbolic-ref','--short','HEAD').stdout.strip()
    require(branch=='main','No automatic branch switch; main required')
    if publish:
        r=git(repo,'ls-remote','--heads','origin','refs/heads/main').stdout.split()
        require(r and r[0]==head,'LOCAL_REMOTE_DIVERGENCE; no automatic pull/rebase')
    return {'head':head,'branch':branch,'dirty_before':git(repo,'status','--porcelain').stdout}

def publish_exact(repo,initial,source_files,result_files,message):
    require(not git(repo,'diff','--cached','--name-only').stdout.strip(),'Index changed during run')
    require(git(repo,'rev-parse','HEAD').stdout.strip()==initial['head'],'HEAD changed during run')
    rels=[];size=0
    for f in sorted(source_files+result_files):
        f=Path(f);require(f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(repo),'Unsafe publish path')
        require(f.suffix in ('.py','.json','.jsonl','.csv','.txt','.md'),'Unsafe publish extension')
        require('__pycache__' not in f.parts and f.stat().st_size<5*1024**2,'Unbounded publish file')
        size+=f.stat().st_size;rels.append(f.relative_to(repo).as_posix())
    require(size<32*1024**2 and len(rels)==len(set(rels)),'Publish size/duplicate violation')
    authored=[f.relative_to(repo).as_posix() for f in source_files if f.suffix in ('.py','.md')]
    if authored:git(repo,'diff','--check','--',*authored)
    # Force-add only audited filenames; never 'git add .' or recursive directory staging.
    git(repo,'add','-f','--',*rels)
    actual=set(git(repo,'diff','--cached','--name-only','-z').stdout.strip('\0').split('\0'))
    require(actual==set(rels),'Unexpected staged file; no commit')
    if authored:git(repo,'diff','--cached','--check','--',*authored)
    git(repo,'commit','-m',message)
    sha=git(repo,'rev-parse','HEAD').stdout.strip()
    p=git(repo,'push','origin','HEAD:refs/heads/main',check=False)
    receipt={'commit':sha,'push_returncode':p.returncode,'stdout':p.stdout[-4000:],'stderr':p.stderr[-4000:],'files':len(rels)}
    if p.returncode:receipt['status']='COMMITTED_PUSH_FAILED';return receipt
    r=git(repo,'ls-remote','--heads','origin','refs/heads/main',check=False)
    receipt['status']='PUSH_VERIFIED' if r.returncode==0 and r.stdout.split() and r.stdout.split()[0]==sha else 'PUSH_OK_VERIFICATION_UNAVAILABLE'
    return receipt
