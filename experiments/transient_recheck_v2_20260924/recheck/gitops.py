"""Exact-file publication. No force, reset, stash, pull, checkout, or broad staging."""
from __future__ import annotations
import os,re,subprocess
from pathlib import Path
from .common import require,ContractError


def git(repo,*args,timeout=90,check=True):
    p=subprocess.run(['git','-C',str(repo),*args],capture_output=True,text=True,timeout=timeout)
    if check and p.returncode:
        raise ContractError('git '+args[0]+' failed: '+p.stderr[-2000:])
    return p


def empty_index(repo):
    q=git(repo,'diff','--cached','--name-only','-z')
    require(not q.stdout,'EXISTING_STAGED_CHANGES: no execution/publication over another index')


def validate_repository(repo,expected_repository,source_commit,publish):
    repo=Path(repo).resolve()
    root=Path(git(repo,'rev-parse','--show-toplevel').stdout.strip()).resolve()
    require(root==repo,'--repo must be the exact Git root')
    url=git(repo,'remote','get-url','origin').stdout.strip()
    accepted={f'https://github.com/{expected_repository}',f'https://github.com/{expected_repository}.git',
              f'git@github.com:{expected_repository}',f'git@github.com:{expected_repository}.git',
              f'ssh://git@github.com/{expected_repository}',f'ssh://git@github.com/{expected_repository}.git'}
    require(url.lower() in {u.lower() for u in accepted},'Wrong repository origin')
    empty_index(repo)
    branch=git(repo,'symbolic-ref','--short','HEAD').stdout.strip()
    require(branch=='main','Only existing main is supported; no automatic branch switch')
    head=git(repo,'rev-parse','HEAD').stdout.strip()
    require(git(repo,'merge-base','--is-ancestor',source_commit,head,check=False).returncode==0,
            'Pinned source commit missing/not ancestor; no fetch or reset is performed')
    if publish:
        remote=git(repo,'ls-remote','--heads','origin','refs/heads/'+branch,timeout=120).stdout.strip()
        require(remote.split() and remote.split()[0]==head,'Remote/main and local HEAD differ; manual reconciliation required')
    return {'head':head,'branch':branch,'origin':url,'dirty_before':git(repo,'status','--porcelain=v1').stdout,
            'git_dir':git(repo,'rev-parse','--absolute-git-dir').stdout.strip()}


def publish_exact(repo,head,branch,files,commit_message):
    """Caller already validates remote identity. This function is unit-tested on local bare Git."""
    empty_index(repo)
    require(git(repo,'rev-parse','HEAD').stdout.strip()==head,'HEAD changed during execution; refuse publication')
    require(git(repo,'symbolic-ref','--short','HEAD').stdout.strip()==branch,'Branch changed during execution')
    require(files and len(set(files))==len(files),'Invalid publication allowlist')
    for rel in files:
        p=Path(repo)/rel
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(Path(repo).resolve()),'Unsafe publication file')
        require('__pycache__' not in p.parts and p.suffix not in ('.pyc','.pt','.pth','.npz'),'Disallowed artifact')
    git(repo,'diff','--check','--',*files)
    # Repository ignores *.log. Force-add ONLY these validated, bounded source/result files; never directories.
    git(repo,'add','-f','--',*sorted(files))
    staged=set(x for x in git(repo,'diff','--cached','--name-only','-z').stdout.split('\0') if x)
    require(staged==set(files),'Staged files differ from exact new-file allowlist; no commit made')
    git(repo,'diff','--cached','--check')
    git(repo,'commit','-m',commit_message,timeout=120)
    commit=git(repo,'rev-parse','HEAD').stdout.strip()
    p=git(repo,'push','origin',f'HEAD:refs/heads/{branch}',timeout=120,check=False)
    receipt={'commit':commit,'push_returncode':p.returncode,'push_stdout':p.stdout[-3000:],'push_stderr':p.stderr[-3000:],
             'n_published_files':len(files),'branch':branch}
    if p.returncode:
        receipt['status']='COMMITTED_PUSH_FAILED';return receipt
    remote=git(repo,'ls-remote','--heads','origin','refs/heads/'+branch,timeout=120,check=False)
    receipt['status']='PUSH_VERIFIED' if remote.returncode==0 and remote.stdout.split() and remote.stdout.split()[0]==commit else 'PUSH_RETURNED_OK_REMOTE_VERIFICATION_UNAVAILABLE'
    return receipt
