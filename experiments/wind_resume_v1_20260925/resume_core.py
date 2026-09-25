"""Author-approved, one-time WIND acquisition retry; original science is immutable.

Standard-library orchestration only. Never remove the old .started marker, overwrite
prior results, change the scientific config, or execute other topic workers.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

NAME = 'wind_resume_v1_20260925'
AUTHORIZATION = 'wind-explicit-local-files-after-c085531-once-v1'
ALLOWED_SUFFIXES = {'.py', '.json', '.jsonl', '.csv', '.txt', '.md'}

class ContractError(RuntimeError):
    pass

def require(condition, message):
    if not condition:
        raise ContractError(message)

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def git(repo, *args, check=True, timeout=120):
    p = subprocess.run(['git', '-C', str(repo), *args], capture_output=True,
                       text=True, timeout=timeout)
    if check and p.returncode:
        raise ContractError('git '+args[0]+' failed: stdout='+p.stdout[-4000:]+' stderr='+p.stderr[-4000:])
    return p

def git_blob_sha(path):
    b = Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()

def verify_package(root):
    manifest = read_json(root / 'PACKAGE_MANIFEST.json')
    require(manifest['name'] == NAME, 'Wrong retry package')
    for rel, wanted in manifest['files'].items():
        p = root / rel
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts, 'Unsafe manifest path')
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve()), 'Missing/unsafe package file: '+rel)
        require('__pycache__' not in p.parts and p.suffix != '.pyc', 'Derived bytecode cannot be pinned')
        require(digest(p) == wanted, 'PACKAGE_MISMATCH: '+rel)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    require(actual == set(manifest['files']) | {'PACKAGE_MANIFEST.json'}, 'Unexpected package files')
    return manifest

def verify_engine(repo, pin):
    source = repo / pin['source_path']
    require(source.is_dir() and not source.is_symlink(), 'Original topic_axis package missing; do not switch to domain_axis')
    for rel, wanted in pin['source_files_sha256'].items():
        p = source / rel
        require(p.is_file() and not p.is_symlink(), 'Missing pinned original file: '+rel)
        require(digest(p) == wanted, 'ORIGINAL_SOURCE_CHANGED: '+rel)
    require(not git(repo, 'ls-files', '--others', '--exclude-standard', '--', str(source.relative_to(repo))).stdout.strip(),
            'Unexpected untracked original source files')
    cfg = read_json(source / 'RUN_CONFIG.json')
    require(cfg['topic_timeout_seconds'] == 900, 'Original time budget changed')
    require(cfg['new_lora_fit'] is False and cfg['new_peft_fit'] is False, 'Unexpected adaptation scope')
    return source, cfg

def snapshot_tree(root):
    require(root.is_dir() and not root.is_symlink(), 'Missing prior result directory')
    out = {}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(), 'Symlink in prior results')
        if p.is_file():
            out[p.relative_to(root).as_posix()] = digest(p)
    return out

def verify_parent(repo, pin):
    parent = repo / pin['parent_result_path']
    wind_path = parent / 'wind' / 'RESULT.json'
    require(git_blob_sha(wind_path) == pin['parent_wind_git_blob_sha1'], 'Parent WIND result changed')
    result = read_json(wind_path)
    require(result['axis_status'] == 'BLOCKED_DATA_OR_ENVIRONMENT', 'Retry not authorized for scored WIND run')
    require(result['reason'] == pin['allowed_parent_reason'], 'Parent failure is not the approved acquisition failure')
    require(result.get('scientific_no_go') is False, 'Parent marked scientific failure')
    for forbidden in ('PRE_TEST_SEAL.json', 'SCORE_ROWS.csv', 'DATA_QC.json'):
        require(not (parent / 'wind' / forbidden).exists(), 'Prior WIND advanced beyond acquisition: '+forbidden)
    return parent, snapshot_tree(parent)

def inspect_input(path, entry):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing input CSV: '+str(path))
    require(path.name == entry['name'], 'Do not rename input to masquerade as official file')
    require(path.stat().st_size == entry['bytes'], 'Official file byte-size mismatch: '+path.name)
    sha = hashlib.sha256(); md5 = hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(block); md5.update(block)
    return {'filename':path.name, 'bytes':path.stat().st_size,
            'sha256':sha.hexdigest(), 'md5':md5.hexdigest()}

def check_cached_metadata(meta_path, entries, inputs):
    # Original fetcher previously read this fixed API version. No new request/download.
    require(meta_path.is_file() and not meta_path.is_symlink(),
            'MISSING_ORIGINAL_FIGSHARE_METADATA: return receipt; do not fetch a different release')
    meta = read_json(meta_path)
    files = meta.get('files', [])
    selected = []
    for key, entry in entries.items():
        matches = [x for x in files if x.get('name') == entry['name'] and x.get('size') == entry['bytes']]
        require(len(matches) == 1, 'Official cached metadata selection is not unique: '+key)
        f = matches[0]
        expected = f.get('computed_md5') or f.get('supplied_md5')
        require(isinstance(expected,str) and re.fullmatch('[0-9a-fA-F]{32}', expected) is not None,
                'No usable official MD5 for '+key)
        require(inputs[key]['md5'] == expected.lower(), 'MD5_MISMATCH: '+key)
        selected.append({'role':key,'name':f['name'],'bytes':f['size'],'md5':expected.lower(),
                         'file_id':f.get('id'),'download_url':f.get('download_url')})
    return {'metadata_sha256':digest(meta_path), 'selected_files':selected,
            'verification_scope':'matches original local cache of fixed Figshare API version; no fresh server verification'}

def claim_once(cache, record):
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / (AUTHORIZATION+'.started.json')
    try:
        with marker.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(record, f, ensure_ascii=False, indent=2); f.write('\n')
    except FileExistsError as e:
        raise ContractError('THIS_WIND_RETRY_ALREADY_STARTED; preserve marker and return '+str(marker)) from e
    return marker

def worker_command(py, source, repo, private_cache, public, csv_path, locations):
    # No --topic supplied by CLI. This literal WIND call cannot run other topics.
    return [str(py), '-B', '-m', 'triage.worker', '--topic', 'wind',
            '--repo', str(repo), '--cache', str(private_cache), '--public', str(public),
            '--config', str(source / 'RUN_CONFIG.json'),
            '--wind-csv', str(csv_path), '--wind-locations', str(locations)]

def run_worker(command, source, public, timeout=900):
    require(command[command.index('--topic')+1] == 'wind', 'Only WIND is authorized')
    require(timeout == 900, 'Must preserve original worker time budget')
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(source),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    started = time.monotonic()
    with (public / 'CONSOLE.txt').open('w', encoding='utf-8', newline='\n') as f:
        p = subprocess.Popen(command, cwd=source, env=env, stdout=f,
                             stderr=subprocess.STDOUT, start_new_session=True)
        timeout_hit = False
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timeout_hit = True
            os.killpg(p.pid, signal.SIGTERM)
            try: p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
            rc = 124
    result_path = public / 'RESULT.json'
    if timeout_hit:
        # Never overwrite a result already emitted at the timeout boundary.
        if not result_path.exists():
            write_json(result_path, {'topic':'wind','axis_status':'BLOCKED_RESOURCE_TIMEOUT',
                                    'scientific_no_go':False,'timeout_s':timeout})
    if not result_path.exists():
        write_json(result_path, {'topic':'wind','axis_status':'IMPLEMENTATION_OR_CONTRACT_ERROR',
                                'scientific_no_go':False,'reason':'worker exited without RESULT.json'})
    return {'command':command,'returncode':rc,'timeout':timeout_hit,
            'elapsed_s':time.monotonic()-started,'only_topic':'wind'}

def copy_retry_package(pkg, repo, manifest):
    dest = repo / 'experiments' / NAME
    require(not dest.exists(), 'Retry source destination already exists; do not overwrite/retry')
    dest.mkdir(parents=True)
    for rel in [*manifest['files'], 'PACKAGE_MANIFEST.json']:
        p = dest / rel; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((pkg / rel).read_bytes())
    return dest

def preserve_checks(repo, pin, source, parent, old_snapshot, original_marker, old_marker_hash, paths, input_info):
    verify_engine(repo, pin)
    require(snapshot_tree(parent) == old_snapshot, 'PRIOR_RESULTS_CHANGED_DURING_RETRY')
    require(original_marker.is_file() and digest(original_marker) == old_marker_hash,
            'ORIGINAL_STARTED_MARKER_CHANGED')
    for key, path in paths.items():
        require(digest(path) == input_info[key]['sha256'], 'INPUT_CHANGED_DURING_RETRY: '+key)

def execute(pkg, args):
    pkg = pkg.resolve(); repo = Path(args.repo).resolve()
    manifest = verify_package(pkg); pin = read_json(pkg / 'ENGINE_PIN.json')
    cache = repo / '.cache' / NAME
    # All preconditions are read-only in the tracked repository.
    source, cfg = verify_engine(repo, pin)
    sys.path.insert(0, str(source))
    from triage.gitops import validate, publish_exact
    initial = validate(repo, args.publish)
    require(git(repo,'merge-base','--is-ancestor',pin['parent_commit'],initial['head'],check=False).returncode==0,
            'Parent run commit is not an ancestor of current checkout')
    parent, old_snapshot = verify_parent(repo, pin)
    original_marker = repo / pin['original_marker_path']
    require(original_marker.is_file() and not original_marker.is_symlink(), 'Original .started marker missing; do not reconstruct it')
    old_marker_hash = digest(original_marker)
    paths = {'wind_csv':Path(args.wind_csv).resolve(), 'wind_locations':Path(args.wind_locations).resolve()}
    for p in paths.values():
        if p.is_relative_to(repo):
            require(not git(repo, 'ls-files', '--', p.relative_to(repo).as_posix()).stdout.strip(), 'Raw input must not be tracked in Git')
    input_info = {key:inspect_input(path,pin['inputs'][key]) for key,path in paths.items()}
    metadata = check_cached_metadata(repo / pin['original_metadata_path'], pin['inputs'], input_info)
    record = {'authorization':AUTHORIZATION,'created':utc(),'parent_commit':pin['parent_commit'],
              'parent_run':pin['parent_result_path'],'original_science_config_sha256':digest(source/'RUN_CONFIG.json'),
              'package_sha256':digest(pkg/'PACKAGE_MANIFEST.json'),'topic':'wind','inputs':input_info,
              'old_marker_sha256':old_marker_hash,'attempt_limit':1}
    marker = claim_once(cache, record)
    runid = 'run_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = repo / 'results' / NAME / runid
    out.mkdir(parents=True, exist_ok=False)
    public = out / 'wind'; public.mkdir()
    write_json(cache / 'LATEST_ATTEMPT.json', {'marker':str(marker),'result_relative_path':out.relative_to(repo).as_posix()})
    copied = None; process = None; failure = None; integrity_ok = False
    t0 = time.monotonic()
    try:
        copied = copy_retry_package(pkg, repo, manifest)
        write_json(out / 'RETRY_AUTHORIZATION.json',record)
        write_json(out / 'INPUT_VERIFICATION.json', {'files':input_info,**metadata})
        write_json(out / 'PRIOR_ARTIFACTS_SHA256.json',old_snapshot)
        py = repo / '.venv' / 'bin' / 'python'
        require(py.is_file(), 'Existing repository .venv required')
        cmd = worker_command(py, source, repo, cache/runid/'worker_cache',public,paths['wind_csv'],paths['wind_locations'])
        process = run_worker(cmd,source,public,timeout=cfg['topic_timeout_seconds'])
        write_json(public/'PROCESS.json',process)
    except Exception as e:
        import traceback
        failure = str(e)
        (out/'EXECUTION_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf-8',newline='\n')
    try:
        preserve_checks(repo,pin,source,parent,old_snapshot,original_marker,old_marker_hash,paths,input_info)
        if copied:
            verify_package(copied)
        integrity_ok = True
    except Exception as e:
        failure = (failure+'; ' if failure else '')+str(e)
        write_json(out/'INTEGRITY_FAILURE.json',{'error':str(e),'publish_allowed':False})
    # A new summary, not a replacement or a merge overwriting previous topic outputs.
    result = read_json(public/'RESULT.json') if (public/'RESULT.json').is_file() else {
        'topic':'wind','axis_status':'IMPLEMENTATION_OR_CONTRACT_ERROR','scientific_no_go':False,'reason':failure}
    write_json(out/'WIND_ONLY.json',{'wind':result,'other_topics':'NOT_RERUN; see immutable parent run',
               'parent_run':pin['parent_result_path'],'failure':failure})
    write_json(out/'RUN_MANIFEST.json',{'authorization':AUTHORIZATION,'run_id':runid,'git_start':initial,
               'original_config':cfg,'original_config_sha256':digest(source/'RUN_CONFIG.json'),
               'executed_topics':['wind'] if process else [],'process':process,'elapsed_s':time.monotonic()-t0,
               'original_lock_preserved':integrity_ok,'original_results_and_source_preserved':integrity_ok,
               'input_hashes_preserved':integrity_ok,'failure':failure,'scientific_parameters_changed':False,
               'prior_results_are_not_new_replications':True,'packages_installed_by_launcher':0,
               'dataset_downloads_by_launcher':0,'new_lora_fit':0,'new_peft_fit':0})
    receipt={'run':runid,'result':out.relative_to(repo).as_posix(),'status':'LOCAL_RESULTS_ONLY',
             'worker_returncode':None if process is None else process['returncode'],
             'axis_status':result.get('axis_status'),'action':result.get('action'),
             'authorization_marker':str(marker),'parent_run':pin['parent_result_path']}
    if not integrity_ok:
        receipt.update(status='PUBLISH_REFUSED_INTEGRITY_FAILURE',error=failure)
    elif args.publish:
        try:
            authored = [] if copied is None else [copied/r for r in [*manifest['files'],'PACKAGE_MANIFEST.json']]
            produced=[p for p in out.rglob('*') if p.is_file() and p.suffix in ALLOWED_SUFFIXES]
            receipt.update(publish_exact(repo,initial,authored,produced,
                 f'Resume blocked WIND acquisition once ({runid}); unchanged topic-axis protocol'))
        except Exception as e:
            receipt.update(status='PUBLISH_FAILED_NO_AUTORETRY',error=str(e))
    write_json(cache/(runid+'_RECEIPT.json'),receipt)
    write_json(cache/'EXECUTION_RECEIPT.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
    if receipt['status'] not in ('PUSH_VERIFIED','LOCAL_RESULTS_ONLY'):
        return 2
    return 0 if process and process['returncode']==0 and failure is None else 3
