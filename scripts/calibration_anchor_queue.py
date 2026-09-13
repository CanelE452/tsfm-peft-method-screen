"""One-command serial queue; bounded attempts, durable state, no implicit retry."""
import argparse
import fcntl
import gzip
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from tsfm_peft_screen.backbone import MODEL_ID, REVISION
from tsfm_peft_screen.reproducibility import ROOT, sha, digest, write_json
from run_calibration_anchor_followup import CFG, OUT, CACHE, oo

TERMINAL = {'COMPLETE', 'COMPLETE_WITH_ERRORS', 'STOPPED', 'TIMED_OUT'}


def stage_data():
    source = json.loads((ROOT/CFG['data_receipt']).read_text())
    directory = CACHE/'data'
    directory.mkdir()
    rows = {}
    for name in CFG['datasets']:
        row = source['datasets'][name]
        path = ROOT/row['file']
        assert sha(path) == row['sha256']
        if name == 'traffic':
            values = np.loadtxt(gzip.open(path, 'rt'), delimiter=',', usecols=range(4), dtype=np.float32)
        else:
            values = pd.read_csv(path).iloc[:, 1:5].to_numpy(dtype=np.float32)
        assert values.shape == (row['rows'], 4) and np.isfinite(values).all()
        scale = np.maximum(values[min(oo('train')):max(oo('train'))+48].std(0, dtype=np.float64), 1e-6)
        dev = directory/(name+'_development.npz')
        held = directory/(name+'_evaluation.npz')
        np.savez_compressed(dev, values=values[:max(oo('validation'))+48], scale=scale)
        np.savez_compressed(held, values=values[:max(oo('evaluation'))+48], scale=scale)
        rows[name] = dict(raw_sha256=sha(path), development_sha256=sha(dev),
            evaluation_sha256=sha(held), train_scale=scale.tolist(),
            scale_interval_half_open=[min(oo('train')), max(oo('train'))+48])
    return rows


def initialize():
    assert not OUT.exists() and not CACHE.exists(), 'Existing experiment: use status; never overwrite/retry'
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip(), 'Commit before execution'
    smoke = json.loads((ROOT/'research/calibration_anchor_20260914/smoke.json').read_text())
    assert smoke['status'] == 'PASS' and smoke['updates'] == CFG['smoke_updates']
    for name, expected in smoke['source_hashes'].items():
        assert sha(ROOT/name) == expected, 'Smoke does not cover current code'
    OUT.mkdir()
    CACHE.mkdir()
    historical = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/'results').rglob('*')
                  if p.is_file() and OUT not in p.parents}
    sources = {str(p.relative_to(ROOT)): sha(p) for folder in ['src', 'configs', 'scripts', 'tests']
               for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    sources[CFG['data_receipt']] = sha(ROOT/CFG['data_receipt'])
    sources['research/calibration_anchor_20260914/smoke.json'] = sha(ROOT/'research/calibration_anchor_20260914/smoke.json')
    from huggingface_hub import hf_hub_download
    model_files = json.loads((ROOT/'results/screening_summary/common_integrity.json').read_text())['model_files']
    assert all(sha(hf_hub_download(MODEL_ID, f, revision=REVISION, local_files_only=True)) == h for f, h in model_files.items())
    contract = dict(execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        config=CFG, source_hashes=sources, historical_result_hashes=historical, model_revision=REVISION,
        model_files=model_files, staged_data=stage_data(), created_unix=time.time(),
        smoke_receipt_sha256=sha(ROOT/'research/calibration_anchor_20260914/smoke.json'))
    write_json(OUT/'contract.json', contract)


def run_queue():
    (ROOT/'.cache').mkdir(exist_ok=True)
    lock = open(ROOT/'.cache/gpu.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another project queue owns .cache/gpu.lock; no duplicate launched')
    initialize()
    started = time.monotonic()
    state = dict(status='RUNNING', pid=os.getpid(), started_unix=time.time(), jobs=[],
                 maximum_wall_seconds=CFG['max_queue_wall_seconds'])
    process = None

    def emit(**kw):
        state.update(kw)
        state['wall_seconds'] = time.monotonic()-started
        write_json(OUT/'queue_status.json', state)

    def stop_child():
        nonlocal process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    def stopped(signum, frame):
        raise KeyboardInterrupt(f'Queue received {signum}')

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)

    def job(topic, phase):
        nonlocal process
        remaining = CFG['max_queue_wall_seconds']-(time.monotonic()-started)
        if remaining <= 0:
            raise TimeoutError('Queue wall budget exhausted')
        phase_limit = min(CFG['max_phase_wall_seconds'], remaining)
        log_path = ROOT/'.cache'/f'{CFG["run_id"]}_{topic}_{phase}.log'
        with open(log_path, 'x') as log:
            cmd = [str(ROOT/'scripts/with_cuda.sh'), str(ROOT/'.venv/bin/python'),
                   str(ROOT/'scripts/run_calibration_anchor_followup.py'), '--topic', topic, '--phase', phase]
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            begin = time.monotonic()
            record = dict(topic=topic, phase=phase, pid=process.pid, log=str(log_path.relative_to(ROOT)),
                          started_unix=time.time())
            emit(current_job=record)
            while process.poll() is None:
                if time.monotonic()-begin > phase_limit:
                    stop_child()
                    record['timeout'] = True
                    break
                emit()
                time.sleep(5)
            record.update(exit_code=process.returncode, wall_seconds=time.monotonic()-begin,
                          finished_unix=time.time())
            state['jobs'].append(record)
            write_json(OUT/topic/(phase+'_job_exit.json'), record)
            process = None
            emit(current_job=None)
            print('JOB EXIT', topic, phase, record['exit_code'], flush=True)
            return record

    try:
        emit()
        barrier = dict(contract_sha256=sha(OUT/'contract.json'), topics={})
        for topic in CFG['topics']:
            record = job(topic, 'train')
            folder = OUT/topic
            receipts = {'train_job_exit.json': sha(folder/'train_job_exit.json')}
            if (folder/'train_status.json').exists():
                receipts['train_status.json'] = sha(folder/'train_status.json')
            if (folder/'selection_seal.json').exists() and record['exit_code'] == 0:
                receipts['selection_seal.json'] = sha(folder/'selection_seal.json')
                decision = json.loads((folder/'selection_seal.json').read_text())['decision']
            else:
                decision = 'INCONCLUSIVE_EXECUTION'
            barrier['topics'][topic] = dict(decision=decision, receipts=receipts)
        barrier['sealed_unix'] = time.time()
        barrier['sha256'] = digest(barrier)
        write_json(OUT/'evaluation_barrier.json', barrier)
        for topic in CFG['topics']:
            if barrier['topics'][topic]['decision'] == 'READY_FOR_E':
                job(topic, 'evaluate')
        cmd = [str(ROOT/'scripts/with_cuda.sh'), str(ROOT/'.venv/bin/python'),
               str(ROOT/'scripts/finalize_calibration_anchor.py')]
        finalizer = subprocess.run(cmd, cwd=ROOT, timeout=600)
        errors = any(r['exit_code'] != 0 for r in state['jobs']) or finalizer.returncode != 0
        emit(status='COMPLETE_WITH_ERRORS' if errors else 'COMPLETE',
             independent_finalization_exit_code=finalizer.returncode, finished_unix=time.time())
    except KeyboardInterrupt as exc:
        stop_child()
        emit(status='STOPPED', error=str(exc), finished_unix=time.time())
    except BaseException as exc:
        stop_child()
        emit(status='TIMED_OUT' if isinstance(exc, TimeoutError) else 'COMPLETE_WITH_ERRORS',
             error=f'{type(exc).__name__}: {exc}', finished_unix=time.time())
        raise


def status():
    path = OUT/'queue_status.json'
    if not path.exists():
        print('NOT_STARTED')
        return
    s = json.loads(path.read_text())
    print(json.dumps(s, indent=2))
    if s['status'] not in TERMINAL:
        age = time.time()-path.stat().st_mtime
        if age > 60:
            print(f'WARNING: heartbeat stale by {age:.0f}s; process may have stopped')
    for topic in CFG['topics']:
        for phase in ['train', 'evaluate']:
            p = OUT/topic/(phase+'_status.json')
            if p.exists():
                print(topic, phase, json.dumps(json.loads(p.read_text())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['start', 'run', 'status'])
    args = parser.parse_args()
    if args.command == 'status':
        return status()
    if args.command == 'run':
        return run_queue()
    if OUT.exists():
        status()
        raise SystemExit('An immutable run already exists; no duplicate or retry')
    log = ROOT/'.cache'/f'{CFG["run_id"]}_queue.log'
    # start_new_session + file-backed stdio survives terminal disconnection.
    with open(log, 'x') as f:
        p = subprocess.Popen([str(ROOT/'scripts/with_cuda.sh'), str(ROOT/'.venv/bin/python'),
            str(ROOT/'scripts/calibration_anchor_queue.py'), 'run'], cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(2)
    if p.poll() is not None:
        print(log.read_text()[-5000:])
        raise SystemExit(p.returncode or 1)
    print(f'Queue launched: pid={p.pid}; log={log}')


if __name__ == '__main__':
    main()
