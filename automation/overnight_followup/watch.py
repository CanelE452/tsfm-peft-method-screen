"""Resume this exact conversation once, only after a verified all-STOP result.

No model invocation during monitoring. No new session, subagent, or model override.
Existing sandbox/approval controls remain enabled. Source pilot files are read-only.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'results/overnight_20260913'
STATE = ROOT/'.cache/overnight_followup'
TOPICS = {'anchor', 'drift', 'distill'}
STOP = {'PILOT_STOP', 'STOP_NO_TEACHER_HEADROOM'}
TERMINAL = {'COMPLETE', 'COMPLETE_WITH_ERRORS', 'STOPPED', 'TIMED_OUT'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eligibility(queue, verification):
    if queue.get('status') not in TERMINAL:
        return 'WAIT_PARENT'
    if queue.get('status') in {'STOPPED', 'TIMED_OUT'}:
        return 'NO_TRIGGER_PARENT_INTERRUPTED'
    if queue.get('status') != 'COMPLETE' or queue.get('independent_finalization_exit_code') != 0:
        return 'NO_TRIGGER_EXECUTION_ERROR'
    if not verification or verification.get('status') != 'VERIFIED':
        return 'NO_TRIGGER_UNVERIFIED'
    rows = verification.get('topics', [])
    if len(rows) != 3 or {r.get('topic') for r in rows} != TOPICS:
        return 'NO_TRIGGER_INCOMPLETE_VERDICTS'
    verdicts = [r.get('verdict') for r in rows]
    if 'PILOT_PASS' in verdicts:
        return 'NO_TRIGGER_PASS_FOUND'
    if all(v in STOP for v in verdicts):
        return 'ELIGIBLE_ALL_STOP'
    return 'NO_TRIGGER_EXECUTION_ERROR'


def session_idle(path, now=None):
    """Positive evidence of an ended turn, followed by 60 seconds of silence."""
    now = time.time() if now is None else now
    if now-path.stat().st_mtime < 60:
        return False
    with path.open('rb') as f:
        size = path.stat().st_size
        f.seek(max(0, size-2*2**20))
        lines = f.read().decode(errors='replace').splitlines()
    boundary = None
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        payload = event.get('payload', {})
        if event.get('type') == 'event_msg':
            kind = payload.get('type')
            if kind in {'task_started', 'task_complete', 'turn_aborted'}:
                boundary = kind
    return boundary in {'task_complete', 'turn_aborted'}


def parent_released():
    status = read(PARENT/'queue_status.json')
    pid = status.get('pid')
    if pid:
        try:
            os.kill(pid, 0)
            return False
        except ProcessLookupError:
            pass
        except PermissionError:
            return False
    with open(ROOT/'.cache/gpu.lock', 'a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def resume_command(codex, session_id):
    # Workspace-write is restricted to the already-authorized project; no bypass,
    # no ignore-rules or ignore-user-config, no new-session fallback.
    uuid.UUID(session_id)
    return [codex, '--ask-for-approval', 'on-request', '--search',
            'exec', '--sandbox', 'workspace-write', '--cd', str(ROOT),
            'resume', session_id, '-', '--json',
            '--output-last-message', str(STATE/'final_message.md')]


def monitor():
    config = read(STATE/'config.json')
    assert sha(__file__) == config['watch_sha256'], 'Watcher changed after arming'
    lock = open(STATE/'watch.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.monotonic()
    state = dict(status='WAIT_PARENT', pid=os.getpid(), armed_unix=config['armed_unix'],
                 exact_session_id=config['session_id'], continuation_attempts=0)
    def emit(label, **kw):
        state.update(status=label, **kw, checked_unix=time.time())
        write(STATE/'status.json', state)
    while True:
        if (STATE/'cancel').exists():
            emit('CANCELLED')
            return
        if time.monotonic()-start > 12*3600:
            emit('WATCH_EXPIRED')
            return
        if (STATE/'trigger.json').exists():
            emit('ALREADY_TRIGGERED_NO_RETRY')
            return
        queue = read(PARENT/'queue_status.json')
        verification = read(PARENT/'verification.json') if (PARENT/'verification.json').exists() else None
        decision = eligibility(queue, verification)
        emit(decision, parent_status=queue.get('status'))
        if decision == 'WAIT_PARENT':
            time.sleep(30)
            continue
        if decision != 'ELIGIBLE_ALL_STOP':
            return
        if not parent_released():
            emit('WAIT_PARENT_PROCESS_EXIT')
            time.sleep(30)
            continue
        if not session_idle(Path(config['session_file'])):
            emit('WAIT_CURRENT_CONVERSATION_IDLE')
            time.sleep(30)
            continue
        with open(STATE/'verify.log', 'x') as log:
            result = subprocess.run([str(ROOT/'scripts/with_cuda.sh'), str(ROOT/'.venv/bin/python'),
                str(ROOT/'scripts/finalize_overnight.py'), '--verify-only'], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, timeout=600)
        if result.returncode:
            emit('NO_TRIGGER_REPLAY_FAILED', replay_exit_code=result.returncode)
            return
        if (STATE/'cancel').exists():
            emit('CANCELLED')
            return
        if not session_idle(Path(config['session_file'])):
            # Do not silently rerun validation or compete with an active user turn.
            emit('READY_BUT_CURRENT_CONVERSATION_ACTIVE')
            while not session_idle(Path(config['session_file'])):
                if (STATE/'cancel').exists():
                    emit('CANCELLED')
                    return
                if time.monotonic()-start > 12*3600:
                    emit('WATCH_EXPIRED')
                    return
                time.sleep(30)
        queue, verification = read(PARENT/'queue_status.json'), read(PARENT/'verification.json')
        if eligibility(queue, verification) != 'ELIGIBLE_ALL_STOP':
            emit('NO_TRIGGER_CHANGED_VERDICT')
            return
        prompt_path = ROOT/'automation/overnight_followup/continuation.md'
        assert sha(prompt_path) == config['prompt_sha256'], 'Continuation instructions changed after arming'
        trigger = dict(session_id=config['session_id'], triggered_unix=time.time(),
            verification_sha256=sha(PARENT/'verification.json'), queue_status_sha256=sha(PARENT/'queue_status.json'),
            verdicts={r['topic']:r['verdict'] for r in verification['topics']},
            prompt_sha256=sha(prompt_path), max_continuation_attempts=1)
        # Atomic, durable one-shot marker: no duplicate even after interruption.
        with open(STATE/'trigger.json', 'x') as f:
            json.dump(trigger, f, indent=2)
        emit('RESUMING_EXISTING_CONVERSATION', continuation_attempts=1)
        cmd = resume_command(config['codex_executable'], config['session_id'])
        with open(prompt_path, 'rb') as prompt, open(STATE/'continuation.jsonl', 'x') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdin=prompt, stdout=log, stderr=subprocess.STDOUT)
            emit('CONTINUATION_RUNNING', continuation_pid=proc.pid, continuation_attempts=1)
            deadline = time.monotonic()+2*3600
            while proc.poll() is None:
                if (STATE/'cancel').exists() or time.monotonic() > deadline:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=10)
                    emit('CONTINUATION_CANCELLED_OR_TIMED_OUT', continuation_exit_code=proc.returncode)
                    return
                time.sleep(30)
                emit('CONTINUATION_RUNNING', continuation_pid=proc.pid)
        # CLI exit 0 does not prove that subsequent experiments completed.
        emit('CONTINUATION_RETURNED' if proc.returncode == 0 else 'CONTINUATION_ERROR',
             continuation_exit_code=proc.returncode,
             final_message_exists=(STATE/'final_message.md').exists(),
             scientific_completion_not_inferred_from_exit_code=True)
        return


def arm(session_id, session_file):
    uuid.UUID(session_id)
    assert session_id in Path(session_file).name
    assert Path(session_file).is_file()
    assert (PARENT/'queue_status.json').exists()
    assert read(PARENT/'queue_status.json')['status'] not in TERMINAL, 'Arm while parent is running'
    STATE.mkdir(exist_ok=True)
    with open(STATE/'config.json', 'x') as f:
        codex = shutil.which('codex')
        assert codex
        json.dump(dict(session_id=session_id, session_file=str(Path(session_file).resolve()),
            armed_unix=time.time(), codex_executable=codex,
            prompt_sha256=sha(ROOT/'automation/overnight_followup/continuation.md'),
            watch_sha256=sha(__file__)), f, indent=2)
    with open(STATE/'watch.log', 'x') as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'watch'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError('Watcher exited; inspect .cache/overnight_followup/watch.log')
    print(f'One-shot follow-up armed; watcher PID {proc.pid}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['arm', 'watch', 'status', 'cancel'])
    p.add_argument('--session-id')
    p.add_argument('--session-file')
    a = p.parse_args()
    if a.command == 'arm':
        arm(a.session_id, a.session_file)
    elif a.command == 'watch':
        try:
            monitor()
        except BaseException as exc:
            write(STATE/'status.json', dict(status='WATCH_ERROR', error=f'{type(exc).__name__}: {exc}',
                checked_unix=time.time(), scientific_verdict_not_inferred=True))
            raise
    elif a.command == 'cancel':
        STATE.mkdir(exist_ok=True)
        (STATE/'cancel').touch(exist_ok=True)
        print('Follow-up cancelled; original experiment queue is unchanged')
    else:
        print(json.dumps(read(STATE/'status.json'), indent=2) if (STATE/'status.json').exists() else 'NOT_ARMED')


if __name__ == '__main__':
    main()
