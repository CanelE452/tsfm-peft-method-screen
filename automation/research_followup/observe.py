"""Read-only job observation and durable handoff records; never invokes an agent.

No subprocess, model call, training, commit/push, service installation or timer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT/'.cache/research_followup_review'
TERMINAL = {'COMPLETE', 'COMPLETE_WITH_ERRORS', 'FAILED', 'ERROR', 'STOPPED',
            'TIMED_OUT', 'INTERRUPTED', 'INCONCLUSIVE_EXECUTION', 'VERIFICATION_ERROR', 'ARCHIVE_ERROR'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def outcome(queue, verification):
    if queue.get('status') not in TERMINAL:
        return 'WAIT_QUEUE'
    if queue['status'] != 'COMPLETE':
        return 'RECOVERY_REQUIRED'
    if not verification or not str(verification.get('status','')).startswith('VERIFIED'):
        return 'VERIFICATION_REQUIRED'
    return 'FOLLOWUP_REQUIRED'


def event_key(status_path, queue, verification):
    # A heartbeat refresh must not create another continuation request.
    event = dict(path=str(status_path), status=queue.get('status'),
                 started_unix=queue.get('started_unix'), finished_unix=queue.get('finished_unix'),
                 execution_commit=(verification or {}).get('execution_commit'))
    return hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()


def observe(status_path, verification_path=None):
    status_path = Path(status_path).resolve()
    relative = str(status_path.relative_to(ROOT.resolve()))
    queue = read(status_path)
    verification = read(verification_path) if verification_path and Path(verification_path).exists() else None
    state = outcome(queue, verification)
    result = dict(status=state, queue_status=queue.get('status'), status_path=relative,
                  checked_unix=time.time(), automatic_continuation_enabled=False,
                  completed_fits=(verification or {}).get('completed_fits'),
                  training_updates=(verification or {}).get('training_updates'))
    if state != 'WAIT_QUEUE':
        key = event_key(relative, queue, verification)
        request = STATE/'requests'/(key+'.json')
        request.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(event_id=key, **result, queue_sha256=sha(status_path),
                       verification_sha256=sha(verification_path) if verification else None,
                       continuation_status='PENDING_ACTIVATION_APPROVAL')
        try:
            with request.open('x') as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write('\n')
            result['new_request'] = True
        except FileExistsError:
            result['new_request'] = False
        result['request_file'] = str(request)
    write(STATE/'status.json', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--verification', type=Path)
    a = p.parse_args()
    print(json.dumps(observe(a.queue, a.verification), indent=2))


if __name__ == '__main__':
    main()
