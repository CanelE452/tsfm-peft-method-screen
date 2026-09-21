from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

NAME = 'rollout_uncertainty_peft_v1_20260921'
ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'experiments' / NAME
RESULTS = ROOT / 'results' / NAME
CACHE = ROOT / '.cache' / NAME

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def json_default(x):
    if hasattr(x, 'item'):
        return x.item()
    if hasattr(x, 'tolist'):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def command(args):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    return {'args': args, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

def event(kind, **fields):
    record = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'kind': kind, **fields}
    with (RESULTS / 'EVENTS.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, default=json_default, allow_nan=False) + '\n')
    print(json.dumps(record, default=json_default), flush=True)
