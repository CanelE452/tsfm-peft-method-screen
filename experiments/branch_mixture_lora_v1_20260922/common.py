from pathlib import Path
import hashlib
import json
import os
import time

NAME = 'branch_mixture_lora_v1_20260922'
ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'experiments' / NAME
RESULTS = ROOT / 'results' / NAME
CACHE = ROOT / '.cache' / NAME
SEEDS = (92231, 92232)
ARMS = ('COMPONENT', 'MIXTURE')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def default(x):
    if isinstance(x, Path): return str(x)
    if hasattr(x, 'tolist'): return x.tolist()
    if hasattr(x, 'item'): return x.item()
    raise TypeError(type(x).__name__)

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=default, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def event(kind, **fields):
    RESULTS.mkdir(parents=True, exist_ok=True)
    row = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), kind=kind, **fields)
    with (RESULTS / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, default=default, allow_nan=False) + '\n')
    print(json.dumps(row, default=default), flush=True)

def source_hashes():
    return {p.name: sha(p) for p in sorted(EXP.glob('*.py'))}
