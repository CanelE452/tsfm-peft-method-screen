from pathlib import Path
import hashlib
import json
import os
import time

NAME = 'peft_gonogo12_v1_20260922'
ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'experiments' / NAME
RESULTS = ROOT / 'results' / NAME
CACHE = ROOT / '.cache' / NAME
DESIGN_DIR = ROOT / 'research/peft_hypothesis_reset_20260922/gonogo_v1'
CONFIG = json.loads((DESIGN_DIR / 'DESIGN.json').read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=default, allow_nan=False) + '\n', encoding='utf-8')
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
