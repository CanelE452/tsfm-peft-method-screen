import hashlib
import json
import os
from pathlib import Path
import sys
import time

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
NAME = EXP.name
CACHE = ROOT / '.cache' / NAME
RESULTS = ROOT / 'results' / NAME
sys.path.insert(0, str(CACHE / 'python_packages'))
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def event(kind, **kwargs):
    record = {'time': time.time(), 'kind': kind, **kwargs}
    with (RESULTS / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)


class Budget:
    def __init__(self):
        self.path = RESULTS / 'OPTIMIZER_LEDGER.json'
        if not self.path.exists():
            write(self.path, {'smoke': 0, 'main': 0, 'Q': 0, 'T': 0, 'F': 0, 'runs': {}})

    def step(self, optimizer, run, smoke=False):
        d = read(self.path)
        category = 'smoke' if smoke else 'main'
        assert d[category] < (24 if smoke else 8192)
        if not smoke:
            group = run[0]
            assert d[group] < {'Q': 3072, 'T': 2560, 'F': 2560}[group]
            d[group] += 1
        # Reserve before calling step: an interrupted or failed call stays spent.
        d[category] += 1
        d['runs'][run] = d['runs'].get(run, 0) + 1
        write(self.path, d)
        optimizer.step()
