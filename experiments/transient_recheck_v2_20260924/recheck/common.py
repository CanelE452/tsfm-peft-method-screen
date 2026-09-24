"""Small auditable IO helpers; no network or silent retry."""
from __future__ import annotations
import csv, gzip, hashlib, io, json, os, time
from pathlib import Path
from datetime import datetime, timezone

class ContractError(RuntimeError):
    pass

def require(condition, message):
    if not condition:
        raise ContractError(message)

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def canonical_hash(obj):
    return sha_bytes(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, obj):
    """Reject NaN instead of producing non-standard JSON."""
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(data, encoding='utf-8'); os.replace(tmp, p)

def write_csv(path, rows, fields):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
        w.writeheader(); w.writerows(rows)

def write_gzip_csv(path, rows, fields):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic gzip header: mtime 0, no local filename.
    with p.open('wb') as f:
        with gzip.GzipFile(fileobj=f, mode='wb', filename='', mtime=0) as g:
            with io.TextIOWrapper(g, encoding='utf-8', newline='') as t:
                w = csv.DictWriter(t, fieldnames=fields, extrasaction='raise')
                w.writeheader(); w.writerows(rows)

class Ledger:
    def __init__(self, path, budgets):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fit_attempts = 0; self.fit_completed = 0
        self.svd_attempts = 0; self.svd_completed = 0
        self.budgets = budgets

    def event(self, kind, **info):
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'utc': utc(), 'kind': kind, **info}, allow_nan=False) + '\n')
            f.flush()

    def begin_fit(self, **info):
        self.fit_attempts += 1
        self.event('FIT_START', attempt=self.fit_attempts, **info)
        require(self.fit_attempts <= self.budgets['max_fit_attempts'], 'Ridge fit budget exceeded')

    def end_fit(self):
        self.fit_completed += 1; self.event('FIT_COMPLETED', count=self.fit_completed)

    def begin_svd(self, **info):
        self.svd_attempts += 1
        self.event('SVD_START', attempt=self.svd_attempts, **info)
        require(self.svd_attempts <= self.budgets['max_svd_attempts'], 'SVD budget exceeded')

    def end_svd(self):
        self.svd_completed += 1; self.event('SVD_COMPLETED', count=self.svd_completed)

    def counts(self):
        return {'ridge_fit_attempts': self.fit_attempts, 'ridge_fit_completed': self.fit_completed,
                'svd_attempts': self.svd_attempts, 'svd_completed': self.svd_completed,
                'neural_fit': 0, 'optimizer_updates': 0, 'simulation_calls': 0,
                'reserve_measurement_files_opened': 0}
