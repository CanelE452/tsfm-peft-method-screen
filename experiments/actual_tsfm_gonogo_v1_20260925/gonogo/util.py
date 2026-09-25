from __future__ import annotations
import json, hashlib, random, subprocess
from pathlib import Path
import numpy as np
import torch

class ContractError(RuntimeError):
    pass

def require(x, msg):
    if not x:
        raise ContractError(msg)

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
        default=lambda x: float(x) if hasattr(x, 'item') else str(x)), encoding='utf-8')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def pct_gain(old, new):
    return 100.0 * (old - new) / old if old else 0.0

def git(args, cwd, timeout=300):
    p = subprocess.run(['git', *args], cwd=cwd, text=True, capture_output=True, timeout=timeout)
    return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
