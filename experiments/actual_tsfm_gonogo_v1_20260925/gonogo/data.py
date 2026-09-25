from __future__ import annotations
import csv, urllib.request
from pathlib import Path
import numpy as np
from .util import require


def ensure_weather(repo, cfg):
    cache = Path(repo) / '.cache' / 'actual_tsfm_gonogo_v1_20260925'
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / 'weather.csv'
    if not path.exists():
        tmp = cache / 'weather.csv.part'
        if tmp.exists():
            tmp.unlink()
        urllib.request.urlretrieve(cfg['data']['url'], tmp)
        require(tmp.stat().st_size > 1_000_000, 'Weather download too small')
        tmp.replace(path)
    return path


def load_weather(path, cfg):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    require(len(rows) > 10_000 and len(header) > 5, 'Unexpected Weather CSV')
    n_total = len(rows)
    tr = int(n_total * cfg['data']['train_fraction'])
    va = int(n_total * (cfg['data']['train_fraction'] + cfg['data']['val_fraction']))
    # The last 20% TEST rows remain raw strings and are never numerically parsed.
    arr = []
    for row in rows[:va]:
        vals = []
        for x in row[1:]:
            try:
                vals.append(float(x))
            except Exception:
                vals.append(float('nan'))
        arr.append(vals)
    x = np.asarray(arr, np.float32)
    n, c = x.shape
    require(n == va, 'Internal pre-test slice mismatch')
    mu = np.nanmean(x[:tr], axis=0)
    sd = np.nanstd(x[:tr], axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    ii = np.where(~np.isfinite(x))
    x[ii] = mu[ii[1]]
    z = (x - mu) / sd
    cal_mid = tr + (va - tr) // 2
    return {
        'x': z.astype(np.float32), 'n': n, 'n_total': n_total, 'c': c, 'header': header[1:],
        'train_end': tr, 'val_end': va, 'cal_mid': cal_mid,
        'mu': mu.tolist(), 'sd': sd.tolist(), 'test_start': va
    }


def valid_origins(start, stop, L, H):
    a = max(start, L)
    b = stop - H
    require(b > a, 'No legal origins')
    return np.arange(a, b, dtype=np.int64)


def choose_even(origins, n):
    if len(origins) <= n:
        return origins
    return origins[np.linspace(0, len(origins) - 1, n).round().astype(int)]


def sample_batch(x, origins, L, H, batch, rng, channels=None):
    ids = rng.choice(origins, size=batch, replace=len(origins) < batch)
    if channels is None:
        channels = np.arange(x.shape[1])
    ctx = np.stack([x[i-L:i, channels].T for i in ids], 0)
    fut = np.stack([x[i:i+H, channels].T for i in ids], 0)
    return ctx, fut, ids
