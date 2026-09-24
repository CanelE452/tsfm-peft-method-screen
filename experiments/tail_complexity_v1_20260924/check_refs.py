#!/usr/bin/env python
"""Quality check on the S2 reference forecasts: how many origins are finite, and how often the
conditional-EVT reference is missing because a GARCH refit did not converge."""
from __future__ import annotations
import numpy as np
import tc

files = sorted((tc.CACHE/'s2_ref').glob('*.npz'))
print('reference files', len(files))
bad = []
n_cal = n_test = 0
finite_evt = finite_t = 0
for f in files:
    d = np.load(f)
    for split in ('cal', 'test'):
        ev = d[f'{split}_cond_evt']; gt = d[f'{split}_garch_t']
        n = ev.shape[0]
        fe = int(np.isfinite(ev).all(axis=1).sum()); ft = int(np.isfinite(gt).all(axis=1).sum())
        if split == 'cal': n_cal += n
        else:
            n_test += n; finite_evt += fe; finite_t += ft
        if fe < 0.9*n: bad.append((f.stem, split, n, fe, ft))
print(f'CAL origins {n_cal}, TEST origins {n_test}')
print(f'TEST finite conditional-EVT {finite_evt} ({finite_evt/max(n_test,1):.4f}), GARCH-t {finite_t} ({finite_t/max(n_test,1):.4f})')
print('symbols with <90% finite reference:', len(bad))
for row in bad[:10]: print('  ', row)
