"""Toy kernel check only; no model, training, or performance experiment."""
import json
from pathlib import Path
import numpy as np

grids = [np.arange(k, k + 8, dtype=float) for k in (0, 8, 16)]
left = np.array([0., 7.])
center = np.array([3., 4.])
tau = 8.


def kernel(s, t, kind):
    distance = np.abs(s[:, None] - t[None, :]) / tau
    return np.exp(-distance if kind == 'laplace' else -.5 * distance**2).mean()


result = {}
for kind in ('laplace', 'rbf'):
    a = np.array([np.log(kernel(left, g, kind) / kernel(grids[0], g, kind)) for g in grids[1:]])
    b = np.array([np.log(kernel(center, g, kind) / kernel(grids[0], g, kind)) for g in grids[1:]])
    result[kind] = dict(layout_log_bias_difference=(a-b).tolist(), within_row_constant_residual=float(np.ptp(a-b)))
assert len(left) == len(center) and left.mean() == center.mean()
assert result['laplace']['within_row_constant_residual'] < 1e-12
assert result['rbf']['within_row_constant_residual'] > 1e-4
result['scope'] = 'Toy algebra only: zero fits, zero model inference, no predictive evidence'
result['limitation'] = 'One forward half-row example; not a proof that full bidirectional attention always equals a key bias'
Path(__file__).with_name('ALGEBRA_PROBE.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
