from __future__ import annotations
from pathlib import Path
import numpy as np
from .util import write_json, pct_gain


def mse(P, Y, w):
    pred = (P * w[None, :, None, None]).sum(1)
    return float(np.mean((pred - Y) ** 2))


def run(payload, cfg, out):
    Pc, Yc = payload['cal_P'], payload['cal_Y']
    Pp, Yp = payload['pilot_P'], payload['pilot_Y']
    native = np.asarray(payload['native_weights'], float)
    uniform = np.ones(Pc.shape[1], dtype=float) / Pc.shape[1]
    single_losses = [mse(Pc, Yc, np.eye(Pc.shape[1])[i]) for i in range(Pc.shape[1])]
    single = np.eye(Pc.shape[1])[int(np.argmin(single_losses))]

    best_loss = float('inf')
    best_w = None
    # Fixed simplex grid 0.05 for three scales.
    for ia in range(21):
        for ib in range(21 - ia):
            ic = 20 - ia - ib
            w = np.asarray([ia, ib, ic], float) / 20.0
            z = mse(Pc, Yc, w)
            if z < best_loss:
                best_loss, best_w = z, w.copy()
    stack = best_w
    methods = {'C_NATIVE': native, 'C_UNIFORM': uniform, 'C_SINGLE': single, 'C_STACK': stack}
    scores = {
        k: {'weights': v.tolist(), 'cal_mse': mse(Pc, Yc, v), 'pilot_mse': mse(Pp, Yp, v)}
        for k, v in methods.items()
    }
    nat = scores['C_NATIVE']['pilot_mse']
    st = scores['C_STACK']['pilot_mse']
    simple = min(scores['C_UNIFORM']['pilot_mse'], scores['C_SINGLE']['pilot_mse'])
    d = cfg['decision']
    if st <= nat * (1 - d['c_native_improvement_min_pct'] / 100.0) and st <= simple * (1 - d['c_simple_margin_pct'] / 100.0):
        decision = 'STACKING_SIGNAL_WORTH_CONFIRMING'
    elif simple <= st:
        decision = 'SIMPLE_BASELINE_FIRST'
    else:
        decision = 'NO_GO_REWEIGHTING'
    result = {
        'scores': scores,
        'stack_improvement_vs_native_pct': pct_gain(nat, st),
        'decision': decision,
        'additional_neural_training': 0,
    }
    write_json(Path(out) / 'C_DECISION.json', result)
    return result
