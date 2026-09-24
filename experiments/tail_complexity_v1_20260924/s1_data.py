#!/usr/bin/env python
"""S1 step 1: generate the synthetic panel, measure difficulty on the training window only,
compute the oracle (contract 4.1), run S0c (oracle exceedance rate) and build the matched pairs table."""
from __future__ import annotations
import csv, time
import numpy as np
import tc

COND_INDEX = {c: i for i, c in enumerate(tc.CONDITIONS)}


def parse_condition(name: str):
    phi_s, eps, g = name.split('_')
    return float(phi_s[3:]), eps, g[1:]


def series_seed(cond: str, k: int) -> int:
    return tc.SEED_BASE + 1000*(COND_INDEX[cond]+1) + k


def build():
    t0 = time.perf_counter()
    rows, pairs_rows = [], []
    truth = {}
    origins = np.arange(tc.LEN_TRAIN, tc.LEN_TOTAL - 1, tc.EVAL_STRIDE)     # 200 evaluation origins, h = 1
    cal_origins = np.arange(tc.LEN_FIT, tc.LEN_TRAIN - 1)                   # CAL rolling origins (contract 4.1)
    store = {}
    for cond in tc.CONDITIONS:
        phi, eps, garch = parse_condition(cond)
        # "true" xi/theta from one very long series with the same measurement procedure (contract 4.1)
        long_sim = tc.simulate(phi, eps, garch, 2_000_000, seed=tc.SEED_BASE + 77_000 + COND_INDEX[cond])
        long_diff = tc.difficulty_from_series(long_sim['y'], 'synth')
        truth[cond] = dict(xi_true=long_diff['xi'], theta_true=long_diff['theta'], se_true=long_diff['se'],
                           truth_ar_order=long_diff['ar_order'], truth_n_exceedances=long_diff['n_exceedances'])
        del long_sim
        ys, oracles, cal_oracles = [], [], []
        for k in range(tc.N_SERIES):
            sim = tc.simulate(phi, eps, garch, tc.LEN_TOTAL, seed=series_seed(cond, k), keep_state=True)
            y = sim['y']
            d = tc.difficulty_from_series(y[:tc.LEN_TRAIN], 'synth')
            rows.append(dict(condition=cond, series=k, phi=phi, eps=eps, garch=garch, **d, **truth[cond]))
            ys.append(y)
            oracles.append(tc.oracle_quantiles(sim, phi, eps, origins))
            cal_oracles.append(tc.oracle_quantiles(sim, phi, eps, cal_origins))
        store[cond] = dict(y=np.stack(ys), oracle=np.stack(oracles), cal_oracle=np.stack(cal_oracles))
        print(f'{cond}: xi_true={truth[cond]["xi_true"]:.3f} theta_true={truth[cond]["theta_true"]:.3f} '
              f'({time.perf_counter()-t0:.0f}s)', flush=True)
    np.savez_compressed(tc.CACHE/'s1_panel.npz', origins=origins, cal_origins=cal_origins,
                        **{f'{c}__{k}': store[c][k] for c in store for k in ('y', 'oracle', 'cal_oracle')})
    tc.write_json(tc.CACHE/'s1_truth.json', truth)
    with (tc.OUT/'S1_DIFFICULTY.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # S0c: oracle exceedance rate at tau = 0.99 over the evaluation window, per condition
    s0c = {}
    for cond in tc.CONDITIONS:
        y = store[cond]['y'][:, origins+1]                       # realised value one step after each origin
        q99 = store[cond]['oracle'][:, :, tc.TAUS.index(0.99)]
        rate = float((y > q99).mean())
        s0c[cond] = dict(exceedance_rate=rate, n=int(y.size), pass_=bool(0.006 <= rate <= 0.014))
    # matched pairs (same phi -> same spectrum by construction, contract 4.1/7-H2)
    diff = {(r['condition'], r['series']): r for r in rows}
    for phi in tc.PHIS:
        conds = [c for c in tc.CONDITIONS if parse_condition(c)[0] == phi]
        base = conds[0]
        for other in conds[1:]:
            for k in range(tc.N_SERIES):
                a, b = diff[(base, k)], diff[(other, k)]
                pairs_rows.append(dict(phi=phi, cond_a=base, cond_b=other, series=k,
                                       dSE=b['se']-a['se'], dxi=b['xi']-a['xi'], dtheta=b['theta']-a['theta'],
                                       se_a=a['se'], se_b=b['se'], xi_a=a['xi'], xi_b=b['xi']))
    with (tc.OUT/'S1_PAIRS.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(pairs_rows[0])); w.writeheader(); w.writerows(pairs_rows)
    tc.write_json(tc.CACHE/'s0c_oracle.json', s0c)
    print('S0c oracle exceedance at tau=0.99:', {c: round(v['exceedance_rate'], 4) for c, v in s0c.items()})
    print('S0c pass:', all(v['pass_'] for v in s0c.values()), f'total {time.perf_counter()-t0:.0f}s')


if __name__ == '__main__':
    build()
