#!/usr/bin/env python
"""Independent recomputation of every hypothesis table from the series-level CSVs with numpy only
(contract section 9: VERIFY_RECOMPUTE.json, max difference <= 1e-9)."""
from __future__ import annotations
import csv, json, math, time
from pathlib import Path
import numpy as np
import tc

OUT = tc.OUT


def rows(name):
    with (OUT/name).open(encoding='utf-8') as f: return list(csv.DictReader(f))


def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return float('nan')


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float(ra @ rb/math.sqrt((ra @ ra)*(rb @ rb)))


def ols_r2(y, X):
    A = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    res = y - A @ beta
    return beta, float(1 - (res @ res)/(((y - y.mean())**2).sum()))


def gstar_map(scores, arm, tau, extension):
    out = {}
    for r in scores:
        if r['arm'] != arm or abs(fnum(r['tau']) - tau) > 1e-12: continue
        want = extension if tau in tc.OUT_TAUS else 'NONE'
        if r['extension'] != want: continue
        out[(r.get('condition', '') or r.get('symbol', ''), r.get('series', ''))] = fnum(r['gap'])
    return out


def check(panel):
    diff = rows(f'{panel}_DIFFICULTY.csv'); scores = rows(f'{panel}_SCORES.csv')
    reported = tc.read_json(OUT/f'{panel}_HYPOTHESES.json')
    out = {}
    if panel == 'S1':
        out['H1_xi'] = dict(recomputed=spearman([fnum(r['xi']) for r in diff], [fnum(r['xi_true']) for r in diff]),
                            reported=reported['H1']['spearman_xi'])
        out['H1_theta'] = dict(recomputed=spearman([fnum(r['theta']) for r in diff], [fnum(r['theta_true']) for r in diff]),
                               reported=reported['H1']['spearman_theta'])
        pairs = rows('S1_PAIRS.csv')
        out['H2_mean_dSE'] = dict(recomputed=float(np.mean([fnum(r['dSE']) for r in pairs])),
                                  reported=reported['H2']['mean_dSE']['value'])
        out['H2_mean_dxi'] = dict(recomputed=float(np.mean([fnum(r['dxi']) for r in pairs])),
                                  reported=reported['H2']['mean_dxi']['value'])
    else:
        out['H1_xi'] = dict(recomputed=spearman([fnum(r['xi_first']) for r in diff], [fnum(r['xi_second']) for r in diff]),
                            reported=reported['H1']['split_half_xi'])
        out['H1_theta'] = dict(recomputed=spearman([fnum(r['theta_first']) for r in diff], [fnum(r['theta_second']) for r in diff]),
                               reported=reported['H1']['split_half_theta'])
        out['H2_se_xi'] = dict(recomputed=spearman([fnum(r['se']) for r in diff], [fnum(r['xi']) for r in diff]),
                               reported=reported['H2']['spearman_se_xi'])
    idx = {(r.get('condition', '') or r.get('symbol', ''), r.get('series', '')): r for r in diff}
    for arm in ('F0', 'LORA'):
        g = gstar_map(scores, arm, 0.995, 'SEL')
        keys = [k for k, v in g.items() if np.isfinite(v) and k in idx]
        y = np.log1p(np.array([g[k] for k in keys]))
        se = np.array([fnum(idx[k]['se']) for k in keys])
        xi = np.array([fnum(idx[k]['xi']) for k in keys])
        om = 1 - np.array([fnum(idx[k]['theta']) for k in keys])
        ok = np.isfinite(y) & np.isfinite(se) & np.isfinite(xi) & np.isfinite(om)
        b1, r1 = ols_r2(y[ok], np.column_stack([se[ok], xi[ok], om[ok]]))
        _, r0 = ols_r2(y[ok], se[ok][:, None])
        rep = reported['H3'][arm]
        out[f'H3_{arm}_xi_coef'] = dict(recomputed=float(b1[2]), reported=rep['xi_coef'])
        out[f'H3_{arm}_delta_r2'] = dict(recomputed=float(r1-r0), reported=rep['delta_r2'])
        rank = (np.argsort(np.argsort(xi[ok])) + np.argsort(np.argsort(om[ok])))/2.0
        vals = np.array([g[k] for k, o in zip(keys, ok) if o])
        lo, hi = np.quantile(rank, [1/3, 2/3])
        out[f'H5_{arm}_top_median'] = dict(recomputed=float(np.median(vals[rank >= hi])),
                                           reported=reported['H5'][arm]['top_median'])
        out[f'H5_{arm}_bottom_median'] = dict(recomputed=float(np.median(vals[rank <= lo])),
                                              reported=reported['H5'][arm]['bottom_median'])
    for k, v in out.items():
        v['abs_difference'] = abs(v['recomputed'] - v['reported']) if np.isfinite(v['reported']) else float('nan')
    return out


def main():
    res = dict(utc=time.time(), tolerance=1e-9)
    for panel in ('S1', 'S2'):
        if (OUT/f'{panel}_HYPOTHESES.json').exists():
            res[panel] = check(panel)
    diffs = [v['abs_difference'] for panel in ('S1', 'S2') if panel in res for v in res[panel].values()]
    res['max_abs_difference'] = float(np.nanmax(diffs)) if diffs else float('nan')
    res['pass_'] = bool(res['max_abs_difference'] <= 1e-9)
    tc.write_json(OUT/'VERIFY_RECOMPUTE.json', res)
    print('verify max abs difference', res['max_abs_difference'], 'pass', res['pass_'])


if __name__ == '__main__':
    main()
