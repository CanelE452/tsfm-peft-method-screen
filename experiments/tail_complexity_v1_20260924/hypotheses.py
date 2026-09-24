#!/usr/bin/env python
"""H1-H5 evaluation and the M0/M1 regressions (contract sections 6.4 and 7).

Reads only saved CSVs. The thresholds are the contract's proposed values and are frozen in SEAL.json.
"""
from __future__ import annotations
import argparse, csv, math
import numpy as np
import tc

TH = dict(h1_s1_xi=0.8, h1_s1_theta=0.6, h1_s2_split_half=0.5, h2_se_gap=0.01, h2_s2_spearman=0.3,
          h3_delta_r2=0.10, h5_top_median=0.10, h5_bottom_median=0.05)


def rows(name):
    return list(csv.DictReader((tc.OUT/name).open(encoding='utf-8')))


def f(x):
    try: return float(x)
    except (TypeError, ValueError): return float('nan')


def gstar(scores, arm, tau=0.995, extension='SEL'):
    """Series/symbol -> G for the given arm at tau with the SEL extension (contract 6.3)."""
    out = {}
    for r in scores:
        if r['arm'] != arm or f(r['tau']) != tau: continue
        if (r['extension'] if tau in tc.OUT_TAUS else 'NONE') != (extension if tau in tc.OUT_TAUS else 'NONE'): continue
        key = (r.get('condition') or r.get('symbol'), r.get('series', ''))
        out[key] = f(r['gap'])
    return out


def design(diff_rows, keys, use_true=False):
    idx = {}
    for r in diff_rows:
        k = (r.get('condition') or r.get('symbol'), r.get('series', ''))
        idx[k] = r
    se = np.array([f(idx[k]['se']) for k in keys])
    xi = np.array([f(idx[k]['xi_true' if use_true else 'xi']) for k in keys])
    th = np.array([f(idx[k]['theta_true' if use_true else 'theta']) for k in keys])
    return se, xi, 1-th, np.array([idx[k].get('condition', '') for k in keys])


def regression_block(diff_rows, scores, arm, tau, strata_on=True, use_true=False, extension='SEL'):
    g = gstar(scores, arm, tau, extension)
    keys = [k for k, v in g.items() if np.isfinite(v)]
    se, xi, one_minus_theta, cond = design(diff_rows, keys, use_true)
    y = np.log1p(np.array([g[k] for k in keys]))
    ok = np.isfinite(y) & np.isfinite(se) & np.isfinite(xi) & np.isfinite(one_minus_theta)
    y, se, xi, om, cond = y[ok], se[ok], xi[ok], one_minus_theta[ok], cond[ok]
    X0 = se[:, None]; X1 = np.column_stack([se, xi, om])
    res = tc.bootstrap_regression(y, X0, X1, strata=cond if strata_on and cond.size else None)
    res.update(arm=arm, tau=tau, extension=extension if tau in tc.OUT_TAUS else 'NONE',
               terms=['intercept', 'SE', 'xi', '1-theta'], estimator='true' if use_true else 'estimated')
    return res, dict(y=y, se=se, xi=xi, om=om, cond=cond)


def coefficient_difference(diff_rows, scores, arm, draws=2000, seed=20260924):
    """H4: xi coefficient at tau=0.995 minus at tau=0.95, bootstrapped on the same resample."""
    g995 = gstar(scores, arm, 0.995, 'SEL'); g95 = gstar(scores, arm, 0.95, 'NONE')
    keys = [k for k in g995 if k in g95 and np.isfinite(g995[k]) and np.isfinite(g95[k])]
    se, xi, om, cond = design(diff_rows, keys)
    y995 = np.log1p(np.array([g995[k] for k in keys])); y95 = np.log1p(np.array([g95[k] for k in keys]))
    ok = np.isfinite(y995) & np.isfinite(y95) & np.isfinite(se) & np.isfinite(xi) & np.isfinite(om)
    y995, y95, se, xi, om, cond = y995[ok], y95[ok], se[ok], xi[ok], om[ok], cond[ok]
    X = np.column_stack([se, xi, om])
    b995, _ = tc.ols(y995, X); b95, _ = tc.ols(y95, X)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(cond == c) for c in np.unique(cond)] if cond.size and cond[0] != '' else [np.arange(len(y995))]
    diffs = np.empty(draws)
    for d in range(draws):
        idx = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in groups])
        bb995, _ = tc.ols(y995[idx], X[idx]); bb95, _ = tc.ols(y95[idx], X[idx])
        diffs[d] = bb995[2] - bb95[2]
    return dict(xi_coef_995=float(b995[2]), xi_coef_95=float(b95[2]), difference=float(b995[2]-b95[2]),
                ci=[float(np.nanquantile(diffs, .025)), float(np.nanquantile(diffs, .975))], n=int(len(y995)))


def h5_terciles(diff_rows, scores, arm):
    g = gstar(scores, arm, 0.995, 'SEL')
    keys = [k for k, v in g.items() if np.isfinite(v)]
    se, xi, om, cond = design(diff_rows, keys)
    ok = np.isfinite(xi) & np.isfinite(om)
    keys = [k for k, o in zip(keys, ok) if o]; xi, om = xi[ok], om[ok]
    rank = (np.argsort(np.argsort(xi)) + np.argsort(np.argsort(om)))/2.0      # contract 5.6
    vals = np.array([g[k] for k in keys])
    lo, hi = np.quantile(rank, [1/3, 2/3])
    top, bottom = vals[rank >= hi], vals[rank <= lo]
    return dict(top_median=float(np.median(top)), bottom_median=float(np.median(bottom)),
                top_n=int(top.size), bottom_n=int(bottom.size),
                top_ci=tc.bootstrap_mean(top, statistic=np.nanmedian)['ci'],
                bottom_ci=tc.bootstrap_mean(bottom, statistic=np.nanmedian)['ci'],
                pass_=bool(np.median(top) >= TH['h5_top_median'] and np.median(bottom) <= TH['h5_bottom_median']))


def evaluate_s1():
    diff = rows('S1_DIFFICULTY.csv'); scores = rows('S1_SCORES.csv'); pairs = rows('S1_PAIRS.csv')
    out, reg = {}, {}
    # H1 measurability
    xi = np.array([f(r['xi']) for r in diff]); xt = np.array([f(r['xi_true']) for r in diff])
    th = np.array([f(r['theta']) for r in diff]); tt = np.array([f(r['theta_true']) for r in diff])
    out['H1'] = dict(spearman_xi=tc.spearman(xi, xt), spearman_theta=tc.spearman(th, tt),
                     threshold=dict(xi=TH['h1_s1_xi'], theta=TH['h1_s1_theta']), n=int(len(diff)))
    out['H1']['pass_'] = bool(out['H1']['spearman_xi'] >= TH['h1_s1_xi'] and out['H1']['spearman_theta'] >= TH['h1_s1_theta'])
    # H2 non-redundancy on spectrum-matched pairs
    dse = np.array([f(r['dSE']) for r in pairs]); dxi = np.array([f(r['dxi']) for r in pairs])
    bse = tc.bootstrap_mean(dse); bxi = tc.bootstrap_mean(dxi)
    se_ok = (bse['ci'][0] <= 0 <= bse['ci'][1]) or abs(bse['value']) < TH['h2_se_gap']
    xi_ok = not (bxi['ci'][0] <= 0 <= bxi['ci'][1])
    out['H2'] = dict(mean_dSE=bse, mean_dxi=bxi, se_no_systematic_difference=bool(se_ok),
                     xi_differs=bool(xi_ok), pairs=int(len(pairs)), pass_=bool(se_ok and xi_ok))
    # H3 predictive validity, H4 mechanism, H5 practical size
    out['H3'] = {}
    for arm in ('F0', 'LORA'):
        res, _ = regression_block(diff, scores, arm, 0.995)
        reg[f'S1_{arm}_tau0.995'] = res
        reg[f'S1_{arm}_tau0.95'] = regression_block(diff, scores, arm, 0.95)[0]
        reg[f'S1_{arm}_tau0.995_true'] = regression_block(diff, scores, arm, 0.995, use_true=True)[0]
        xi_ok = res['coef'][2] > 0 and res['coef_ci'][2][0] > 0
        om_ok = res['coef'][3] > 0 and res['coef_ci'][3][0] > 0
        out['H3'][arm] = dict(xi_coef=res['coef'][2], xi_ci=res['coef_ci'][2], theta_coef=res['coef'][3],
                              theta_ci=res['coef_ci'][3], delta_r2=res['delta_r2'], delta_r2_ci=res['delta_r2_ci'],
                              pass_=bool((xi_ok or om_ok) and res['delta_r2'] >= TH['h3_delta_r2']))
    out['H3']['pass_'] = bool(all(v['pass_'] for k, v in out['H3'].items() if k in ('F0', 'LORA')))
    d = coefficient_difference(diff, scores, 'F0')
    out['H4'] = dict(**d, pass_=bool(d['difference'] > 0 and d['ci'][0] > 0))
    out['H5'] = {arm: h5_terciles(diff, scores, arm) for arm in ('F0', 'LORA')}
    out['H5']['pass_'] = bool(out['H5']['F0']['pass_'] and out['H5']['LORA']['pass_'])
    return out, reg


def evaluate_s2():
    diff = rows('S2_DIFFICULTY.csv'); scores = rows('S2_SCORES.csv')
    out, reg = {}, {}
    xi_r = tc.spearman([f(r['xi_first']) for r in diff], [f(r['xi_second']) for r in diff])
    th_r = tc.spearman([f(r['theta_first']) for r in diff], [f(r['theta_second']) for r in diff])
    out['H1'] = dict(split_half_xi=xi_r, split_half_theta=th_r, threshold=TH['h1_s2_split_half'],
                     n=len(diff), pass_=bool(xi_r >= TH['h1_s2_split_half'] and th_r >= TH['h1_s2_split_half']))
    se = np.array([f(r['se']) for r in diff]); xi = np.array([f(r['xi']) for r in diff]); om = 1-np.array([f(r['theta']) for r in diff])
    s1_, s2_ = tc.spearman(se, xi), tc.spearman(se, om)
    out['H2'] = dict(spearman_se_xi=s1_, spearman_se_one_minus_theta=s2_, threshold=TH['h2_s2_spearman'],
                     pass_=bool(abs(s1_) < TH['h2_s2_spearman'] and abs(s2_) < TH['h2_s2_spearman']))
    out['H3'] = {}
    for arm in ('F0', 'LORA'):
        res, _ = regression_block(diff, scores, arm, 0.995, strata_on=False)
        reg[f'S2_{arm}_tau0.995'] = res
        reg[f'S2_{arm}_tau0.95'] = regression_block(diff, scores, arm, 0.95, strata_on=False)[0]
        xi_ok = res['coef'][2] > 0 and res['coef_ci'][2][0] > 0
        om_ok = res['coef'][3] > 0 and res['coef_ci'][3][0] > 0
        out['H3'][arm] = dict(xi_coef=res['coef'][2], xi_ci=res['coef_ci'][2], theta_coef=res['coef'][3],
                              theta_ci=res['coef_ci'][3], delta_r2=res['delta_r2'], delta_r2_ci=res['delta_r2_ci'],
                              pass_=bool((xi_ok or om_ok) and res['delta_r2'] >= TH['h3_delta_r2']))
    out['H3']['pass_'] = bool(all(v['pass_'] for k, v in out['H3'].items() if k in ('F0', 'LORA')))
    d = coefficient_difference(diff, scores, 'F0')
    out['H4'] = dict(**d, pass_=bool(d['difference'] > 0 and d['ci'][0] > 0))
    out['H5'] = {arm: h5_terciles(diff, scores, arm) for arm in ('F0', 'LORA')}
    out['H5']['pass_'] = bool(out['H5']['F0']['pass_'] and out['H5']['LORA']['pass_'])
    return out, reg


def verdict(s1, s2):
    """Contract section 0 decision tokens, evaluated in the fixed order S0 -> H1 -> ... -> H5."""
    def first_failure(h):
        for k in ('H1', 'H2', 'H3', 'H4', 'H5'):
            if not h[k]['pass_']: return k
        return None
    f1, f2 = first_failure(s1), first_failure(s2)
    if f1 is None and f2 == 'H1':
        # The contract collides here: section 0 line 34 says any H1 failure is TC_UNSTABLE, while
        # section 7 line 205 says "S1 only passes -> TC_SYNTH_ONLY". TC_SYNTH_ONLY asserts that the
        # difficulty axis brings no benefit on financial data, which cannot be asserted while the
        # difficulty estimate on that panel is itself unreliable, so the H1 rule wins and the
        # collision is reported in REPORT_KO.md.
        return 'TC_UNSTABLE', ('S2 failed at H1 while S1 passed every hypothesis; contract section 0 (TC_UNSTABLE) '
                               'and section 7 (TC_SYNTH_ONLY) disagree on this combination and section 0 is applied '
                               'because an unreliable difficulty estimate cannot support a no-benefit claim')
    if f1 == 'H1' or f2 == 'H1': return 'TC_UNSTABLE', f'H1 failed (S1={f1}, S2={f2})'
    if f1 == 'H2' or f2 == 'H2' or f1 == 'H3' or f2 == 'H3': return 'TC_REDUNDANT', f'H2/H3 failed (S1={f1}, S2={f2})'
    if f1 == 'H5' or f2 == 'H5': return 'TC_NO_GAP', f'H5 failed (S1={f1}, S2={f2})'
    if f1 is None and f2 is None: return 'TC_VALID', 'all hypotheses passed on both panels'
    if f1 is None and f2 is not None: return 'TC_SYNTH_ONLY', f'S1 passed, S2 failed at {f2}'
    return 'INCONCLUSIVE_MIXED', f'S1={f1}, S2={f2}'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('panel', choices=['s1', 's2', 'both'])
    a = p.parse_args(); regs = {}
    if a.panel in ('s1', 'both'):
        h, r = evaluate_s1(); tc.write_json(tc.OUT/'S1_HYPOTHESES.json', h); regs.update(r)
        print('S1', {k: (v.get('pass_') if isinstance(v, dict) else v) for k, v in h.items()})
    if a.panel in ('s2', 'both'):
        h2, r2 = evaluate_s2(); tc.write_json(tc.OUT/'S2_HYPOTHESES.json', h2); regs.update(r2)
        print('S2', {k: (v.get('pass_') if isinstance(v, dict) else v) for k, v in h2.items()})
    if regs:
        old = tc.read_json(tc.OUT/'REGRESSIONS.json') if (tc.OUT/'REGRESSIONS.json').exists() else {}
        old.update(regs); tc.write_json(tc.OUT/'REGRESSIONS.json', old)
