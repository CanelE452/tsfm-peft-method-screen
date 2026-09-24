#!/usr/bin/env python
"""S2 extensions on CAL and TEST scoring against the conditional-EVT reference (contract 6.2, 6.3)."""
from __future__ import annotations
import argparse, csv
import numpy as np
import tc
from s2_run import SPLITS, load_losses, split_mask

ARMS = dict(F0=['f0'], LORA=['lora0', 'lora1'])       # two seeds are scored separately and averaged per symbol


def _grid_index(grid):
    return {t: list(grid).index(t) for t in (0.1, 0.5, 0.9, 0.95, 0.99)}


def extend():
    symbols, data = load_losses()
    res = {}
    for arm, tags in ARMS.items():
        per_arm = {}
        for sym in symbols:
            recs = []
            for tag in tags:
                f = tc.CACHE/f's2_{tag}'/f'{sym}.npz'
                if not f.exists(): continue
                d = np.load(f); gi = _grid_index(d['grid']); cal_q = tc.drop_horizon(d['cal'])
                cal_idx = d['cal_idx']; L = data[sym]['loss']
                y_cal = L[cal_idx]
                qc = {t: cal_q[:, gi[t]] for t in gi}
                par = tc.fit_evt_static(qc, y_cal)
                scores = {}
                for name in ('LIN', 'EVT-S'):
                    if name == 'EVT-S' and par is None: scores[name] = float('inf'); continue
                    q995 = tc.lin_extension(qc, 0.995) if name == 'LIN' else tc.evt_extension(qc, par, 0.995)
                    scores[name] = float(np.mean(tc.pinball(q995, y_cal, 0.995)))
                recs.append(dict(tag=tag, evt=par, cal_scores=scores, sel=min(scores, key=scores.get)))
            if recs:
                sel = recs[0]['sel'] if len(recs) == 1 else min(('LIN', 'EVT-S'),
                                                                key=lambda s: np.mean([r['cal_scores'][s] for r in recs]))
                per_arm[sym] = dict(sel=sel, per_tag=recs)
        res[arm] = per_arm
        counts = {s: sum(1 for v in per_arm.values() if v['sel'] == s) for s in ('LIN', 'EVT-S')}
        print(f'S2 extend {arm}: {counts}', flush=True)
    tc.write_json(tc.CACHE/'s2_extension.json', res)


def s2_inputs_digest():
    parts = {'difficulty': tc.sha(tc.OUT/'S2_DIFFICULTY.csv'), 'extension': tc.sha(tc.CACHE/'s2_extension.json')}
    for tag in ('f0', 'lora0', 'lora1'):
        d = tc.CACHE/f's2_{tag}'
        parts[tag] = tc.digest({p.name: tc.sha(p) for p in sorted(d.glob('*.npz'))}) if d.exists() else None
    parts['ref'] = tc.digest({p.name: tc.sha(p) for p in sorted((tc.CACHE/'s2_ref').glob('*.npz'))})
    return tc.digest(parts)


def score():
    seal_path = tc.OUT/'SEAL.json'
    if not seal_path.exists(): raise SystemExit('REFUSED: SEAL.json is missing (contract 8/12)')
    seal = tc.read_json(seal_path)
    if seal['s2']['inputs_digest'] != s2_inputs_digest():
        raise SystemExit('REFUSED: sealed S2 inputs digest does not match the current artifacts')
    symbols, data = load_losses(); ext = tc.read_json(tc.CACHE/'s2_extension.json'); rows = []
    for arm, tags in ARMS.items():
        for sym in symbols:
            if sym not in ext[arm]: continue
            ref = np.load(tc.CACHE/'s2_ref'/f'{sym}.npz')
            L = data[sym]['loss']
            for tag, rec in zip(tags, ext[arm][sym]['per_tag']):
                d = np.load(tc.CACHE/f's2_{tag}'/f'{sym}.npz'); gi = _grid_index(d['grid'])
                test_idx = d['test_idx']
                common, ia, ib = np.intersect1d(test_idx, ref['test_idx'], return_indices=True)
                if len(common) < 100: continue
                y = L[common]
                test_q = tc.drop_horizon(d['test'])[ia]
                qrow = {t: test_q[:, gi[t]] for t in gi}
                par = rec['evt']; sel = ext[arm][sym]['sel']
                for tau in tc.TAUS:
                    ti = tc.TAUS.index(tau)
                    q_evt_ref = ref['test_cond_evt'][ib][:, ti]
                    q_garch_t = ref['test_garch_t'][ib][:, ti]
                    ok = np.isfinite(q_evt_ref)
                    ref_qs = float(np.mean(tc.pinball(q_evt_ref[ok], y[ok], tau)))
                    ref_qs_t = float(np.mean(tc.pinball(q_garch_t[ok], y[ok], tau)))
                    variants = {}
                    if tau in tc.GRID_TAUS:
                        variants['NONE'] = qrow[tau]
                    else:
                        variants['CLAMP'] = qrow[0.99]
                        variants['LIN'] = tc.lin_extension(qrow, tau)
                        if par is not None: variants['EVT-S'] = tc.evt_extension(qrow, par, tau)
                        variants['SEL'] = tc.evt_extension(qrow, par, tau) if (sel == 'EVT-S' and par is not None) else tc.lin_extension(qrow, tau)
                    for vname, q in variants.items():
                        qs = float(np.mean(tc.pinball(q[ok], y[ok], tau)))
                        rows.append(dict(arm=arm, tag=tag, symbol=sym, series='', tau=tau, extension=vname, qs=qs,
                                         qs_ref=ref_qs, gap=tc.gap(qs, ref_qs), qs_ref_garch_t=ref_qs_t,
                                         gap_vs_garch_t=tc.gap(qs, ref_qs_t),
                                         exceed_rate=float(np.mean(y[ok] > q[ok])), n=int(ok.sum()), sel_choice=sel))
    # average the two LoRA seeds per symbol (contract 6.1)
    merged, seen = [], {}
    for r in rows:
        key = (r['arm'], r['symbol'], r['tau'], r['extension'])
        seen.setdefault(key, []).append(r)
    for key, group in seen.items():
        base = dict(group[0])
        for field in ('qs', 'qs_ref', 'gap', 'qs_ref_garch_t', 'gap_vs_garch_t', 'exceed_rate'):
            base[field] = float(np.mean([g[field] for g in group]))
        base['tag'] = '+'.join(g['tag'] for g in group); base['seeds'] = len(group)
        merged.append(base)
    with (tc.OUT/'S2_SCORES.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(merged[0])); w.writeheader(); w.writerows(merged)
    print('S2_SCORES.csv rows', len(merged))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['extend', 'score', 'digest'])
    a = p.parse_args()
    if a.stage == 'extend': extend()
    elif a.stage == 'score': score()
    else: print(s2_inputs_digest())
