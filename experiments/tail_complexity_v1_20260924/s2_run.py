#!/usr/bin/env python
"""S2 financial panel (contract 4.2, 6.1): difficulty on TRAIN, Chronos-2 F0 and 2 LoRA fits,
GARCH-t and conditional-EVT references, CAL extensions, then (after SEAL) TEST scoring."""
from __future__ import annotations
import argparse, csv, time, warnings
import numpy as np
import torch
import tc

warnings.filterwarnings('ignore')
SPLITS = dict(train=('2005-01-01', '2016-12-31'), cal=('2017-01-01', '2018-12-31'), test=('2019-01-01', '2025-06-30'))
BATCH = 256
GARCH_WINDOW = 1000
FIT_KW = dict(prediction_length=1, finetune_mode='lora', lora_config=None, context_length=tc.CONTEXT,
              learning_rate=1e-5, num_steps=1000, batch_size=32, validation_inputs=None)


def load_losses():
    """Adjusted close -> log returns -> loss L_t = -r_t, aligned to trading days (contract 4.2)."""
    z = np.load(tc.CACHE/'s2_prices.npz', allow_pickle=False)
    symbols = [str(s) for s in z['symbols']]
    data = {}
    for s in symbols:
        days = z[f'{s}__days'].astype('datetime64[D]'); adj = z[f'{s}__adj']
        ok = np.isfinite(adj) & (adj > 0)
        days, adj = days[ok], adj[ok]
        r = np.diff(np.log(adj)); d = days[1:]
        data[s] = dict(days=d, loss=-r)
    return symbols, data


def split_mask(days, name):
    a, b = SPLITS[name]
    return (days >= np.datetime64(a)) & (days <= np.datetime64(b))


def difficulty():
    symbols, data = load_losses(); rows = []
    for s in symbols:
        d = data[s]; m = split_mask(d['days'], 'train'); L = d['loss'][m]; days = d['days'][m]
        full = tc.difficulty_from_series(L, 'fin')
        # contract 7-H1 (S2): split-half reliability between 2005-2010 and 2011-2016
        h1 = days <= np.datetime64('2010-12-31'); h2 = days >= np.datetime64('2011-01-01')
        first = tc.difficulty_from_series(L[h1], 'fin'); second = tc.difficulty_from_series(L[h2], 'fin')
        rows.append(dict(symbol=s, n_train=len(L), n_first=int(h1.sum()), n_second=int(h2.sum()), **full,
                         xi_first=first['xi'], xi_second=second['xi'],
                         theta_first=first['theta'], theta_second=second['theta'],
                         se_first=first['se'], se_second=second['se']))
    with (tc.OUT/'S2_DIFFICULTY.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    rel = dict(xi=tc.spearman([r['xi_first'] for r in rows], [r['xi_second'] for r in rows]),
               theta=tc.spearman([r['theta_first'] for r in rows], [r['theta_second'] for r in rows]),
               se=tc.spearman([r['se_first'] for r in rows], [r['se_second'] for r in rows]), n=len(rows))
    with (tc.OUT/'S2_RELIABILITY.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['quantity', 'spearman_first_vs_second_half', 'n']); w.writeheader()
        for k in ('xi', 'theta', 'se'): w.writerow(dict(quantity=k, spearman_first_vs_second_half=rel[k], n=rel['n']))
    print('S2 difficulty rows', len(rows), 'split-half spearman', {k: round(v, 3) for k, v in rel.items() if k != 'n'})


def pipeline(finetuned=None):
    from chronos import Chronos2Pipeline
    from s1_run import REVISION, ensure_pinned_base
    kw = dict(device_map='cuda', torch_dtype=torch.bfloat16)
    if finetuned is None: kw['revision'] = REVISION
    else: ensure_pinned_base(finetuned)
    return Chronos2Pipeline.from_pretrained(str(finetuned) if finetuned else 'amazon/chronos-2', **kw)


def infer(arm: str, seed: int | None = None):
    symbols, data = load_losses()
    tag = arm if seed is None else f'{arm}{seed}'
    dest_dir = tc.CACHE/f's2_{tag}'; dest_dir.mkdir(exist_ok=True)
    pipe = pipeline(tc.CACHE/f's2_lora_seed{seed}_model' if arm == 'lora' else None)
    grid = list(pipe.quantiles); t0 = time.perf_counter()
    for i, s in enumerate(symbols):
        dest = dest_dir/f'{s}.npz'
        if dest.exists(): continue
        d = data[s]; L = d['loss']; days = d['days']
        idx = {name: np.flatnonzero(split_mask(days, name)) for name in ('cal', 'test')}
        out = {}
        for name, ii in idx.items():
            ii = ii[ii >= tc.CONTEXT]                     # need a full context window before each origin
            ctx = [L[o-tc.CONTEXT:o] for o in ii]         # predict L[o] from the previous 512 losses
            preds = []
            for b in range(0, len(ctx), BATCH):
                chunk = [torch.tensor(c, dtype=torch.float32) for c in ctx[b:b+BATCH]]
                q, _ = pipe.predict_quantiles(chunk, prediction_length=1, quantile_levels=grid)
                preds.append(torch.stack([x[0] for x in q]).float().cpu().numpy())
            out[name] = np.concatenate(preds) if preds else np.zeros((0, len(grid)))
            out[name+'_idx'] = ii
        np.savez_compressed(dest, grid=np.array(grid), **out)
        if (i+1) % 10 == 0: print(f'{tag} {i+1}/{len(symbols)} ({time.perf_counter()-t0:.0f}s)', flush=True)
    del pipe; torch.cuda.empty_cache()


def fit_all():
    """2 LoRA fits pooled over all symbols' TRAIN split, seeds 0 and 1 (contract 6.1)."""
    symbols, data = load_losses()
    ledger = tc.CACHE/'s2_fit_ledger.json'
    done = tc.read_json(ledger) if ledger.exists() else {}
    series = []
    for s in symbols:
        d = data[s]; L = d['loss'][split_mask(d['days'], 'train')]
        if len(L) > tc.CONTEXT: series.append(torch.tensor(L, dtype=torch.float32))
    for seed in (0, 1):
        key = f'seed{seed}'
        if key in done: continue
        model_dir = tc.CACHE/f's2_lora_seed{seed}_model'
        if model_dir.exists(): raise SystemExit(f'PARTIAL_FIT {key}')
        pipe = pipeline()
        torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter(); torch.manual_seed(seed)
        tuned = pipe.fit(series, output_dir=str(tc.SCRATCH/f's2_fit_{seed}'), save_strategy='no', seed=seed, **FIT_KW)
        tuned.model.save_pretrained(model_dir)
        done[key] = dict(seconds=time.perf_counter()-t0, peak_vram_gib=torch.cuda.max_memory_allocated()/2**30,
                         series=len(series), seed=seed, steps=FIT_KW['num_steps'])
        tc.write_json(ledger, done)
        print(f'S2 fit seed {seed}: {done[key]["seconds"]:.0f}s', flush=True)
        del pipe, tuned; torch.cuda.empty_cache()


def references():
    """AR(1)-GARCH(1,1)-t and conditional EVT (QML filter + GPD on standardized residuals), refit monthly
    on a rolling 1000-day window that ends strictly before the origin (contract 6.1, leakage rule S0f)."""
    from arch import arch_model
    from scipy.stats import t as student, genpareto
    symbols, data = load_losses(); t0 = time.perf_counter()
    for i, s in enumerate(symbols):
        dest = tc.CACHE/'s2_ref'/f'{s}.npz'
        if dest.exists(): continue
        dest.parent.mkdir(exist_ok=True)
        d = data[s]; L = d['loss']*100.0; days = d['days']
        rows = {name: np.flatnonzero(split_mask(days, name)) for name in ('cal', 'test')}
        res = {}
        for name, ii in rows.items():
            ii = ii[ii >= GARCH_WINDOW]
            months = days[ii].astype('datetime64[M]')
            q_t = np.full((len(ii), len(tc.TAUS)), np.nan); q_evt = np.full_like(q_t, np.nan)
            params_t = params_q = None; last_month = None
            for j, o in enumerate(ii):
                if months[j] != last_month:                      # refit on the first trading day of each month
                    win = L[o-GARCH_WINDOW:o]
                    try:
                        mt = arch_model(win, mean='AR', lags=1, vol='GARCH', p=1, q=1, dist='t').fit(disp='off')
                        mq = arch_model(win, mean='AR', lags=1, vol='GARCH', p=1, q=1, dist='normal').fit(disp='off')
                        z = mq.std_resid[np.isfinite(mq.std_resid)]
                        u = float(np.quantile(z, 0.90)); exc = z[z > u] - u
                        gpd = genpareto.fit(exc, floc=0) if len(exc) >= 30 else None
                        params_t, params_q = (mt, ), (mq, u, gpd, float((z > u).mean()))
                    except Exception:
                        params_t = params_q = None
                    last_month = months[j]
                if params_t is None: continue
                win = L[o-GARCH_WINDOW:o]
                # one-step update without refitting: recompute the conditional mean/variance recursion on the
                # window that ends strictly before this origin, using the parameters of the monthly fit
                mu_t, sig_t, nu = _recurse(params_t[0], win)
                mu_q, sig_q, _ = _recurse(params_q[0], win)
                for ti, tau in enumerate(tc.TAUS):
                    q_t[j, ti] = mu_t + sig_t*student.ppf(tau, nu)/np.sqrt(nu/(nu-2))
                    mq_, u, gpd, rate = params_q
                    if gpd is not None and tau > 1 - rate:
                        zt = u + genpareto.ppf(1 - (1-tau)/rate, gpd[0], loc=0, scale=gpd[2])
                    else:
                        zt = float(np.quantile(mq_.std_resid[np.isfinite(mq_.std_resid)], tau))
                    q_evt[j, ti] = mu_q + sig_q*zt
            res[name+'_idx'] = ii; res[name+'_garch_t'] = q_t/100.0; res[name+'_cond_evt'] = q_evt/100.0
        np.savez_compressed(dest, **res)
        if (i+1) % 10 == 0: print(f'ref {i+1}/{len(symbols)} ({time.perf_counter()-t0:.0f}s)', flush=True)


def _recurse(fitted, window):
    """One-step-ahead conditional mean and volatility from a fitted AR(1)-GARCH(1,1) applied to `window`."""
    p = fitted.params
    ar_key = [k for k in p.index if k.endswith('[1]') and not k.startswith(('alpha', 'beta', 'gamma'))][0]
    mu, phi = float(p['Const']), float(p[ar_key])
    omega, alpha, beta = float(p['omega']), float(p['alpha[1]']), float(p['beta[1]'])
    nu = float(p['nu']) if 'nu' in p.index else np.nan
    resid = window[1:] - (mu + phi*window[:-1])
    sig2 = np.empty(len(resid)); sig2[0] = omega/max(1e-8, 1-alpha-beta)
    for i in range(1, len(resid)):
        sig2[i] = omega + alpha*resid[i-1]**2 + beta*sig2[i-1]
    sig2_next = omega + alpha*resid[-1]**2 + beta*sig2[-1]
    return mu + phi*window[-1], float(np.sqrt(sig2_next)), nu


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['difficulty', 'infer-f0', 'fit', 'infer-lora0', 'infer-lora1', 'references'])
    a = p.parse_args()
    if a.stage == 'difficulty': difficulty()
    elif a.stage == 'infer-f0': infer('f0')
    elif a.stage == 'fit': fit_all()
    elif a.stage == 'infer-lora0': infer('lora', 0)
    elif a.stage == 'infer-lora1': infer('lora', 1)
    else: references()
