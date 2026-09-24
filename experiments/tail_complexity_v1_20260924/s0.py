#!/usr/bin/env python
"""S0 tool verification (contract section 8). S0a/S0b come from premise_audit.py; this adds S0c-S0f
and writes S0_REPORT.json. Any failure means STOP_DEBUG."""
from __future__ import annotations
import hashlib, json, time, urllib.request
import numpy as np
import tc

UA = {'User-Agent': 'Mozilla/5.0 (research; tail-complexity TCV-1)'}
FIN_START, FIN_END = '2005-01-01', '2025-06-30'
SPLITS = dict(train=('2005-01-01', '2016-12-31'), cal=('2017-01-01', '2018-12-31'), test=('2019-01-01', '2025-06-30'))


def get(url, timeout=40):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def s0d_pinball():
    """Three hand-computed pinball values (contract 6.3): QS = (1{y<q} - tau)(q - y)."""
    cases = [dict(q=1.0, y=0.0, tau=0.9, expected=(1-0.9)*(1.0-0.0)),      # y < q  -> 0.1*1  = 0.1
             dict(q=1.0, y=2.0, tau=0.9, expected=(0-0.9)*(1.0-2.0)),      # y >= q -> -0.9*-1 = 0.9
             dict(q=-0.5, y=-2.0, tau=0.995, expected=(1-0.995)*(-0.5+2.0))]
    rows = []
    for c in cases:
        got = float(tc.pinball(np.array([c['q']]), np.array([c['y']]), c['tau'])[0])
        rows.append(dict(**c, got=got, pass_=abs(got-c['expected']) < 1e-12))
    return dict(cases=rows, pass_=all(r['pass_'] for r in rows))


def sp100_symbols():
    page = get('https://en.wikipedia.org/wiki/S%26P_100').decode('utf-8', 'replace')
    import re
    block = page.split('id="Components"')[-1]
    start = block.find('<table'); end = block.find('</table>', start)
    table = block[start:end]
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table, flags=re.S)
    syms = []
    for row in rows:                      # the symbol is the plain text of the first cell
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, flags=re.S)
        if not cells: continue
        text = re.sub(r'<[^>]+>', '', cells[0]).strip()
        if re.fullmatch(r'[A-Z][A-Z\.\-]{0,5}', text): syms.append(text)
    seen, out = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s); out.append(s)
    return out, hashlib.sha256(page.encode()).hexdigest(), len(page)


def fetch_prices(symbol):
    raw = get(f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol.replace(".", "-")}'
              f'?period1=1104537600&period2=1751241600&interval=1d&events=div%2Csplit')
    d = json.loads(raw)['chart']['result'][0]
    ts = np.array(d['timestamp'], dtype='int64')
    adj = np.array([np.nan if v is None else v for v in d['indicators']['adjclose'][0]['adjclose']], dtype=float)
    days = ts.astype('datetime64[s]').astype('datetime64[D]')
    return days, adj, hashlib.sha256(raw).hexdigest(), len(raw)


def s0e_financial_snapshot(reuse_cache=True):
    symbols, page_hash, page_bytes = sp100_symbols()
    t0 = time.perf_counter(); kept, dropped, records, store, raw_meta = [], {}, {}, {}, {}
    cached = None
    cache_file = tc.CACHE/'s2_prices_raw.npz'
    if reuse_cache and cache_file.exists():
        cached = np.load(cache_file, allow_pickle=False)
    for i, sym in enumerate(symbols):
        if cached is not None and f'{sym}__days' in cached:
            days = cached[f'{sym}__days'].astype('datetime64[D]'); adj = cached[f'{sym}__adj']
            h, nbytes = str(cached[f'{sym}__sha'][0]), int(cached[f'{sym}__bytes'][0])
        else:
            try:
                days, adj, h, nbytes = fetch_prices(sym)
            except Exception as e:
                dropped[sym] = f'fetch_failed: {type(e).__name__}'; continue
            time.sleep(0.15)
        store[sym] = (days, adj); raw_meta[sym] = (h, nbytes)
        if (i+1) % 20 == 0: print(f'  prices {i+1}/{len(symbols)} ({time.perf_counter()-t0:.0f}s)', flush=True)
    np.savez_compressed(cache_file, **{f'{s}__days': store[s][0].astype('int64') for s in store},
                        **{f'{s}__adj': store[s][1] for s in store},
                        **{f'{s}__sha': np.array([raw_meta[s][0]]) for s in store},
                        **{f'{s}__bytes': np.array([raw_meta[s][1]]) for s in store})
    # trading-day coverage per split, estimated by the cross-sectional median (contract 4.2: <5% missing)
    counts = {name: [] for name in SPLITS}
    for sym, (days, adj) in store.items():
        for name, (a, b) in SPLITS.items():
            counts[name].append(int(((days >= np.datetime64(a)) & (days <= np.datetime64(b))).sum()))
    expected = {name: float(np.median(v)) for name, v in counts.items()}
    half_expected = float(np.median([int(((d >= np.datetime64('2005-01-01')) & (d <= np.datetime64('2010-12-31'))).sum())
                                     for d, _ in store.values()]))
    for sym, (days, adj) in store.items():
        ok = True; miss = {}
        for name, (a, b) in SPLITS.items():
            m = (days >= np.datetime64(a)) & (days <= np.datetime64(b))
            n = int(m.sum()); nan = float(np.isnan(adj[m]).mean()) if n else 1.0
            cover = n/expected[name] if expected[name] else 0.0
            miss[name] = dict(rows=n, nan_fraction=nan, coverage=cover)
            if n == 0 or nan >= 0.05 or cover < 0.95: ok = False
        first_half = int(((days >= np.datetime64('2005-01-01')) & (days <= np.datetime64('2010-12-31'))).sum())
        miss['train_first_half'] = dict(rows=first_half, coverage=first_half/half_expected if half_expected else 0.0)
        if miss['train_first_half']['coverage'] < 0.95: ok = False      # H1 split-half needs both halves
        h, nbytes = raw_meta[sym]
        if ok:
            kept.append(sym); records[sym] = dict(sha256=h, bytes=nbytes, rows=int(len(days)),
                                                  first=str(days[0]), last=str(days[-1]), splits=miss)
        else:
            dropped[sym] = miss
    np.savez_compressed(tc.CACHE/'s2_prices.npz', symbols=np.array(kept),
                        **{f'{s}__days': store[s][0].astype('int64') for s in kept},
                        **{f'{s}__adj': store[s][1] for s in kept})
    return dict(source='https://query1.finance.yahoo.com/v8/finance/chart/{symbol} (adjclose, daily)',
                universe_source='https://en.wikipedia.org/wiki/S%26P_100', universe_sha256=page_hash,
                universe_bytes=page_bytes, symbols_listed=len(symbols), eligible=len(kept), eligible_symbols=kept,
                dropped=dropped, splits=SPLITS, records=records, seconds=time.perf_counter()-t0,
                eligibility_rule='per split: >=95% of the cross-sectional median trading-day count and <5% NaN; '
                                 'the 2005-2010 half must also reach 95% coverage because H1 (S2) needs both halves',
                snapshot_utc=time.time(), pass_=len(kept) >= 80)


def s0f_leakage_design():
    """Index-level statement of what each estimator may see (verified again inside each stage)."""
    return dict(
        difficulty=dict(s1='training window y[0:3000] only (AR residuals, threshold, GPD, intervals estimator)',
                        s2='TRAIN split only (2005-01-01..2016-12-31)'),
        evt_static=dict(s1='CAL origins 2500..2998 of the training window', s2='CAL split 2017-2018'),
        sel=dict(rule='chosen per arm and series on CAL tau=0.995 pinball; TEST never used'),
        garch=dict(s2='rolling 1000-day window ending strictly before each origin; refit on the first trading day of each month'),
        lora=dict(s1='FIT part of the training window only (0..2499)', s2='TRAIN split only'),
        scoring=dict(rule='TEST scored only after SEAL.json exists and its hash matches'),
        pass_=True)


def main():
    report = dict(contract_sha256=tc.sha(tc.ROOT/'experiments'/'tail_complexity_v1_20260924'/'00_MASTER_CLI_tail_complexity_v1_20260924.txt'),
                  utc=time.time())
    audit = tc.read_json(tc.OUT/'PREMISE_AUDIT.json')
    report['S0a_out_of_grid'] = dict(pass_=audit['Q1_out_of_grid_equals_099']['status'] == 'CONFIRMED',
                                     detail=audit['Q1_out_of_grid_equals_099'])
    report['S0b_estimators'] = dict(pass_=audit['Q3_armax_extremal_index']['status'] == 'CONFIRMED'
                                    and audit['Q4_genpareto_sign']['status'] == 'CONFIRMED',
                                    gpd=audit['Q4_genpareto_sign'], theta=audit['Q3_armax_extremal_index'])
    s0c = tc.read_json(tc.CACHE/'s0c_oracle.json')
    report['S0c_oracle'] = dict(pass_=all(v['pass_'] for v in s0c.values()), per_condition=s0c,
                                gate='oracle exceedance rate at tau=0.99 within [0.006, 0.014]')
    report['S0d_pinball'] = s0d_pinball()
    print('S0d pinball pass:', report['S0d_pinball']['pass_'])
    report['S0e_financial_snapshot'] = s0e_financial_snapshot()
    print('S0e eligible symbols:', report['S0e_financial_snapshot']['eligible'])
    report['S0f_leakage'] = s0f_leakage_design()
    fails = [k for k, v in report.items() if isinstance(v, dict) and v.get('pass_') is False]
    report['verdict'] = 'S0_PASS' if not fails else 'STOP_DEBUG'
    report['failures'] = fails
    tc.write_json(tc.OUT/'S0_REPORT.json', report)
    print('S0 verdict:', report['verdict'], fails)


if __name__ == '__main__':
    main()
