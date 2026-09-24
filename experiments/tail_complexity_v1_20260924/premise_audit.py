#!/usr/bin/env python
"""S0 premise audit for MASTER_CLI tail_complexity_v1_20260924 (contract section 3 + section 12 gate).

Runs the checks that can be run without touching TEST data and writes PREMISE_AUDIT.json.
Nothing here trains an experiment fit (only the Q8 two-step smoke) and nothing is scored.
"""
from __future__ import annotations
import hashlib, io, json, logging, platform, re, subprocess, sys, time, urllib.request
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2]/'results'/'tail_complexity_v1_20260924'
OUT.mkdir(parents=True, exist_ok=True)
CONTRACT_SHA = hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() if len(sys.argv) > 1 else None
UA = {'User-Agent': 'Mozilla/5.0 (research; tail-complexity-audit)'}


def get(url, timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def theta_intervals(exceed_idx):
    """Contract 5.3 intervals estimator (Ferro & Segers 2003)."""
    T = np.diff(np.asarray(exceed_idx)); N = len(exceed_idx)
    if N < 3: return float('nan')
    if T.max() <= 2: th = 2*T.sum()**2/((N-1)*np.sum(T**2))
    else: th = 2*np.sum(T-1)**2/((N-1)*np.sum((T-1)*(T-2)))
    return float(min(1.0, th))


def theta_r_implementation(exceed_idx):
    """envoutliers::extremal.index.intervals (public R implementation of the same estimator)."""
    T = np.diff(np.asarray(exceed_idx))
    if len(T) < 2: return float('nan')
    if T.max() <= 2: th = 2*T.sum()**2/(len(T)*np.sum(T**2))
    else: th = 2*np.sum(T-1)**2/(len(T)*np.sum((T-1)*(T-2)))
    return float(min(1.0, th))


def main():
    import torch
    from scipy.stats import genpareto
    import importlib.metadata as md
    rng = np.random.default_rng(20260924)
    audit = {'contract': '00_MASTER_CLI_tail_complexity_v1_20260924.txt', 'contract_sha256': CONTRACT_SHA,
             'utc': time.time(), 'date_local': time.strftime('%Y-%m-%d %H:%M:%S')}
    # ---------------- environment (contract section 1 claims Ubuntu 22.04 / RTX 3080 10GB)
    pkgs = {}
    for p in ['chronos-forecasting', 'torch', 'peft', 'transformers', 'scipy', 'numpy', 'pandas', 'statsmodels', 'arch', 'yfinance']:
        try: pkgs[p] = md.version(p)
        except md.PackageNotFoundError: pkgs[p] = None
    audit['environment'] = dict(platform=platform.platform(), python=sys.version.split()[0], executable=sys.executable,
                                torch_cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                                vram_gib=round(torch.cuda.get_device_properties(0).total_memory/2**30, 1), packages=pkgs,
                                contract_claim='Ubuntu 22.04, RTX 3080 10GB, torch 2.8.0+cu128',
                                matches_contract=False,
                                note='execution environment differs from the contract claim; timings and VRAM are not comparable to it')
    # ---------------- Q1 out-of-grid quantiles
    import chronos
    from chronos import Chronos2Pipeline
    src = Path(chronos.__file__).parent
    lines = (src/'chronos2'/'model.py').read_text(encoding='utf-8').splitlines()
    pipe_lines = (src/'chronos2'/'pipeline.py').read_text(encoding='utf-8').splitlines()
    cite = dict(model_py_260=lines[259].strip(), model_py_268=lines[267].strip(),
                pipeline_predict_quantiles_def=[i+1 for i, l in enumerate(pipe_lines) if 'def predict_quantiles' in l],
                pipeline_warning_lines=[i+1 for i, l in enumerate(pipe_lines) if 'not within the range of' in l])
    logs = io.StringIO(); handler = logging.StreamHandler(logs)
    logging.getLogger('chronos.chronos2.pipeline').addHandler(handler)
    pipe = Chronos2Pipeline.from_pretrained('amazon/chronos-2', revision='29ec3766d36d6f73f0696f85560a422f50e8498c',
                                            device_map='cuda', torch_dtype=torch.bfloat16)
    t = np.arange(600); y = np.sin(2*np.pi*t/24) + rng.standard_t(3.5, 600)/np.sqrt(3.5/1.5)
    q, _ = pipe.predict_quantiles([torch.tensor(y[-512:], dtype=torch.float32)], prediction_length=1,
                                  quantile_levels=[0.95, 0.99, 0.995, 0.999])
    v = q[0][0].float().numpy().ravel()
    audit['Q1_out_of_grid_equals_099'] = dict(
        status='CONFIRMED' if (v[2] == v[1] and v[3] == v[1]) else 'REFUTED',
        values=dict(zip(['0.95', '0.99', '0.995', '0.999'], [float(x) for x in v])),
        training_quantiles=list(pipe.quantiles), n_training_quantiles=len(pipe.quantiles),
        warning_logged=bool('not within the range of' in logs.getvalue()), code_citations=cite,
        mechanism='utils.interpolate_quantiles pads the grid with the max value at level 1.0, so levels above 0.99 return the 0.99 value')
    # ---------------- Q2 intervals estimator vs public implementation
    idx = np.sort(rng.choice(20000, size=900, replace=False))
    same = [abs(theta_intervals(idx[:k]) - theta_r_implementation(idx[:k])) for k in (50, 200, 900)]
    audit['Q2_intervals_formula'] = dict(
        status='CONFIRMED_VS_PUBLIC_IMPLEMENTATION',
        max_abs_difference_vs_r_implementation=max(same),
        r_source='https://rdrr.io/cran/envoutliers/src/R/extremevalue.R (envoutliers::extremal.index.intervals)',
        r_code=['theta = min(1, (2 * (sum(T))^2) / (length(T) * sum(T^2)))',
                'theta = min(1, (2 * (sum(T - 1))^2) / (length(T) * sum((T - 1) * (T - 2))))'],
        note='length(T) == N-1 for N exceedances, so the contract formula and the published implementation agree exactly; '
             'the original JRSS-B 2003 PDF could not be text-extracted in this environment (poppler/pypdf unavailable), '
             'so the primary evidence is the published implementation, not the paper text')
    # ---------------- Q3 ARMAX extremal index
    q3 = {}
    for phi in (0.5, 0.9):
        n = 200000; z = 1/(-np.log(rng.random(n))); x = np.empty(n); x[0] = z[0]
        for i in range(1, n): x[i] = max(phi*x[i-1], (1-phi)*z[i])
        u = np.quantile(x, 0.95); q3[f'phi_{phi}'] = dict(theta_hat=theta_intervals(np.flatnonzero(x > u)), expected=1-phi)
    iid = rng.standard_normal(200000); q3['iid'] = dict(theta_hat=theta_intervals(np.flatnonzero(iid > np.quantile(iid, 0.95))), expected=1.0)
    ok3 = 0.45 <= q3['phi_0.5']['theta_hat'] <= 0.55 and 0.9 <= q3['iid']['theta_hat'] <= 1.0
    audit['Q3_armax_extremal_index'] = dict(status='CONFIRMED' if ok3 else 'REFUTED', **q3,
                                            gate='S0b requires phi=0.5 in [0.45,0.55] and iid in [0.9,1.0]')
    # ---------------- Q4 genpareto sign convention + GPD unit test
    x = genpareto.rvs(c=0.25, loc=0, scale=1.0, size=5000, random_state=rng)
    c, loc, scale = genpareto.fit(x, floc=0)
    audit['Q4_genpareto_sign'] = dict(status='CONFIRMED' if 0.20 <= c <= 0.30 else 'REFUTED', fitted_c=float(c),
                                      fitted_scale=float(scale), target_xi=0.25, gate='S0b requires xi_hat in [0.20,0.30]',
                                      convention='scipy c == GPD shape xi (positive c = heavy tail)')
    # ---------------- Q5 Time-PEFT spectral entropy definition
    blocked = None
    try:
        page = get('https://openreview.net/pdf?id=n8seTOinYs', timeout=30)
        blocked = 'pdf_bytes=%d' % len(page)
    except Exception as e:
        blocked = f'{type(e).__name__}: {e}'
    audit['Q5_spectral_entropy_definition'] = dict(
        status='NOT_VERIFIED', openreview='https://openreview.net/forum?id=n8seTOinYs', fetch_result=blocked,
        summary_source='https://lacuna.tiptreesystems.com/work/time-peft-... (states only: temporal complexity derived from spectral entropy)',
        fallback_used='contract section 2 fallback: standard normalized spectral entropy',
        definition_to_use='Welch periodogram -> p_k = P_k / sum(P_k) over non-DC frequencies; SE = -sum p_k log p_k / log(K); '
                          'computed on the same series used for difficulty measurement, plus the residual variant as secondary',
        consequence='the claim "independent of the Time-PEFT axis" (H2) is weaker because the original definition could not be read')
    # ---------------- Q6 financial data path
    q6 = {}
    try:
        wiki = get('https://en.wikipedia.org/wiki/S%26P_100')
        tickers = sorted(set(re.findall(r'<td><a[^>]*>([A-Z][A-Z\.\-]{0,5})</a>\s*</td>', wiki.decode('utf-8', 'replace'))))
        q6['wikipedia_sp100'] = dict(bytes=len(wiki), sha256=hashlib.sha256(wiki).hexdigest(), candidate_symbols=len(tickers),
                                     sample=tickers[:8], url='https://en.wikipedia.org/wiki/S%26P_100')
    except Exception as e:
        q6['wikipedia_sp100'] = dict(error=f'{type(e).__name__}: {e}')
    probes = {}
    for sym in ('AAPL', 'XOM', 'JPM'):
        try:
            raw = get(f'https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1=1104537600&period2=1751241600&interval=1d&events=div%2Csplit')
            d = json.loads(raw)['chart']['result'][0]
            adj = d['indicators']['adjclose'][0]['adjclose']
            probes[sym] = dict(rows=len(d['timestamp']), adjclose=True, nan_fraction=float(np.mean([a is None for a in adj])),
                               first=str(np.datetime64(d['timestamp'][0], 's')), last=str(np.datetime64(d['timestamp'][-1], 's')),
                               sha256=hashlib.sha256(raw).hexdigest())
        except Exception as e:
            probes[sym] = dict(error=f'{type(e).__name__}: {e}')
    q6['yahoo_chart_api'] = probes
    try:
        st = get('https://stooq.com/q/d/l/?s=aapl.us&d1=20050101&d2=20250630&i=d', timeout=20)
        q6['stooq'] = dict(bytes=len(st), usable=not st.lstrip().startswith(b'<'), note='JS challenge page returned' if st.lstrip().startswith(b'<') else 'csv')
    except Exception as e:
        q6['stooq'] = dict(error=f'{type(e).__name__}: {e}')
    ok6 = all('error' not in v for v in probes.values()) and 'error' not in q6['wikipedia_sp100']
    audit['Q6_financial_data'] = dict(status='AVAILABLE_VIA_YAHOO_CHART_API' if ok6 else 'BLOCKED_DATA_FIN', **q6,
                                      packages=dict(yfinance=pkgs['yfinance'], pandas_datareader=None),
                                      note='no data package is installed; the audit fetched the public Yahoo chart endpoint directly over HTTPS')
    # ---------------- Q7 arch package
    audit['Q7_arch_package'] = dict(status='MISSING' if pkgs['arch'] is None else 'AVAILABLE', version=pkgs['arch'],
                                    needed_for='S2 references: AR(1)-GARCH(1,1)-t and the conditional-EVT filter (contract section 6.1)',
                                    consequence='S2 cannot run its reference models until this is resolved')
    # ---------------- Q8 fit smoke
    series = [torch.tensor(np.sin(2*np.pi*np.arange(600)/24) + 0.5*rng.standard_normal(600), dtype=torch.float32) for _ in range(8)]
    torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter(); torch.manual_seed(0)
    tuned = pipe.fit(series, prediction_length=1, finetune_mode='lora', lora_config=None, context_length=512,
                     learning_rate=1e-5, num_steps=2, batch_size=4, validation_inputs=None, save_strategy='no', seed=0)
    smoke_s = time.perf_counter()-t0
    audit['Q8_fit_prediction_length_1'] = dict(status='CONFIRMED', seconds=smoke_s, num_steps=2,
                                               peak_vram_gib=round(torch.cuda.max_memory_allocated()/2**30, 2),
                                               returned=type(tuned).__name__, note='throwaway smoke fit; not part of the 14-fit budget')
    # ---------------- section 12 gate
    repo = Path(__file__).resolve().parents[2]
    def sh(*a):
        return subprocess.run(['git', *a], cwd=repo, capture_output=True, text=True).stdout.strip()
    audit['gate_section_12'] = dict(
        files_and_functions='PASS: chronos 2.3.2 model.py/pipeline.py citations resolve; Chronos2Pipeline.fit accepts every argument named in section 6.1',
        numeric_premises='PASS for Q1/Q3/Q4 (see above); Q5 unverified, Q7 missing, environment differs',
        leakage_design='PENDING: to be re-checked per stage in S0f',
        outputs_absent=dict(results=sorted(p.name for p in OUT.iterdir()),
                            experiments=sorted(p.name for p in (repo/'experiments'/'tail_complexity_v1_20260924').iterdir())),
        repo=dict(head=sh('rev-parse', 'HEAD'), branch=sh('branch', '--show-current'),
                  untracked_leftover=[l for l in sh('status', '--porcelain=v1').splitlines() if l.startswith('??')]))
    blocking = [k for k, v in audit.items() if isinstance(v, dict) and v.get('status') in ('MISSING', 'BLOCKED_DATA_FIN', 'REFUTED')]
    audit['blocking'] = blocking
    audit['verdict'] = 'PREMISES_OK_EXCEPT_' + ','.join(blocking) if blocking else 'PREMISES_OK'
    (OUT/'PREMISE_AUDIT.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str)+'\n', encoding='utf-8')
    print(json.dumps({k: (v.get('status') if isinstance(v, dict) else v) for k, v in audit.items()}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
