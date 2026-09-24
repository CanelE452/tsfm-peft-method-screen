#!/usr/bin/env python
"""ENVIRONMENT.json and DATA_MANIFEST.json (contract section 9)."""
from __future__ import annotations
import platform, subprocess, sys, time
import numpy as np
import tc

PKGS = ['torch', 'chronos', 'transformers', 'peft', 'accelerate', 'numpy', 'scipy', 'statsmodels', 'arch', 'pandas', 'matplotlib']


def environment():
    import importlib
    vers = {}
    for name in PKGS:
        try: vers[name] = getattr(importlib.import_module(name), '__version__', 'unknown')
        except Exception as e: vers[name] = f'missing ({type(e).__name__})'
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu-only'
    vram = torch.cuda.get_device_properties(0).total_memory/2**30 if torch.cuda.is_available() else 0.0
    try:
        commit = subprocess.run(['git', '-C', str(tc.ROOT), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        commit = 'unknown'
    return dict(utc=time.time(), date=time.strftime('%Y-%m-%d %H:%M:%S'),
                platform=platform.platform(), python=sys.version.split()[0], packages=vers,
                gpu=gpu, vram_gib=round(float(vram), 2), cuda=getattr(torch.version, 'cuda', None),
                repo=str(tc.ROOT), repo_commit=commit,
                model='amazon/chronos-2', revision='29ec3766d36d6f73f0696f85560a422f50e8498c',
                note='the contract section 1 recorded Ubuntu 22.04 / RTX 3080 10GB / torch 2.8.0; this run differs')


def data_manifest():
    s0 = tc.read_json(tc.OUT/'S0_REPORT.json')
    fin = s0['S0e_financial_snapshot']
    s1 = dict(kind='synthetic, generated in this run', generator='experiments/tail_complexity_v1_20260924/tc.py:simulate',
              conditions=tc.CONDITIONS, series_per_condition=tc.N_SERIES, seed_base=tc.SEED_BASE,
              lengths=dict(total=tc.LEN_TOTAL, train=tc.LEN_TRAIN, fit=tc.LEN_FIT, burn_in=tc.LEN_BURN),
              truth_run_length=2_000_000, panel_file=str(tc.CACHE/'s1_panel.npz'),
              panel_sha256=tc.sha(tc.CACHE/'s1_panel.npz'))
    per_symbol = {sym: dict(sha256=rec['sha256'], bytes=rec['bytes'], rows=rec['rows'],
                            first=rec['first'], last=rec['last'])
                  for sym, rec in fin['records'].items()}
    s2 = dict(kind='public daily adjusted close', source=fin['source'], universe_source=fin['universe_source'],
              universe_sha256=fin['universe_sha256'], snapshot_utc=fin['snapshot_utc'],
              symbols_listed=fin['symbols_listed'], eligible=fin['eligible'],
              eligibility_rule=fin['eligibility_rule'], splits=fin['splits'],
              target='daily loss = -100 * log return (contract 4.2)',
              survivorship_bias='current S&P 100 constituents only; delisted names are absent, which narrows the '
                                'difficulty range and lowers the power of H3',
              cache=str(tc.CACHE/'s2_prices.npz'), cache_sha256=tc.sha(tc.CACHE/'s2_prices.npz'),
              per_symbol=per_symbol, dropped=list(fin['dropped'].keys()))
    return dict(utc=time.time(), S1=s1, S2=s2,
                redistribution='raw prices and model weights are not committed; only hashes and derived tables are')


if __name__ == '__main__':
    tc.write_json(tc.OUT/'ENVIRONMENT.json', environment())
    tc.write_json(tc.OUT/'DATA_MANIFEST.json', data_manifest())
    print('ENVIRONMENT.json and DATA_MANIFEST.json written')
