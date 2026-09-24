#!/usr/bin/env python
"""Write SEAL.json (contract section 8): hypothesis thresholds, conditions/seeds, symbols, split dates,
estimation settings and the SEL choices. TEST scoring is only allowed after this file exists."""
from __future__ import annotations
import time
import numpy as np
import tc
import hypotheses
import s1_run
import s2_score


def main():
    if (tc.OUT/'SEAL.json').exists():
        print('SEAL.json already exists; refusing to re-seal'); return
    s0 = tc.read_json(tc.OUT/'S0_REPORT.json')
    if s0['verdict'] != 'S0_PASS': raise SystemExit('REFUSED: S0 did not pass')
    ext_s1 = {arm: tc.read_json(tc.CACHE/f's1_extension_{arm}.json') for arm in ('f0', 'lora')}
    sel_s1 = {arm: {cond: [p['sel'] for p in rows] for cond, rows in ext.items()} for arm, ext in ext_s1.items()}
    ext_s2 = tc.read_json(tc.CACHE/'s2_extension.json')
    sel_s2 = {arm: {sym: rec['sel'] for sym, rec in d.items()} for arm, d in ext_s2.items()}
    seal = dict(
        utc=time.time(), date=time.strftime('%Y-%m-%d %H:%M:%S'),
        contract_sha256=s0['contract_sha256'],
        hypothesis_thresholds=hypotheses.TH,
        decision_tokens=['TC_VALID', 'TC_UNSTABLE', 'TC_REDUNDANT', 'TC_NO_GAP', 'TC_SYNTH_ONLY', 'STOP_DEBUG', 'INCONCLUSIVE_*'],
        s1=dict(conditions=tc.CONDITIONS, series_per_condition=tc.N_SERIES, seed_base=tc.SEED_BASE,
                lengths=dict(total=tc.LEN_TOTAL, train=tc.LEN_TRAIN, fit=tc.LEN_FIT, burn_in=tc.LEN_BURN),
                eval_stride=tc.EVAL_STRIDE, context=tc.CONTEXT, lora_fits=12, lora_seed=0,
                sel_choices=sel_s1, inputs_digest=s1_run.s1_inputs_digest()),
        s2=dict(symbols=list(tc.read_json(tc.OUT/'S0_REPORT.json')['S0e_financial_snapshot']['eligible_symbols']),
                splits=s2_score.SPLITS, context=tc.CONTEXT, lora_fits=2, lora_seeds=[0, 1],
                reference_primary='conditional EVT (AR(1)-GARCH(1,1) QML + GPD on standardized residuals)',
                reference_secondary='AR(1)-GARCH(1,1)-t', garch_window=1000, refit='first trading day of each month',
                sel_choices=sel_s2, inputs_digest=s2_score.s2_inputs_digest()),
        estimation=dict(threshold_quantile=tc.U_Q, min_exceedances=tc.MIN_EXCEEDANCES, taus=tc.TAUS,
                        grid_taus=tc.GRID_TAUS, out_of_grid_taus=tc.OUT_TAUS,
                        spectral_entropy='standard normalized spectral entropy (Q5 fallback, PREMISE_AUDIT.json)',
                        bootstrap_draws=2000, bootstrap_seed=tc.SEED_BASE),
        note='TEST scores are computed only after this file exists; the scoring scripts refuse to run otherwise')
    tc.write_json(tc.OUT/'SEAL.json', seal)
    print('SEAL written; s1 digest', seal['s1']['inputs_digest'][:16], 's2 digest', seal['s2']['inputs_digest'][:16])


if __name__ == '__main__':
    main()
