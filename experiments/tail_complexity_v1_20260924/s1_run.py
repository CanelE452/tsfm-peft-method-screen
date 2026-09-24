#!/usr/bin/env python
"""S1 step 2: F0 and LoRA forecasts on the synthetic panel, CAL-based extensions, then (after SEAL) scoring.

Stages: `infer f0` | `fit` (12 LoRA fits) | `infer lora` | `extend` | `score`.
Nothing here reads the evaluation window except `score`, which refuses to run without SEAL.json.
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch
import tc

GRID = None      # filled with the 21 training quantile levels at runtime
BATCH = 256
FIT_KW = dict(prediction_length=1, finetune_mode='lora', lora_config=None, context_length=tc.CONTEXT,
              learning_rate=1e-5, num_steps=1000, batch_size=32, validation_inputs=None)


def load_panel():
    z = np.load(tc.CACHE/'s1_panel.npz')
    return z, z['origins'], z['cal_origins']


REVISION = '29ec3766d36d6f73f0696f85560a422f50e8498c'


def ensure_pinned_base(model_dir: Path):
    """peft resolves the base model at `adapter_config.revision` (peft/auto.py:115,156); `fit` leaves it
    null, which would take the hub's current main. Pin it so the LoRA arm sits on the same weights as F0."""
    cfg_path = model_dir/'adapter_config.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    if cfg.get('base_model_name_or_path') != 'amazon/chronos-2':
        raise SystemExit(f'UNEXPECTED_BASE {model_dir}: {cfg.get("base_model_name_or_path")}')
    if cfg.get('revision') != REVISION:
        cfg['revision'] = REVISION
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')


def pipeline(finetuned: Path | None = None):
    from chronos import Chronos2Pipeline
    kw = dict(device_map='cuda', torch_dtype=torch.bfloat16)
    if finetuned:
        ensure_pinned_base(finetuned)
    else:
        kw['revision'] = REVISION
    return Chronos2Pipeline.from_pretrained(str(finetuned) if finetuned else 'amazon/chronos-2', **kw)


def forecast_grid(pipe, contexts: list[np.ndarray]) -> np.ndarray:
    """Returns [n, 21] quantile forecasts at the training grid for h = 1."""
    global GRID
    GRID = list(pipe.quantiles)
    out = []
    for i in range(0, len(contexts), BATCH):
        chunk = [torch.tensor(c, dtype=torch.float32) for c in contexts[i:i+BATCH]]
        q, _ = pipe.predict_quantiles(chunk, prediction_length=1, quantile_levels=GRID)
        out.append(torch.stack([x[0] for x in q]).float().cpu().numpy())
    return np.concatenate(out)


def contexts_for(y: np.ndarray, origins: np.ndarray) -> list[np.ndarray]:
    return [y[max(0, o+1-tc.CONTEXT):o+1] for o in origins]


def infer(arm: str):
    z, origins, cal_origins = load_panel()
    out_dir = tc.CACHE/f's1_{arm}'; out_dir.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    for cond in tc.CONDITIONS:
        dest = out_dir/f'{cond}.npz'
        if dest.exists(): continue
        pipe = pipeline(tc.CACHE/'s1_lora'/f'{cond}_model' if arm == 'lora' else None)
        y = z[f'{cond}__y']
        ev, cal = [], []
        for k in range(tc.N_SERIES):
            ev.append(forecast_grid(pipe, contexts_for(y[k], origins)))
            cal.append(forecast_grid(pipe, contexts_for(y[k], cal_origins)))
        np.savez_compressed(dest, eval=np.stack(ev), cal=np.stack(cal), grid=np.array(GRID))
        del pipe; torch.cuda.empty_cache()
        print(f'{arm} {cond} done ({time.perf_counter()-t0:.0f}s)', flush=True)


def fit_all():
    """12 LoRA fits (contract 6.1): one per condition, FIT part of the training window only."""
    z, _, _ = load_panel()
    ledger = tc.CACHE/'s1_fit_ledger.json'
    done = tc.read_json(ledger) if ledger.exists() else {}
    for cond in tc.CONDITIONS:
        if cond in done: continue
        model_dir = tc.CACHE/'s1_lora'/f'{cond}_model'
        if model_dir.exists(): raise SystemExit(f'PARTIAL_FIT {cond}: refusing to overwrite {model_dir}')
        y = z[f'{cond}__y'][:, :tc.LEN_FIT]                     # FIT part only
        series = [torch.tensor(y[k], dtype=torch.float32) for k in range(tc.N_SERIES)]
        pipe = pipeline()
        torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter(); torch.manual_seed(0)
        tuned = pipe.fit(series, output_dir=str(tc.SCRATCH/f'fit_{cond}'), save_strategy='no', seed=0, **FIT_KW)
        tuned.model.save_pretrained(model_dir)
        done[cond] = dict(seconds=time.perf_counter()-t0, peak_vram_gib=torch.cuda.max_memory_allocated()/2**30,
                          steps=FIT_KW['num_steps'], seed=0, series=len(series), fit_length=tc.LEN_FIT,
                          model_dir=str(model_dir.relative_to(tc.ROOT)))
        tc.write_json(ledger, done)
        print(f'fit {cond}: {done[cond]["seconds"]:.0f}s peak {done[cond]["peak_vram_gib"]:.2f} GiB '
              f'({len(done)}/12 fits)', flush=True)
        del pipe, tuned; torch.cuda.empty_cache()


def extend(arm: str):
    """CAL-only extension fitting and SEL choice (contract 6.2). No evaluation-window value is read."""
    z, origins, cal_origins = load_panel()
    res = {}
    for cond in tc.CONDITIONS:
        d = np.load(tc.CACHE/f's1_{arm}'/f'{cond}.npz')
        grid = list(d['grid']); y = z[f'{cond}__y']
        gi = {t: grid.index(t) for t in (0.1, 0.5, 0.9, 0.95, 0.99)}
        cal_truth = y[:, cal_origins+1]
        per_series = []
        for k in range(tc.N_SERIES):
            cal_q = tc.drop_horizon(d['cal'][k])
            qc = {t: cal_q[:, gi[t]] for t in gi}
            par = tc.fit_evt_static(qc, cal_truth[k])
            scores = {}
            for name in ('LIN', 'EVT-S'):
                if name == 'EVT-S' and par is None: scores[name] = np.inf; continue
                q995 = tc.lin_extension(qc, 0.995) if name == 'LIN' else tc.evt_extension(qc, par, 0.995)
                scores[name] = float(np.mean(tc.pinball(q995, cal_truth[k], 0.995)))
            sel = min(scores, key=scores.get)
            per_series.append(dict(series=k, evt=par, cal_scores=scores, sel=sel))
        res[cond] = per_series
        print(f'extend {arm} {cond}: SEL counts ' +
              str({s: sum(1 for p in per_series if p['sel'] == s) for s in ('LIN', 'EVT-S')}), flush=True)
    tc.write_json(tc.CACHE/f's1_extension_{arm}.json', res)


def apply_extension(qrow: dict, par, sel: str, tau: float) -> np.ndarray:
    if sel == 'EVT-S' and par is not None: return tc.evt_extension(qrow, par, tau)
    return tc.lin_extension(qrow, tau)


def score():
    seal_path = tc.OUT/'SEAL.json'
    if not seal_path.exists(): raise SystemExit('REFUSED: SEAL.json is missing (contract 8/12)')
    seal = tc.read_json(seal_path)
    if seal['s1']['inputs_digest'] != s1_inputs_digest():
        raise SystemExit('REFUSED: sealed inputs digest does not match the current artifacts')
    z, origins, cal_origins = load_panel()
    rows = []
    for arm in ('F0', 'LORA'):
        ext = tc.read_json(tc.CACHE/f's1_extension_{arm.lower()}.json')
        for cond in tc.CONDITIONS:
            d = np.load(tc.CACHE/f's1_{arm.lower()}'/f'{cond}.npz')
            grid = list(d['grid']); gi = {t: grid.index(t) for t in (0.1, 0.5, 0.9, 0.95, 0.99)}
            y_next = z[f'{cond}__y'][:, origins+1]
            oracle = z[f'{cond}__oracle']
            for k in range(tc.N_SERIES):
                ev_q = tc.drop_horizon(d['eval'][k])
                qrow = {t: ev_q[:, gi[t]] for t in gi}
                par = ext[cond][k]['evt']; sel = ext[cond][k]['sel']
                for tau in tc.TAUS:
                    ti = tc.TAUS.index(tau)
                    ref = float(np.mean(tc.pinball(oracle[k][:, ti], y_next[k], tau)))
                    variants = {}
                    if tau in tc.GRID_TAUS:
                        variants['NONE'] = qrow[tau]
                    else:
                        variants['CLAMP'] = qrow[0.99]
                        variants['LIN'] = tc.lin_extension(qrow, tau)
                        if par is not None: variants['EVT-S'] = tc.evt_extension(qrow, par, tau)
                        variants['SEL'] = apply_extension(qrow, par, sel, tau)
                    for vname, q in variants.items():
                        qs = float(np.mean(tc.pinball(q, y_next[k], tau)))
                        rows.append(dict(arm=arm, condition=cond, series=k, tau=tau, extension=vname,
                                         qs=qs, qs_ref=ref, gap=tc.gap(qs, ref),
                                         exceed_rate=float(np.mean(y_next[k] > q)), sel_choice=sel))
    with (tc.OUT/'S1_SCORES.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('S1_SCORES.csv rows', len(rows))


def s1_inputs_digest():
    parts = {'panel': tc.sha(tc.CACHE/'s1_panel.npz'), 'difficulty': tc.sha(tc.OUT/'S1_DIFFICULTY.csv')}
    for arm in ('f0', 'lora'):
        parts[f'ext_{arm}'] = tc.sha(tc.CACHE/f's1_extension_{arm}.json')
        parts[f'pred_{arm}'] = tc.digest({c: tc.sha(tc.CACHE/f's1_{arm}'/f'{c}.npz') for c in tc.CONDITIONS})
    return tc.digest(parts)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['infer-f0', 'fit', 'infer-lora', 'extend', 'score', 'digest'])
    a = p.parse_args()
    if a.stage == 'infer-f0': infer('f0')
    elif a.stage == 'fit': fit_all()
    elif a.stage == 'infer-lora': infer('lora')
    elif a.stage == 'extend': extend('f0'); extend('lora')
    elif a.stage == 'score': score()
    else: print(s1_inputs_digest())
