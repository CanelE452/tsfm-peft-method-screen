from __future__ import annotations
import gc
from pathlib import Path
import numpy as np
import torch
from .backend import load_chronos, direct_quantiles, deterministic_contract
from .lora import install_scale_lora, set_scale
from .data import valid_origins, choose_even, sample_batch
from .util import seed_all, require, count_trainable, write_json


def downsample(t, factor):
    if factor == 1:
        return t
    n = (t.shape[-1] // factor) * factor
    require(n > 0, 'Downsample length collapsed')
    return t[..., -n:].reshape(*t.shape[:-1], n // factor, factor).mean(-1)


def upsample(t, factor, horizon):
    if factor == 1:
        return t[..., :horizon]
    return t.repeat_interleave(factor, dim=-1)[..., :horizon]


def batch_predict(base, wrapped, ctx, scales, med_idx, horizon, device):
    B, C, _ = ctx.shape
    preds = []
    for si, factor in enumerate(scales):
        set_scale(wrapped, si)
        c = downsample(ctx, factor)
        h = max(1, horizon // factor)
        rows = c.reshape(B * C, c.shape[-1])
        gids = torch.arange(B, device=device).repeat_interleave(C)
        q = direct_quantiles(base, rows, h, gids)
        p = q[:, med_idx, :].reshape(B, C, h)
        preds.append(upsample(p, factor, horizon))
    return preds


def eval_variant(base, wrapped, weights, x, origins, channels, cfg, med_idx, device):
    H = cfg['data']['horizon']
    L = cfg['data']['context']
    bs = 6
    all_scale = [[] for _ in cfg['candidate_a']['scales']]
    ys = []
    for k in range(0, len(origins), bs):
        ids = origins[k:k + bs]
        cc = np.stack([x[i-L:i, channels].T for i in ids])
        yy = np.stack([x[i:i+H, channels].T for i in ids])
        ctx = torch.tensor(cc, device=device)
        with torch.no_grad():
            pp = batch_predict(base, wrapped, ctx, cfg['candidate_a']['scales'], med_idx, H, device)
        for s, p in enumerate(pp):
            all_scale[s].append(p.cpu().numpy())
        ys.append(yy)
    P = np.stack([np.concatenate(v, 0) for v in all_scale], 1)  # N,S,C,H
    Y = np.concatenate(ys, 0)
    w = np.asarray(weights, float)
    comb = (P * w[None, :, None, None]).sum(1)
    return {
        'mse': float(np.mean((comb - Y) ** 2)),
        'mae': float(np.mean(np.abs(comb - Y))),
        'per_scale': P,
        'y': Y,
    }


def train_variant(name, spec, data, cfg, out):
    seed_all(cfg['seed'])
    pipe, device = load_chronos(cfg)
    base = pipe.inner_model
    scales = cfg['candidate_a']['scales']
    alpha = cfg['candidate_a']['alpha_over_rank'] * spec['rank']
    wrapped = install_scale_lora(base, len(scales), spec['rank'], alpha, spec['mode'])
    logits = torch.nn.Parameter(torch.zeros(len(scales), device=device))
    params = [p for p in base.parameters() if p.requires_grad]
    require(params, 'No LoRA params')
    opt = torch.optim.AdamW(params, lr=cfg['candidate_a']['lr'], weight_decay=0)
    optw = torch.optim.Adam([logits], lr=cfg['candidate_a']['scale_weight_lr'])
    x = data['x']
    L = cfg['data']['context']
    H = cfg['data']['horizon']
    train_orig = valid_origins(L, data['train_end'], L, H)
    channels = np.arange(min(cfg['data']['a_channels'], data['c']))
    require(len(channels) >= 2, 'Need >=2 Weather channels for A')
    rng = np.random.default_rng(cfg['seed'] + sum(map(ord, name)))
    qlevels = [float(v) for v in pipe.quantiles]
    med_idx = min(range(len(qlevels)), key=lambda i: abs(qlevels[i] - 0.5))

    c0, _, _ = sample_batch(x, train_orig, L, H, 1, rng, channels)
    tc = torch.tensor(c0, device=device)
    rows = tc.reshape(len(channels), L)
    gids = torch.zeros(len(channels), device=device, dtype=torch.long)
    set_scale(wrapped, 0)
    deterministic_diff = deterministic_contract(base, rows, H, gids)

    trace = []
    steps = int(cfg['candidate_a']['steps'])
    for step in range(steps):
        cc, yy, _ = sample_batch(x, train_orig, L, H, cfg['model']['batch_size'], rng, channels)
        ctx = torch.tensor(cc, device=device)
        y = torch.tensor(yy, device=device)
        opt.zero_grad(set_to_none=True)
        losses = []
        w_detached = torch.softmax(logits.detach(), 0)
        for si, factor in enumerate(scales):
            set_scale(wrapped, si)
            c = downsample(ctx, factor)
            yf = downsample(y, factor)
            h = yf.shape[-1]
            B, C = c.shape[:2]
            flat = c.reshape(B * C, c.shape[-1])
            gids = torch.arange(B, device=device).repeat_interleave(C)
            q = direct_quantiles(base, flat, h, gids)
            pred = q[:, med_idx, :].reshape(B, C, h)
            loss = ((pred - yf) ** 2).mean()
            (w_detached[si] * loss).backward()
            losses.append(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        opt.step()

        optw.zero_grad(set_to_none=True)
        surrogate = (torch.softmax(logits, 0) * torch.stack(losses)).sum()
        surrogate.backward()
        optw.step()

        if step in (0, 7, 15, 31, steps - 1):
            trace.append({
                'step': step + 1,
                'losses': [float(z) for z in losses],
                'weights': torch.softmax(logits.detach(), 0).cpu().tolist(),
            })

    cal = choose_even(valid_origins(data['train_end'], data['cal_mid'], L, H), cfg['data']['eval_origins_per_half'])
    pilot = choose_even(valid_origins(data['cal_mid'], data['val_end'], L, H), cfg['data']['eval_origins_per_half'])
    weights = torch.softmax(logits.detach(), 0).cpu().numpy()
    er_cal = eval_variant(base, wrapped, weights, x, cal, channels, cfg, med_idx, device)
    er_pilot = eval_variant(base, wrapped, weights, x, pilot, channels, cfg, med_idx, device)
    result = {
        'name': name,
        'spec': spec,
        'trainable_lora_params': count_trainable(base),
        'trainable_scale_weight_params': int(logits.numel()),
        'scale_weights': weights.tolist(),
        'cal_mse': er_cal['mse'],
        'pilot_mse': er_pilot['mse'],
        'pilot_mae': er_pilot['mae'],
        'optimizer_steps': steps,
        'trace': trace,
        'deterministic_contract_max_abs': deterministic_diff,
    }
    c_payload = None
    if name == 'I8':
        c_payload = {
            'cal_P': er_cal['per_scale'], 'cal_Y': er_cal['y'],
            'pilot_P': er_pilot['per_scale'], 'pilot_Y': er_pilot['y'],
            'native_weights': weights,
        }
    del base, pipe, wrapped
    gc.collect()
    torch.cuda.empty_cache()
    return result, c_payload


def run(data, cfg, out):
    results = {}
    payload = None
    for name, spec in cfg['candidate_a']['variants'].items():
        result, p = train_variant(name, spec, data, cfg, out)
        results[name] = result
        if p is not None:
            payload = p
        write_json(Path(out) / 'A_PROGRESS.json', results)

    I8, P8, S8, I2 = results['I8'], results['P8'], results['S8'], results['I2']
    reduction = 100.0 * (I8['trainable_lora_params'] - P8['trainable_lora_params']) / I8['trainable_lora_params']
    simple = min(S8['pilot_mse'], I2['pilot_mse'])
    d = cfg['decision']
    keep = (
        P8['pilot_mse'] <= I8['pilot_mse'] * (1 + d['a_accuracy_tolerance_pct'] / 100.0)
        and reduction >= d['a_param_reduction_min_pct']
        and P8['pilot_mse'] <= simple * (1 - d['a_simple_margin_pct'] / 100.0)
    )
    if keep:
        decision = 'GO_TO_CONFIRMATION'
    elif simple <= P8['pilot_mse']:
        decision = 'SIMPLE_BASELINE_FIRST'
    else:
        decision = 'NO_GO_CURRENT_FORM'
    report = {
        'variants': results,
        'p8_lora_param_reduction_vs_i8_pct': reduction,
        'decision': decision,
        'meaning': 'One-seed screening only; GO is confirmation-worthy, not paper-level evidence.',
    }
    write_json(Path(out) / 'A_DECISION.json', report)
    require(payload is not None, 'I8 payload missing for candidate C')
    return report, payload
