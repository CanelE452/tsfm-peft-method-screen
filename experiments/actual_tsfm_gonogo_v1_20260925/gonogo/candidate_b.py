from __future__ import annotations
import gc, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .backend import load_chronos, direct_quantiles
from .data import valid_origins, choose_even, sample_batch
from .util import seed_all, require, count_trainable, write_json, pct_gain, sync


def pca_weights(train, K):
    x = torch.tensor(train, dtype=torch.float32)
    cov = x.T @ x / max(1, len(x) - 1)
    _, vec = torch.linalg.eigh(cov)
    return vec[:, -K:].T.contiguous()  # K,C


class CompressedForecast(nn.Module):
    def __init__(self, base, C, K, L, H, mode, V):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.C, self.K, self.L, self.H, self.mode = C, K, L, H, mode
        self.E = nn.Linear(C, K, bias=False)
        self.D = nn.Linear(K, C, bias=False)
        with torch.no_grad():
            self.E.weight.copy_(V)
            self.D.weight.copy_(V.T)
        if mode in ('PCA4_R', 'LINEAR'):
            self.E.requires_grad_(False)
            self.D.requires_grad_(False)
        self.head = None
        if mode in ('R4', 'X4', 'PCA4_R', 'LINEAR'):
            self.head = nn.Linear(L, H)

    def forward(self, x, med_idx):
        # x: B,C,L. Shared temporal head applies identically per channel.
        if self.mode == 'LINEAR':
            return self.head(x)
        latent = torch.einsum('kc,bcl->bkl', self.E.weight, x)
        B, K, L = latent.shape
        rows = latent.reshape(B * K, L)
        gids = torch.arange(B, device=x.device).repeat_interleave(K)
        q = direct_quantiles(self.base, rows, self.H, gids)
        lp = q[:, med_idx, :].reshape(B, K, self.H)
        pred = torch.einsum('ck,bkh->bch', self.D.weight, lp)
        if self.head is not None:
            if self.mode in ('R4', 'PCA4_R'):
                rec = torch.einsum('ck,bkl->bcl', self.D.weight, latent)
                pred = pred + self.head(x - rec)
            elif self.mode == 'X4':
                pred = pred + self.head(x)
        return pred


def evaluate(model, x, origins, L, H, med_idx, device, bs=4):
    ps, ys = [], []
    for k in range(0, len(origins), bs):
        ids = origins[k:k + bs]
        cc = np.stack([x[i-L:i].T for i in ids])
        yy = np.stack([x[i:i+H].T for i in ids])
        t = torch.tensor(cc, device=device)
        with torch.no_grad():
            p = model(t, med_idx).cpu().numpy()
        ps.append(p)
        ys.append(yy)
    P = np.concatenate(ps)
    Y = np.concatenate(ys)
    return float(np.mean((P - Y) ** 2)), float(np.mean(np.abs(P - Y)))


def runtime(model, x, origin, L, med_idx, device):
    cc = np.stack([x[origin-L:origin].T] * 2)
    t = torch.tensor(cc, device=device)
    for _ in range(2):
        with torch.no_grad():
            model(t, med_idx)
    sync()
    vals = []
    for _ in range(6):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(t, med_idx)
        sync()
        vals.append(time.perf_counter() - t0)
    return float(np.median(vals))


def train_one(name, data, cfg, V):
    seed_all(cfg['seed'] + sum(map(ord, name)))
    pipe, device = load_chronos(cfg)
    base = pipe.inner_model
    C = data['c']
    L = cfg['data']['context']
    H = cfg['data']['horizon']
    K = 8 if name == 'U8' else 4
    model = CompressedForecast(base, C, K, L, H, name, V[K]).to(device)
    qlevels = [float(x) for x in pipe.quantiles]
    med_idx = min(range(len(qlevels)), key=lambda i: abs(qlevels[i] - 0.5))
    train_orig = valid_origins(L, data['train_end'], L, H)
    rng = np.random.default_rng(cfg['seed'] + K + len(name))
    params = [p for p in model.parameters() if p.requires_grad]
    require(params, f'No trainable parameters for {name}')
    opt = torch.optim.AdamW(params, lr=cfg['candidate_b']['lr'], weight_decay=0)
    trace = []
    steps = int(cfg['candidate_b']['steps'])
    for step in range(steps):
        cc, yy, _ = sample_batch(data['x'], train_orig, L, H, cfg['model']['batch_size'], rng)
        x = torch.tensor(cc, device=device)
        y = torch.tensor(yy, device=device)
        opt.zero_grad(set_to_none=True)
        pred = model(x, med_idx)
        loss = ((pred - y) ** 2).mean()
        require(bool(torch.isfinite(loss)), f'Nonfinite loss {name}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        opt.step()
        if step in (0, 7, 15, 31, steps - 1):
            trace.append({'step': step + 1, 'loss': float(loss.detach())})

    cal = choose_even(valid_origins(data['train_end'], data['cal_mid'], L, H), cfg['data']['eval_origins_per_half'])
    pilot = choose_even(valid_origins(data['cal_mid'], data['val_end'], L, H), cfg['data']['eval_origins_per_half'])
    cal_mse, cal_mae = evaluate(model, data['x'], cal, L, H, med_idx, device)
    pilot_mse, pilot_mae = evaluate(model, data['x'], pilot, L, H, med_idx, device)
    rt = runtime(model, data['x'], int(pilot[len(pilot)//2]), L, med_idx, device)
    result = {
        'name': name,
        'trainable_params': count_trainable(model),
        'cal_mse': cal_mse,
        'cal_mae': cal_mae,
        'pilot_mse': pilot_mse,
        'pilot_mae': pilot_mae,
        'runtime_s_batch2': rt,
        'optimizer_steps': steps,
        'trace': trace,
    }
    del model, base, pipe
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(data, cfg, out):
    require(data['c'] >= 8, 'Weather must have >=8 numeric channels for B')
    V = {k: pca_weights(data['x'][:data['train_end']], k) for k in (4, 8)}
    results = {}
    for name in cfg['candidate_b']['variants']:
        result = train_one(name, data, cfg, V)
        results[name] = result
        write_json(Path(out) / 'B_PROGRESS.json', results)

    R4, U4, U8 = results['R4'], results['U4'], results['U8']
    X4, PCA4, LINEAR = results['X4'], results['PCA4_R'], results['LINEAR']
    d = cfg['decision']
    runtime_reduction = pct_gain(U8['runtime_s_batch2'], R4['runtime_s_batch2'])
    simple = min(U4['pilot_mse'], X4['pilot_mse'], PCA4['pilot_mse'], LINEAR['pilot_mse'])
    keep = (
        R4['pilot_mse'] <= U8['pilot_mse'] * (1 + d['b_accuracy_tolerance_pct'] / 100.0)
        and R4['pilot_mse'] <= U4['pilot_mse'] * (1 - d['b_simple_margin_pct'] / 100.0)
        and runtime_reduction >= d['b_runtime_reduction_min_pct']
        and R4['pilot_mse'] <= simple * (1 - d['b_simple_margin_pct'] / 100.0)
    )
    if keep:
        decision = 'GO_TO_CONFIRMATION'
    elif simple <= R4['pilot_mse']:
        decision = 'SIMPLE_BASELINE_FIRST'
    else:
        decision = 'NO_GO_CURRENT_FORM'
    report = {
        'variants': results,
        'r4_runtime_reduction_vs_u8_pct': runtime_reduction,
        'decision': decision,
        'meaning': 'One-seed screening only; GO is confirmation-worthy, not paper-level evidence.',
    }
    write_json(Path(out) / 'B_DECISION.json', report)
    return report
