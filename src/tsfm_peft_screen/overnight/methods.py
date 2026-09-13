"""Candidate mechanisms; known ingredients, no claim of established novelty."""
import math
import numpy as np
import torch
from torch import nn
from ..backbone import QUANTILES, load_base
from ..lora import attach, LowRank, disabled

TOPICS = {
    'anchor': ['native', 'raw', 'l2_anchor', 'prediction_anchor'],
    'drift': ['native', 'raw', 'moment_gate', 'drift_gate'],
    'distill': ['raw', 'uniform_distill', 'reversed_distill', 'temporal_distill'],
}
PROPOSED = {'anchor': 'prediction_anchor', 'drift': 'drift_gate', 'distill': 'temporal_distill'}


def raw_loss(prediction, target, scale):
    """Exactly the reporting objective for finite, complete targets (mean Q)."""
    p = prediction.float().sort(dim=1).values
    e = target[:, None, :].float() - p
    q = torch.tensor(QUANTILES, device=p.device)[None, :, None]
    return (2 * torch.maximum(q * e, (q - 1) * e) / scale[:, None, None]).mean()


def context_features(x, kind):
    """Six causal features; no future data, origin index, dataset or seed labels."""
    x = x.float()
    center = x.mean(-1, keepdim=True)
    scale = x.std(-1, keepdim=True, unbiased=False).clamp_min(1e-5)
    z = (x - center) / scale
    recent, previous = z[:, -64:], z[:, -128:-64]
    slow = z[:, -512:-128]
    stats = lambda a: (a.mean(-1), a.std(-1, unbiased=False))
    mr, sr = stats(recent)
    mp, sp = stats(previous)
    ms, ss = stats(slow)
    if kind == 'moment_gate':
        result = torch.stack((mr, sr, mp, sp, ms, ss), -1)
    elif kind == 'drift_gate':
        # Contrasts are deterministically recoverable from the generic moments.
        # This deliberately tests inductive bias, not privileged information.
        result = torch.stack((mr-mp, mr-ms, mp-ms,
                              torch.log((sr+.05)/(sp+.05)),
                              torch.log((sr+.05)/(ss+.05)),
                              torch.log((sp+.05)/(ss+.05))), -1)
    else:
        raise ValueError(kind)
    return result.clamp(-6, 6)


class ConditionedLowRank(LowRank):
    def __init__(self, old):
        nn.Module.__init__(self)
        self.base, self.lora_A, self.lora_B = old.base, old.lora_A, old.lora_B
        self.enabled = True
        self.group_attention = old.group_attention
        self.condition = nn.Linear(6, 8, device=self.lora_A.device)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)
        self.features = None

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        if self.features is None:
            raise RuntimeError('Missing causal gate features')
        g = 2 * torch.sigmoid(self.condition(self.features))
        g = g[None, :, :] if self.group_attention else g[:, None, :]
        a = nn.functional.linear(x, self.lora_A) * g
        return out + 2 * nn.functional.linear(a, self.lora_B)


class PilotModel(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        self.arm = arm
        self.base = load_base()
        attach(self.base, seed=seed)
        if arm in ('moment_gate', 'drift_gate'):
            for index, (name, old) in enumerate(list(self.base.named_modules())):
                if isinstance(old, LowRank):
                    torch.manual_seed(seed + 4000 + index)
                    parent, leaf = name.rsplit('.', 1)
                    setattr(self.base.get_submodule(parent), leaf, ConditionedLowRank(old))

    def forward(self, x, groups, frozen=False):
        if self.arm in ('moment_gate', 'drift_gate') and not frozen:
            features = context_features(x, self.arm)
            for layer in self.base.modules():
                if isinstance(layer, ConditionedLowRank):
                    layer.features = features
        # No activation checkpointing: no recomputation can see mutable gate state.
        from contextlib import nullcontext
        with disabled(self.base) if frozen else nullcontext():
            enc, (loc, scale), _, _ = self.base.encode(context=x, group_ids=groups, num_output_patches=3)
            h = enc.last_hidden_state[:, -3:]
            z = self.base.output_patch_embedding(h).reshape(len(x), 3, 21, 16)
            z = z.permute(0, 2, 1, 3).reshape(len(x), 21, 48).float()
            prediction = z.sinh() * scale[:, None, :] + loc[:, None, :]
        return z, prediction, loc, scale


def mixture_quantiles(components, samples_per_component=256):
    """Equal CDF mixture, NOT an average of quantiles; deterministic quadrature.

    Input K,N,C,Q,H. Piecewise-linear inverse CDF with constant endpoint tails.
    The same finite-grid distribution approximation is used for all teachers.
    This represents marginal horizon distributions, not joint forecast paths.
    """
    a = np.sort(np.asarray(components, dtype=np.float64), axis=3)
    if a.ndim != 5 or a.shape[3] != len(QUANTILES) or not np.isfinite(a).all():
        raise ValueError('Invalid component predictions')
    u = (np.arange(samples_per_component) + .5) / samples_per_component
    q = np.array(QUANTILES)
    hi = np.searchsorted(q, u, side='right').clip(1, len(q)-1)
    lo = hi - 1
    w = ((u-q[lo])/(q[hi]-q[lo])).clip(0, 1)
    samples = a[:, :, :, lo, :] * (1-w)[None, None, None, :, None]
    samples += a[:, :, :, hi, :] * w[None, None, None, :, None]
    pooled = samples.transpose(1, 2, 4, 0, 3).reshape(*a.shape[1:3], a.shape[-1], -1)
    return np.quantile(pooled, q, axis=-1).transpose(1, 2, 0, 3).astype(np.float32)


def teacher_weights(components, scale):
    """Downweight patches where context-length teachers disagree; no labels."""
    medians = np.sort(components, axis=3)[:, :, :, 10, :]
    disagreement = medians.std(axis=0) / np.asarray(scale)[None, :, None]
    patch = disagreement.reshape(*disagreement.shape[:2], 3, 16).mean(-1)
    weights = np.repeat(1/(1+patch), 16, axis=-1)
    # Equal total KD mass per series, so the ablation tests its time placement.
    return (weights / weights.mean(-1, keepdims=True)).astype(np.float32)


def verdict_from_rows(rows, selections, topic, config):
    """No choice of candidate, metrics, datasets or thresholds after evaluation."""
    proposed = PROPOSED[topic]
    checks = []
    for dataset in config['datasets']:
        dr = [r for r in rows if r['dataset'] == dataset]
        arms = sorted(set(r['arm'] for r in dr))
        means = {a: float(np.mean([r['metrics']['scaled_2pinball'] for r in dr if r['arm'] == a])) for a in arms}
        best = min(means[a] for a in arms if a != proposed)
        q = means[proposed]
        seed_checks = []
        for seed in config['seeds']:
            values = {r['arm']: r['metrics']['scaled_2pinball'] for r in dr if r['seed'] in (None, seed)}
            selected = next(s for s in selections if s['dataset'] == dataset and s['seed'] == seed and s['arm'] == proposed)
            seed_checks.append(selected['step'] > 0 and values[proposed] < values['F0']
                               and values[proposed] <= config['gate']['max_seed_ratio'] * min(v for a, v in values.items() if a != proposed))
        checks.append(dict(dataset=dataset, means=means, gain_vs_best=1-q/best,
                           mean_pass=q <= (1-config['gate']['min_gain']) * best,
                           seed_checks=seed_checks))
    passed = all(r['mean_pass'] and all(r['seed_checks']) for r in checks)
    return dict(verdict='PILOT_PASS' if passed else 'PILOT_STOP', checks=checks,
                novelty_status='UNRESOLVED_KNOWN_INGREDIENTS',
                scope='Development continuation only. Not a publication PASS or a revision of historical FAIL.')
