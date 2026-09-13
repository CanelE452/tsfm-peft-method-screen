"""Train-only calibration selects where to preserve the frozen function.

An exploratory hypothesis using known calibration/distillation ingredients.
No conformal, conditional coverage, independence, or novelty guarantee.
"""
import numpy as np
import torch
from ..backbone import QUANTILES
from ..overnight.methods import PilotModel

TOPICS = {'calibration': ['native', 'raw', 'l2_anchor', 'full_anchor',
                         'weighted_loss', 'shuffled_anchor', 'calibration_anchor']}
PROPOSED = {'calibration': 'calibration_anchor'}


def calibration_weights(prediction, target):
    """Nonoverlapping target windows (every second stride-32 origin), 3 patches.

    No future/test labels. Mean preservation weight = 1 within every channel,
    including the shuffled control. Coverage is empirical, not a confidence bound.
    """
    p = np.sort(np.asarray(prediction, dtype=np.float64), axis=2)[::2]
    y = np.asarray(target, dtype=np.float64)[::2]
    assert p.ndim == 4 and p.shape[2:] == (21,48)
    assert y.shape == p.shape[:2] + (48,)
    assert np.isfinite(p).all() and np.isfinite(y).all()
    coverage = (y[:,:,None,:] <= p).reshape(len(p),p.shape[1],21,3,16).mean((0,4))
    q = np.asarray(QUANTILES)[None,:,None]
    standardized_error = np.abs(coverage-q) / np.sqrt(q*(1-q))
    reliability = np.maximum(np.exp(-4*standardized_error), .1)
    weights = reliability / reliability.mean((1,2),keepdims=True)
    weights = np.repeat(weights,16,axis=2).astype(np.float32)
    return weights, dict(origins_used=len(p),coverage=coverage.tolist(),
                         weight_min=float(weights.min()),weight_max=float(weights.max()),
                         train_only=True,confidence_interval_claim=False)


def shuffled_weights(weights):
    return np.roll(np.roll(weights,7,axis=1),16,axis=2).copy()


def weighted_pinball(prediction,target,scale,weight):
    p=prediction.float().sort(dim=1).values
    error=target[:,None,:].float()-p
    q=torch.tensor(QUANTILES,device=p.device)[None,:,None]
    return (2*torch.maximum(q*error,(q-1)*error)/scale[:,None,None]*weight).mean()


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
