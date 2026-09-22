"""Small, CPU-testable numerical components for the A/B design.

NOT a complete experiment runner: no datasets, pretrained models, LoRA wiring,
forecast-time availability audit, GPU benchmark, or empirical results are supplied.
Atoms use [..., n_atoms]. Weights must be broadcastable to that exact shape.
The scores are for finite discrete predictive distributions, not exact scores
of an unknown continuous forecast distribution or of an entire joint trajectory.
"""
from __future__ import annotations
from itertools import combinations
import math
import torch
from torch import nn


def checked_weights(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    if z.ndim < 1 or z.shape[-1] < 1 or not torch.isfinite(z).all():
        raise ValueError('Atoms must be finite and nonempty.')
    p = torch.broadcast_to(p.to(device=z.device, dtype=z.dtype), z.shape)
    if not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError('Weights must be finite and nonnegative.')
    total = p.sum(-1, keepdim=True)
    if (total <= 0).any():
        raise ValueError('Each distribution needs positive total mass.')
    return p / total


def weighted_crps(z: torch.Tensor, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Exact discrete CRPS, differentiable in atoms and positive weights.

    Uses sorting, O(n log n), rather than a quadratic pairwise tensor.
    Returns one score for each leading dimension of z.
    """
    p = checked_weights(z, p)
    y = torch.as_tensor(y, device=z.device, dtype=z.dtype)
    if y.shape != z.shape[:-1] or not torch.isfinite(y).all():
        raise ValueError('Targets must have the leading atom dimensions.')
    zs, idx = z.sort(-1)
    ps = p.gather(-1, idx)
    mass_before = ps.cumsum(-1) - ps
    value_before = (ps * zs).cumsum(-1) - ps * zs
    half_pair_distance = (ps * (zs * mass_before - value_before)).sum(-1)
    return (p * (z - y[..., None]).abs()).sum(-1) - half_pair_distance


def cdf_l2(z: torch.Tensor, p: torch.Tensor,
           t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Integral of squared CDF difference (half the energy distance).

    Cross term is quadratic in support counts: batch or chunk for large supports.
    Inputs must have equal leading dimensions, but support counts can differ.
    """
    if z.shape[:-1] != t.shape[:-1]:
        raise ValueError('Predictive distributions must align.')
    p, w = checked_weights(z, p), checked_weights(t, w)
    cross = ((z[..., :, None] - t[..., None, :]).abs()
             * p[..., :, None] * w[..., None, :]).sum((-2, -1))
    zero = torch.zeros_like(z[..., 0])
    half_zz = (p * z.abs()).sum(-1) - weighted_crps(z, p, zero)
    half_tt = (w * t.abs()).sum(-1) - weighted_crps(t, w, zero)
    return cross - half_zz - half_tt


def discrete_quantiles(z: torch.Tensor, p: torch.Tensor,
                       levels: torch.Tensor) -> torch.Tensor:
    """Left inverse of a weighted discrete CDF; not native linear interpolation.

    Metrics/readout only. Do not train routing weights through this searchsorted
    operation: use weighted_crps or cdf_l2 instead.
    """
    p = checked_weights(z, p)
    q = torch.as_tensor(levels, device=z.device, dtype=z.dtype)
    if q.ndim != 1 or ((q <= 0) | (q >= 1)).any():
        raise ValueError('Levels must be a vector strictly between zero and one.')
    zs, idx = z.sort(-1)
    cdf = p.gather(-1, idx).cumsum(-1).contiguous()
    values = q.expand(*cdf.shape[:-1], len(q)).contiguous()
    where = torch.searchsorted(cdf, values, right=False).clamp_max(z.shape[-1] - 1)
    return zs.gather(-1, where)


def equal_atoms(z: torch.Tensor) -> torch.Tensor:
    return torch.full_like(z, 1.0 / z.shape[-1])


def medoids3(paths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact best 3-of-9 medoids of nine complete paths, inference-only.

    Distance is mean squared difference along the full path. Tie-breaking follows
    lexicographic medoid indices; assignment ties use the first medoid.
    Medoid weights are the fractions of original paths assigned to them.
    """
    if paths.ndim != 3 or paths.shape[1] != 9 or not torch.isfinite(paths).all():
        raise ValueError('Expected [batch, 9, lead] finite paths.')
    tuples = list(combinations(range(9), 3))
    distance = (paths[:, :, None] - paths[:, None, :]).square().mean(-1)
    costs = torch.stack([distance[:, :, list(ids)].min(-1).values.sum(-1)
                         for ids in tuples], -1)
    best = costs.argmin(-1)
    indices = torch.tensor(tuples, device=paths.device)[best]
    batch = torch.arange(len(paths), device=paths.device)
    centers = paths[batch[:, None], indices]
    assignments = (paths[:, :, None] - centers[:, None]).square().mean(-1).argmin(-1)
    weights = torch.nn.functional.one_hot(assignments, 3).to(paths.dtype).mean(1)
    return centers, weights, indices


class PathRouter(nn.Module):
    """A: context-independent or context-conditioned convex path router.

    All learnable prototypes use the SAME mixture coefficients at all 64 leads.
    Pass normalized first-call paths. Construct raw-valued prototypes separately
    from the returned attention and the raw F0 paths; keep router gradients.
    """
    def __init__(self, conditional: bool, leads: int = 64):
        super().__init__()
        init = torch.full((3, 9), 0.1 / 8)
        init[torch.arange(3), torch.tensor([1, 4, 7])] = 0.9
        bias = torch.cat([init.log().reshape(-1), torch.zeros(3)])
        self.conditional, self.leads = conditional, leads
        if conditional:
            self.down = nn.Linear(9 * leads, 8)
            self.up = nn.Linear(8, 30)
            nn.init.zeros_(self.up.weight)
            with torch.no_grad():
                self.up.bias.copy_(bias)
        else:
            self.logits = nn.Parameter(bias)

    def forward(self, paths: torch.Tensor):
        if paths.ndim != 3 or paths.shape[1:] != (9, self.leads):
            raise ValueError('Wrong first-call path shape.')
        logits = (self.up(torch.nn.functional.gelu(self.down(paths.flatten(1))))
                  if self.conditional else self.logits.expand(len(paths), -1))
        assignment = logits[:, :27].reshape(-1, 3, 9).softmax(-1)
        weights = logits[:, 27:].softmax(-1)
        prototypes = torch.einsum('bkm,bmh->bkh', assignment, paths)
        return prototypes, weights, assignment


class ScenarioPool(nn.Module):
    """B: target-conditioned, member-permutation-invariant scenario compressor.

    Inputs: standardized COMPLETE weather paths [B,M,D] and target-only context
    summaries [B,context_dim]. No target truth or observed future weather inputs.
    Returned prototypes are convex averages of complete input scenarios.
    This is a design component, NOT a verified novel architecture.
    """
    def __init__(self, path_dim: int, context_dim: int = 80, k: int = 3):
        super().__init__()
        self.k, self.path_dim = k, path_dim
        self.encoder = nn.Sequential(nn.Linear(path_dim, 32), nn.GELU(), nn.Linear(32, 16))
        self.queries = nn.Linear(context_dim, k * 16)
        self.mix = nn.Linear(context_dim, k)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, paths: torch.Tensor, context: torch.Tensor):
        if paths.ndim != 3 or paths.shape[-1] != self.path_dim or len(paths) != len(context):
            raise ValueError('Scenario shape mismatch.')
        keys = self.encoder(paths)
        queries = self.queries(context).reshape(len(paths), self.k, 16)
        a = (torch.einsum('bkd,bmd->bkm', queries, keys) / math.sqrt(16)).softmax(-1)
        prototypes = torch.einsum('bkm,bmd->bkd', a, paths)
        return prototypes, self.mix(context).softmax(-1), a
