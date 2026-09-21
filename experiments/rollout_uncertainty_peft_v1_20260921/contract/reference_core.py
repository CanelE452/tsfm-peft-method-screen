"""CPU-testable parts of the proposed rollout experiment.

This is NOT a Chronos trainer. In particular it does not download data, attach
LoRA, call the official native branching predictor, or verify GPU parity.
All arrays of quantiles in this module use [batch, horizon, quantile].
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch
from torch import nn

LEVELS = np.arange(1, 10, dtype=np.float64) / 10
CONTEXT = 512
BLOCK = 64
STEPS = 4
HORIZON = BLOCK * STEPS
PATCH = 16


def ordered_quantiles(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    if q.ndim != 3 or q.shape[-1] != 9 or not np.isfinite(q).all():
        raise ValueError('Expected finite [batch, horizon, 9] quantiles')
    return np.sort(q, axis=-1)


@dataclass(frozen=True)
class RolloutState:
    values: np.ndarray
    generated: np.ndarray
    lead: np.ndarray
    width: np.ndarray
    initial_scale: np.ndarray
    completed: int = 0

    @classmethod
    def initial(cls, x: np.ndarray, train_scale: np.ndarray) -> 'RolloutState':
        x = np.asarray(x, dtype=np.float64)
        scale = np.asarray(train_scale, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or x.shape[1] != CONTEXT:
            raise ValueError('Initial input must have exactly 512 observations')
        if len(scale) != len(x) or np.any(scale <= 0):
            raise ValueError('Positive TRAIN scale per series required')
        if not np.isfinite(x).all() or not np.isfinite(scale).all():
            raise ValueError('Nonfinite inputs')
        s0 = np.maximum(np.std(x, axis=1, ddof=0), 1e-6 * scale)
        z = np.zeros_like(x)
        return cls(x.copy(), z.copy(), z.copy(), z.copy(), s0, 0)

    def append(self, quantiles: np.ndarray) -> 'RolloutState':
        if self.completed >= STEPS:
            raise ValueError('Rollout already complete')
        q = ordered_quantiles(quantiles)
        if q.shape[:2] != (len(self.values), BLOCK):
            raise ValueError('Exactly 64 predictions per call required')
        median = q[..., 4]
        width = q[..., 8] - q[..., 0]
        leads = np.arange(self.completed * BLOCK + 1,
                          (self.completed + 1) * BLOCK + 1)
        leads = np.broadcast_to(leads, median.shape).astype(float)
        # Keep the complete initial history. Context lengths for the four calls
        # are 512, 576, 640, 704; no artificial 512-step rolling truncation.
        return RolloutState(
            np.concatenate([self.values, median], axis=1),
            np.concatenate([self.generated, np.ones_like(median)], axis=1),
            np.concatenate([self.lead, leads], axis=1),
            np.concatenate([self.width, width], axis=1),
            self.initial_scale.copy(), self.completed + 1,
        )

    def metadata(self, mode: str) -> np.ndarray:
        if mode not in {'STATE', 'UNCERTAINTY'}:
            raise ValueError(mode)
        if self.values.shape[1] % PATCH:
            raise ValueError('Unaligned context')
        m = self.generated
        a = m * self.lead / HORIZON
        b = m * self.completed / (STEPS - 1)
        # STATE uses a genuine horizon-only basis, not an unused dummy weight.
        last = (m * (self.lead / HORIZON)**2 if mode == 'STATE'
                else m * np.log1p(self.width / self.initial_scale[:, None]))
        z = np.stack([m, a, b, last], axis=-1)
        return z.reshape(len(m), -1, PATCH, 4).mean(axis=2)


def rollout_numpy(forecast: Callable, x: np.ndarray, train_scale: np.ndarray,
                  mode: str = 'UNCERTAINTY') -> np.ndarray:
    """No targets are accepted by the forecasting interface."""
    state = RolloutState.initial(x, train_scale)
    out = []
    for _ in range(STEPS):
        q = ordered_quantiles(forecast(state.values.copy(), state.metadata(mode)))
        out.append(q)
        state = state.append(q)
    return np.concatenate(out, axis=1)


class MetadataAdapter(nn.Module):
    """Proposed shared bottleneck, applied before encoder attention.

The metadata builder chooses STATE or UNCERTAINTY. This module is identical
in the two arms. The pretrained model and LoRA are outside this class.
"""
    def __init__(self, d_model: int = 512, rank: int = 8):
        super().__init__()
        self.down = nn.Linear(d_model + 4, rank)
        self.up = nn.Linear(rank, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        if h.shape[:-1] != metadata.shape[:-1] or metadata.shape[-1] != 4:
            raise ValueError('Patch metadata and patch embeddings must align')
        z = metadata.to(device=h.device, dtype=h.dtype)
        delta = self.up(torch.nn.functional.gelu(self.down(torch.cat([h, z], dim=-1))))
        return h + z[..., :1] * delta


def scaled_pinball(y: np.ndarray, q: np.ndarray, scale: np.ndarray) -> float:
    """Mean over origins, series, horizons, quantiles; equal-size panels only.
    q [origin, series, horizon, 9], y [origin, series, horizon].
    """
    y, q, scale = np.asarray(y), np.asarray(q), np.asarray(scale)
    if q.shape != y.shape + (9,) or y.ndim != 3 or scale.shape != (y.shape[1],):
        raise ValueError('Shape mismatch')
    if np.any(scale <= 0) or not all(np.isfinite(a).all() for a in (y, q, scale)):
        raise ValueError('Nonfinite or nonpositive scale')
    e = y[..., None] - q
    loss = 2 * np.maximum(LEVELS * e, (LEVELS - 1) * e)
    return float(np.mean(loss / scale[None, :, None, None]))


def calibration_metrics(y: np.ndarray, q: np.ndarray, scale: np.ndarray) -> dict:
    """PCE computed within each series then averaged, not pooled across scales."""
    y, q, scale = np.asarray(y), np.asarray(q), np.asarray(scale)
    if q.shape != y.shape + (9,) or y.ndim != 3:
        raise ValueError('Shape mismatch')
    if np.any(np.diff(q, axis=-1) < 0):
        raise ValueError('Report raw crossing separately; feed ordered quantiles')
    freq = np.mean(y[..., None] <= q, axis=(0, 2))
    pce = float(np.mean(np.abs(freq - LEVELS)))
    cover = np.mean((q[..., 0] <= y) & (y <= q[..., -1]), axis=(0, 2))
    width = np.mean((q[..., -1] - q[..., 0]) / scale[None, :, None])
    return {'pce': pce, 'coverage80': float(np.mean(cover)),
            'overconfidence80': float(0.8 - np.mean(cover)),
            'width80_train_scaled': float(width)}


def affine_calibration(q: np.ndarray, scale: np.ndarray,
                       alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    q, scale = np.asarray(q), np.asarray(scale)
    if q.ndim != 4 or q.shape[-2:] != (HORIZON, 9):
        raise ValueError('Expected [origin, series, 256, 9]')
    if np.shape(alphas) != (STEPS,) or np.shape(betas) != (STEPS,) or np.any(alphas <= 0):
        raise ValueError('Four positive width multipliers and four shifts required')
    out = q.copy()
    for k in range(STEPS):
        sl = slice(k * BLOCK, (k + 1) * BLOCK)
        if alphas[k] == 1 and betas[k] == 0:
            continue  # preserve the identity option bit-for-bit
        part = q[:, :, sl]
        med = part[..., 4:5]
        out[:, :, sl] = med + betas[k] * scale[None, :, None, None] + alphas[k] * (part - med)
    return out


def role_origins(T: int) -> dict[str, np.ndarray]:
    """Explicit experimental 60/10/10/20 split, NOT the official ETT split.
    Every role contains all horizon-256 targets; historical contexts may precede
    role starts. Daily stride is a compute design choice, not event selection.
    """
    edges = [0, int(.6*T), int(.7*T), int(.8*T), T]
    roles = ['TRAIN', 'CALIBRATION', 'VALIDATION', 'TEST']
    out = {}
    for name, lo, hi in zip(roles, edges[:-1], edges[1:]):
        first = max(lo, CONTEXT)
        first = ((first + 23) // 24) * 24
        out[name] = np.arange(first, hi - HORIZON + 1, 24, dtype=np.int64)
    return out
