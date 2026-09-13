"""Implementation for the fixed equal-time plan; historical models unchanged."""
from contextlib import contextmanager
import random
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .model import ForecastModel
from .checkpoint_diagnostic import DiagnosticModel


def adapted_head(head, h, enabled):
    return checkpoint(head, h, use_reentrant=False) if enabled else head(h)


def adapted_side(side, features, time_mask, group_mask, base, enabled):
    h = torch.zeros(*features[0].shape[:2], 64, device=base.device, dtype=base.dtype)
    for layer, frozen in zip(side.layers, features):
        args = (h, frozen, time_mask, group_mask)
        h = checkpoint(layer, *args, use_reentrant=False) if enabled else layer(*args)
    return base + side.out(nn.functional.layer_norm(h[:, -3:], (64,)))


class EqualTimeModel(DiagnosticModel):
    def __init__(self, arm, seed):
        assert arm in ('standard', 'head', 'side', 'query')
        ForecastModel.__init__(self, arm, seed)
        self.checkpoint_enabled = False
        self.observer = None

    def forward(self, x, g):
        if self.arm in ('standard', 'query'):
            return super().forward(x, g)
        c, loc, scale = self.encode_frozen(x, g)
        if self.arm == 'head':
            h = adapted_head(self.adapter, c['base'], self.checkpoint_enabled)
        else:
            h = adapted_side(self.adapter, c['features'], c['time_mask'],
                             c['group_mask'], c['base'], self.checkpoint_enabled)
        z = self.base.output_patch_embedding(h).reshape(len(x),3,21,16).permute(0,2,1,3).reshape(len(x),21,48).float()
        return z, z.sinh()*scale[:,None,:]+loc[:,None,:], loc, scale


class TrainingClock:
    """Whole updates cross boundaries; excessive overshoot invalidates the run."""
    def __init__(self, boundaries=(10.,20.,30.), tolerance=.5, max_updates=4096):
        self.boundaries = list(boundaries)
        self.tolerance = tolerance
        self.max_updates = max_updates
        self.elapsed = 0.
        self.updates = 0
        self.index = 0

    @property
    def complete(self):
        return self.index == len(self.boundaries)

    def add(self, seconds):
        if self.complete:
            raise RuntimeError('No updates after the final time checkpoint')
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError('Invalid measured step time')
        self.elapsed += seconds
        self.updates += 1
        boundary = self.boundaries[self.index]
        if self.elapsed >= boundary:
            if self.elapsed-boundary > self.tolerance:
                raise RuntimeError('INVALID_TIMING: checkpoint overshoot')
            self.index += 1
            return boundary
        if self.updates >= self.max_updates:
            raise RuntimeError('INVALID_TIMING: update cap before time budget')
        return None


def choose_storage(records, limit):
    candidates = []
    for cp in (False, True):
        rows = [r for r in records if r['checkpoint'] == cp]
        assert len(rows) == 3
        peak = max(r['peak_allocated_bytes'] for r in rows)
        if peak <= limit:
            candidates.append(dict(checkpoint=cp, median_seconds=float(np.median([r['seconds'] for r in rows])),
                                   peak_allocated_bytes=peak))
    if not candidates:
        raise RuntimeError('No numerically valid storage option within the memory cap')
    fastest = min(r['median_seconds'] for r in candidates)
    tied = [r for r in candidates if r['median_seconds'] <= fastest*1.02]
    return min(tied, key=lambda r:(r['peak_allocated_bytes'], r['checkpoint']))


@contextmanager
def preserve_rng():
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])
