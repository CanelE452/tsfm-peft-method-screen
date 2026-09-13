"""Train/V-only factorial anchoring and bounded event-branch diagnostics."""
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from .backbone import load_base, forecast, native_loss
from .lora import attach, disabled
from .candidates.dualclock import EventAdapter, event_features
from .overnight.methods import raw_loss


class DiagnosticEventAdapter(EventAdapter):
    mode = 'normal'

    def forward(self, h, state):
        if self.mode == 'drop_adapter':
            return h
        seq, lengths, summary = state
        if self.mode == 'rotate_events':
            seq, lengths, summary = seq.roll(1, 0), lengths.roll(1, 0), summary.roll(1, 0)
        if self.full:
            packed = nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
            _, e = self.encoder(packed)
            e = e[-1]
        else:
            e = torch.tanh(self.encoder(summary))
        if self.mode == 'zero_events':
            e = torch.zeros_like(e)
        e = e[:, None, :].expand(-1, h.shape[1], -1)
        return h + self.up(torch.nn.functional.silu(self.down(torch.cat([h, e], -1))))


class DiagnosticModel(nn.Module):
    def __init__(self, topic, arm, seed):
        super().__init__()
        self.topic, self.arm = topic, arm
        self.core = load_base()
        attach(self.core, seed=seed)
        self.adapter = None
        if topic == 'dualclock' and arm != 'standard':
            torch.manual_seed(seed + 2000)
            self.adapter = DiagnosticEventAdapter(self.core.config.d_model, arm == 'dualclock').cuda()

    def forward(self, x, groups, frozen=False, ablation='normal'):
        with disabled(self.core) if frozen else nullcontext():
            if self.topic == 'dualclock':
                adapter = None if frozen else self.adapter
                if adapter is not None:
                    adapter.mode = ablation
                try:
                    return forecast(self.core, x, groups, adapter=adapter,
                                    state=event_features(x) if adapter is not None else None)
                finally:
                    if adapter is not None:
                        adapter.mode = 'normal'
            enc, (loc, scale), _, _ = self.core.encode(context=x, group_ids=groups, num_output_patches=3)
            h = enc.last_hidden_state[:, -3:]
            z = self.core.output_patch_embedding(h).reshape(len(x), 3, 21, 16)
            z = z.permute(0, 2, 1, 3).reshape(len(x), 21, 48).float()
            return z, z.sinh() * scale[:, None, :] + loc[:, None, :], loc, scale


def objective(topic, arm, z, p, y, loc, scale, channel_scale, teacher=None, coefficient=.1):
    if topic == 'dualclock':
        # Keep original optimizer scaling for the exact 360-step prefix comparison.
        return native_loss(z, y, loc, scale), p.new_zeros(())
    task = native_loss(z, y, loc, scale)/21 if arm.startswith('native') else raw_loss(p, y, channel_scale)
    reg = p.new_zeros(())
    if arm.endswith('_anchor'):
        assert teacher is not None and teacher.shape == p.shape
        reg = coefficient * ((p.sort(dim=1).values-teacher.sort(dim=1).values).abs()
                             / channel_scale[:, None, None]).mean()
    return task, reg


def schedule(topic, origins, seed, steps):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(steps):
        if topic == 'dualclock':
            oo = rng.choice(np.asarray(origins), 8).tolist()
            cs = [[int(c)] for c in rng.integers(0, 256, 8)]
        else:
            oo = [origins[i] for i in rng.integers(len(origins), size=2)]
            cs = [list(range(4)) for _ in oo]
        rows.append(dict(origins=oo, channels=cs))
    return rows


def choose(rows, maximum_step=None):
    pool = rows if maximum_step is None else [r for r in rows if r['step'] <= maximum_step]
    return min(pool, key=lambda r: (r['metrics']['scaled_2pinball'], r['step'], r['lr']))
