from __future__ import annotations
import math
import torch
from torch import nn
from .util import require


class ScaleLoRALinear(nn.Module):
    def __init__(self, base, n_scales, rank, alpha, mode):
        super().__init__()
        require(isinstance(base, nn.Linear), 'Expected linear')
        self.base = base
        self.base.requires_grad_(False)
        self.n_scales = n_scales
        self.rank = rank
        self.alpha = alpha
        self.mode = mode
        self.active = 0
        if mode == 'independent':
            self.A = nn.Parameter(torch.empty(n_scales, rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
            self.B = nn.Parameter(torch.zeros(n_scales, base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
            for s in range(n_scales):
                nn.init.kaiming_uniform_(self.A[s], a=math.sqrt(5))
        elif mode in ('shared', 'gated_shared'):
            self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
            self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            if mode == 'gated_shared':
                self.gate = nn.Parameter(torch.ones(n_scales, rank, device=base.weight.device, dtype=base.weight.dtype))
        else:
            raise ValueError(mode)

    def forward(self, x):
        if self.mode == 'independent':
            a = torch.nn.functional.linear(x, self.A[self.active])
            u = torch.nn.functional.linear(a, self.B[self.active])
        else:
            a = torch.nn.functional.linear(x, self.A)
            if self.mode == 'gated_shared':
                a = a * self.gate[self.active]
            u = torch.nn.functional.linear(a, self.B)
        return self.base(x) + (self.alpha / self.rank) * u


def install_scale_lora(base, n_scales, rank, alpha, mode):
    import chronos.chronos2.layers as layers
    base.requires_grad_(False)
    wrapped = []
    parents = [(n, m) for n, m in base.named_modules() if isinstance(m, (layers.TimeSelfAttention, layers.GroupSelfAttention))]
    require(parents, 'No Chronos attention modules found')
    for name, parent in parents:
        att = parent.self_attention
        for p in ('q', 'k', 'v', 'o'):
            old = getattr(att, p)
            new = ScaleLoRALinear(old, n_scales, rank, alpha, mode)
            setattr(att, p, new)
            wrapped.append((name + '.' + p, new))
    require(len(wrapped) > 0, 'No attention projections wrapped')
    return wrapped


def set_scale(wrapped, s):
    for _, m in wrapped:
        m.active = int(s)
