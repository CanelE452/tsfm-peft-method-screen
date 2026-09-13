"""Monotone center/gap feature PEFT and transactional shape-update acceptance."""
import copy
import math
import torch
from torch import nn
from ..backbone import QUANTILES

class Branch(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.down = nn.Linear(768, 8, bias=False)
        self.up = nn.Linear(8, outputs, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden):
        return self.up(nn.functional.silu(self.down(hidden.float())))

class SplitAdapter(nn.Module):
    def __init__(self, seed=30000, center_only=False):
        super().__init__()
        torch.manual_seed(seed + 4000)
        self.center = Branch(16)
        self.shape = Branch(320)
        self.center_only = center_only
        if center_only:
            self.shape.requires_grad_(False)

    def forward(self, h, f0):
        """h: N,3,768 standardized frozen features; f0: N,21,48 raw/context-scale."""
        center = f0[:, 10:11] + self.center(h).reshape(-1, 1, 48)
        gap_delta = self.shape(h).reshape(-1, 3, 20, 16).permute(0, 2, 1, 3).reshape(-1, 20, 48)
        # Fixed, common bound for all arms. It is not the proposed contribution.
        log_multiplier = .5 * torch.tanh(gap_delta / .5)
        gaps = (f0[:, 1:] - f0[:, :-1]) * torch.exp(log_multiplier)
        left = torch.flip(torch.cumsum(torch.flip(gaps[:, :10], [1]), 1), [1])
        right = torch.cumsum(gaps[:, 10:], 1)
        p = torch.cat((center - left, center, center + right), 1)
        return p, log_multiplier

def per_origin_loss(p, target, channels=4):
    """Native asinh 2-pinball; mean horizon, sum quantiles, mean channels."""
    valid = torch.isfinite(target)
    safe = torch.where(valid, target, torch.zeros_like(target))
    residual = safe.asinh()[:, None] - p.asinh()
    q = torch.as_tensor(QUANTILES, device=p.device, dtype=p.dtype)[None, :, None]
    losses = (2 * torch.maximum(q * residual, (q - 1) * residual) * valid[:, None]).mean(-1).sum(-1)
    return losses.reshape(-1, channels).mean(1)

def acceptance(differences, block_ids, block_margin=1.):
    """A heuristic block variability penalty, NOT a valid confidence interval.

    Reused training labels and dependent histories prevent independent testing.
    Equal-size blocks ensure the margin=0 control has the same pooled mean.
    """
    blocks = torch.unique(block_ids, sorted=True)
    means = torch.stack([differences[block_ids == b].double().mean() for b in blocks])
    assert len(means) >= 2 and torch.isfinite(means).all()
    mean = means.mean()
    penalty = means.std(unbiased=True) / math.sqrt(len(means))
    bound = mean + block_margin * penalty
    return dict(accept=bool(bound <= 0), mean_change=float(mean),
                block_standard_error_heuristic=float(penalty), criterion=float(bound),
                block_changes=means.cpu().tolist())

def snapshot_transaction(module, optimizer):
    return ({n: p.detach().clone() for n, p in module.named_parameters()},
            copy.deepcopy(optimizer.state_dict()))

def rollback_transaction(module, optimizer, snapshot):
    parameters, optimizer_state = snapshot
    with torch.no_grad():
        for n, p in module.named_parameters():
            p.copy_(parameters[n])
    optimizer.load_state_dict(optimizer_state)
