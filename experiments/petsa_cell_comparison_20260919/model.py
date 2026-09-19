"""Offline Chronos transfer of PETSA's GCM cell, not its full TTA method."""
import math
import torch
from torch import nn

class GCM(nn.Module):
    """The official var_wise=True cell; correction exposed for exact identity."""
    def __init__(self, window_len, n_var=1, low_rank=16, gating_init=.01):
        super().__init__()
        self.gating = nn.Parameter(torch.full((n_var,), gating_init))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))
        self.lora_A = nn.Parameter(torch.empty(window_len, low_rank))
        self.lora_B = nn.Parameter(torch.zeros(low_rank, window_len, n_var))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def correction(self, x):
        weight = torch.einsum('ik,kjl->ijl', self.lora_A, self.lora_B)
        return torch.einsum('biv,iov->bov', torch.tanh(self.gating*x), weight) + self.bias

    def forward(self, x):
        return x + self.correction(x)

class PetsaForecast(nn.Module):
    def __init__(self, b0, seed):
        super().__init__()
        self.b0 = b0.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+200000)
            self.in_cali = GCM(512)
            self.out_cali = GCM(64)

    @property
    def base(self):
        return self.b0.base

    def forward(self, x, sigma, off=False):
        if off:
            return self.b0(x, sigma)
        assert x.ndim == 2 and x.shape[-1] == 512
        assert sigma.shape == x.shape[:1] and torch.isfinite(sigma).all() and (sigma > 0).all()
        center = x.mean(-1, keepdim=True).detach()
        scale = sigma[:, None]
        # Same observation-only coordinates as the earlier delta XY control.
        # Add only the correction, avoiding normalization round-trip at step 0.
        xx = x + scale*self.in_cali.correction(((x-center)/scale)[..., None])[..., 0]
        pred = self.b0(xx, sigma)
        normalized = ((pred-center[:, None, :])/scale[:, None, :]).reshape(-1, 64, 1)
        correction = self.out_cali.correction(normalized).reshape_as(pred)
        # One shared module across the nine quantiles; no rank sorting or E calibration.
        return pred + scale[:, None, :]*correction
