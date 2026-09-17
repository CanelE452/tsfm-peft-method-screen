"""Explicit Chronos-Bolt patch adapter; no generator metadata enters forward.

Initial CPU parity and actual GPU training checks are audited separately.
"""
import hashlib
import math
import torch
from torch import nn


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class LoRA(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(8, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.b = nn.Parameter(torch.zeros(base.out_features, 8, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + 2 * nn.functional.linear(nn.functional.linear(x, self.a), self.b)


def attach_lora(base, seed):
    targets = [(name, module) for name, module in base.named_modules()
               if isinstance(module, nn.Linear) and name.split('.')[-1] in {'q','v'}
               and any(t in name for t in ['SelfAttention', 'EncDecAttention'])]
    assert len(targets) == 36
    assert all(tuple(m.weight.shape) == (512,512) for _,m in targets)
    receipt = []
    # Common LoRA initialization is independent of any subsequent extra adapter.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for name, module in targets:
            parent, leaf = name.rsplit('.',1)
            setattr(base.get_submodule(parent), leaf, LoRA(module))
            receipt.append(dict(name=name, shape=list(module.weight.shape), rank=8, alpha=16))
    return receipt


def robust_scale(x, sigma):
    med = x.quantile(.5, dim=-1, keepdim=True)
    scale = torch.maximum(1.4826*(x-med).abs().quantile(.5, dim=-1, keepdim=True), .1*sigma[:,None])
    return med, scale


def preprocess(x, sigma, arm):
    if arm in {'A0','A1','FROZEN_RAW'}:
        return x
    med, scale = robust_scale(x, sigma)
    if arm in {'A2','A4','A5','FROZEN_CLIP6'}:
        return torch.maximum(torch.minimum(x, med+6*scale), med-6*scale)
    if arm in {'A3','FROZEN_HAMPEL'}:
        # Single-pass centered window, truncated at the observed context edges.
        windows = nn.functional.pad(x, (12,12), value=float('nan')).unfold(-1,25,1)
        local_med = torch.nanquantile(windows, .5, dim=-1)
        local_scale = torch.maximum(1.4826*torch.nanquantile((windows-local_med[...,None]).abs(), .5, dim=-1), .1*sigma[:,None])
        return torch.where((x-local_med).abs()>4*local_scale, local_med, x)
    raise ValueError(arm)


class BoundedAdapter(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+100000)
            self.down = nn.Linear(32,8)
            self.up = nn.Linear(8,512)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, embeds, features):
        cap = embeds.square().mean(-1).sqrt().quantile(.5,dim=-1,keepdim=True).clamp_min(1e-6).detach()[...,None]
        delta = cap*torch.tanh(self.up(nn.functional.gelu(self.down(features))))
        return embeds+delta


class ForecastModel(nn.Module):
    def __init__(self, base, arm, seed):
        super().__init__()
        self.base, self.arm = base, arm
        self.adapter = BoundedAdapter(seed) if arm in {'A4','A5'} else None

    def forward(self, observed, sigma, residual_mode="normal"):
        assert observed.ndim == 2 and observed.shape[-1] == 512
        assert sigma.shape == observed.shape[:1]
        assert torch.isfinite(observed).all() and (sigma>0).all()
        b = self.base
        context = preprocess(observed, sigma, self.arm)
        mask = torch.ones_like(context)
        normalized, loc_scale = b.instance_norm(context)
        normalized = normalized.to(b.dtype)
        z = b.patch(normalized)
        patch_mask = b.patch(mask)
        assert z.shape == (len(observed),32,16) and patch_mask.shape == z.shape
        embeds = b.input_patch_embedding(torch.cat([z, patch_mask], dim=-1))
        if self.adapter is not None:
            first = torch.asinh(z).clamp(-6,6)
            if self.arm == 'A4':
                second = torch.tanh(z)
            else:
                _, scale = robust_scale(observed, sigma)
                discarded = b.patch((observed-context)/scale)
                second = torch.asinh(discarded).clamp(-6,6)
                if residual_mode == "zero": second = torch.zeros_like(second)
                elif residual_mode == "permute":
                    permutation = torch.randperm(32, generator=torch.Generator().manual_seed(82400)).to(second.device)
                    second = second[:, permutation]
                else: assert residual_mode == "normal"
            embeds = self.adapter(embeds, torch.cat([first,second],dim=-1))
        attention_mask = (patch_mask.sum(dim=-1)>0).to(b.dtype)
        if b.chronos_config.use_reg_token:
            reg = torch.full((len(observed),1), b.config.reg_token_id, device=observed.device, dtype=torch.long)
            embeds = torch.cat([embeds,b.shared(reg)],dim=1)
            attention_mask = torch.cat([attention_mask,torch.ones_like(reg,dtype=b.dtype)],dim=1)
        hidden = b.encoder(attention_mask=attention_mask,inputs_embeds=embeds)[0]
        output = b.decode(embeds,attention_mask,hidden)
        quantiles = b.output_patch_embedding(output).reshape(len(observed),9,64)
        return b.instance_norm.inverse(quantiles.flatten(1), loc_scale).reshape(len(observed),9,64)


def loss_2pinball(pred, target, sigma, quantiles):
    error = target[:,None,:]-pred
    q = quantiles[None,:,None].to(pred)
    return (2*torch.maximum(q*error,(q-1)*error)/sigma[:,None,None]).mean()
