"""Fixed MAG implementation from the paper, independent of experiment runners.

Supported contract: Chronos-Bolt-small FP32, complete 512-value univariate
contexts, 32 patches, 64 future points and nine quantiles. No training driver.
"""
import hashlib
import math
from pathlib import Path
import torch
from torch import nn

CONFIG = dict(context=512, patch=16, hidden=512, rank=8, horizon=64,
              quantiles=9, threshold=3., mad_multiplier=1.4826, scale_floor=.1)

def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def validate_inputs(x, sigma):
    if x.ndim != 2 or x.shape[-1] != 512 or sigma.shape != x.shape[:1] or len(x) == 0:
        raise ValueError('Expected nonempty observed [batch,512] and TRAIN sigma [batch]')
    if x.dtype != torch.float32 or sigma.dtype != torch.float32 or x.device != sigma.device:
        raise ValueError('Observed and sigma must be FP32 on the same device')
    if not torch.isfinite(x).all() or not torch.isfinite(sigma).all() or not (sigma > 0).all():
        raise ValueError('Complete finite observations and positive finite TRAIN sigma are required')

def magnitude_gate(observed, sigma):
    validate_inputs(observed, sigma)
    center = observed.quantile(.5, dim=-1, keepdim=True)
    scale = torch.maximum(1.4826 * (observed-center).abs().quantile(.5, dim=-1, keepdim=True), .1*sigma[:, None])
    return 1 - (((observed-center)/scale).abs() > 3).to(observed).reshape(len(observed), 32, 16).mean(-1)

class MagnitudeResidual(nn.Module):
    """The existing 8,712-parameter residual, with identical RNG and state keys."""
    def __init__(self, seed=0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+200000)
            self.down = nn.Linear(512, 8)
            self.up = nn.Linear(8, 512)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, hidden, gate):
        cap = hidden.square().mean(-1).sqrt().quantile(.5, dim=-1, keepdim=True).clamp_min(1e-6).detach()[..., None]
        correction = cap*torch.tanh(self.up(nn.functional.gelu(self.down(hidden))))
        return hidden + gate[..., None]*correction

class MAGForecast(nn.Module):
    """Wrap an already adapted B0; the base is frozen and raw input is retained."""
    def __init__(self, base, seed=0):
        super().__init__()
        if base.dtype != torch.float32 or base.config.d_model != 512:
            raise ValueError('Only the audited FP32 Chronos-Bolt-small contract is supported')
        self.base = base.requires_grad_(False).eval()
        self.adapter = MagnitudeResidual(seed).to(device=next(base.parameters()).device)
        self.eval()

    def forward(self, observed, sigma, *, adapter_off=False):
        validate_inputs(observed, sigma)
        b = self.base
        normalized, loc_scale = b.instance_norm(observed)
        patches = b.patch(normalized.to(b.dtype))
        patch_mask = b.patch(torch.ones_like(observed))
        if patches.shape[1:] != (32, 16):
            raise ValueError('Backbone patch contract does not match the fixed method')
        hidden = b.input_patch_embedding(torch.cat([patches, patch_mask], dim=-1))
        if not adapter_off:
            hidden = self.adapter(hidden, magnitude_gate(observed, sigma))
        attention_mask = (patch_mask.sum(dim=-1) > 0).to(b.dtype)
        if b.chronos_config.use_reg_token:
            reg = torch.full((len(observed), 1), b.config.reg_token_id, device=observed.device, dtype=torch.long)
            hidden = torch.cat([hidden, b.shared(reg)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(reg, dtype=b.dtype)], dim=1)
        encoded = b.encoder(attention_mask=attention_mask, inputs_embeds=hidden)[0]
        output = b.decode(hidden, attention_mask, encoded)
        quantiles = b.output_patch_embedding(output).reshape(len(observed), 9, 64)
        return b.instance_norm.inverse(quantiles.flatten(1), loc_scale).reshape(len(observed), 9, 64)

class _B0LoRA(nn.Module):
    """Restores the existing rank8/alpha16 q/v B0; not a new LoRA method."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(8, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, 8))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + 2*nn.functional.linear(nn.functional.linear(x, self.a), self.b)

def _attach_existing_lora(base):
    targets = [(n, m) for n, m in base.named_modules() if isinstance(m, nn.Linear)
               and n.split('.')[-1] in {'q', 'v'} and any(t in n for t in ['SelfAttention', 'EncDecAttention'])]
    if len(targets) != 36 or not all(tuple(m.weight.shape) == (512, 512) for _, m in targets):
        raise ValueError('B0 LoRA target contract mismatch')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        for name, layer in targets:
            parent, leaf = name.rsplit('.', 1)
            setattr(base.get_submodule(parent), leaf, _B0LoRA(layer))

def save_bundle(model, destination, snapshot, provenance):
    """Export only B0 LoRA and MAG tensors; the original snapshot is separate."""
    snapshot = Path(snapshot)
    lora = {n: p.detach().cpu().clone() for n, p in model.base.named_parameters() if n.endswith(('.a', '.b'))}
    if len(lora) != 72 or sum(t.numel() for t in lora.values()) != 294912:
        raise ValueError('Expected the already adapted 294,912-parameter B0')
    bundle = dict(schema=1, method='MAG_ONLY_FIXED', config=CONFIG,
                  snapshot_files={n: file_hash(snapshot/n) for n in ['config.json', 'model.safetensors']},
                  base_state_sha256=state_hash(model.base.state_dict()),
                  lora=lora, adapter={n: t.detach().cpu().clone() for n, t in model.adapter.state_dict().items()},
                  provenance=dict(provenance))
    torch.save(bundle, destination)

def load_bundle(snapshot, bundle_path, device='cpu'):
    """Load hash-matched local assets only. No download, optimizer or fitting."""
    from chronos import ChronosBoltPipeline
    snapshot = Path(snapshot)
    if not snapshot.is_dir():
        raise ValueError('A local pretrained snapshot directory is required')
    bundle = torch.load(bundle_path, map_location='cpu', weights_only=True)
    if bundle['schema'] != 1 or bundle['method'] != 'MAG_ONLY_FIXED' or bundle['config'] != CONFIG:
        raise ValueError('Unsupported or modified method contract')
    for name in ['config.json', 'model.safetensors']:
        if file_hash(snapshot/name) != bundle['snapshot_files'][name]:
            raise ValueError('Pretrained snapshot hash mismatch: '+name)
    base = ChronosBoltPipeline.from_pretrained(str(snapshot), device_map='cpu', torch_dtype=torch.float32, local_files_only=True).model
    base.eval().requires_grad_(False)
    for module in base.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.
        elif isinstance(getattr(module, 'dropout', None), float):
            module.dropout = 0.
    _attach_existing_lora(base)
    lora = {n: p for n, p in base.named_parameters() if n.endswith(('.a', '.b'))}
    if lora.keys() != bundle['lora'].keys():
        raise ValueError('B0 state keys mismatch')
    with torch.no_grad():
        for name, value in bundle['lora'].items():
            lora[name].copy_(value)
    if state_hash(base.state_dict()) != bundle['base_state_sha256']:
        raise ValueError('Restored B0 state mismatch')
    model = MAGForecast(base)
    model.adapter.load_state_dict(bundle['adapter'], strict=True)
    return model.eval().to(device)
