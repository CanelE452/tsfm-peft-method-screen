from __future__ import annotations
import math
import torch
from .util import require


def load_chronos(cfg):
    import importlib.metadata as md
    ver = md.version('chronos-forecasting')
    require(ver == cfg['model']['chronos_version_expected'], f'chronos version {ver}')
    from chronos import Chronos2Pipeline
    device = 'cuda' if cfg['model']['device'] == 'cuda' and torch.cuda.is_available() else 'cpu'
    require(device == 'cuda', 'CUDA required for this actual training screen')
    pipe = Chronos2Pipeline.from_pretrained(cfg['model']['id'], device_map=device, dtype=torch.float32)
    return pipe, device


def direct_quantiles(base, context, horizon, group_ids):
    rows = context.shape[0]
    z = torch.zeros((rows, horizon), device=context.device, dtype=context.dtype)
    out = base(
        context=context,
        context_mask=torch.ones_like(context),
        future_covariates=z,
        future_covariates_mask=torch.zeros_like(z),
        future_target=z,
        future_target_mask=torch.ones_like(z),
        group_ids=group_ids,
        num_output_patches=math.ceil(horizon / base.chronos_config.output_patch_size),
    )
    return out.quantile_preds[..., :horizon]


@torch.no_grad()
def deterministic_contract(base, context, horizon, group_ids):
    a = direct_quantiles(base, context, horizon, group_ids)
    b = direct_quantiles(base, context.clone(), horizon, group_ids)
    d = float((a - b).abs().max())
    require(d <= 1e-7, f'Non-deterministic direct inference diff={d}')
    return d
