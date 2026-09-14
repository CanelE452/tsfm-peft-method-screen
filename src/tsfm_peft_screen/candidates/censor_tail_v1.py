from __future__ import annotations
import torch


def log_survival_from_quantiles(
    prediction: torch.Tensor,
    threshold: torch.Tensor,
    levels: torch.Tensor,
    observed_train_scale: torch.Tensor,
    *,
    gap_floor_relative: float = 1e-4,
) -> torch.Tensor:
    """Return log P(Y > threshold) for a continuous, exponential-tail proxy.

    prediction: [batch, quantile, horizon]; threshold: [batch, horizon]
    levels: increasing quantile levels in (0, 1)
    observed_train_scale: [batch], derived from capped training observations only.
    This is a distribution-completion assumption, not an identified count model.
    """
    if prediction.ndim != 3 or threshold.shape != (prediction.shape[0], prediction.shape[2]):
        raise ValueError('Expected prediction [B,K,H] and threshold [B,H]')
    if levels.ndim != 1 or len(levels) != prediction.shape[1] or len(levels) < 2:
        raise ValueError('Quantile levels must have shape [K], K >= 2')
    if observed_train_scale.shape != (prediction.shape[0],):
        raise ValueError('Expected observed_train_scale [B]')
    if not (gap_floor_relative > 0):
        raise ValueError('Gap floor must be positive')
    # Deliberate small-tensor FP64 computation, also for a FP32 forecasting model.
    p = prediction.transpose(1, 2).to(torch.float64).sort(dim=-1).values
    y = threshold.to(device=p.device, dtype=torch.float64)
    tau = levels.to(device=p.device, dtype=torch.float64)
    scale = observed_train_scale.to(device=p.device, dtype=torch.float64)
    if not all(bool(torch.isfinite(v).all()) for v in (p, y, tau, scale)):
        raise ValueError('Inputs must be finite; filter missing observations first')
    if not (bool((scale > 0).all()) and bool((tau > 0).all())
            and bool((tau < 1).all()) and bool((tau[1:] > tau[:-1]).all())):
        raise ValueError('Invalid scale or quantile levels')
    eps = gap_floor_relative * scale.clamp_min(1.0)[:, None, None]
    gap = torch.maximum(p[..., 1:] - p[..., :-1], eps)
    knots = torch.cat((p[..., :1], p[..., :1] + gap.cumsum(dim=-1)), dim=-1)
    b_left = gap[..., 0] * tau[0] / (tau[1] - tau[0])
    b_right = gap[..., -1] * (1 - tau[-1]) / (tau[-1] - tau[-2])
    left = y <= knots[..., 0]
    right = y >= knots[..., -1]
    middle = ~(left | right)
    out = torch.empty_like(y)
    # Compute only active branches: avoid overflow in an unused torch.where branch.
    if bool(left.any()):
        log_cdf = tau[0].log() + (y[left] - knots[..., 0][left]) / b_left[left]
        out[left] = torch.log1p(-torch.exp(log_cdf))
    if bool(right.any()):
        out[right] = torch.log1p(-tau[-1]) - (y[right] - knots[..., -1][right]) / b_right[right]
    if bool(middle.any()):
        k = knots[middle].contiguous()
        t = y[middle]
        hi_idx = torch.searchsorted(k, t[:, None].contiguous(), right=True).squeeze(-1)
        lo_idx = hi_idx - 1
        lo = k.gather(-1, lo_idx[:, None]).squeeze(-1)
        hi = k.gather(-1, hi_idx[:, None]).squeeze(-1)
        frac = (t - lo) / (hi - lo)
        cdf = tau[lo_idx] + frac * (tau[hi_idx] - tau[lo_idx])
        out[middle] = torch.log1p(-cdf)
    return out


def survival_loss(raw, sale, censored, observed_scale):
    """Only active finite censored positions enter the probability calculation."""
    from ..backbone import QUANTILES
    valid = torch.isfinite(sale)
    if bool((censored & ~valid).any()):
        raise ValueError('Censored targets must have finite observed lower bounds')
    n = int(valid.sum())
    if n == 0:
        raise ValueError('No valid target positions')
    if not bool(censored.any()):
        return raw.reshape(-1)[:0].sum().double()
    # One position per temporary batch: no calculation on missing/uncensored raw.
    prediction = raw.transpose(1, 2)[censored][:, :, None]
    lower = sale[censored][:, None]
    scales = observed_scale[:, None].expand_as(sale)[censored]
    return -log_survival_from_quantiles(prediction, lower,
        raw.new_tensor(QUANTILES, dtype=torch.float64), scales).sum() / n


def objective(arm, z, raw, sale, censored, loc, scale, observed_scale, lambda_c):
    from ..backbone import native_loss
    if arm not in ('NAIVE', 'DROP', 'TAIL'):
        raise ValueError(arm)
    if bool((censored & ~torch.isfinite(sale)).any()):
        raise ValueError('Missing censored lower bound')
    task = native_loss(z, sale if arm == 'NAIVE' else sale.masked_fill(censored, float('nan')), loc, scale)
    # Exact shared graph when lambda=0, including dtype and optimizer arithmetic.
    if arm != 'TAIL' or lambda_c == 0 or not bool(censored.any()):
        return task, task, raw.reshape(-1)[:0].sum().double()
    tail = survival_loss(raw, sale, censored, observed_scale)
    return task + lambda_c * tail, task, tail


def tail_diagnostics(raw, sale, censored, observed_scale):
    """Detached output autograd only; no extra model backward or optimizer."""
    from ..backbone import QUANTILES
    p = raw.detach().transpose(1, 2)[censored].double().sort(-1).values
    if not len(p):
        return dict(censored=0, right=0, left=0, floored_gaps=0, gaps=0,
                    max_knot_shift=0., strong_right=0, strong_shift_gradient_max=None)
    scales = observed_scale[:, None].expand_as(sale)[censored].double()
    eps = 1e-4 * scales.clamp_min(1.)[:, None]
    gaps = p[:, 1:] - p[:, :-1]
    actual = torch.maximum(gaps, eps)
    knots = torch.cat((p[:, :1], p[:, :1] + actual.cumsum(-1)), -1)
    tau = p.new_tensor(QUANTILES)
    br = actual[:, -1]*(1-tau[-1])/(tau[-1]-tau[-2])
    y = sale[censored].double()
    strong = y > knots[:, -1] + br
    shift_max = None
    if bool(strong.any()):
        # Each example has its own common-knot shift; no cancellation across examples.
        shift = torch.zeros(int(strong.sum()), device=p.device, dtype=torch.float64, requires_grad=True)
        loss = -log_survival_from_quantiles((p[strong] + shift[:, None])[:, :, None],
                  y[strong, None], tau, scales[strong]).sum()
        gradient = torch.autograd.grad(loss, shift)[0]
        if not bool(torch.isfinite(gradient).all() & (gradient < 0).all()):
            raise FloatingPointError('Invalid strong upper-tail common-shift gradient')
        shift_max = float(gradient.max())
    return dict(censored=len(p), right=int((y>=knots[:, -1]).sum()),
                left=int((y<=knots[:, 0]).sum()), floored_gaps=int((gaps<eps).sum()),
                gaps=gaps.numel(), max_knot_shift=float((knots-p).abs().max()),
                strong_right=int(strong.sum()), strong_shift_gradient_max=shift_max)
