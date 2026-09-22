"""Synthetic-testable primitives; NOT a Chronos runner or a validated PEFT method.

Clock convention: at integer time t, only y[j] with j < t has arrived.
Forecast issued at s predicts indices s,...,s+H-1.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CONTEXT = 256
HORIZON = 64
RECORD_DIM = 274


def visible_mask(issue: int, now: int, horizon: int = HORIZON) -> Tensor:
    if horizon <= 0 or issue > now:
        raise ValueError("Invalid issue/clock/horizon")
    return torch.arange(issue, issue + horizon) < now


def a_support_issues(now: int) -> list[int]:
    return [now - 256, now - 192, now - 128, now - 64]


def b_support_issues(now: int) -> list[int]:
    return [now - 64, now - 32, now - 16, now - 8]


def observed_scale(context: Tensor) -> Tensor:
    """Past-only episode scale, with no future or evaluation-series statistic."""
    if context.ndim != 1 or not torch.isfinite(context).all():
        raise ValueError("A finite one-dimensional context is required")
    floor = torch.maximum(context.abs().mean() * 1e-6, context.new_tensor(1e-6))
    return torch.maximum(context.std(unbiased=False), floor)


def safe_masked_pinball(q: Tensor, target: Tensor, mask: Tensor, scale: Tensor) -> Tensor:
    """Observed cells are selected BEFORE subtraction, so hidden NaNs cannot leak.

    q: [batch,H,9], target/mask: [batch,H], scale: [batch].
    Mean per nonempty support task, then mean across tasks.
    """
    if q.ndim != 3 or q.shape[-1] != 9:
        raise ValueError("Expected [batch,H,9] forecasts")
    if target.shape != q.shape[:-1] or mask.shape != target.shape:
        raise ValueError("Shape mismatch")
    if scale.shape != (q.shape[0],) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid scale")
    if not torch.isfinite(q).all():
        raise ValueError("Nonfinite model output")
    levels = torch.arange(1, 10, device=q.device, dtype=q.dtype) / 10
    terms = []
    for k in range(q.shape[0]):
        idx = mask[k].bool()
        if not idx.any():
            continue
        observed = target[k, idx]
        if not torch.isfinite(observed).all():
            raise ValueError("Nonfinite observed target")
        error = observed[:, None] - q[k, idx]
        terms.append((2 * torch.maximum(levels * error, (levels - 1) * error)).mean() / scale[k])
    if not terms:
        raise ValueError("No observed labels: skip the optimizer update, do not fake zero loss")
    return torch.stack(terms).mean()


def residual_record(context: Tensor, qref: Tensor, target: Tensor, mask: Tensor,
                    issue_age: int, scale: Tensor, no_error: bool = False) -> Tensor:
    """One 274-D support record. qref is an immutable diagnostic forecaster output.

    Hidden target values are never included in arithmetic. The record contains
    forecast median/width, observed residuals/mask, 16 past means, age/maturity.
    """
    if context.shape != (CONTEXT,) or qref.shape != (HORIZON, 9):
        raise ValueError("Unexpected context/forecast shape")
    if target.shape != (HORIZON,) or mask.shape != (HORIZON,):
        raise ValueError("Unexpected target/mask shape")
    if not torch.isfinite(context).all() or not torch.isfinite(qref).all():
        raise ValueError("Nonfinite past or forecast")
    if scale.numel() != 1 or not torch.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid episode scale")
    if issue_age < 0:
        raise ValueError("Support cannot come from the future")
    observed = mask.bool()
    if not torch.isfinite(target[observed]).all():
        raise ValueError("Nonfinite observed target")
    residual = torch.zeros_like(target)
    if not no_error:
        residual[observed] = (target[observed] - qref[observed, 4]) / scale
    median = qref[:, 4] / scale
    # Native quantile channels are not sorted for the training objective.
    width = (qref[:, 8] - qref[:, 0]).clamp_min(0) / scale
    means = context.reshape(16, 16).mean(1) / scale
    meta = context.new_tensor([issue_age / CONTEXT, float(observed.float().mean())])
    record = torch.cat([median, width, residual, observed.to(context.dtype), means, meta])
    if record.shape != (RECORD_DIM,) or not torch.isfinite(record).all():
        raise AssertionError("Invalid feature record")
    return record


class CoefficientGenerator(nn.Module):
    """Same feature encoder/head; TIME has GRU, SET has a near-size-matched MLP.

    With hidden=32, the SET aggregation MLP has just one more parameter than GRU.
    Returned coefficient initially equals 1, preserving a WARM-STARTED LoRA.
    """
    def __init__(self, n_modules: int, rank: int = 8, mode: str = "time"):
        super().__init__()
        if n_modules <= 0 or rank <= 0 or mode not in {"set", "time"}:
            raise ValueError("Invalid module count/rank/mode")
        self.n_modules, self.rank, self.mode = n_modules, rank, mode
        self.encoder = nn.Sequential(nn.Linear(RECORD_DIM, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        if mode == "time":
            self.aggregate = nn.GRU(32, 32, batch_first=True)
        else:
            self.aggregate = nn.Sequential(nn.Linear(32, 97), nn.GELU(), nn.Linear(97, 32))
        self.head = nn.Linear(32, n_modules * rank)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, records: Tensor) -> Tensor:
        if records.ndim != 3 or records.shape[-1] != RECORD_DIM:
            raise ValueError("Expected [batch,support,274]")
        z = self.encoder(records)
        if self.mode == "time":
            _, hidden = self.aggregate(z)
            state = hidden[-1]
        else:
            state = self.aggregate(z.mean(1))
        return (1 + self.head(state)).reshape(-1, self.n_modules, self.rank)


def conditional_lora(x: Tensor, a: Tensor, b: Tensor, coefficient: Tensor,
                     alpha_over_rank: float = 2.0) -> Tensor:
    """Return only the adapter delta, keeping the generator's autograd graph intact.

    Do not install generated coefficients using .data, copy_, load_state_dict,
    nn.Parameter(...) or torch.tensor(existing_tensor): these sever gradients.
    """
    if x.ndim != 3 or coefficient.shape != (x.shape[0], a.shape[0]):
        raise ValueError("Expected x [batch,tokens,in], coefficient [batch,rank]")
    if a.shape[1] != x.shape[-1] or b.shape[1] != a.shape[0]:
        raise ValueError("Low-rank shape mismatch")
    return alpha_over_rank * F.linear(F.linear(x, a) * coefficient[:, None, :], b)


@dataclass(frozen=True)
class IssuedForecast:
    series: str
    issue: int
    horizon: int
    digest: str


class IssuedLedger:
    """Tiny reference immutability check. Real runner must durably fsync its ledger."""
    def __init__(self) -> None:
        self._items: dict[tuple[str, int], IssuedForecast] = {}

    def issue(self, series: str, now: int, prediction: Tensor) -> IssuedForecast:
        if prediction.ndim != 2 or prediction.shape[1] != 9 or not torch.isfinite(prediction).all():
            raise ValueError("Invalid issued prediction")
        key = (series, now)
        if key in self._items:
            raise ValueError("An issued forecast is immutable")
        payload = prediction.detach().cpu().contiguous().numpy().tobytes()
        entry = IssuedForecast(series, now, prediction.shape[0], hashlib.sha256(payload).hexdigest())
        self._items[key] = entry
        return entry


def split_ids(ids: Iterable[str]) -> tuple[list[str], list[str], list[str]]:
    unique = sorted(set(ids), key=lambda s: hashlib.sha256(("feedback-peft-v1|" + s).encode()).hexdigest())
    if len(unique) < 48:
        raise ValueError("Need at least 48 eligible series")
    return unique[:32], unique[32:40], unique[40:48]


def break_even(extra_preparation_seconds: float, baseline_adapt_seconds: float,
               generated_adapt_seconds: float) -> int | None:
    if min(extra_preparation_seconds, baseline_adapt_seconds, generated_adapt_seconds) < 0:
        raise ValueError("Negative time")
    saving = baseline_adapt_seconds - generated_adapt_seconds
    if saving <= 0:
        return None
    return max(0, math.ceil(extra_preparation_seconds / saving))
