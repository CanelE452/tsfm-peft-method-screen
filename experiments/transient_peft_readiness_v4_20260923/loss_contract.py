"""TARGET_ROW_MEAN_CORRECTION — native quantile loss 는 그대로 두고 row reduction 만 고친다.

native Chronos `_compute_loss` 는 마지막에 batch(행) 평균을 취한다. 공변량 행은 마스킹되어
분자에 0 을 기여하지만 분모에는 포함되므로, INPUT arm 이 같은 target 오차에 더 작은 scalar loss
를 받는다. 이 모듈은 active target row 수로 다시 정규화한다.

**"완전히 native scalar loss 와 동일" 이라고 부르지 않는다.** underlying quantile loss 는 동일하고
row reduction 만 공정하게 고친 것이다.
"""
import torch

CONTRACT_NAME = "TARGET_ROW_MEAN_CORRECTION"
CONTRACT_VERSION = "v4.2026-09-23"

def active_target_rows(future_target_mask: torch.Tensor) -> torch.Tensor:
    """(rows, H) 마스크에서 유효 target 이 하나라도 있는 행 수."""
    return (future_target_mask.sum(dim=-1) > 0).sum()

def correction_factor(future_target_mask: torch.Tensor) -> torch.Tensor:
    """native 의 분모(total_rows) 를 active_target_rows 로 바꾸는 배수."""
    total = future_target_mask.shape[0]
    active = active_target_rows(future_target_mask)
    if active == 0:
        raise ValueError("active target row 가 0 이다. loss 를 정의할 수 없다.")
    return torch.as_tensor(total, dtype=future_target_mask.dtype, device=future_target_mask.device) / active

def corrected_loss(native_loss: torch.Tensor, future_target_mask: torch.Tensor):
    """방법 A: native scalar 에 total/active 를 곱한다.
    inactive row 가 정확히 0 loss 라는 계약이 확인된 경우에만 유효하며, 그 확인은
    test_loss_contract.py 가 수행한다."""
    f = correction_factor(future_target_mask)
    return native_loss * f, {"total_rows": int(future_target_mask.shape[0]),
                             "active_target_rows": int(active_target_rows(future_target_mask)),
                             "correction_factor": float(f)}

def reference_rowmean(per_row_loss: torch.Tensor, future_target_mask: torch.Tensor):
    """방법 B: native 수식을 재현하되 마지막 row mean 만 active target rows 로 계산.
    방법 A 와의 수학적 동치를 검산하는 데 쓴다."""
    act = (future_target_mask.sum(dim=-1) > 0)
    if act.sum() == 0: raise ValueError("active target row 0")
    return per_row_loss[act].sum() / act.sum()
