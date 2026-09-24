"""v4 공통 빌더."""
import torch
from pathlib import Path
OUT = Path(__file__).resolve().parents[2] / "results/transient_peft_readiness_v4_20260923"
LORA_TARGETS = ["self_attention.q","self_attention.k","self_attention.v","self_attention.o"]

def build(seed=0, learned_state=False, with_modulator=False, rank=8):
    """모든 arm 공통 진입점: 같은 base checkpoint, 같은 LoRA init seed, 출력층 제외."""
    from chronos import Chronos2Pipeline
    from peft import LoraConfig, get_peft_model
    from transient_model import TransientPEFTModel, ResponseState, Modulator
    from fixture import TAU_INIT
    torch.manual_seed(seed)
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu", dtype=torch.float32)
    base = pipe.inner_model
    for q in base.parameters(): q.requires_grad_(False)
    pm = get_peft_model(base, LoraConfig(r=rank, lora_alpha=16, lora_dropout=0.0,
                                         bias="none", target_modules=LORA_TARGETS))
    sg = ResponseState(TAU_INIT, learned=learned_state)
    md = Modulator(len(TAU_INIT), rank) if with_modulator else None
    return TransientPEFTModel(pm, sg, md, rank=rank), pipe

def frozen_hash(tp):
    import hashlib
    b = tp.peft_model.base_model.model
    h = hashlib.sha256()
    for _, q in b.output_patch_embedding.named_parameters(): h.update(q.detach().numpy().tobytes())
    return h.hexdigest()[:16]

def fwd(tp, ctx, fc, fcm, ft, ftm, gids, n_out, c_rows=None):
    if c_rows is not None: tp.set_state(c_rows, gids)
    try:
        out = tp.peft_model(context=ctx, group_ids=gids, future_covariates=fc,
                            future_covariates_mask=fcm, future_target=ft,
                            future_target_mask=ftm, num_output_patches=n_out)
    finally:
        tp.clear_state()
    return out
