"""Pinned, frozen native Chronos-2 encoder/head. No future inputs."""
import torch
from chronos import Chronos2Pipeline
from chronos.chronos2.layers import MHA
MODEL_ID='amazon/chronos-2'
REVISION='29ec3766d36d6f73f0696f85560a422f50e8498c'
QUANTILES=[.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.99]
def load_base(device='cuda'):
    m=Chronos2Pipeline.from_pretrained(MODEL_ID,revision=REVISION,device_map=device,torch_dtype=torch.float32,local_files_only=True).model
    m.requires_grad_(False);m.eval();m.config.dropout_rate=0.
    for v in m.modules():
        if isinstance(v,torch.nn.Dropout):v.p=0.
        if isinstance(v,MHA):v.dropout=0.;v.config.dropout_rate=0.
    assert list(m.chronos_config.quantiles)==QUANTILES and m.config.num_layers==12
    assert m.chronos_config.output_patch_size==16 and m.chronos_config.use_arcsinh
    return m

def forecast(m,x,groups,phase=0,adapter=None,state=None,return_hidden=False):
    assert x.ndim==2 and x.shape[-1]==336
    original=None
    if phase:
        from .candidates.patchphase import prepare_context
        original=m._prepare_patched_context
        m._prepare_patched_context=lambda context,context_mask=None:prepare_context(m,context,context_mask,phase)
    try:
        enc,(loc,scale),_,_=m.encode(context=x,group_ids=groups,num_output_patches=3)
    finally:
        if original is not None:m._prepare_patched_context=original
    h=enc.last_hidden_state[:,-3:]
    if adapter is not None:
        adjusted=adapter(h,state)
        # Preserve native head input strides: avoids GEMM layout-dependent
        # float32 differences between a zero residual and the frozen path.
        h=torch.empty_strided(h.size(),h.stride(),device=h.device,dtype=h.dtype).copy_(adjusted)
    z=m.output_patch_embedding(h).reshape(len(x),3,21,16).permute(0,2,1,3).reshape(len(x),21,48).float()
    raw=z.sinh()*scale[:,None,:]+loc[:,None,:]
    if not torch.isfinite(raw).all():raise FloatingPointError('Nonfinite forecast')
    out=(z,raw,loc,scale)
    return out+(h.detach(),) if return_hidden else out

def native_loss(z,y,loc,scale):
    norm=((y-loc)/scale).asinh()[:,None,:];valid=torch.isfinite(norm)
    v=torch.where(valid,norm,0.);e=v-z
    q=torch.as_tensor(QUANTILES,device=z.device)[None,:,None]
    return (2*torch.maximum(q*e,(q-1)*e)*valid).mean(-1).sum(-1).mean()
