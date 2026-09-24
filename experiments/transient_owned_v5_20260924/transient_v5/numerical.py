"""Pinned reduction tests using FIXED predictions, not two different model predictions."""
from __future__ import annotations
import copy
import torch
from torch import nn
from .model import OwnedLoRALinear
from .util import require


def normalized_quantile_loss(pred,target,mask,quantiles,future_cov_mask):
    """Inputs are already normalized. Native horizon mean and quantile sum retained."""
    h=pred.shape[-1]
    target=nn.functional.pad(target,(0,h-target.shape[-1]))
    mask=nn.functional.pad(mask,(0,h-mask.shape[-1]))
    future_cov_mask=nn.functional.pad(future_cov_mask,(0,h-future_cov_mask.shape[-1]))
    effective=mask*(1-future_cov_mask)
    per=(2*((target[:,None,:]-pred)*((target[:,None,:]<=pred).to(pred.dtype)-quantiles[None,:,None])).abs()*effective[:,None,:]).mean(-1).sum(-1)
    active=effective.bool().any(-1)
    require(bool(active.any()),'No active target rows')
    return per.mean(),per[active].mean(),per,active


def check_loss_reduction(native_base=None):
    dtype=torch.float64
    q=torch.tensor([.1,.5,.9],dtype=dtype)
    target=torch.tensor([[.2,-.5,.1,.7,.8]],dtype=dtype)
    # H=5, padded output length=8 tests that horizon padding is retained, not silently renormalized.
    pred0=torch.linspace(-.8,1.2,24,dtype=dtype).reshape(1,3,8)
    original=None; values=[]
    for extra in (0,1,2,4):
        pred=torch.cat([pred0,torch.full((extra,3,8),17.,dtype=dtype)],0).requires_grad_()
        y=torch.cat([target,torch.zeros(extra,5,dtype=dtype)],0)
        mask=torch.cat([torch.ones_like(target),torch.zeros(extra,5,dtype=dtype)],0)
        fmask=torch.cat([torch.zeros_like(target),torch.ones(extra,5,dtype=dtype)],0)
        native,corrected,per,active=normalized_quantile_loss(pred,y,mask,q,fmask)
        gn=torch.autograd.grad(native,pred,retain_graph=True)[0]
        gc=torch.autograd.grad(corrected,pred)[0]
        if original is None: original=(float(corrected.detach()),gc[0].clone())
        require(abs(float(corrected.detach())-original[0])<1e-10,'Corrected loss is not row-invariant')
        require(torch.allclose(gc[0],original[1],atol=1e-10,rtol=1e-10),'Corrected gradient is not row-invariant')
        require(torch.equal(gc[1:],torch.zeros_like(gc[1:])),'Covariate-only prediction gradient nonzero')
        require(torch.allclose(gn[0]*(extra+1),gc[0],atol=1e-10,rtol=1e-10),'Native 1/N gradient relation failed')
        values.append({'extra_rows':extra,'native':float(native.detach()),'corrected':float(corrected.detach()),
                       'target_gradient_norm':float(gc[0].norm()),'native_gradient_norm':float(gn[0].norm()),
                       'padded_horizon_denominator':8,'observed_horizon':5})
    result={'status':'PASS','fixed_prediction_cases':values,'native_chronos_comparison':'NOT_RUN'}
    if native_base is not None and hasattr(native_base,'_compute_loss'):
        nq=len(native_base.quantiles);p=native_base.chronos_config.output_patch_size
        for n in (1,3):
            raw=torch.linspace(-.3,.7,p+1).repeat(n,1)
            tm=torch.zeros(n,p+1);tm[0]=1
            fcm=torch.zeros(n,2,p);fcm[1:]=1
            predictions=torch.linspace(-.6,.9,n*nq*2*p).reshape(n,nq,2*p)
            loc_scale=(torch.zeros(n,1),torch.ones(n,1))
            yn,_=native_base.instance_norm(raw,loc_scale)
            direct=native_base._compute_loss(predictions,raw,tm,fcm,loc_scale,2)
            computed,*_=normalized_quantile_loss(predictions,yn,tm,native_base.quantiles,fcm.reshape(n,-1))
            require(torch.allclose(direct,computed,atol=1e-6,rtol=1e-6),'Local native quantile reduction differs from source contract')
        result['native_chronos_comparison']='PASS'
    return result


def check_peft_parity():
    from peft.tuners.lora.layer import Linear
    torch.manual_seed(1997)
    base=nn.Linear(7,5,bias=True).double()
    own=OwnedLoRALinear(copy.deepcopy(base),3,6.,False)
    official=Linear(copy.deepcopy(base),'default',r=3,lora_alpha=6,lora_dropout=0.,init_lora_weights=True).double().eval()
    # This is an isolated formula test, not a mutation of any lifecycle/trained model.
    with torch.no_grad():
        own.B.weight.copy_(torch.linspace(-.03,.04,15).reshape(5,3))
        official.lora_A['default'].weight.copy_(own.A.weight)
        official.lora_B['default'].weight.copy_(own.B.weight)
    x=torch.linspace(-1,1,42,dtype=torch.float64).reshape(2,3,7).requires_grad_()
    a,b=own(x),official(x)
    require(torch.allclose(a,b,atol=1e-10,rtol=1e-10),'Owned standard LoRA disagrees with installed PEFT')
    ga=torch.autograd.grad(a.square().sum(),(x,own.A.weight,own.B.weight))
    gb=torch.autograd.grad(b.square().sum(),(x,official.lora_A['default'].weight,official.lora_B['default'].weight))
    for i,(aa,bb) in enumerate(zip(ga,gb)):
        require(torch.allclose(aa,bb,atol=1e-10,rtol=1e-10),f'PEFT gradient parity failed: {i}')
    return {'status':'PASS','max_abs_output_diff':float((a-b).abs().max().detach()),
            'max_abs_gradient_diff':[float((aa-bb).abs().max()) for aa,bb in zip(ga,gb)],
            'scope':'dropout=0, bias unchanged, standard rank/alpha LoRA; not full training equivalence'}
