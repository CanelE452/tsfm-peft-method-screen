"""CPU-only diagnosis; no method changes, GPU fits, or new E predictions."""
import json
import torch
from tsfm_peft_screen.backbone import native_loss,QUANTILES
from tsfm_peft_screen.reproducibility import ROOT,write_json

def probe(sanitize):
    z=torch.zeros(2,21,48,requires_grad=True)
    loc=torch.zeros(2,1,requires_grad=True);scale=torch.ones(2,1,requires_grad=True)
    y=torch.ones(2,48);y[:,24:]=float('nan')
    if not sanitize:
        # Exact executed formula from commit 4f0854b, preserved for diagnosis.
        norm=((y-loc)/scale).asinh()[:,None,:];valid=torch.isfinite(norm);v=torch.where(valid,norm,0.);e=v-z;q=torch.tensor(QUANTILES)[None,:,None]
        loss=(2*torch.maximum(q*e,(q-1)*e)*valid).mean(-1).sum(-1).mean()
    else:
        valid=torch.isfinite(y)[:,None,:]
        safe=torch.where(torch.isfinite(y),y,loc.expand_as(y).detach())
        norm=((safe-loc)/scale).asinh()[:,None,:];q=torch.tensor(QUANTILES)[None,:,None];e=norm-z
        loss=(2*torch.maximum(q*e,(q-1)*e)*valid).mean(-1).sum(-1).mean()
    loss.backward()
    return {'finite_loss':bool(torch.isfinite(loss)),'finite_prediction_gradient':bool(torch.isfinite(z.grad).all()),'finite_loc_gradient':bool(torch.isfinite(loc.grad).all()),'finite_scale_gradient':bool(torch.isfinite(scale.grad).all())}
observed=probe(False);alternative=probe(True)
assert observed['finite_loss'] and observed['finite_prediction_gradient'] and not observed['finite_loc_gradient']
assert all(alternative.values())
write_json(ROOT/'results/candidate_05/failure_diagnosis.json',{'status':'REPRODUCED_ON_CPU','executed_native_loss':observed,'sanitized_target_formula_cpu_probe':alternative,'root_cause':'Unrevealed targets are NaN. Masking after normalization leaves 0*NaN derivatives into trainable input-normalization loc/scale of the TAFAS input GCM. Frozen-input LoRA paths do not differentiate loc/scale.','scope':'Historical execution source and sealed recipes preserved. Main now sanitizes labels before normalization; train-only GPU gradient verification is in repair_verification.json. No stream was repeated and the Maturity stream was not reached.','next_action':'A separately reviewed fresh Round1 run would be needed for method evidence; current repair has no new streaming result.'})
print(json.dumps({'current':observed,'sanitized_cpu_probe':alternative},indent=2))
