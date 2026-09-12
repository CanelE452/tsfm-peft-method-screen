import json,platform,subprocess
from pathlib import Path
import numpy as np
import torch
from tsfm_peft_screen.backbone import load_base,forecast,native_loss,MODEL_ID,REVISION,QUANTILES
from tsfm_peft_screen.lora import attach,audit,snapshot,restore
from tsfm_peft_screen.reproducibility import ROOT,sha,write_json,seed_all,guard,source_hashes
from tsfm_peft_screen.metrics import score,replay
seed_all(30000);guard()
m=load_base();x=torch.randn(8,336,device='cuda');x[0,10:20]=float('nan');groups=torch.arange(8,device='cuda')//4
y=torch.randn(8,48,device='cuda')
with torch.no_grad():
    z,f0,l,s=forecast(m,x,groups);official=m(context=x,group_ids=groups,num_output_patches=3,future_target=y)
    official_loss=float(official.loss);ours=float(native_loss(z,y,l,s))
attach(m)
with torch.no_grad():initial=forecast(m,x,groups)[1]
identity=float((initial-f0).abs().max());assert identity<=1e-6;assert abs(official_loss-ours)<=1e-6
state=snapshot(m);cache=ROOT/'.cache/common';cache.mkdir(exist_ok=True)
torch.save(state,cache/'step0.pt')
params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=0)
frozen={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
z,p,l,s=forecast(m,x,groups);loss=native_loss(z,y,l,s);loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step()
assert all(torch.equal(p,frozen[n]) for n,p in m.named_parameters() if not p.requires_grad)
trained=snapshot(m);torch.save(trained,cache/'step1.pt')
with torch.no_grad():expected=forecast(m,x,groups)[1].cpu().numpy()
restore(torch.load(cache/'step0.pt',weights_only=True),m);restore(torch.load(cache/'step1.pt',weights_only=True),m)
with torch.no_grad():actual=forecast(m,x,groups)[1].cpu().numpy()
assert np.array_equal(expected,actual)
p=actual.reshape(2,4,21,48);target=y.cpu().numpy().reshape(2,4,48);sc=np.array([1.,2.,3.,4.]);np.savez_compressed(cache/'prediction.npz',prediction=p,target=target,scale=sc)
metric_error=replay(cache/'prediction.npz')
modelroot=Path.home()/'.cache/huggingface/hub/models--amazon--chronos-2/snapshots'/REVISION
record=dict(status='PASS',model=MODEL_ID,revision=REVISION,model_files={p.name:sha(p) for p in modelroot.iterdir() if p.is_file()},identity_max_abs=identity,native_loss_parity_abs=abs(official_loss-ours),checkpoint_replay_max_abs=float(np.max(abs(actual-expected))),metric_replay_abs=metric_error,frozen_unchanged=True,trainable_audit=audit(m),python=platform.python_version(),torch=torch.__version__,gpu=torch.cuda.get_device_name(),resources=guard(),source_hashes=source_hashes(),cuda_userspace={p.name:sha(p) for p in (ROOT/'.cache/nvidia-580.173.02').glob('*.580.173.02')},cuda_note='Official 580.173.02 runfile extracted only; matching local userspace, no system driver modifications')
write_json(ROOT/'results/screening_summary/common_integrity.json',record);print(json.dumps(record,indent=2))
