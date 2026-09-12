"""One train-only GPU gradient smoke; zero optimization/stream runs."""
import torch
from tsfm_peft_screen.backbone import load_base,native_loss
from tsfm_peft_screen.candidates.maturity import Calibration
from tsfm_peft_screen.runners.streaming_eval import stream_forecast
from tsfm_peft_screen.data import Panel
from tsfm_peft_screen.reproducibility import ROOT,write_json,seed_all,guard,source_hashes
seed_all(30000);guard();panel=Panel('jena');x,y=panel.window(int(panel.origins['train'][0]));x=torch.tensor(x,device='cuda');y=torch.tensor(y,device='cuda');y[:,24:]=float('nan')
m=load_base();cal=Calibration(4).cuda();g=torch.zeros(4,device='cuda',dtype=torch.long)
z,p,l,s=stream_forecast(m,cal,x,g);loss=native_loss(z,y,l,s);loss.backward()
finite=all(v.grad is not None and torch.isfinite(v.grad).all() for v in cal.parameters());assert finite
write_json(ROOT/'results/candidate_05/repair_verification.json',dict(status='GPU_GRADIENT_SMOKE_PASS',loss=float(loss.detach()),all_calibration_gradients_finite=finite,optimizer_updates=0,stream_runs=0,E_opened=False,source_hashes=source_hashes(),candidate_verdict_unchanged='IMPLEMENTATION_BLOCKED',scope='Fix validated on train-only partial-target gradient, not a repeated online pilot'))
print('Partial-label GPU gradients finite; zero optimizer updates / E predictions.')
