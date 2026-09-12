import gc,json
import torch
from tsfm_peft_screen.runners.common_fit import build,batch,predict,train_schedule,DATA,ARMS
from tsfm_peft_screen.data import Panel
from tsfm_peft_screen.backbone import native_loss,forecast
from tsfm_peft_screen.lora import disabled
from tsfm_peft_screen.reproducibility import ROOT,write_json,guard,seed_all
from tsfm_peft_screen.candidates.maturity import Calibration
from tsfm_peft_screen.runners.streaming_eval import stream_forecast
checks=[]
for candidate,arms in [(1,ARMS[1]),(2,ARMS[2][1:]),(3,ARMS[3][1:])]:
    panel=Panel(DATA[candidate]);origins,channels,variant=train_schedule(panel,candidate)[1]
    x,y,g,r,sc,caps=batch(panel,origins,channels,candidate,variant)
    for arm in arms:
        m,a=build(candidate,arm);z,p,l,s=predict(m,a,candidate,x,g,variant,r)
        with torch.no_grad(),disabled(m):f0=forecast(m,x,g,phase=variant if candidate==3 else 0)[1]
        error=float((p-f0).detach().abs().max());assert error<=1e-6,(candidate,arm,error)
        loss=native_loss(z,y,l,s);loss.backward()
        params=[v for model in [m,a] if model is not None for v in model.parameters() if v.requires_grad]
        assert any(v.grad is not None and v.grad.abs().sum()>0 for v in params)
        checks.append(dict(candidate=candidate,arm=arm,identity=error,loss=float(loss.detach()),finite_gradients=all(v.grad is None or torch.isfinite(v.grad).all().item() for v in params)))
        print(checks[-1],flush=True)
        del m,a,z,p,l,s,f0,loss,params;gc.collect();torch.cuda.empty_cache();guard()
seed_all(30000);m,a=build(5,'F0');cal=Calibration(4).cuda();panel=Panel('jena');x,y=panel.window(int(panel.origins['train'][0]));x=torch.tensor(x,device='cuda');y=torch.tensor(y,device='cuda');g=torch.zeros(4,device='cuda',dtype=torch.long)
z,p,l,s=stream_forecast(m,cal,x,g)
with torch.no_grad():f0=forecast(m,x,g)[1]
assert torch.equal(p,f0);native_loss(z,y,l,s).backward();assert cal.w_out.grad.abs().sum()>0
checks.append(dict(candidate=5,arm='TAFAS_LIKE',identity=0.,output_gradient_nonzero=True))
write_json(ROOT/'results/screening_summary/candidate_preflight.json',dict(status='PASS',checks=checks,scope='forward/backward smoke only; no optimizer steps, no V/E access'))
