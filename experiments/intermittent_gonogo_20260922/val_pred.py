"""validation 구간(d_1886~d_1913) Chronos-2 예측 — α* 계산용. context = d_1..d_1885."""
import sys,time
from pathlib import Path
import numpy as np, torch
M5=Path("/home/minjae/Documents/github/m5dataset"); sys.path.insert(0,str(M5))
from common.data_split import get_split
from chronos import Chronos2Pipeline
from chronos.utils import interpolate_quantiles
OUT=Path(__file__).resolve().parent
H=28; CONTEXT=512; BS=256; VAL_END=1913; VAL_START=1886
QSPEC=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
ids,train,test=get_split()
ctx=train[:, VAL_START-1-CONTEXT : VAL_START-1]   # d_1374..d_1885
assert ctx.shape[1]==CONTEXT
print("context 끝 = d_%d  (validation 시작 d_%d 이전)"%(VAL_START-1,VAL_START),flush=True)
p=Chronos2Pipeline.from_pretrained("amazon/chronos-2",device_map="cuda",dtype=torch.bfloat16)
q=list(p.quantiles)
inp=[torch.tensor(ctx[i],dtype=torch.float32) for i in range(ctx.shape[0])]
t0=time.time(); qp,_=p.predict_quantiles(inp,prediction_length=H,quantile_levels=q,batch_size=BS)
torch.cuda.synchronize(); Q=np.stack([x[0].float().cpu().numpy() for x in qp])
print("추론 %.1fs  Q%s"%(time.time()-t0,Q.shape),flush=True)
lv=np.asarray(QSPEC,float)
Qs=interpolate_quantiles(QSPEC,q,torch.from_numpy(Q)).numpy()
lve=np.concatenate([[0.],lv,[1.]]); ve=np.concatenate([Qs[...,:1],Qs,Qs[...,-1:]],-1)
qm=np.clip(np.trapezoid(ve,x=lve,axis=-1),0,None)
np.save(OUT/"val_zs_qmean.npy",qm.astype(np.float32))
val_true=train[:, VAL_START-1 : VAL_END]
np.save(OUT/"val_true.npy",val_true.astype(np.float32))
print("saved val_zs_qmean %s  val_true %s"%(qm.shape,val_true.shape))
print("val sum/true = %.4f"%(qm.sum()/val_true.sum()))
