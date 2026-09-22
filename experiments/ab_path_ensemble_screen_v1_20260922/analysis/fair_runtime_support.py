"""Inference-equivalent removal of baseline wrapper overhead, without retraining."""
import numpy as np
import torch
from references import features,cpu_predict

def cpu_once(arm,packet,models=None):
    if arm=='NAIVE':return cpu_predict(arm,packet,models)
    xx=features(packet)
    z=np.stack([m.predict(xx) for m in models],-1).reshape(len(packet['x']),8,9).astype(np.float32)
    return dict(z=z,p=np.full_like(z,1/9),**{k:packet[k] for k in ['y','sigma','pairs']})

def direct_cpu_input(model,packet):
    q,_=model.pipeline.predict_quantiles([torch.from_numpy(v[None]) for v in packet['x']],prediction_length=128,
        quantile_levels=[i/10 for i in range(1,10)],batch_size=len(packet['x']),context_length=512,cross_learning=False)
    z=torch.cat(q,0).cpu().numpy()
    return dict(z=z,p=np.full_like(z,1/9),sigma=packet['sigma'])
