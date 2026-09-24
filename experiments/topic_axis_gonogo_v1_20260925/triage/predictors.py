"""Native Chronos-2 zero-shot. No model downloads, fine-tuning, or fallback model."""
from __future__ import annotations
import importlib.metadata as meta
from dataclasses import replace
import numpy as np
from .common import Blocked,require
from .coordinates import canonical,rotate,heading

class Native:
    def __init__(self,device='auto'):
        try:
            import torch
            from chronos import Chronos2Pipeline
            from huggingface_hub import snapshot_download
            require(meta.version('chronos-forecasting')=='2.3.2','This package validates Chronos API 2.3.2 only; no installation allowed')
            path=snapshot_download('amazon/chronos-2',revision='29ec3766d36d6f73f0696f85560a422f50e8498c',local_files_only=True)
            self.device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
            torch.set_num_threads(2)
            self.pipe=Chronos2Pipeline.from_pretrained(path,device_map=self.device,dtype=torch.float32,local_files_only=True)
            self.pipe.inner_model.eval();self.forwards=0;self.windows=0
            self.hook=self.pipe.inner_model.register_forward_hook(self._count)
            self.info={'model':'amazon/chronos-2','snapshot':'29ec3766d36d6f73f0696f85560a422f50e8498c',
                 'device':self.device,'torch':torch.__version__,'chronos':meta.version('chronos-forecasting'),
                 'downloads':0,'optimizer_steps':0,'known_future_covariates':False}
        except Exception as e:raise Blocked('NATIVE_CHRONOS_UNAVAILABLE: '+str(e)) from e
    def _count(self,*args):self.forwards+=1
    def predict(self,cases,canonicalize=False):
        import torch
        if not cases:return []
        H,T=cases[0].y.shape
        require(all(c.y.shape==(H,T) for c in cases),'Mixed tasks in native batch')
        result=[]
        for start in range(0,len(cases),8):
            batch=cases[start:start+8];inputs=[];angles=[]
            for c in batch:
                x=c.x.copy()
                if canonicalize:
                    require(T==2 and x.shape[1]==2,'Canonical baseline only for 2D trajectories')
                    angle=np.rad2deg(heading(x));x=rotate(x,-angle);angles.append(angle)
                target=np.asarray(x[:,:T].T,dtype=np.float32)
                d={'target':target}
                if x.shape[1]>T:d['past_covariates']={f'c{j}':np.asarray(x[:,j],np.float32) for j in range(T,x.shape[1])}
                inputs.append(d)
            with torch.inference_mode():
                qs,_=self.pipe.predict_quantiles(inputs,prediction_length=H,quantile_levels=[.5],batch_size=8,cross_learning=False)
            require(len(qs)==len(batch),'Wrong Chronos output count')
            for k,q in enumerate(qs):
                require(tuple(q.shape)==(T,H,1),f'Unexpected quantile axes {tuple(q.shape)}')
                pred=q.detach().float().cpu().numpy()[:,:,0].T
                if canonicalize:pred=rotate(pred,angles[k])
                require(np.isfinite(pred).all(),'Nonfinite native predictions')
                result.append(pred)
            self.windows+=len(batch)
            require(self.windows<=2400,'Native forecast window budget exceeded')
        return result
    def close(self):
        self.hook.remove()
