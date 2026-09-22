import gc
import numpy as np
import torch
from chronos import Chronos2Pipeline
from common import *
from evaluation import packet,predict,predpath,calibration,score
from ledger import PhaseTimer


class DirectReference:
    def __init__(self):
        manifest=read(ROOT/'results/rollout_uncertainty_peft_v1_20260921/SOURCE_AND_MODEL_MANIFEST.json')
        info=manifest['models']['amazon/chronos-2']
        for f in info['files']:assert sha(Path(info['path'])/f['name'])==f['sha256']
        self.pipeline=Chronos2Pipeline.from_pretrained(info['path'],device_map='cuda',torch_dtype=torch.float32,local_files_only=True)
        self.info=info

    @torch.no_grad()
    def predict(self,xs,batch_size=16):
        out=[]
        for start in range(0,len(xs),batch_size):
            inputs=[torch.as_tensor(row,dtype=torch.float32)[None] for row in xs[start:start+batch_size]]
            qs,_=self.pipeline.predict_quantiles(inputs,prediction_length=24,quantile_levels=[i/10 for i in range(1,10)],
                    batch_size=batch_size,context_length=192,cross_learning=False)
            out.append(torch.cat(qs,dim=0).cpu().numpy())
        result=np.concatenate(out)
        assert result.shape==(len(xs),24,9) and np.isfinite(result).all()
        return result


def prepare_reference(d):
    if (RESULTS/'REFERENCE_SELECTION.json').exists():return
    try:m=DirectReference()
    except Exception as e:
        save(RESULTS/'REFERENCE_SELECTION.json',dict(status='UNAVAILABLE_PRACTICAL_UNVERIFIED',error=repr(e)))
        return
    results={}
    for candidate in ['h1','h2']:
        timer=PhaseTimer(candidate,'chronos2_reference_prepare')
        try:
            p=packet(d,candidate,'dev','cal')
            index=[0,len(p['x'])//2] if candidate=='h2' else [0,1]
            x=p['x'][index];a=m.predict(x,batch_size=2);b=m.predict(x,batch_size=1)
            try:
                np.testing.assert_allclose(a/p['sigma'][index,None,None],b/p['sigma'][index,None,None],atol=1e-5,rtol=1e-5)
            except AssertionError as e:
                results[candidate]=dict(status='UNVERIFIED_NATIVE_BATCH_PARITY',error=str(e))
                continue
            cal=predict(m,p,predpath(candidate,'dev',0,'CHRONOS2_DIRECT',2,'cal'))
            val=predict(m,packet(d,candidate,'dev','validation'),predpath(candidate,'dev',0,'CHRONOS2_DIRECT',2,'validation'))
            coefficients=calibration(cal);choices={mode:score(val,coefficients,mode) for mode in ['RAW','CAL']}
            mode=min(choices,key=lambda name:(choices[name],name!='RAW'))
            ec=predict(m,packet(d,candidate,'eval','cal'),predpath(candidate,'eval',0,'CHRONOS2_DIRECT',2,'cal'))
            results[candidate]=dict(status='AVAILABLE',mode=mode,development_options=choices,eval_calibration=calibration(ec),
                                   native_batch_invariance_max_abs=float(np.max(np.abs(a-b))))
        finally:timer.close()
    save(RESULTS/'REFERENCE_SELECTION.json',dict(status='AVAILABLE',model=m.info,candidates=results,official_direct_api=True,test_scores_accessed=False))
    del m;gc.collect();torch.cuda.empty_cache()


def test_reference(d):
    selection=read(RESULTS/'REFERENCE_SELECTION.json')
    if selection['status']!='AVAILABLE':return
    m=DirectReference()
    for candidate in ['h1','h2']:
        if selection['candidates'][candidate]['status']!='AVAILABLE':continue
        timer=PhaseTimer(candidate,'chronos2_reference_test')
        try:
            if candidate=='h1':
                predict(m,packet(d,candidate,'eval','test'),predpath(candidate,'eval',0,'CHRONOS2_DIRECT',2,'test'))
            else:
                for condition in ['BLOCK48','IID48','ALIGNED48_LAST','CLEAN']:
                    predict(m,packet(d,candidate,'eval','test',condition),predpath(candidate,'eval',0,'CHRONOS2_DIRECT',2,'test_'+condition))
        finally:timer.close()
    del m;gc.collect();torch.cuda.empty_cache()
