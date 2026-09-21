"""Post-run inference verification using TRAIN inputs; consumes zero updates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'/'rollout_uncertainty_peft_v1_20260921'))
import gc
import numpy as np
import torch
from common import *
from model import *
from data import load,schedule
from smoke import batch,close
from runner import selected_model

def main():
    setup()
    assert read_json(RESULTS/'VERIFICATION.json')['status']=='PASS'
    ledger_before=sha(RESULTS/'UPDATE_LEDGER.jsonl')
    selected=read_json(RESULTS/'MODEL_SELECTION.json')
    folder=CACHE/'supplementary_audit'; folder.mkdir(exist_ok=True)
    records=[]
    for source in ['Electricity','ETTh1']:
        d=load(source); x,y,sigma=batch(d,schedule(source,92120)[0])
        for name in ['F0_NATIVE','R_NATIVE']:
            m=RolloutModel('F0',92120,lora=False) if name=='F0_NATIVE' else selected_model(selected[f'{source}/R_ROLLOUT_LORA/92121'])
            contexts=[]
            def capture(module,args,kwargs):
                contexts.append(kwargs['context'].detach().cpu().clone())
            h=m.pipeline.model.register_forward_pre_hook(capture,with_kwargs=True)
            trace=[]
            output=native(m,x,trace)
            h.remove()
            qlevels=torch.arange(1,10,dtype=torch.float32)/10
            blocks=[trace[0]['quantiles']]
            for item in trace[1:]:
                blocks.append(torch.quantile(item['quantiles'].reshape(8,81,64),qlevels,dim=1,interpolation='linear').transpose(0,1))
            rebuilt=torch.cat(blocks,dim=-1).transpose(1,2)
            aggregation=close(output,rebuilt)
            beam=x.cpu()[:,None,:].repeat(1,9,1)
            checks=[]
            for k in range(1,4):
                beam=torch.cat([beam,blocks[k-1]],dim=-1)
                checks.append(close(contexts[k],beam.reshape(72,-1)))
                assert torch.equal(contexts[k][:,:512],x.cpu()[:,None,:].repeat(1,9,1).reshape(72,512))
            p=folder/f'{source}_{name}.npz'
            np.savez_compressed(p,initial_context=x.cpu().numpy(),native_output=output.numpy(),**{f'conditional_raw_{i}':item['quantiles'].numpy() for i,item in enumerate(trace)},**{f'context_{i}':v.numpy() for i,v in enumerate(contexts)})
            records.append({'source':source,'method':name,'role':'TRAIN','model_parameter_count':sum(v.numel() for v in m.pipeline.model.parameters()),'aggregation_parity':aggregation,'context_retention_checks':checks,'per_call_shapes':[list(t['quantiles'].shape) for t in trace],'raw_trace_path':str(p.relative_to(ROOT)),'sha256':sha(p)})
            del m; gc.collect(); torch.cuda.empty_cache()
    p=load_direct(); calls=[]
    def direct_trace(module,args,kwargs,out):
        calls.append({k:({'shape':list(v.shape),'unique_values':torch.unique(v).cpu().tolist() if 'group' in k else None,'finite_count':int(torch.isfinite(v).sum()) if v.is_floating_point() else None} if torch.is_tensor(v) else str(v)) for k,v in kwargs.items()})
    hook=p.model.register_forward_hook(direct_trace,with_kwargs=True)
    q=direct(p,x)
    hook.remove()
    assert len(calls)==1 and q.shape==(8,256,9)
    assert calls[0]['group_ids']['unique_values']==list(range(8))
    assert calls[0]['future_covariates']['finite_count']==0
    after=sha(RESULTS/'UPDATE_LEDGER.jsonl')
    assert ledger_before==after
    save_json(RESULTS/'SUPPLEMENTARY_MODEL_VERIFICATION.json',{'status':'PASS','optimizer_updates_added':0,'ledger_sha256_before_after':after,'native_audits':records,'chronos2_parameter_count':sum(v.numel() for v in p.model.parameters()),'chronos2_direct_calls':calls,'chronos2_prediction_shape':list(q.shape),'script_sha256':sha(Path(__file__))})
    print('Supplementary TRAIN-only inference audit PASS; zero updates')

if __name__=='__main__': main()
