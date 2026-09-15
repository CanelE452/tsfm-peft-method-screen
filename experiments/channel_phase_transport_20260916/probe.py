"""Train-input-only operator activation probe; zero optimizer updates."""
import json
from pathlib import Path
import numpy as np
import torch
from model import ROOT,make
from common import read,save,sha
from rank2_data import load,batch
OUT=ROOT/'results/channel_phase_transport_20260916'

def main():
    c=read(OUT/'seal.json');rows=[]
    for d,dc in c['data'].items():
        values,std=load(ROOT/'.cache/channel_sharing_screen_v1_20260915',d,'train')
        m=make('CONDITIONED',dc['channel_ids'],41000,'cpu').eval()
        selected=dc['origins']['train'][:64]
        x,y=batch(values,selected,'cpu');mask=torch.ones(len(selected),96,dtype=torch.bool)
        with torch.no_grad():
            normalized=m.normalizer(x=x,mask=mask,mode='norm')
            patches=m.tokenizer(normalized);t=m.adapter.weights(patches)
            idx=torch.arange(12);u=((idx[:,None]-idx[None,:]).remainder(3)==0).float()/4
            entropy=-(t*t.clamp_min(torch.finfo(t.dtype).tiny).log()).sum(-1)
            diff=(t-u).abs().sum(-1)
            rows.append(dict(dataset=d,origins=selected,input_only=True,fit_updates=0,mean_row_l1_from_uniform=float(diff.mean()),max_row_l1_from_uniform=float(diff.max()),mean_self_weight=float(t.diagonal(dim1=-2,dim2=-1).mean()),mean_entropy=float(entropy.mean()),uniform_entropy=float(np.log(4)),nonuniform_row_fraction=float((diff>1e-6).float().mean())))
        del m,x,y,values
    save(OUT/'train_operator_probe.json',dict(rows=rows,source_sha256=sha(Path(__file__)),training_contract_sha256=sha(OUT/'seal.json'),note='First64 sealed train origins only; no V/E labels, optimization, or model selection. Descriptive activity is not predictive benefit.'))
    print(json.dumps(rows,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
