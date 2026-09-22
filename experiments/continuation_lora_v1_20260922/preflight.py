import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import gc
import importlib.metadata as md
import inspect
import json
import platform
import subprocess
import sys
import urllib.request
import warnings
import numpy as np
import torch
import chronos.chronos_bolt as bolt
from common import *
from model import BranchModel, setup, tensor_hash, update, branch_context
import data

def cmd(args):
    p = subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
    return dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)

def provenance():
    RESULTS.mkdir(parents=True,exist_ok=True)
    prior = ROOT/'results/rollout_uncertainty_peft_v1_20260921/SOURCE_AND_MODEL_MANIFEST.json'
    old = read(prior)
    for info in old['models'].values():
        for f in info['files']:
            assert sha(Path(info['path'])/f['name'])==f['sha256']
    url = 'https://api.github.com/repos/amazon-science/chronos-forecasting/commits/main'
    with urllib.request.urlopen(url,timeout=60) as r: revision=json.load(r)['sha']
    source_url = f'https://raw.githubusercontent.com/amazon-science/chronos-forecasting/{revision}/src/chronos/chronos_bolt.py'
    with urllib.request.urlopen(source_url,timeout=60) as r: source=r.read()
    dest = CACHE/'upstream/chronos_bolt.py'
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(source)
    installed = Path(inspect.getfile(bolt))
    manifest = dict(models=old['models'],reused_pretrained_cache_only=True,previous_manifest_sha=sha(prior),
                    official_commit=revision,source_url=source_url,upstream_sha=sha(dest),installed_source=str(installed),
                    installed_sha=sha(installed),installed_matches_current_upstream=sha(dest)==sha(installed),
                    repo_head=cmd(['git','rev-parse','HEAD'])['stdout'].strip())
    save(RESULTS/'SOURCE_MANIFEST.json',manifest)
    assert manifest['installed_matches_current_upstream'], 'IMPLEMENTATION_BLOCK: installed/current source mismatch requires audit'
    assert torch.cuda.is_available(), 'RESOURCE_BLOCK: CUDA unavailable'
    env = dict(python=sys.version,executable=sys.executable,platform=platform.platform(),torch_cuda=torch.version.cuda,
               packages={n:md.version(n) for n in ['torch','chronos-forecasting','peft','transformers','numpy','pandas','pytest','matplotlib','huggingface_hub']},
               gpu=torch.cuda.get_device_name(0),vram=torch.cuda.get_device_properties(0).total_memory,
               nvidia_smi=cmd(['nvidia-smi']),pip_check=cmd([sys.executable,'-m','pip','check']))
    save(RESULTS/'ENVIRONMENT.json',env)
    assert sys.version_info[:2]==(3,11) and env['pip_check']['returncode']==0
    lock=cmd([sys.executable,'-m','pip','freeze'])
    assert lock['returncode']==0
    (RESULTS/'requirements-lock.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cu128\n'+lock['stdout'],encoding='utf-8')

def gpu_batch(d, pairs):
    return tuple(torch.as_tensor(v,device='cuda',dtype=torch.float32) for v in data.batch(d,pairs))

@torch.no_grad()
def parity(m,x,sigma):
    ours,_=m.forecast(x)
    if m.mode=='CONTINUATION' and m.has_lora:
        with m.network.disable_adapter():
            first=m.pipeline.predict(x,prediction_length=64).to(x.device)
        ctx=branch_context(x,first)
        second=m.pipeline.predict(ctx,prediction_length=64).to(x.device).reshape(len(x),81,64)
        reduced=torch.quantile(second,torch.arange(1,10,device=x.device)/10,dim=1).transpose(0,1)
        official=torch.cat([first,reduced],dim=-1).transpose(1,2)
        native64=first.transpose(1,2)
    else:
        official=m.pipeline.predict(x,prediction_length=128).transpose(1,2).to(x.device)
        native64=m.pipeline.predict(x,prediction_length=64).transpose(1,2).to(x.device)
    delta=(ours-official).abs()
    torch.testing.assert_close(ours,official,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(ours[:,:64],native64,atol=1e-5,rtol=1e-5)
    return dict(max_abs=float(delta.max()),max_train_scaled=float((delta/sigma[:,None,None]).max()),
                reference='official first64 adapter-off + official branch64 adapter-on' if m.mode=='CONTINUATION' else 'official native64/128')

def smoke():
    if (RESULTS/'SMOKE_STARTED.json').exists():
        raise RuntimeError('Smoke already started: inspect receipts; never silently repeat optimizer updates')
    d=data.load()
    x,y,sigma=gpu_batch(d,data.schedule(92241)[0])
    base=BranchModel(lora=False)
    base_parity=parity(base,x,sigma)
    with torch.no_grad(): initial_base=base.forecast(x)[0].cpu()
    del base
    gc.collect(); torch.cuda.empty_cache()
    rows=[]; initial=[]
    save(RESULTS/'SMOKE_STARTED.json',dict(cap=4,consumed=0))
    for arm in ARMS:
        m=BranchModel(seed=92240,mode=arm)
        frozen=m.frozen_hash()
        initial.append(tensor_hash(m.learned().items()))
        with torch.no_grad():
            q=m.forecast(x)[0].cpu()
            torch.testing.assert_close(q,initial_base,atol=1e-5,rtol=1e-5)
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=0.)
        before=parity(m,x,sigma)
        for j in range(2):
            row=update(m,opt,x,y,sigma,arm)
            rows.append(dict(arm=arm,step=j+1,**row))
            save(RESULTS/'SMOKE_STARTED.json',dict(cap=4,consumed=len(rows),updates=rows))
        assert frozen==m.frozen_hash()
        assert tensor_hash(m.learned().items())!=initial[-1]
        if arm=='CONTINUATION':
            with torch.no_grad():
                after=m.forecast(x)[0].cpu()
                torch.testing.assert_close(after[:,:64],initial_base[:,:64],atol=0,rtol=0)
        save(RESULTS/f'SMOKE_{arm}.json',dict(initial_parity=before,trained_parity=parity(m,x,sigma),frozen_unchanged=True,
                 initial_f0_first64_preserved=arm=='CONTINUATION',
                 trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),finite_gradients=True))
        del m,opt
        gc.collect(); torch.cuda.empty_cache()
    assert len(set(initial))==1
    save(RESULTS/'PREFLIGHT.json',dict(status='PASS',base_parity=base_parity,same_seed_initial_lora_hash=initial[0],
         zero_lora_matches_base=True,smoke_updates=len(rows),future_targets_not_in_forecast_api=True,
         data_audit_sha=sha(RESULTS/'DATA_AUDIT.json'),source_sha=sha(RESULTS/'SOURCE_MANIFEST.json')))

if __name__=='__main__':
    warnings.filterwarnings('ignore',message='We recommend keeping prediction length')
    setup()
    provenance()
    data.prepare()
    tests=cmd([sys.executable,'-m','pytest',str(EXP/'test_model.py'),str(EXP/'test_data.py'),str(EXP/'test_metrics.py'),'-q'])
    save(RESULTS/'CPU_TESTS.json',tests)
    assert tests['returncode']==0, tests
    smoke()
    event('preflight_complete',smoke_updates=4)
