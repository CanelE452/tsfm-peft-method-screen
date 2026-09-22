import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import importlib.metadata as md
import inspect
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import chronos.chronos_bolt as bolt
from common import *
from model import ForecastModel, setup
from metrics import affine, point_loss
import data


def command(args):
    r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
    return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)


def provenance():
    prior=ROOT/'results/rollout_uncertainty_peft_v1_20260921/SOURCE_AND_MODEL_MANIFEST.json'
    old=read(prior);info=old['models']['amazon/chronos-bolt-small']
    assert info['revision']==CONFIG['model']['revision']
    for f in info['files']:
        assert sha(Path(info['path'])/f['name'])==f['sha256']
    installed=Path(inspect.getfile(bolt))
    assert sha(installed)=='93b3af9e4521ea981c4228b1acaa252321f7a190512f956088394426182c8b8c', 'IMPLEMENTATION_FAILURE: pinned Chronos source mismatch'
    assert torch.cuda.is_available(), 'RESOURCE_BLOCK: CUDA unavailable'
    env=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
             packages={n:md.version(n) for n in ['torch','chronos-forecasting','peft','transformers','numpy','pandas','pytest','matplotlib']},
             gpu=torch.cuda.get_device_name(0),vram=torch.cuda.get_device_properties(0).total_memory,
             torch_cuda=torch.version.cuda,nvidia_smi=command(['nvidia-smi']),
             pip_check=command([sys.executable,'-m','pip','check']))
    assert sys.version_info[:2]==(3,11) and env['pip_check']['returncode']==0
    save(RESULTS/'ENVIRONMENT.json',env)
    lock=command([sys.executable,'-m','pip','freeze']);assert lock['returncode']==0
    (RESULTS/'requirements-lock.txt').write_text(lock['stdout'],encoding='utf-8')
    save(RESULTS/'SOURCE_MANIFEST.json',dict(model=info,upstream_commit='10afa9ebe016e514f9d7dc1aa873f66af57e116b',
         installed_source=str(installed),installed_sha256=sha(installed),prior_manifest_sha256=sha(prior),
         plan_sha256=sha(DESIGN_DIR/'PLAN_KO.md'),design_sha256=sha(DESIGN_DIR/'DESIGN.json'),
         repo_head=command(['git','rev-parse','HEAD'])['stdout'].strip(),prior_models_untouched=True))


@torch.no_grad()
def parity(model,x,sigma):
    x=torch.as_tensor(x,device='cuda',dtype=torch.float32)
    ours=model(x).cpu();native=model.pipeline.predict(x,prediction_length=24).transpose(1,2)
    scale=torch.as_tensor(sigma,dtype=torch.float32)[:,None,None]
    torch.testing.assert_close(ours/scale,native/scale,atol=1e-5,rtol=1e-5)
    return dict(max_abs=float((ours-native).abs().max()),max_scaled=float(((ours-native)/scale).abs().max()))


def daily_pairs(d,part):
    return [(i,int(o)) for i in CONFIG['data']['development_ids'] for o in d.origins('dev',part,stride=24)]


def h1_precheck(m,d):
    start=time.perf_counter();profiles=[];packets={}
    for part in ['adapt_a','adapt_b']:
        pairs=daily_pairs(d,part)
        x,y,sigma,_=d.batch(pairs,group='dev',condition='CLEAN',role='P0_H1')
        q=m.predict(x);p=[]
        for i in CONFIG['data']['development_ids']:
            ix=np.array([sid==i for sid,_ in pairs]);a,b=affine(q[ix],y[ix])
            residual=(y[ix]-(a*q[ix,...,4]+b))/sigma[ix,None]
            p.append(residual.mean(0))
        p=np.stack(p);profiles.append(p-p.mean(0,keepdims=True))
        packets[part+'_q']=q;packets[part+'_y']=y;packets[part+'_sigma']=sigma
        packets[part+'_pairs']=np.asarray(pairs)
    rows=[]
    for j,i in enumerate(CONFIG['data']['development_ids']):
        a,b=profiles[0][j],profiles[1][j];den=np.linalg.norm(a)*np.linalg.norm(b)
        cosine=float(np.dot(a,b)/den) if den>0 else 0.
        ra=float(np.sqrt(np.mean(a*a)));rb=float(np.sqrt(np.mean(b*b)))
        rows.append(dict(series=i,cosine=cosine,rms_a=ra,rms_b=rb,pass_client=cosine>0 and min(ra,rb)>=.02))
    passed=sum(r['pass_client'] for r in rows)>=2
    np.savez_compressed(CACHE/'P0_H1.npz',**packets,profiles=np.stack(profiles))
    out=dict(status='PASS_PROBLEM_GATE' if passed else 'STOP_SETUP_H1_NO_STABLE_HETEROGENEITY',
             passed=passed,clients_passing=sum(r['pass_client'] for r in rows),required_clients=2,rows=rows,
             seconds=time.perf_counter()-start,scope='DEV ADAPT only; no model training, no EVAL forecast scores',
             predictions_sha256=sha(CACHE/'P0_H1.npz'))
    save(RESULTS/'H1_PROBLEM_GATE.json',out);event('h1_problem_gate',**out)
    return out


def h2_precheck(m,d):
    start=time.perf_counter();pairs=daily_pairs(d,'adapt_a')+daily_pairs(d,'adapt_b')
    x,y,sigma,_=d.batch(pairs,group='dev',condition='CLEAN',role='P0_H2')
    clean=m.predict(x);cleanloss=point_loss(clean,y,sigma)
    blocks=[];interps=[];partial=[];packets=dict(clean=clean,y=y,sigma=sigma,pairs=np.asarray(pairs))
    for repeat in range(2):
        bx,by,bs,_=d.batch(pairs,group='dev',condition='BLOCK48',role='P0_H2',repeat=repeat)
        ix,iy,iss,_=d.batch(pairs,group='dev',condition='BLOCK48',role='P0_H2',repeat=repeat,interpolate=True)
        np.testing.assert_array_equal(by,y);np.testing.assert_array_equal(iy,y)
        obs=np.isfinite(bx).reshape(-1,12,16).sum(-1)
        partial.extend(((obs>0)&(obs<16)).sum(-1).tolist())
        bq=m.predict(bx);iq=m.predict(ix)
        blocks.append(point_loss(bq,y,bs));interps.append(point_loss(iq,y,iss))
        packets[f'block_{repeat}']=bq;packets[f'interp_{repeat}']=iq;packets[f'mask_{repeat}']=np.isfinite(bx)
    blockloss=np.mean(blocks,axis=0);interploss=np.mean(interps,axis=0)
    cs=float(cleanloss.mean());bs=float(blockloss.mean());ins=float(interploss.mean())
    degradation=100*(bs-cs)/cs;remaining=100*(ins-cs)/cs;mp=float(np.mean(partial))
    passed=degradation>=1 and remaining>=.5 and mp>=1
    np.savez_compressed(CACHE/'P0_H2.npz',**packets)
    rows=[]
    for i in CONFIG['data']['development_ids']:
        ix=np.array([sid==i for sid,_ in pairs])
        rows.append(dict(series=i,clean=float(cleanloss[ix].mean()),block=float(blockloss[ix].mean()),interp=float(interploss[ix].mean())))
    out=dict(status='PASS_PROBLEM_GATE' if passed else 'STOP_SETUP_H2_NO_UNRESOLVED_MASK_GAP',passed=passed,
             clean_score=cs,block_score=bs,interpolated_score=ins,block_degradation_pct=degradation,
             interpolated_gap_pct=remaining,mean_partial_patches=mp,rows=rows,seconds=time.perf_counter()-start,
             scope='DEV ADAPT only; synthetic missingness; no EVAL forecast scores',predictions_sha256=sha(CACHE/'P0_H2.npz'))
    save(RESULTS/'H2_PROBLEM_GATE.json',out);event('h2_problem_gate',**out)
    return out


def main():
    if (RESULTS/'P0_COMPLETE.json').exists():
        raise RuntimeError('P0 already complete: use archived receipts; do not rerun gates')
    RESULTS.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    setup();provenance();data.audit();d=data.load_data();m=ForecastModel()
    pairs=daily_pairs(d,'adapt_a')[:8];parities={}
    for kind in ['CLEAN','IID48','BLOCK48','ALIGNED48_LAST']:
        x,_,s,_=d.batch(pairs,group='dev',condition=kind,role='PARITY')
        parities[kind]=parity(m,x,s)
    save(RESULTS/'NATIVE_PARITY.json',dict(status='PASS',conditions=parities,optimizer_updates=0))
    event('native_parity_pass',conditions=parities)
    h1=h1_precheck(m,d);h2=h2_precheck(m,d)
    payload=dict(status='COMPLETE',h1=h1['status'],h2=h2['status'],new_neural_fits=0,new_optimizer_updates=0,
                 native_parity=True,source_hashes=source_hashes(),no_test_scores_accessed=True)
    save(RESULTS/'P0_COMPLETE.json',payload);save(RESULTS/'PREFLIGHT.json',payload)


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        event('preflight_error',error=repr(e),traceback=traceback.format_exc())
        raise
