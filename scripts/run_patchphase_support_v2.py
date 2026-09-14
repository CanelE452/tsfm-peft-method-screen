"""Bounded support-complete patchphase experiment, no retry or v3 continuation."""
import argparse,fcntl,gc,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
import run_temporal_transfer_gradient_v1 as G
from tsfm_peft_screen.candidates.patchphase_support_v2 import PhaseModel,prepare_context
from tsfm_peft_screen.backbone import native_loss,REVISION
from tsfm_peft_screen.data import Panel
from tsfm_peft_screen.metrics import score,independent
from tsfm_peft_screen.reproducibility import ROOT,sha,digest,seed_all,write_json,guard
OUT=ROOT/'results/patchphase_v2_support_complete';CACHE=ROOT/'.cache/patchphase_v2_support_complete'
RESEARCH=ROOT/'research/reopen_review_20260914'
ARMS=['standard','augmented','conditioned'];STEPS=[0,60,120,240,360,540,720]


def params(m):return {n:p for n,p in m.named_parameters() if p.requires_grad}
def state(m):return {n:p.detach().cpu().clone() for n,p in params(m).items()}
def batch(panel,o):
    x,y=panel.window(int(o));return torch.tensor(x,device='cuda'),torch.tensor(y,device='cuda'),torch.zeros(len(x),dtype=torch.long,device='cuda')
def read(p):return json.loads(Path(p).read_text())


def main(mode):
    folder=RESEARCH/'patch_smoke' if mode=='smoke' else OUT
    assert not folder.exists(),'No overwrite or retry';folder.mkdir(parents=True)
    if mode=='run':
        assert read(RESEARCH/'patch_smoke/receipt.json')['status']=='COMPLETED'
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'Commit/push protocol/code before run'
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT)==subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT)
        CACHE.mkdir()
    lock=open(ROOT/'.cache/gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    G.OUT=folder;watch=G.Budget();start=time.monotonic();count=dict(fits_started=0,fits_completed=0,updates=0,forward=0,evaluation_opens=0)
    def save(status='RUNNING',**kw):write_json(folder/'receipt.json',dict(status=status,counts=count,seconds=time.monotonic()-start,**kw))
    def check(label):watch.resource(label);guard(start)
    check('startup')
    panel=Panel('ettm2');assert len(panel.channels)==7
    for f,h in panel.meta['files'].items():assert sha(panel.root/f)==h
    history={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'results').rglob('*') if p.is_file() and OUT not in p.parents}
    source=G.snapshot_history() if mode=='run' else {}
    contract=dict(model_revision=REVISION,execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protocol_sha256=sha(RESEARCH/'PATCHPHASE_V2_PROTOCOL.md'),code_sha256=sha(__file__),module_sha256=sha(ROOT/'src/tsfm_peft_screen/candidates/patchphase_support_v2.py'),
        data_manifest_sha256=sha(panel.root/'manifest.json'),train=list(map(int,panel.origins['train'])),V=list(map(int,panel.origins['validation'])),E=list(map(int,panel.origins['evaluation'])),
        phases=dict(train=list(range(0,16,2)),V=[1,5,9,13],E=[3,7,11,15]),history=history)
    write_json(folder/'contract.json',contract)
    traj=[];chosen=[];predindex=[]
    def evaluate(m,split,phases,tag):
        vals=[];preds=[]
        with torch.no_grad(),G.torch.random.fork_rng(devices=[0]):
            for phase in phases:
                pp=[];yy=[]
                for i,o in enumerate(panel.origins[split]):
                    if i%8==0:check('evaluate')
                    x,y,g=batch(panel,o);count['forward']+=1
                    p=m(x,g,phase)[1];assert torch.isfinite(p).all()
                    pp.append(p.cpu().numpy());yy.append(y.cpu().numpy())
                p=np.stack(pp);y=np.stack(yy);s=score(p,y,panel.scale)['scaled_2pinball']
                assert abs(s-independent(p,y,panel.scale))<=1e-10
                path=CACHE/f'{tag}_phase{phase}.npz';np.savez_compressed(path,prediction=p,target=y,scale=panel.scale,origins=panel.origins[split])
                predindex.append(dict(tag=tag,phase=phase,split=split,path=str(path.relative_to(ROOT)),sha256=sha(path),loss=s))
                vals.append(s);preds.append(p)
        return dict(primary=float(np.mean(vals)),per_phase=dict(zip(map(str,phases),vals)),
                    variance=float(np.var(np.stack(preds).astype(np.float64)/panel.scale[None,None,:,None,None],axis=0).mean()))
    try:
        if mode=='smoke':
            receipts=[]
            for arm in ARMS:
                seed_all(30000);m=PhaseModel(arm,30000);opt=torch.optim.AdamW(params(m).values(),lr=3e-5,weight_decay=0.)
                x,y,g=batch(panel,panel.origins['train'][0])
                old=m.core._prepare_patched_context(x);new=prepare_context(m.core,x,None,0)
                for a,b in zip(old[:2],new[:2]):torch.testing.assert_close(a,b,rtol=0,atol=0)
                for phase in range(16):
                    with torch.no_grad():p=m(x,g,phase)[1];count['forward']+=1;assert torch.isfinite(p).all()
                before=state(m)
                for phase in [0,2]:
                    opt.zero_grad(set_to_none=True);z,p,l,s=m(x,g,phase);count['forward']+=1
                    loss=native_loss(z,y,l,s);loss.backward();torch.nn.utils.clip_grad_norm_(params(m).values(),1.,error_if_nonfinite=True)
                    count['updates']+=1;opt.step()
                assert any(not torch.equal(p,before[n].to(p)) for n,p in params(m).items())
                if arm=='conditioned':assert m.adapter.gate.weight.grad is not None and torch.isfinite(m.adapter.gate.weight.grad).all()
                receipts.append(dict(arm=arm,parameters=sum(p.numel() for p in params(m).values()),loss=float(loss.detach())))
                del m,opt,x,y,g,z,p,l,s,loss;gc.collect();torch.cuda.empty_cache()
            save('COMPLETED',arms=receipts,resources=guard(start));return
        schedules={}
        for seed in [30000,30001]:
            rng=np.random.default_rng(seed);schedules[str(seed)]=[dict(origin=int(rng.choice(panel.origins['train'])),phase=int(2*(i%8))) for i in range(720)]
        write_json(OUT/'schedules.json',schedules)
        for seed in [30000,30001]:
            for arm in ARMS:
                records=[]
                for lr in [3e-5,1e-4]:
                    fid=f'{arm}_{seed}_{lr:g}';count['fits_started']+=1;save(fit=fid)
                    seed_all(seed);m=PhaseModel(arm,seed);opt=torch.optim.AdamW(params(m).values(),lr=lr,weight_decay=0.)
                    frozen=lambda:G.tensor_hash({n:p for n,p in m.named_parameters() if not p.requires_grad})
                    frozen_before=frozen()
                    for step in range(721):
                        if step in STEPS:
                            met=evaluate(m,'validation',[1,5,9,13],f'{fid}_V_{step}')
                            cp=CACHE/f'{fid}_{step}.pt';torch.save(state(m),cp)
                            row=dict(arm=arm,seed=seed,lr=lr,step=step,checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=sha(cp),**met)
                            records.append(row);traj.append(row);write_json(OUT/'trajectory.json',traj)
                            save(fit=fid,step=step);print(json.dumps(dict(fit=fid,step=step,V=met['primary'],updates=count['updates'])),flush=True)
                        if step==720:break
                        if step%10==0:check('train')
                        r=schedules[str(seed)][step];x,y,g=batch(panel,r['origin']);opt.zero_grad(set_to_none=True)
                        count['forward']+=1;z,p,l,s=m(x,g,r['phase']);loss=native_loss(z,y,l,s)
                        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(params(m).values(),1.,error_if_nonfinite=True)
                        count['updates']+=1;opt.step()
                        del x,y,g,z,p,l,s,loss
                    assert frozen()==frozen_before
                    count['fits_completed']+=1;del m,opt;gc.collect();torch.cuda.empty_cache()
                chosen.append(min(records,key=lambda r:(r['primary'],r['step'],r['lr'])))
        seal=dict(contract_sha256=sha(OUT/'contract.json'),selections=chosen,evaluation_opens=count['evaluation_opens']);seal['sha256']=digest(seal);write_json(OUT/'selection_seal.json',seal)
        assert len(chosen)==6 and count['updates']==8640
        assert digest({k:v for k,v in seal.items() if k!='sha256'})==seal['sha256']
        with np.load(panel.root/'evaluation.npz',allow_pickle=False) as f:panel.values=np.concatenate([panel.values,f['tail']])
        count['evaluation_opens']+=1;results=[]
        for row in chosen:
            seed_all(row['seed']);m=PhaseModel(row['arm'],row['seed']);assert sha(ROOT/row['checkpoint'])==row['checkpoint_sha256']
            G.restore(params(m),torch.load(ROOT/row['checkpoint'],map_location='cpu',weights_only=True))
            met=evaluate(m,'evaluation',[3,7,11,15],f"{row['arm']}_{row['seed']}_E")
            canonical=evaluate(m,'evaluation',[0],f"{row['arm']}_{row['seed']}_canonical")
            results.append(dict(selection=row,unseen=met,canonical=canonical))
            write_json(OUT/'evaluation.json',results);del m;gc.collect();torch.cuda.empty_cache()
        write_json(OUT/'prediction_index.json',predindex)
        assert all(sha(ROOT/p)==h for p,h in history.items())
        assert sha(ROOT/'src/tsfm_peft_screen/candidates/patchphase_support_v2.py')==contract['module_sha256']
        save('COMPLETED',resources=guard(start),historical_files_unchanged=len(history),exit_code=0)
    except BaseException as e:save('EXECUTION_ERROR',error=repr(e),exit_code=1);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','run']);main(p.parse_args().mode)
