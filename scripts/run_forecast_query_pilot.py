"""GPU-monitored chronological16-fit pilot. No E access before sealing V choices."""
import fcntl,gc,json,subprocess,time
from contextlib import nullcontext
import numpy as np
import torch
from tsfm_peft_screen.reproducibility import ROOT,sha,write_json,source_hashes,digest,seed_all,guard
from tsfm_peft_screen.forecast_query.model import ForecastModel
from tsfm_peft_screen.forecast_query.gpu import GPUWatch
from tsfm_peft_screen.backbone import native_loss
from tsfm_peft_screen.metrics import score
OUT=ROOT/'results/forecast_query_pilot';CACHE=ROOT/'.cache/forecast_query_pilot'
CFG=json.loads((ROOT/'configs/forecast_query_pilot.json').read_text())
def origins(key):
    q=CFG[key];return list(range(q['start'],q['stop_inclusive']+1,q['stride']))
def state(model):return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
def restore(model,s):
    p={n:p for n,p in model.named_parameters() if p.requires_grad};assert p.keys()==s.keys()
    with torch.no_grad():
        for n,v in s.items():p[n].copy_(v.to(p[n]))
def batch(values,os):
    x=np.concatenate([values[o-4096:o,:4].T for o in os]).astype(np.float32);y=np.concatenate([values[o:o+48,:4].T for o in os]).astype(np.float32)
    return torch.tensor(x,device='cuda'),torch.tensor(y,device='cuda'),torch.arange(len(os),device='cuda').repeat_interleave(4)
def evaluate(model,values,scale,os,tag,f0=False):
    predictions=[];targets=[];losses=[]
    with torch.no_grad():
        for j in range(0,len(os),2):
            watch.check('evaluation_'+tag);batchos=os[j:j+2];x,y,g=batch(values,batchos)
            with torch.autocast('cuda',dtype=torch.bfloat16):z,p,l,s=model.f0(x,g) if f0 else model(x,g);loss=native_loss(z,y,l,s)
            predictions.append(p.float().cpu().numpy().reshape(len(batchos),4,21,48));targets.append(y.cpu().numpy().reshape(len(batchos),4,48));losses.extend([float(loss)]*len(batchos))
    pred=np.concatenate(predictions);target=np.concatenate(targets);metrics=score(pred,target,scale);metrics['native_loss']=float(np.mean(losses))
    np.savez_compressed(CACHE/f'{tag}.npz',prediction=pred,target=target,scale=scale)
    return metrics

def train_step(model,opt,x,y,g):
    opt.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):z,p,l,s=model(x,g);loss=native_loss(z,y,l,s)
    assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad and p.grad is not None)
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();torch.cuda.synchronize()
    return dict(loss=float(loss.detach()),seconds=time.monotonic()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gradient_norm=float(norm))

lock=open(ROOT/'.cache/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not OUT.exists(),'Immutable experiment'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'Commit before running'
OUT.mkdir();CACHE.mkdir();watch=GPUWatch(OUT/'gpu_monitor.json');watch.wait_idle();start=time.monotonic();fits=[];smokes=[];trajectories=[]
contract=dict(execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hashes=source_hashes(),config=CFG,config_sha256=sha(ROOT/'configs/forecast_query_pilot.json'),data_manifest_hashes={n:sha(ROOT/'data/processed'/n/'manifest.json') for n in CFG['datasets']},fit_data_hashes={n:sha(ROOT/'data/processed'/n/'fit.npz') for n in CFG['datasets']},data_scope='chronological train/V; evaluation tail loaded only after all8 choices sealed; development periods previously exposed',fit_count_cap=16)
write_json(OUT/'contract.json',contract);fitdata={}
for name in CFG['datasets']:
    with np.load(ROOT/'data/processed'/name/'fit.npz') as f:fitdata[name]=(f['values'],f['scale'][:4])
try:
    seed_all(CFG['seed'])
    # Minimal real-model checks, two diagnostic steps per arm, discarded afterwards.
    for arm in CFG['arms']:
        watch.check('smoke_'+arm,True);m=ForecastModel(arm,CFG['seed']);values,scale=fitdata['ettm2'];x,y,g=batch(values,[4352,4608])
        with torch.no_grad():
            z=m(x,g)[0];f=m.f0(x,g)[0];fp32rel=float((z-f).norm()/f.norm());fp32max=float((z-f).abs().max())
            assert fp32rel<=1e-5 and fp32max<=1e-4,(arm,fp32rel,fp32max)
            with torch.autocast('cuda',dtype=torch.bfloat16):za=m(x,g)[0];fa=m.f0(x,g)[0]
            amprel=float((za-fa).norm()/fa.norm());assert amprel<=.02
        initial=state(m);opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=CFG['recipes'][arm][0],weight_decay=0)
        grad_tokens=[];handles=[]
        if arm=='query':
            for block in m.base.encoder.block:
                def observe(module,args,out):
                    if torch.is_grad_enabled():grad_tokens.append(args[0].shape[1])
                handles.append(block.layer[2].register_forward_hook(observe))
            c,_,_=m.encode_frozen(x,g);assert all(not v.requires_grad for kind in ['k','v'] for v in c[kind]);del c
        first=train_step(m,opt,x,y,g);second=train_step(m,opt,x,y,g)
        for h in handles:h.remove()
        if arm=='query':assert grad_tokens and set(grad_tokens)=={4}
        trained=state(m);changed=sum(not torch.equal(v,trained[n]) for n,v in initial.items());assert changed>0
        with torch.no_grad():
            after=m(x,g)[0];effect=float((after-z).abs().max());assert effect>0
            modified=x.clone();modified[4:]+=100;isolated=m(modified,g)[0]
            group_error=float((after[:4]-isolated[:4]).abs().max());assert group_error<=1e-4
            if arm=='query':
                restore(m,initial);missing=x.clone();missing[:,:16]=float('nan');a=m(missing,g)[0];b=m.f0(missing,g)[0];maskerror=float((a-b).norm()/b.norm());assert maskerror<=1e-5
            else:maskerror=None
        smokes.append(dict(arm=arm,parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),fp32_initial_relative=fp32rel,fp32_initial_max_abs=fp32max,bf16_initial_relative=amprel,changed_parameter_tensors=changed,forecast_change_max_abs=effect,group_isolation_max_abs=group_error,missing_key_parity_relative=maskerror,query_backward_mlp_token_counts=sorted(set(grad_tokens)),diagnostic_optimizer_updates=2,step_resources=[first,second]))
        write_json(OUT/'smoke.json',smokes);print('SMOKE PASS',arm,fp32rel,amprel,flush=True)
        del m,opt,x,y,g,initial;gc.collect();torch.cuda.empty_cache();guard(start)
    # Full learning, with identical schedules and separate immutable fit summaries.
    schedule=np.random.default_rng(CFG['seed']).choice(origins('train_origins'),size=(CFG['steps'],2),replace=True).tolist();write_json(OUT/'train_schedule.json',schedule)
    for name in CFG['datasets']:
        values,scale=fitdata[name]
        for arm in CFG['arms']:
            for recipe,lr in enumerate(CFG['recipes'][arm]):
                watch.check('fit_start',True);seed_all(CFG['seed']);m=ForecastModel(arm,CFG['seed']);opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=lr,weight_decay=0)
                fid=f'{name}_{arm}_r{recipe}';fitstart=time.monotonic();best=None;resources=[];validation_seconds=0
                for step in range(CFG['steps']+1):
                    if step:
                        watch.check(fid);x,y,g=batch(values,schedule[step-1]);r=train_step(m,opt,x,y,g);resources.append(r);guard(fitstart)
                    if step in CFG['checkpoints']:
                        tick=time.monotonic();metric=evaluate(m,values,scale,origins('validation_origins'),f'V_{fid}_{step}');validation_seconds+=time.monotonic()-tick
                        row=dict(fit=fid,dataset=name,arm=arm,recipe=recipe,step=step,elapsed_seconds=time.monotonic()-fitstart,metrics=metric);trajectories.append(row);write_json(OUT/'trajectories.json',trajectories)
                        key=(metric['scaled_2pinball'],step)
                        if best is None or key<(best['metrics']['scaled_2pinball'],best['step']):
                            best=row;torch.save(state(m),CACHE/f'{fid}_best.pt')
                        print('FIT',fid,'step',step,'V',round(metric['scaled_2pinball'],6),flush=True)
                record=dict(fit=fid,dataset=name,arm=arm,recipe=recipe,lr=lr,seed=CFG['seed'],steps=CFG['steps'],best_step=best['step'],best_validation=best['metrics'],checkpoint_sha256=sha(CACHE/f'{fid}_best.pt'),wall_seconds=time.monotonic()-fitstart,validation_seconds=validation_seconds,median_step_seconds=float(np.median([r['seconds'] for r in resources])),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in resources),peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in resources),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),step_resources=resources)
                fits.append(record);write_json(OUT/'fits.json',fits);print('FIT COMPLETE',fid,'best',best['step'],'peak MiB',round(record['peak_allocated_bytes']/2**20,1),flush=True)
                del m,opt,x,y,g;gc.collect();torch.cuda.empty_cache()
    assert len(fits)==16 and source_hashes()==contract['source_hashes']
    selections=[]
    for name in CFG['datasets']:
        for arm in CFG['arms']:
            chosen=min([r for r in fits if r['dataset']==name and r['arm']==arm],key=lambda r:(r['best_validation']['scaled_2pinball'],r['best_step'],r['recipe']))
            selections.append({k:chosen[k] for k in ['fit','dataset','arm','recipe','best_step','best_validation','checkpoint_sha256']})
    seal=dict(selections=selections,source_hashes=source_hashes(),contract_sha256=sha(OUT/'contract.json'));seal['seal_sha256']=digest(seal);write_json(OUT/'selection_seal.json',seal)
    evaluation=[]
    for name in CFG['datasets']:
        prior,scale=fitdata[name]
        with np.load(ROOT/'data/processed'/name/'evaluation.npz') as f:values=np.concatenate([prior,f['tail']])
        for arm in CFG['arms']:
            watch.check('development_E',True);m=ForecastModel(arm,CFG['seed']);choice=next(r for r in selections if r['dataset']==name and r['arm']==arm);path=CACHE/f"{choice['fit']}_best.pt";assert sha(path)==choice['checkpoint_sha256'];restore(m,torch.load(path,weights_only=True))
            if arm=='standard':evaluation.append(dict(dataset=name,arm='F0',metrics=evaluate(m,values,scale,origins('evaluation_origins'),f'E_{name}_F0',f0=True)))
            evaluation.append(dict(dataset=name,arm=arm,fit=choice['fit'],step=choice['best_step'],metrics=evaluate(m,values,scale,origins('evaluation_origins'),f'E_{name}_{arm}')));write_json(OUT/'evaluation.json',evaluation)
            del m;gc.collect();torch.cuda.empty_cache()
    decisions=[]
    for name in CFG['datasets']:
        ev={r['arm']:r for r in evaluation if r['dataset']==name};sel={r['arm']:r for r in selections if r['dataset']==name};fr={arm:next(f for f in fits if f['fit']==choice['fit']) for arm,choice in sel.items()}
        q=ev['query']['metrics']['scaled_2pinball'];standard=ev['standard']['metrics']['scaled_2pinball'];simple=min(ev[a]['metrics']['scaled_2pinball'] for a in ['head','side']);f0=ev['F0']['metrics']['scaled_2pinball'];reduction=1-fr['query']['peak_allocated_bytes']/fr['standard']['peak_allocated_bytes']
        passed=sel['query']['best_step']>0 and q<f0 and q<=1.01*standard and reduction>=.20 and q<=.995*simple
        decisions.append(dict(dataset=name,query_vs_standard_loss_ratio=q/standard,query_vs_best_simple_loss_ratio=q/simple,query_vs_f0_loss_ratio=q/f0,query_peak_reduction=reduction,query_step=sel['query']['best_step'],pass_gate=passed))
    watch.check('complete',True);write_json(OUT/'status.json',dict(status='COMPLETE',verdict='PASS' if all(d['pass_gate'] for d in decisions) else 'FAIL',fit_count=16,training_optimizer_updates=3840,smoke_optimizer_updates=8,decisions=decisions,wall_seconds=time.monotonic()-start,e_scope='reused development chronological periods',expanded=False))
except Exception as e:
    write_json(OUT/'status.json',dict(status='IMPLEMENTATION_BLOCKED',error=str(e),completed_fits=len(fits),completed_smoke_arms=len(smokes)));raise
