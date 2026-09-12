import csv,gc,json,time
from pathlib import Path
import numpy as np
import torch
from ..backbone import load_base,forecast,native_loss
from ..lora import attach,audit,disabled,set_state,snapshot,restore
from ..metrics import score,replay
from ..data import Panel
from ..reproducibility import ROOT,seed_all,guard,write_json,sha,source_hashes,digest
from ..selection import choose,seal
from ..candidates.freshness import corrupt,token_state
from ..candidates.dualclock import event_features,EventAdapter
from ..candidates.patchphase import PhaseAdapter,prepare_context,patch_phase,unpatch
from ..candidates.fr_lora import regularizer
from ..candidates.censor import objective as censor_objective,cdf
ARMS={1:['STANDARD_LORA','FEATURE_LORA','FRESHNESS_GATED_LORA'],2:['STANDARD_LORA','EVENT_SUMMARY_LORA','DUALCLOCK_ADAPTER'],3:['STANDARD_LORA','PHASE_AUGMENTED_ADAPTER','PHASE_CONDITIONED_ADAPTER'],4:['STANDARD_LORA','RAW_STABILITY_LORA','F0_ANCHOR_LORA','FR_LORA'],7:['NAIVE_LORA','DROP_CENSORED_LORA','CENSORED_LOSS_LORA','CENSOR_PRESERVE_LORA']}
DATA={1:'jena',2:'m5',3:'ettm2',4:'jena',5:'jena',6:'electricity',7:'m5'}
CHECKPOINTS=[0,4,8,15,30,60,120,180,240,360]

def csv_write(path,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'w') as f:
        w=csv.DictWriter(f,fieldnames=fields or ['not_run']);w.writeheader();w.writerows(rows)

def build(candidate,arm,seed=30000):
    seed_all(seed);m=load_base();adapter=None
    if arm!='F0':
        mode={'FEATURE_LORA':'feature','FRESHNESS_GATED_LORA':'freshness'}.get(arm,'standard');attach(m,seed,mode)
        torch.manual_seed(seed+2000)
        if arm in ['EVENT_SUMMARY_LORA','DUALCLOCK_ADAPTER']:adapter=EventAdapter(m.config.d_model,arm=='DUALCLOCK_ADAPTER').cuda()
        if arm in ['PHASE_AUGMENTED_ADAPTER','PHASE_CONDITIONED_ADAPTER']:adapter=PhaseAdapter(m.config.d_model,arm=='PHASE_CONDITIONED_ADAPTER').cuda()
    return m,adapter

def predict(m,adapter,candidate,x,groups,variant,observations=None):
    state=None
    if candidate==1:
        assert observations is not None
        set_state(m,token_state(observations,m.chronos_config.use_reg_token))
    if candidate==2 and adapter is not None:state=event_features(x)
    if candidate==3:state=int(variant)
    return forecast(m,x,groups,phase=int(variant) if candidate==3 else 0,adapter=adapter,state=state)

def batch(panel,origins,channels,candidate,variant):
    xs=[];ys=[];rs=[];sc=[];caps=[];groups=[]
    for i,(o,cs) in enumerate(zip(origins,channels)):
        x,y=panel.window(int(o),cs)
        if candidate==1:x,obs=corrupt(x,variant);rs.append(obs)
        if candidate==7:x=np.minimum(x,panel.caps[cs,None])
        xs.append(x);ys.append(y);groups.extend([i]*len(cs) if panel.name!='m5' else range(len(groups),len(groups)+len(cs)))
        sc.extend(panel.scale[cs]);caps.extend(panel.caps[cs])
    tensor=lambda a:torch.as_tensor(np.concatenate(a) if isinstance(a,list) and isinstance(a[0],np.ndarray) else a,device='cuda',dtype=torch.float32)
    return tensor(xs),tensor(ys),torch.tensor(groups,device='cuda'),tensor(rs) if rs else None,tensor(sc),tensor(caps)

def eval_variants(candidate,split):
    if candidate==1:return ['block6','refresh2'] if split=='validation' else ['block12','block24','refresh4','refresh8','clean']
    if candidate==3:return [0,8] if split=='validation' else [0,4,12]
    return ['clean']

def predictions(m,adapter,panel,candidate,split,variant,origins=None):
    pp=[];yy=[];origins=panel.origins[split] if origins is None else origins
    for o in origins:
        chunks=[];targets=[]
        for cs in np.array_split(np.arange(len(panel.channels)),max(1,int(np.ceil(len(panel.channels)/32)))):
            x,y,g,r,sc,caps=batch(panel,[o],[cs],candidate,variant)
            with torch.no_grad():p=predict(m,adapter,candidate,x,g,variant,r)[1]
            chunks.append(p.cpu().numpy());targets.append(y.cpu().numpy())
        pp.append(np.concatenate(chunks));yy.append(np.concatenate(targets))
    return np.stack(pp),np.stack(yy)

def validation(m,adapter,panel,candidate):
    vals=[]
    for variant in eval_variants(candidate,'validation'):
        p,y=predictions(m,adapter,panel,candidate,'validation',variant)
        vals.append(score(p,y,panel.scale)['scaled_2pinball'])
    return float(np.mean(vals))

def gate(candidate,panel,out,cache):
    if candidate not in (1,3):return {'status':'PASS','reason':'Data/construct unit tests pass; no numerical F0 problem threshold specified'}
    seed_all(30000);m=load_base();variants=['clean','block6','block12','refresh2','refresh4','stale'] if candidate==1 else [0,4,8,12]
    pred={};loss={}
    if candidate==3:
        x,_,_,_,_,_=batch(panel,[panel.origins['gate'][0]],[np.arange(len(panel.channels))],candidate,0)
        for phase in variants:
            torch.testing.assert_close(unpatch(patch_phase(x,phase)),x,rtol=0,atol=0,equal_nan=True)
        native=m._prepare_patched_context(x);custom=prepare_context(m,x,None,0)
        for a,b in zip(native[:2],custom[:2]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    for variant in variants:
        p,y=predictions(m,None,panel,candidate,'gate',variant);pred[variant]=p;loss[str(variant)]=score(p,y,panel.scale)['scaled_2pinball']
        np.savez_compressed(cache/f'gate_{variant}.npz',prediction=p,target=y,scale=panel.scale)
        replay(cache/f'gate_{variant}.npz');guard();print('GATE',candidate,variant,loss[str(variant)],flush=True)
    if candidate==1:
        degradation={k:100*(v-loss['clean'])/loss['clean'] for k,v in loss.items() if k!='clean'}
        passed=np.mean(list(degradation.values()))>=2 and sum(v>=1 for v in degradation.values())>=2
        result=dict(status='PASS' if passed else 'NO_PROBLEM',loss=loss,degradation_percent=degradation,mean_degradation_percent=float(np.mean(list(degradation.values()))))
    else:
        span=100*(max(loss.values())-min(loss.values()))/loss['0']
        discrepancy=100*float(np.mean([np.mean(abs(pred[v]-pred[0])/panel.scale[None,:,None,None]) for v in [4,8,12]]))
        result=dict(status='PASS' if span>=.5 or discrepancy>=.25 else 'NO_PROBLEM',loss=loss,loss_range_percent=span,normalized_discrepancy_percent=discrepancy,construct='Exact values, original time encodings, masks, context information, forecast origin and future tokens preserved; only masked padding changes patch membership',origins=128)
    del m;gc.collect();torch.cuda.empty_cache();write_json(out/'round0.json',result);return result

def train_schedule(panel,candidate):
    rng=np.random.default_rng(30000);schedule=[]
    for step in range(360):
        if candidate==4:
            o=int(rng.choice(panel.origins['train']));origins=[o,o+24];channels=[np.arange(4),np.arange(4)]
        elif panel.name=='m5':
            origins=rng.choice(panel.origins['train'],8).tolist();channels=[np.array([int(c)]) for c in rng.integers(0,256,8)]
        else:
            n=2 if panel.name=='jena' else 1;origins=rng.choice(panel.origins['train'],n).tolist();channels=[np.arange(len(panel.channels)) for _ in origins]
        variant=['block6','refresh2'][step%2] if candidate==1 else [0,8][step%2] if candidate==3 else 'clean'
        schedule.append((origins,channels,variant))
    return schedule

def train(candidate):
    out=ROOT/'results'/f'candidate_{candidate:02}';cache=ROOT/'.cache'/f'candidate_{candidate:02}';cache.mkdir(parents=True,exist_ok=True)
    assert json.loads((ROOT/'results/screening_summary/common_integrity.json').read_text())['status']=='PASS'
    if (out/'selection.json').exists():raise FileExistsError('Candidate already sealed; no training rerun')
    panel=Panel(DATA[candidate]);start=time.monotonic();g=gate(candidate,panel,out,cache)
    contract=dict(candidate=candidate,dataset=panel.name,manifest_hash=sha(panel.root/'manifest.json'),common=json.loads((ROOT/'configs/common_screen.yaml').read_text()),source_hashes=source_hashes(),arms=ARMS[candidate],seed=30000,round0=g,validation_variants=eval_variants(candidate,'validation'),evaluation_variants=eval_variants(candidate,'evaluation'),selection='lowest V scaled 2pinball, then earliest step, then lowest LR',effect_denominator='F0 primary',effective_batch_series=7 if candidate==3 else 8)
    write_json(out/'contract.json',contract)
    if g['status']!='PASS':
        write_json(out/'status.json',dict(verdict=g['status'],fit_count=0,stream_count=0,problem_gate=g['status'],novelty_collision=False));write_json(out/'integrity.json',dict(status='PASS',round0_only=True,common='screening_summary/common_integrity.json'));write_json(out/'resource_usage.json',dict(wall_seconds=time.monotonic()-start,fit_count=0,**guard()));return
    schedule=train_schedule(panel,candidate);schedule_serial=[dict(origins=o,channels=[c.tolist() for c in cs],variant=v) for o,cs,v in schedule];write_json(out/'sampling_manifest.json',schedule_serial);contract['sampling_hash']=digest(schedule_serial)
    # Fix regularization scales using only the first declared train batch.
    recipe={};seed_all(30000)
    if candidate in (4,7):
        m=load_base();o,cs,v=schedule[0];x,y,groups,r,sc,caps=batch(panel,o,cs,candidate,v)
        with torch.no_grad():z,p,l,s=forecast(m,x,groups);task=float(native_loss(z,y,l,s))
        if candidate==4:
            reg=float(regularizer('RAW_STABILITY_LORA',p,p,sc));recipe={'lambda':float(np.clip(.01*task/max(reg,1e-6),.001,1.)),'train_task':task,'train_raw_stability':reg}
        else:
            sale=torch.minimum(y,caps[:,None]);censored=y>caps[:,None];task=float(native_loss(z,sale.masked_fill(censored,float('nan')),l,s));survival=float((-(1-cdf(p,sale)).clamp_min(1e-6).log()*censored).mean())
            recipe={'lambda_c':float(np.clip(.1*task/max(survival,1e-6),.001,100.)),'lambda_p':float(.02*task),'train_task':task,'train_censor':survival,'preserve_reference':'unit standardized correction SmoothL1=0.5; 1% task weight'}
        del m;gc.collect();torch.cuda.empty_cache()
    contract['recipe']=recipe;write_json(out/'contract.json',contract)
    allrows=[];winners=[];integrity=[];fitusage=[];base_cache={};attempts=[]
    for arm in ARMS[candidate]:
        records=[]
        for lr in [3e-5,1e-4]:
            attempts.append(dict(arm=arm,lr=lr,status='RUNNING'));write_json(out/'attempts.json',attempts)
            fit_start=time.monotonic();guard();torch.cuda.reset_peak_memory_stats();m,adapter=build(candidate,arm)
            fit_id=f'{arm}_{lr:g}';fitdir=cache/fit_id;fitdir.mkdir(exist_ok=True)
            # Initial identity on actual inputs, with all proposed components enabled.
            o,cs,v=schedule[0];x,y,groups,r,sc,caps=batch(panel,o,cs,candidate,v)
            with torch.no_grad():
                p=predict(m,adapter,candidate,x,groups,v,r)[1]
                with disabled(m):f0=forecast(m,x,groups,phase=int(v) if candidate==3 else 0)[1]
            err=float((p-f0).abs().max());assert err<=1e-6
            audit_info=audit(m,standard=arm not in ['FEATURE_LORA','FRESHNESS_GATED_LORA'])
            frozen_hashes={n:digest([list(t.shape),float(t.double().sum()),float(t.double().square().sum())]) for n,t in m.named_parameters() if not t.requires_grad}
            params=[p for model in [m,adapter] if model is not None for p in model.parameters() if p.requires_grad]
            opt=torch.optim.AdamW(params,lr=lr,weight_decay=0);best=float('inf');beststate=None;beststep=None
            for step in range(361):
                if step in CHECKPOINTS:
                    val=validation(m,adapter,panel,candidate);assert np.isfinite(val)
                    row=dict(arm=arm,lr=lr,step=step,validation_loss=val,train_loss=None if step==0 else float(loss.detach()))
                    allrows.append(row);records.append(dict(arm=arm,lr=lr,step=step,validation_loss=val,checkpoint=str(fitdir/'best.pt')))
                    if val<best:best=val;beststep=step;beststate=snapshot(m,adapter);torch.save(beststate,fitdir/'best.pt')
                    csv_write(out/'trajectories.csv',allrows);print('FIT',candidate,arm,lr,step,val,flush=True)
                if step==360:break
                guard(fit_start);o,cs,v=schedule[step];x,y,groups,r,sc,caps=batch(panel,o,cs,candidate,v)
                if candidate in (4,7):
                    if step not in base_cache:
                        with torch.no_grad(),disabled(m):base_cache[step]=forecast(m,x,groups)[1].cpu()
                    base=base_cache[step].to('cuda')
                opt.zero_grad(set_to_none=True);z,p,l,s=predict(m,adapter,candidate,x,groups,v,r);loss=native_loss(z,y,l,s)
                if candidate==4 and arm!='STANDARD_LORA':loss=loss+recipe['lambda']*regularizer(arm,p,base,sc)
                if candidate==7:
                    sale=torch.minimum(y,caps[:,None]);censored=y>caps[:,None]
                    loss=censor_objective(arm,z,p,sale,censored,l,s,base,sc,recipe['lambda_c'],recipe['lambda_p'])
                if not torch.isfinite(loss):raise FloatingPointError(f'{fit_id} nonfinite loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise FloatingPointError('Nonfinite gradient')
                torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
            assert beststate is not None
            # Validate the selected state from disk before E access.
            restore(beststate,m,adapter)
            with torch.no_grad():expected=predict(m,adapter,candidate,x,groups,v,r)[1].cpu()
            restore(torch.load(fitdir/'best.pt',weights_only=True),m,adapter)
            with torch.no_grad():actual=predict(m,adapter,candidate,x,groups,v,r)[1].cpu()
            replay_err=float((actual-expected).abs().max());assert replay_err==0
            for n,t in m.named_parameters():
                if not t.requires_grad:assert frozen_hashes[n]==digest([list(t.shape),float(t.double().sum()),float(t.double().square().sum())])
            integrity.append(dict(fit=fit_id,identity_max_abs=err,checkpoint_replay_max_abs=replay_err,frozen_unchanged=True,**audit_info,adapter_trainable_count=sum(p.numel() for p in adapter.parameters()) if adapter else 0))
            fitusage.append(dict(fit=fit_id,wall_seconds=time.monotonic()-fit_start,optimizer_steps=360,**guard()))
            write_json(out/'integrity.json',dict(status='PASS',fits=integrity));write_json(out/'resource_usage.json',dict(fits=fitusage,fit_count=len(fitusage),wall_seconds=time.monotonic()-start))
            attempts[-1]['status']='COMPLETE';write_json(out/'attempts.json',attempts)
            del m,adapter,opt,params,beststate;gc.collect();torch.cuda.empty_cache()
        # Each fit saves only its own V best; choose returns that step on its LR.
        winner=choose(records);winner['checkpoint_sha256']=sha(winner['checkpoint']);winners.append(winner)
    seal(out/'selection.json',winners,contract);csv_write(out/'selections.csv',winners)
    from .common_eval import evaluate
    evaluate(candidate,panel,contract,winners,out,cache)
    resources=json.loads((out/'resource_usage.json').read_text());resources['wall_seconds']=time.monotonic()-start;resources['stream_count']=0;write_json(out/'resource_usage.json',resources)
