"""Exactly twelve fixed-budget fits; original experiments are immutable."""
import csv,gc,json,subprocess,time
import numpy as np
import torch
from ..backbone import load_base,forecast,native_loss
from ..data import Panel
from ..metrics import score,replay
from ..lora import LowRank,set_state,snapshot,restore,disabled
from ..selection import seal
from ..reproducibility import ROOT,sha,digest,write_json,source_hashes,seed_all,guard
from ..candidates.freshness_v2 import corrupt,token_state,attach_v2,AffineLowRank,RULES,TRAIN_VARIANTS,E_VARIANTS
OUT=ROOT/'results/candidate_01_v2';CACHE=ROOT/'.cache/candidate_01_v2'
CONFIG=ROOT/'configs/candidate_01_v2.json'

def csv_write(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'w') as f:
        w=csv.DictWriter(f,fieldnames=keys or ['not_run'],lineterminator='\n');w.writeheader();w.writerows(rows)

def build(arm,seed):
    seed_all(seed);m=load_base()
    info=attach_v2(m,arm,seed) if arm!='F0' else dict(trainable_count=0)
    return m,info

def batch(panel,origins,variant):
    xx=[];yy=[];rr=[]
    for origin in origins:
        x,y=panel.window(int(origin));a,r=corrupt(x,variant,int(origin));xx.append(a);yy.append(y);rr.append(r)
    return (torch.tensor(np.concatenate(xx),device='cuda'),torch.tensor(np.concatenate(yy),device='cuda'),
            torch.arange(len(origins),device='cuda').repeat_interleave(4),torch.tensor(np.concatenate(rr),device='cuda'))

def predict(m,x,g,r):
    set_state(m,token_state(r,m.chronos_config.use_reg_token))
    return forecast(m,x,g)

def predictions(m,panel,split,variant):
    pp=[];yy=[]
    for origin in panel.origins[split]:
        x,y,g,r=batch(panel,[int(origin)],variant)
        with torch.no_grad():p=predict(m,x,g,r)[1]
        pp.append(p.cpu().numpy());yy.append(y.cpu().numpy())
    return np.stack(pp),np.stack(yy)

def validation(m,panel):
    values={}
    for variant in TRAIN_VARIANTS:
        p,y=predictions(m,panel,'validation',variant)
        values[variant]=score(p,y,panel.scale)['scaled_2pinball']
    return float(np.mean(list(values.values()))),values

def schedule(panel,seed,steps):
    rng=np.random.default_rng(seed)
    return [dict(origins=[int(v) for v in rng.choice(panel.origins['train'],2)],variant=TRAIN_VARIANTS[s%4]) for s in range(steps)]

def frozen_digest(m):
    return {n:digest([list(p.shape),float(p.double().sum()),float(p.double().square().sum())]) for n,p in m.named_parameters() if not p.requires_grad}

def norm(params,gradient=False):
    tensors=[p.grad if gradient else p.detach() for p in params]
    tensors=[p for p in tensors if p is not None]
    return float(torch.stack([p.float().square().sum() for p in tensors]).sum().sqrt()) if tensors else 0.

def branch_probe(m,panel,origin,arm):
    x,y,g,r=batch(panel,[origin],'combined_train')
    with torch.no_grad():
        actual=predict(m,x,g,r)[1]
        clean=r.new_tensor([1.,0.,1.]).expand_as(r)
        removed=predict(m,x,g,clean)[1]
        error=float(abs(actual-removed).max())
        set_state(m,token_state(r,m.chronos_config.use_reg_token))
        multipliers=[];shifts=[]
        for layer in m.modules():
            if isinstance(layer,AffineLowRank):
                a,b=layer.modulation(layer.state);multipliers.append(a.flatten());shifts.append(b.flatten())
        detail={}
        if multipliers:
            a=torch.cat(multipliers);b=torch.cat(shifts)
            detail=dict(multiplier_min=float(a.min()),multiplier_max=float(a.max()),multiplier_mean=float(a.mean()),shift_abs_max=float(b.abs().max()))
            assert torch.isfinite(a).all() and torch.isfinite(b).all() and float(a.min())>=0 and float(a.max())<=2
    return dict(train_state_replacement_prediction_max_abs=error,**detail)

def run():
    if (OUT/'contract.json').exists():raise FileExistsError('Existing v2 contract; no overwrite or automatic rerun')
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(CONFIG.read_text());panel=Panel('jena');start=time.monotonic()
    schedules={str(seed):schedule(panel,seed,cfg['max_steps']) for seed in cfg['seeds']}
    write_json(OUT/'sampling_manifest.json',schedules)
    execution=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    contract=dict(candidate='01_v2',execution_commit=execution,config=cfg,rules=RULES,train_variants=TRAIN_VARIANTS,evaluation_variants=E_VARIANTS,manifest_sha256=sha(panel.root/'manifest.json'),source_hashes=source_hashes(),sampling_sha256=sha(OUT/'sampling_manifest.json'),authorization='User: 해줘, after explicit discussion of Candidate01 v2 and maximum12 fits',evaluation_scope='Reused Jena development E; not independent holdout evidence',formula='W0h + 2B[(1+0.5(tanh(Wa r)-tanh(Wa r_clean)))Ah + Wb(r-r_clean)]',clean_reference=[1,0,1],new_parameter_counts=dict(feature=3072,affine=4608),novelty='Conditional affine modulation is established (FiLM); no novelty certification',diagnostics='First eight updates of each scheduled fit double as train-only functionality smoke; no additional fitted trials')
    write_json(OUT/'contract.json',contract)
    # No E: quantify the asynchronous construct on the historical gate split.
    m,_=build('F0',30000);gate_losses={}
    for variant in TRAIN_VARIANTS:
        p,y=predictions(m,panel,'gate',variant)
        path=CACHE/f'gate_{variant}.npz';np.savez_compressed(path,prediction=p,target=y,scale=panel.scale)
        replay(path);gate_losses[variant]=score(p,y,panel.scale)['scaled_2pinball']
    degradation={k:100*(v-gate_losses['clean'])/gate_losses['clean'] for k,v in gate_losses.items() if k!='clean'}
    gatepass=np.mean(list(degradation.values()))>=2 and sum(v>=1 for v in degradation.values())>=2
    write_json(OUT/'round0.json',dict(status='PASS' if gatepass else 'NO_PROBLEM',loss=gate_losses,degradation_percent=degradation,mean_degradation_percent=float(np.mean(list(degradation.values())))))
    del m;gc.collect();torch.cuda.empty_cache()
    if not gatepass:
        write_json(OUT/'status.json',dict(verdict='NO_PROBLEM',fit_count=0,round2_executed=False));return
    attempts=[];trajectories=[];diagnostics=[];integrity=[];resources=[];winners=[]
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            arm_records=[]
            for recipe in cfg['recipes']:
                fit=f'{seed}_{arm}_{recipe["id"]}';fitdir=CACHE/fit;fitdir.mkdir()
                attempts.append(dict(fit=fit,seed=seed,arm=arm,recipe=recipe['id'],status='RUNNING'));write_json(OUT/'attempts.json',attempts)
                print('START',fit,flush=True);fit_start=time.monotonic();guard();torch.cuda.reset_peak_memory_stats()
                m,info=build(arm,seed);frozen=frozen_digest(m)
                conditional=[p for n,p in m.named_parameters() if p.requires_grad and '.condition.' in n]
                lora=[p for n,p in m.named_parameters() if p.requires_grad and '.condition.' not in n]
                groups=[dict(params=lora,lr=recipe['lora_lr'],name='lora')]
                if conditional:groups.append(dict(params=conditional,lr=recipe['conditional_lr'],name='conditional'))
                opt=torch.optim.AdamW(groups,weight_decay=0);params=lora+conditional
                assert len({id(p) for g in opt.param_groups for p in g['params']})==len([p for p in m.parameters() if p.requires_grad])
                probe_origin=int(panel.origins['train'][0]);x,y,g,r=batch(panel,[probe_origin],'combined_train')
                with torch.no_grad():
                    pp=predict(m,x,g,r)[1]
                    with disabled(m):base=forecast(m,x,g)[1]
                identity=float(abs(pp-base).max());assert identity==0
                best=None;beststate=None;smoke=None;lastloss=None
                for step in range(cfg['max_steps']+1):
                    if step in cfg['checkpoints']:
                        val,parts=validation(m,panel);assert np.isfinite(val)
                        row=dict(fit=fit,seed=seed,arm=arm,recipe=recipe['id'],step=step,validation_loss=val,train_loss=lastloss,**{f'V_{k}':v for k,v in parts.items()})
                        trajectories.append(row);csv_write(OUT/'trajectories.csv',trajectories)
                        if best is None or val<best['validation_loss']:
                            best=dict(seed=seed,arm=arm,recipe=recipe['id'],step=step,validation_loss=val,checkpoint=str(fitdir/'best.pt'))
                            beststate=snapshot(m);torch.save(beststate,fitdir/'best.pt')
                        print('V',fit,step,val,flush=True)
                        if step>0:
                            probe=branch_probe(m,panel,probe_origin,arm)
                            diagnostics.append(dict(fit=fit,step=step,kind='branch_probe',conditional_parameter_norm=norm(conditional),**probe));csv_write(OUT/'branch_diagnostics.csv',diagnostics)
                            if step==8 and conditional:
                                assert probe['train_state_replacement_prediction_max_abs']>0,'Condition does not affect predictions after eight scheduled updates'
                                assert norm(conditional)>0,'Conditional weights unchanged after smoke'
                                smoke=probe
                    if step==cfg['max_steps']:break
                    guard(fit_start);sample=schedules[str(seed)][step]
                    x,y,g,r=batch(panel,sample['origins'],sample['variant'])
                    opt.zero_grad(set_to_none=True);z,p,l,s=predict(m,x,g,r);loss=native_loss(z,y,l,s)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite train loss')
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise FloatingPointError('Nonfinite gradient')
                    record=(step+1 in cfg['checkpoints'] or step<2)
                    if record:
                        cg=norm(conditional,True);lg=norm(lora,True);before=[p.detach().clone() for p in conditional]
                        detail={}
                        if arm=='AFFINE_V2':
                            detail=dict(scale_gradient_norm=norm([p.grad[:8] for p in conditional]),shift_gradient_norm=norm([p.grad[8:] for p in conditional]))
                    torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip']);opt.step();lastloss=float(loss.detach())
                    if record:
                        update=norm([p.detach()-b for p,b in zip(conditional,before)])
                        diagnostics.append(dict(fit=fit,step=step+1,kind='optimizer',variant=sample['variant'],conditional_gradient_norm=cg,lora_gradient_norm=lg,conditional_to_lora_gradient_ratio=cg/max(lg,1e-30),conditional_update_norm=update,**detail))
                        csv_write(OUT/'branch_diagnostics.csv',diagnostics)
                        if step==7 and conditional:assert cg>0 and update>0
                assert frozen_digest(m)==frozen,'Frozen weights changed'
                restore(beststate,m)
                with torch.no_grad():expected=predict(m,x,g,r)[1]
                restore(torch.load(fitdir/'best.pt',weights_only=True),m)
                with torch.no_grad():actual=predict(m,x,g,r)[1]
                error=float(abs(expected-actual).max());assert error==0
                selected_probe=branch_probe(m,panel,probe_origin,arm)
                best['checkpoint_sha256']=sha(fitdir/'best.pt');arm_records.append(best)
                integrity.append(dict(fit=fit,identity_max_abs=identity,checkpoint_replay_max_abs=error,frozen_unchanged=True,optimizer_covers_all_trainables=True,conditional_smoke=smoke,selected_probe=selected_probe,**info))
                resources.append(dict(fit=fit,optimizer_steps=cfg['max_steps'],wall_seconds=time.monotonic()-fit_start,**guard()))
                write_json(OUT/'integrity.json',dict(status='PASS',fits=integrity));write_json(OUT/'resource_usage.json',dict(fits=resources,fit_count=len(resources),wall_seconds=time.monotonic()-start))
                attempts[-1]['status']='COMPLETE';write_json(OUT/'attempts.json',attempts)
                del m,opt,params,lora,conditional,beststate,groups;gc.collect();torch.cuda.empty_cache()
            winner=min(arm_records,key=lambda a:(a['validation_loss'],a['step'],a['recipe']));winners.append(winner)
    assert len(attempts)==cfg['fit_cap'] and len(winners)==6
    assert source_hashes()==contract['source_hashes'],'Execution source changed during fitting'
    seal(OUT/'selection.json',winners,contract);csv_write(OUT/'selections.csv',winners)
    panel.open_e(OUT/'selection.json',contract)
    write_json(OUT/'evaluation_open.json',dict(selection_sha256=sha(OUT/'selection.json'),evaluation_sha256=sha(panel.root/'evaluation.npz'),all_twelve_fits_complete_before_open=True))
    rows=[];errors=[]
    for selection in [dict(arm='F0',seed=30000)]+winners:
        arm=selection['arm'];seed=selection['seed'];m,_=build(arm,seed)
        if arm!='F0':
            assert sha(selection['checkpoint'])==selection['checkpoint_sha256']
            restore(torch.load(selection['checkpoint'],weights_only=True),m)
        for variant in E_VARIANTS:
            p,y=predictions(m,panel,'evaluation',variant);path=CACHE/f'E_{seed}_{arm}_{variant}.npz'
            np.savez_compressed(path,prediction=p,target=y,scale=panel.scale,origins=panel.origins['evaluation'])
            errors.append(replay(path));metrics=score(p,y,panel.scale)
            rows.append(dict(seed=seed,arm=arm,variant=variant,**metrics,prediction_sha256=sha(path)))
            csv_write(OUT/'metrics.csv',rows);print('E',seed,arm,variant,metrics['scaled_2pinball'],flush=True)
        del m;gc.collect();torch.cuda.empty_cache();guard()
    status=judge(rows,cfg)
    write_json(OUT/'status.json',status)
    write_json(OUT/'integrity.json',dict(status='PASS',fits=integrity,metric_replay_max_abs=max(errors),selection_before_e=True,source_unchanged=True,saved_predictions=len(rows)))
    write_json(OUT/'resource_usage.json',dict(fits=resources,fit_count=len(resources),optimizer_steps=sum(u['optimizer_steps'] for u in resources),wall_seconds=time.monotonic()-start,stream_count=0))
    print('FINAL',json.dumps(status),flush=True)

def judge(rows,cfg):
    def loss(arm,seed,clean=False):
        rr=[float(r['scaled_2pinball']) for r in rows if r['arm']==arm and (arm=='F0' or int(r['seed'])==seed) and ((r['variant']=='clean')==clean)]
        assert len(rr)==(1 if clean else 4)
        return float(np.mean(rr))
    f0=loss('F0',30000);details=[]
    for seed in cfg['seeds']:
        baseline=min(cfg['arms'][:2],key=lambda a:loss(a,seed));p=loss('AFFINE_V2',seed)
        gains={a:100*(loss(a,seed)-p)/f0 for a in cfg['arms'][:2]}
        clean=100*(loss('AFFINE_V2',seed,True)-min(loss(a,seed,True) for a in cfg['arms'][:2]))/loss('F0',30000,True)
        details.append(dict(seed=seed,strongest_baseline=baseline,baseline_primary=loss(baseline,seed),proposed_primary=p,gain_percent_f0=gains[baseline],gains_percent_f0=gains,clean_degradation_percent_f0=clean,gain_over_f0=100*(f0-p)/f0))
    mean=float(np.mean([r['gain_percent_f0'] for r in details]))
    positive=all(r['gain_percent_f0']>0 for r in details)
    passed=positive and mean>=1 and all(r['clean_degradation_percent_f0']<=.5 for r in details)
    return dict(verdict='PASS' if passed else 'WEAK' if positive else 'FAIL',fit_count=12,stream_count=0,seeds=details,mean_gain_percent_f0=mean,f0_primary=f0,round2_executed=False,new_source_eligible_for_review=passed,scope='Fixed-budget revised development candidate; reused development E, no statistical significance or novelty certification')
