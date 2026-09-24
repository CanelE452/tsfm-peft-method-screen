"""Exactly one model instance is built, updated, and saved. No retry or parameter injection."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import copy
import json
import os
import torch
from .model import TransientModel,ModelSpec,Task
from .backend import load_base
from .fixture import make_tasks
from .checkpoint import save_trained,load_trained,reference,compare_reference
from .util import require,write_json,state_hash,tensor_hash
from .ledger import Ledger,summarize
from .numerical import check_loss_reduction


def call(model,tasks,ledger,phase,row_order=None,grad=False):
    with ledger.operation('forward',phase=phase):
        with torch.set_grad_enabled(grad):
            return model(tasks,row_order=row_order)


def unchanged(a,b,tasks):
    vals={}
    for t in tasks:
        scale=max(float(t.y_context.std()),1.)
        delta=float((a[t.task_id]-b[t.task_id]).abs().max())/scale
        require(delta<=1e-5,f'Task {t.task_id} invariance failed: {delta}')
        vals[str(t.task_id)]=delta
    return vals


def inference_contract_checks(model,tasks,ledger):
    """Checks run on the actually trained instance; they do not modify parameters."""
    before=state_hash(model.compact_state())
    a=call(model,tasks,ledger,'invariance_reference')
    rows=model.last_debug['rows']
    d0=copy.deepcopy(model.last_debug)
    perm=list(reversed(rows))
    b=call(model,tasks,ledger,'row_permutation',row_order=perm)
    report={'row_permutation':unchanged(a['predictions'],b['predictions'],tasks)}
    # Reorder task list and rename IDs using a non-order-preserving bijection.
    renamed=[replace(tasks[1],task_id=40),replace(tasks[0],task_id=2)]
    r=call(model,renamed,ledger,'task_id_rename')
    require(torch.allclose(a['predictions'][tasks[0].task_id],r['predictions'][2],atol=1e-5,rtol=1e-5),'Task-ID lookup is order-dependent')
    report['task_id_rename']='PASS'
    changed=[tasks[0],replace(tasks[1],command=tasks[1].command*1.7+.6)]
    c=call(model,changed,ledger,'cross_task_isolation')
    report['other_task_command']=unchanged({tasks[0].task_id:a['predictions'][tasks[0].task_id]},
                                          {tasks[0].task_id:c['predictions'][tasks[0].task_id]},[tasks[0]])
    poisoned=[replace(t,y_future=t.y_future+200) for t in tasks]
    p=call(model,poisoned,ledger,'future_target_poison')
    report['future_target_poison']=unchanged(a['predictions'],p['predictions'],tasks)
    again=call(model,tasks,ledger,'no_stale_state')
    report['no_carryover']=unchanged(a['predictions'],again['predictions'],tasks)
    if model.modulator is not None:
        require(d0['c'] is not None and float(d0['c'].abs().max())>0,'Trained MOD path is not active; invariance would be vacuous')
        byid={tid:[i for i,key in enumerate(d0['rows']) if key[0]==tid] for tid in (tasks[0].task_id,tasks[1].task_id)}
        for tid,inds in byid.items():
            require(torch.equal(d0['c'][inds[0]],d0['c'][inds[1]]),'Same-task raw rows received different state')
        require(not torch.equal(d0['c'][byid[tasks[0].task_id][0]],d0['c'][byid[tasks[1].task_id][0]]),'Distinct tasks received identical modulation')
        require(torch.equal(d0['c'][~d0['valid']],torch.zeros_like(d0['c'][~d0['valid']])),'Neutral token modulation is nonzero')
        report['active_modulation_max']=float(d0['c'].abs().max())
        # row_count == token_count counterexample: 3 tasks * 2 raw rows == 6 tokens.
        third=replace(tasks[0],task_id=117,command=tasks[0].command*.8,y_context=tasks[0].y_context+2)
        equal_tasks=[*tasks,third]
        e=call(model,equal_tasks,ledger,'rows_equal_tokens')
        eq_debug=copy.deepcopy(model.last_debug)
        require(eq_debug['c'].shape[0]==eq_debug['c'].shape[1],'Equal-axis counterexample was not exercised')
        er=call(model,equal_tasks,ledger,'rows_equal_tokens_permute',row_order=list(reversed(eq_debug['rows'])))
        report['equal_axis_invariance']=unchanged(e['predictions'],er['predictions'],equal_tasks)
        # Real all-masked command patch, separate from padding. MOD only; INPUT gap support is rejected.
        mask=torch.ones_like(tasks[0].command,dtype=torch.bool)
        length=tasks[0].y_context.numel();left=(-length)%model.p_in
        start=model.p_in-left;end=2*model.p_in-left
        mask[start:end]=False
        masked=[replace(tasks[0],command_mask=mask),tasks[1]]
        call(model,masked,ledger,'masked_command_patch')
        dm=model.last_debug
        ti=[i for i,(tid,_) in enumerate(dm['rows']) if tid==tasks[0].task_id]
        require(torch.equal(dm['c'][ti,1],torch.zeros_like(dm['c'][ti,1])),'Fully unobserved command patch is not neutral')
        report['masked_patch_after_real_updates']='PASS'
        # Unknown values beneath an explicit mask must not influence states or predictions.
        modified=tasks[0].command.clone();modified[~mask]=10000
        d=call(model,[replace(tasks[0],command=modified,command_mask=mask),tasks[1]],ledger,'hidden_mask_value')
        dmasked=call(model,masked,ledger,'hidden_mask_reference')
        report['hidden_mask_values']=unchanged(d['predictions'],dmasked['predictions'],tasks)
        require(all(not l.modulated for n,l in model._projections.items() if n in model.scope['group']),
                'Group attention unexpectedly conditionally modulated')
        report['group_scope']='PLAIN_LORA_ONLY'
    require(state_hash(model.compact_state())==before,'Inference checks modified trained parameters')
    return report


def run_arm(backend: dict,spec: ModelSpec,public: Path,private: Path,steps=6,lr=.001):
    public.mkdir(parents=True,exist_ok=True);private.mkdir(parents=True,exist_ok=True)
    ledger=Ledger(public/'events.jsonl')
    with ledger.operation('model_load',phase='parent'):
        base,tt,gt=load_base(backend)
    write_json(public/'loss_contract.json',check_loss_reduction(base))
    model=TransientModel(base,spec,tt,gt)
    tasks=make_tasks(model.p_in,model.p_out)
    initial_hash=model.frozen_state_hash()
    inv=model.inventory()
    parameters=[p for _,p in model.named_parameters() if p.requires_grad]
    ids=[id(p) for p in parameters]
    require(len(ids)==len(set(ids)),'Duplicate optimizer parameter ownership')
    opt=torch.optim.AdamW(parameters,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False)
    actual={id(p) for g in opt.param_groups for p in g['params']}
    require(actual==set(ids),'Optimizer excludes/adds parameters')
    write_json(public/'ownership.json',{'inventory':inv,'optimizer_exact_ownership':True,
               'response_instances':int(model.response is not None),'shared_response_parameters':True,
               'scope':model.scope,'modulated_projection_count':sum(x.modulated for x in model._projections.values()),
               'native_frozen_sha256':initial_hash,'backend':backend})
    initial=model.compact_state()
    ini=call(model,tasks,ledger,'initial')
    # A separate parameter-free switch exposes the original native paths.
    for layer in model._projections.values():layer.disabled=True
    try:
        ref=call(model,tasks,ledger,'same_view_native_reference')
    finally:
        for layer in model._projections.values():layer.disabled=False
    parity=unchanged(ini['predictions'],ref['predictions'],tasks)
    write_json(public/'initial_parity.json',{'same_view_native_parity':parity,'input_view':'raw+state_init' if spec.arm in ('FI','LI') else 'raw',
                                         'manual_mutations_after_initialization':0})
    trace=[]
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        out=call(model,tasks,ledger,f'update_{step+1}',grad=True)
        require(bool(torch.isfinite(out['loss'])),'Non-finite loss')
        with ledger.operation('backward',step=step+1):out['loss'].backward()
        grad={}
        for n,p in model.named_parameters():
            if p.requires_grad:
                require(p.grad is not None and bool(torch.isfinite(p.grad).all()),f'Missing/nonfinite gradient {n}')
                grad[n]=float(p.grad.detach().double().norm())
        # Clip only by the same predeclared rule in every arm. No learning-rate search.
        torch.nn.utils.clip_grad_norm_(parameters,max_norm=1.,error_if_nonfinite=True)
        with ledger.operation('optimizer_step',step=step+1):opt.step()
        delta={n:float((p.detach().double()-before[n].double()).norm()) for n,p in model.named_parameters() if p.requires_grad}
        for n,p in model.named_parameters():
            require(bool(torch.isfinite(p).all()),f'Non-finite parameter after optimizer step {n}')
        item={'step':step+1,'native_loss':float(out['native_loss'].detach()),'corrected_loss':float(out['loss'].detach()),
              'gradient_norms_before_clip':grad,'parameter_deltas_from_optimizer_only':delta,
              'state_log_tau':None if model.response is None else model.response.log_tau.detach().cpu().tolist(),
              'compact_sha256':state_hash(model.compact_state()),'n_target_rows':model.last_debug['active_target_rows'],
              'n_rows':model.last_debug['total_rows'],'manual_injections':0}
        trace.append(item);write_json(public/'update_trace.json',trace)
    final=call(model,tasks,ledger,'trained_checkpoint_reference')
    require(model.frozen_state_hash()==initial_hash,'Frozen backbone/head changed during updates')
    final_state=model.compact_state()
    changed={n:not torch.equal(initial[n],v) for n,v in final_state.items()}
    categories={cat:[n for n,info in inv.items() if info['category']==cat] for cat in ('lora','state','modulator')}
    checks={}
    for cat,names in categories.items():
        if names:
            checks[cat]={'any_optimizer_delta':any(changed[n] for n in names),
                         'any_finite_nonzero_gradient':any(any(t['gradient_norms_before_clip'].get(n,0)>0 for n in names) for t in trace)}
    if spec.arm in ('FI','FM'):
        require(torch.equal(initial['response.log_tau'],final_state['response.log_tau']),'Fixed response state changed')
    # Save the exact just-trained model even if its tiny-fixture delta is inconclusive.
    metadata=save_trained(model,backend,tasks,final,private/'checkpoint',steps)
    write_json(public/'trained_checkpoint.json',{'steps':steps,'compact_sha256':metadata['compact_sha256'],
               'frozen_sha256':metadata['frozen_sha256'],'keys':list(final_state),'parameter_changes':changed,
               'categories':checks,'same_object_updated_and_saved':True,'manual_parameter_injections':0})
    confirmed=all(v['any_optimizer_delta'] and v['any_finite_nonzero_gradient'] for v in checks.values())
    invariance={}
    if confirmed:
        invariance=inference_contract_checks(model,tasks,ledger)
        write_json(public/'invariance.json',invariance)
    else:
        write_json(public/'invariance.json',{'status':'NOT_RUN_ACTIVE_PATH_UNCONFIRMED','reason':'No extra steps, random injection or larger LR is allowed.'})
    write_json(public/'arm_result.json',{'status':'PASS_LOCAL_LIFECYCLE' if confirmed else 'INCONCLUSIVE_UPDATE',
               'arm':spec.arm,'backend_kind':backend['kind'],'categories':checks,'native_frozen_preserved':True,
               'same_model_saved':True,'counts':summarize(public/'events.jsonl'),
               'scientific_performance_evaluated':False})
    return confirmed


def run_restore(private: Path,public: Path):
    public.mkdir(parents=True,exist_ok=True)
    ledger=Ledger(public/'restore_events.jsonl')
    with ledger.operation('model_load',phase='new_process_restore'):
        model,tasks,metadata=load_trained(private/'checkpoint')
    actual=call(model,tasks,ledger,'new_process_restored_prediction')
    expected=torch.load(private/'checkpoint'/'reference.pt',map_location='cpu',weights_only=True)
    checks=compare_reference(reference(model,tasks,actual),expected)
    write_json(public/'roundtrip.json',{'status':'PASS_INFERENCE_ROUNDTRIP','pid':os.getpid(),
               'checks':checks,'optimizer_steps_before_saving':metadata['steps_from_optimizer'],
               'restored_compact_sha256':state_hash(model.compact_state()),'training_resume_claimed':False,
               'counts':summarize(public/'restore_events.jsonl')})
