"""Read-only audit of completed artifacts, additional to the sealed scorer."""
from runtime import *

def audit():
    s=state_json();assert s['status']=='COMPLETE'
    c=contract();fits=read(OUT/'fits.json');curves=read(OUT/'training_curves.json');evaluation=read(OUT/'evaluation.json');selected=read(OUT/'selection_seal.json')['selections']
    targets={};target_checks=0
    for r in evaluation:
        with np.load(ROOT/r['prediction_path']) as z:target=z['target'];origins=z['origins'];std=z['std']
        key=(r['dataset'],r['panel'])
        if key not in targets:targets[key]=(target.copy(),origins.copy(),std.copy())
        a,b,cc=targets[key]
        assert np.array_equal(a,target,equal_nan=True) and np.array_equal(b,origins) and np.array_equal(cc,std)
        assert origins.tolist()==c['data'][r['dataset']]['panels'][r['panel']]
        target_checks+=1
    resume=[];budget=[];cost=[]
    for f in fits:
        if f['mode']=='NEW':
            st=torch.load(ROOT/f['resume_path'],map_location='cpu',weights_only=False)
            cp=next(r for r in curves if r['fit']==f['fit'] and r['epoch']==20)
            state=torch.load(ROOT/cp['checkpoint_path'],map_location='cpu',weights_only=True)
            assert tensor_hash(st['parameters'])==tensor_hash(state)
            assert st['epoch']==21 and st['offset']==0 and st['fit_record']['updates']==f['updates']
            assert st['scheduler']['last_epoch']==20
            assert st['optimizer']['param_groups'][0]['lr']==f['lr']*.5**4
            assert len(st['steps'])==f['updates'] and all(k in st['rng'] for k in ['python','numpy','torch','cuda'])
            resume.append(dict(fit=f['fit'],updates=f['updates'],complete_epoch_batch_position=True,parameters_equal_epoch20=True,optimizer_scheduler_rng_present=True,resume_sha256=sha(ROOT/f['resume_path'])))
        for panel in ['V_FIXED','V_MIXED']:
            rr=[r for r in curves if r['fit']==f['fit'] and r['panel']==panel];best=min(rr,key=lambda r:(r['metrics']['mse'],r['epoch']))
            budget.append(dict(fit=f['fit'],panel=panel,best_epoch=best['epoch'],status='BUDGET_LIMITED' if best['epoch']==20 else 'FIXED20_COMPLETE'))
    for r in selected:
        matching=[f for f in fits if f['dataset']==r['dataset'] and f['arm']==r['arm'] and f['seed']==r['seed']]
        assert len(matching)==2
        cost.append(dict(dataset=r['dataset'],arm=r['arm'],seed=r['seed'],policy=r['policy'],selected_lr=r['lr'],selected_epoch=r['epoch'],selected_prefix_updates=r['updates'],two_lr_total_updates=sum(f['updates'] for f in matching),two_lr_new_updates=sum(f['updates'] for f in matching if f['mode']=='NEW'),two_lr_reused_updates=sum(f['updates'] for f in matching if f['mode']=='REUSE')))
    save(OUT/'resume_audit.json',resume);csvwrite(OUT/'budget_status.csv',budget);csvwrite(OUT/'selection_cost.csv',cost)
    resource=read(OUT/'resources.json');macro=read(OUT/'evaluation.json');trades=[]
    for d in c['data']:
        lh=next(r for r in resource if r['dataset']==d and r['option']=='LH-current')
        for option in ['SIDE-fast','PRIOR-current','LH-native-checkpoint']:
            q=next(r for r in resource if r['dataset']==d and r['option']==option)
            if q['status']!='MEASURED':continue
            a=option.split('-')[0]
            get=lambda arm:next(r['metrics']['mse'] for r in macro if r['dataset']==d and r['panel']=='E_MIXED' and r['arm']==arm and r['policy']=='P_MAIN' and r['seed']==41000)
            trades.append(dict(dataset=d,option=option,baseline='LH-current',seed=41000,mse=get(a),lh_mse=get('LH'),accuracy_gain_pct=100*(1-get(a)/get('LH')),training_memory_saving_pct=100*(1-q['peak_allocated']/lh['peak_allocated']),step_time_saving_pct=100*(1-q['median_seconds']/lh['median_seconds'])))
    csvwrite(OUT/'resource_tradeoffs.csv',trades)
    save(OUT/'completion_audit.json',dict(target_alignment_checks=target_checks,complete_new_resume_states=len(resume),budget_rows=len(budget),selected_cost_rows=len(cost),updates_total=sum(f['updates'] for f in fits),status='PASS',scope='independent final target alignment and full resume state inspection; no forward or optimizer update'))
    print('COMPLETION_AUDIT_PASS',len(resume),target_checks,flush=True)

if __name__=='__main__':audit()
