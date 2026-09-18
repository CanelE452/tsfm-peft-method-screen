"""Independent accounting, raw-score aggregation, selections and immutable parents."""
import pandas as pd
from .common import *
from .train import choice

def audit():
    seal=check_seal();journal=[json.loads(l) for l in open(OUT/'UPDATE_LEDGER.jsonl')];smoke=[json.loads(l) for p in OUT.glob('smoke_*.jsonl') for l in open(p)]
    assert len(journal)==30720 and len({(r['fit'],r['step']) for r in journal})==30720 and len(smoke)==12
    fits=[];parameterrows=[];checkpoint_count=0
    for p in sorted((OUT/'fits').glob('*/receipt.json')):
        r=read(p);assert r['status']=='COMPLETE' and r['updates']==1024 and r['frozen_unchanged'];assert [j['step'] for j in journal if j['fit']==r['fit']]==list(range(1,1025));assert min(r['checkpoints'],key=lambda c:(c['objective'],c['step']))==r['selected'];assert read(p.parent/'update_intent.json')==dict(fit=r['fit'],status='JOURNALED',completed_step=1024)
        for c in r['checkpoints']:assert sha(ROOT/c['checkpoint'])==c['sha256'];checkpoint_count+=1
        delta=read(p.parent/'parameter_change.json');state=torch.load(ROOT/r['selected']['checkpoint'],map_location='cpu',weights_only=True)
        assert sum(v.numel() for v in state.values())=={'POS_ONLY':8744,'MAG_ONLY':8712,'OUTPUT_CONTEXT':4673}[r['arm']]
        for n,v in delta['max_abs_change'].items():assert np.isfinite(v)
        for n in (['position_logits','adapter.up.weight','adapter.down.weight'] if r['arm']=='POS_ONLY' else ['output_linear.weight','gamma'] if r['arm']=='OUTPUT_CONTEXT' else ['adapter.up.weight','adapter.down.weight']):assert delta['max_abs_change'][n]>0,(r['fit'],n)
        fits.append({k:r[k] for k in ['fit','source','arm','seed','lr','updates','optimizer_seconds','validation_seconds','checkpoint_io_seconds','invocation_wall_seconds','peak_allocated','peak_reserved']})
        entry=dict(fit=r['fit'],source=r['source'],arm=r['arm'],seed=r['seed'],lr=r['lr'],selected_step=r['selected']['step'],selected_V=r['selected']['objective'],final_gate=delta['position_gate'],final_gamma=delta['gamma'],parameter_max_abs_change=delta['max_abs_change'])
        if r['arm']=='POS_ONLY':entry['selected_gate']=state['position_logits'].sigmoid().tolist()
        if r['arm']=='OUTPUT_CONTEXT':entry['selected_gamma']=float(state['gamma']);entry['selected_tanh_gamma']=float(state['gamma'].tanh())
        grad=[j['gate_gradient_norm'] for j in journal if j['fit']==r['fit'] and j['gate_gradient_norm'] is not None];entry['component_gradient_nonzero_steps']=sum(v>0 for v in grad);entry['component_gradient_max']=max(grad) if grad else None
        parameterrows.append(entry)
    assert len(fits)==30 and checkpoint_count==150;csvwrite(OUT/'FIT_LEDGER.csv',fits);save(OUT/'PARAMETER_DYNAMICS.json',parameterrows)
    for s,d in read(OUT/'LR_SELECTION.json').items():
        for a,r in d.items():assert choice(s,a,81550)==r
    models=read(OUT/'MODEL_SELECTION.json');assert len(models)==18 and all(r['seed'] in [81551,81552,81553] for r in models)
    for r in models:
        receipt=read(OUT/'fits'/r['fit']/'receipt.json');assert r['checkpoint']==receipt['selected']['checkpoint'];assert sha(ROOT/r['checkpoint'])==r['sha256']
    ev=read(OUT/'EVALUATION_SEAL.json');marker=read(OUT/'ALL_PREDICTIONS_SAVED.json');assert ev['at']<marker['at'] and marker['manifest_sha256']==sha(OUT/'PREDICTIONS.json');assert ev['model_selection_sha256']==sha(OUT/'MODEL_SELECTION.json') and ev['lr_selection_sha256']==sha(OUT/'LR_SELECTION.json')
    preds=read(OUT/'PREDICTIONS.json');checks=read(OUT/'MODEL_REPLAY.json');assert len(preds)==54 and set(preds)==set(checks)
    for key,r in preds.items():assert sha(ROOT/r['path'])==r['sha256'] and all(checks[key].values())
    frame=pd.read_csv(OUT/'ALL_ORIGIN_SCORES.csv.gz',dtype={'channel':str});fault=frame[frame.condition.str.startswith(('POINT','BURST'))].assign(condition='FAULT');f=pd.concat([frame,fault]);keys=['panel','kind','condition','arm','seed'];raw=pd.read_csv(OUT/'RAW_SCORES.csv').set_index(keys).sort_index()
    agg=f.groupby(keys)[['nmae','mae','normalized_mse','pinball','crossing']].mean().sort_index();assert agg.index.equals(raw.index);np.testing.assert_allclose(agg.to_numpy(),raw[agg.columns].to_numpy(),rtol=1e-10,atol=1e-12)
    nr=f.groupby(keys+['channel']).normalized_mse.mean().pow(.5).groupby(level=list(range(5))).mean().sort_index();np.testing.assert_allclose(nr.to_numpy(),raw.nrmse.to_numpy(),rtol=1e-10,atol=1e-12)
    pair=pd.read_csv(OUT/'PAIRED_ORIGIN_DIFFERENCES.csv');effects=pd.read_csv(OUT/'MATCHED_CONTRASTS.csv');replayed=0
    for (p,k,c,b),g in pair.groupby(['panel','kind','condition','baseline']):
        expected=100*(1-g.new_nmae.mean()/g.baseline_nmae.mean());r=effects[(effects.panel==p)&(effects.kind==k)&(effects.condition==c)&(effects.new=='C3')&(effects.baseline==b)];np.testing.assert_allclose(r.gain_pct,expected,rtol=1e-10,atol=1e-10);replayed+=len(r)
    events=[json.loads(l) for l in (OUT/'gpu_run.jsonl').read_text().splitlines()];assert not any(any(not a['own'] and a['name']!='/usr/share/rustdesk/rustdesk' for a in r['apps']) for r in events)
    sv=read(OUT/'SCORE_VERIFICATION.json');assert sv['status']=='VERIFIED';check_seal()
    save(OUT/'VERIFICATION.json',dict(status='VERIFIED',main_fits=30,main_updates=30720,smoke_updates=12,total_updates=30732,selected_models=18,new_prediction_views=54,new_checkpoint_hashes=150,full_raw_score_rows_reaggregated=len(raw),paired_effect_rows_replayed=replayed,scalar_rows=sv['scalar_rows'],scalar_metrics_per_row=2,all_prediction_hashes=True,initial_off_restore_and_frozen_checks=True,parents_unchanged=True,sealed_hashes_checked=len(seal['hashes']),minimum_free_gpu_mib=min(r['free_mib'] for r in events),unapproved_external_compute_samples=0,E_previously_exposed=True,automatic_successor=False))
    costs=pd.DataFrame(fits);save(OUT/'COST.json',dict(new_fits=30,new_main_updates=30720,new_smoke_updates=12,total_updates=30732,training_compute_seconds=float(costs.optimizer_seconds.sum()),validation_seconds=float(costs.validation_seconds.sum()),checkpoint_io_seconds=float(costs.checkpoint_io_seconds.sum()),wall=read(OUT/'run_wall.json'),old_B0_costs_in=str((PRIOR/'COST_ACCOUNTING.json').relative_to(ROOT)),no_shared_B0_retraining=True))
    save(OUT/'status.json',dict(execution='VERIFIED',completed_fits=30,main_updates=30720,smoke_updates=12,predictions=54,automatic_successor=False));print('FINAL_VERIFICATION_COMPLETE',len(raw),replayed,flush=True)
if __name__=='__main__':audit()
