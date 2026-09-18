"""Reconstruct published summaries from origin sums independently of scoring loops."""
import pandas as pd
from .common import *

def audit():
    seal=check_seal();pred=read(OUT/'PREDICTIONS.json');assert len(pred)==60;assert sha(OUT/'PREDICTIONS.json')==read(OUT/'ALL_PREDICTIONS_SAVED.json')['manifest_sha256']
    checks=read(OUT/'MODEL_CHECKS.json');assert set(checks)==set(pred)
    for k,r in pred.items():
        assert sha(ROOT/r['path'])==r['sha256'];assert checks[k]['optimizer_updates']==0
        if r['arm'] not in ['PERSISTENCE','SEASONAL']:assert checks[k]['all_weights_frozen'] and checks[k]['state_unchanged'] and checks[k]['restore_forward_verified']
    c=pd.read_csv(OUT/'CONDITIONAL_EFFECTS.csv');s=pd.read_csv(OUT/'CONDITIONAL_SEEDS.csv');o=pd.read_csv(OUT/'CONDITIONAL_ORIGIN_SUMS.csv.gz');keys=['panel','kind','condition','stratum'];replayed=0
    totals=o.groupby(keys+['seed'])[['contexts','sum_A','sum_B','sum_C','sum_D']].sum()
    for row in s.to_dict('records'):
        z=totals.loc[tuple(row[k] for k in keys)+ (row['seed'],)];a,b,cc,d=[z[k]/z.contexts for k in ['sum_A','sum_B','sum_C','sum_D']]
        np.testing.assert_allclose([a,d,100*(1-a/d),d-a,((b-a)+(d-cc))/2,((cc-a)+(d-b))/2],[row[k] for k in ['C3_nmae','RECENCY_nmae','gain_pct','total','gate','weights']],rtol=1e-8,atol=1e-10);assert z.contexts==row['contexts'];replayed+=1
    for key,g in c.groupby(keys[:3]):
        full=g[g.stratum=='ALL'].iloc[0];parts=g[(g.stratum!='ALL')&(g.contexts>0)];assert parts.contexts.sum()==full.contexts
        for k in ['total','gate','weights']:np.testing.assert_allclose(np.dot(parts[k],parts.contexts)/full.contexts,full[k],rtol=1e-8,atol=1e-10)
        for z in g[g.contexts>0].to_dict('records'):
            ss=s
            for k in keys:ss=ss[ss[k]==z[k]]
            assert len(ss)==3;a=ss.C3_nmae.mean();b=ss.RECENCY_nmae.mean();np.testing.assert_allclose([a,b,100*(1-a/b)],[z['C3_nmae'],z['RECENCY_nmae'],z['gain_pct']],rtol=1e-8,atol=1e-10)
    new=pd.read_csv(OUT/'EFFECTS.csv');previous=pd.read_csv(ext.OUT/'EFFECTS.csv');previous_raw=pd.read_csv(ext.OUT/'RAW_SCORES.csv');full=c[c.stratum=='ALL'];cross=0
    for z in full.to_dict('records'):
        f=new if z['panel']==NEW else previous[previous.ci_type=='time'];f=f[(f.panel==z['panel'])&(f.kind==z['kind'])&(f.condition==z['condition'])&(f.new=='C3')&(f.baseline=='M_RECENCY')]
        if len(f)==1:expected=f.iloc[0].gain_pct
        else:
            assert len(f)==0 and z['condition'].startswith(('POINT','BURST')) and z['panel']!=NEW
            rr=previous_raw[(previous_raw.panel==z['panel'])&(previous_raw.kind==z['kind'])&(previous_raw.condition==z['condition'])].groupby('arm').nmae.mean();expected=100*(1-rr['C3']/rr['M_RECENCY'])
        np.testing.assert_allclose(z['gain_pct'],expected,rtol=1e-8,atol=1e-10);cross+=1
    events=[json.loads(l) for l in (OUT/'gpu_evaluation.jsonl').read_text().splitlines()];bad=[r for r in events if any(not a['own'] and a['name']!='/usr/share/rustdesk/rustdesk' for a in r['apps'])];assert not bad;assert min(r['free_mib'] for r in events)>=1024
    m=np.load(data_path(NEW)/'E_DISCOVERY_inputs.npz');assert len(set(m['origins']//24))==128;assert min(m['origins'])>=8760 and max(m['origins'])+64<=13104;assert np.ptp(np.bincount(m['origins']%24,minlength=24))<=1
    oldsigma=np.load(ext.data_path('neso_2025')/'E_DISCOVERY_inputs.npz')['sigma'];np.testing.assert_array_equal(m['sigma'],oldsigma)
    x,sig=inputs(NEW,'shape');block=2*64;np.testing.assert_array_equal(x[7*block:8*block],x[8*block:9*block]);delta=np.load(panel_path(NEW,'shape')/'offset.npy');assert (delta[7*block:8*block]==0).all()
    v=dict(status='VERIFIED',new_fits=0,optimizer_updates=0,new_prediction_views=60,GPU_prediction_views=56,CPU_prediction_views=4,all_prediction_hashes=True,conditional_seed_rows_independently_replayed=replayed,conditional_aggregate_to_original_effect_checks=cross,all_parent_hashes_preserved=len(seal['hashes']),minimum_free_gpu_mib=min(r['free_mib'] for r in events),unapproved_external_compute_samples=len(bad),approved_exception='RustDesk only',data_days=128,old_sigma_preserved=True,paired_pulse_shift_inputs_equal=True,all_predictions_saved_before_scoring=True,score_verification=read(OUT/'SCORE_VERIFICATION.json'),diagnostic_verification=read(OUT/'DIAGNOSTIC_VERIFICATION.json'),automatic_successor=False)
    save(OUT/'VERIFICATION.json',v);save(OUT/'status.json',dict(execution='VERIFIED',views=60,fits=0,updates=0,automatic_successor=False))
    hashes={str(p.relative_to(ROOT)):sha(p) for base in [OUT,EXP] for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='FINAL_ARTIFACT_AUDIT.json'}
    save(OUT/'FINAL_ARTIFACT_AUDIT.json',dict(status='VERIFIED',hashes=hashes,local_prediction_manifest_sha256=sha(OUT/'PREDICTIONS.json'),github_alone_full_numerical_replay=False));print(json.dumps(v,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':audit()
