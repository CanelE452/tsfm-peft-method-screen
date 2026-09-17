"""Post-report artifact audit. No training, inference, selection, or score mutation."""
import csv,gzip,re
from datetime import datetime,timezone
import pandas as pd
from .common import *

def main():
    seal=check_seal()
    v=read(OUT/'VERIFICATION.json');assert v['status']=='VERIFIED'
    assert (v['optimizer_updates'],v['new_prediction_views'],v['new_origin_rows'],v['model_checks'])==(0,111,438336,105)
    predictions=read(OUT/'PREDICTIONS.json');assert len(predictions)==111
    for k,r in predictions.items():assert sha(ROOT/r['path'])==r['sha256'],k
    marker=read(OUT/'ALL_PREDICTIONS_SAVED.json')
    assert marker['manifest_sha256']==sha(OUT/'PREDICTIONS.json')
    assert marker['at']<(OUT/'NEW_ORIGIN_SCORES.csv.gz').stat().st_mtime
    models=read(OUT/'MODEL_CHECKS.json');assert len(models)==105
    for k,r in models.items():
        assert r['optimizer_updates']==0,k
        for flag in ['all_weights_frozen','original_forward_exact','restored_fixed_gate_forward','state_unchanged']:assert r[flag],(k,flag)
    with gzip.open(OUT/'NEW_ORIGIN_SCORES.csv.gz','rt') as f:
        reader=csv.reader(f);next(reader);n=sum(1 for _ in reader)
    assert n==438336
    fx=pd.read_csv(OUT/'FACTORIAL_DECOMPOSITION.csv');assert len(fx)==42
    A,B,C,D=[fx[x].to_numpy() for x in ['C3weights_C3gate','C3weights_RECENCYgate','RECENCYweights_C3gate','RECENCYweights_RECENCYgate']]
    for col,val in [('gate_nmae_benefit',((B-A)+(D-C))/2),('weights_nmae_benefit',((C-A)+(D-B))/2),('total_nmae_benefit',D-A),('interaction',A-B-C+D)]:
        np.testing.assert_allclose(fx[col],val,atol=1e-12,rtol=1e-10)
    np.testing.assert_allclose(fx.gate_nmae_benefit+fx.weights_nmae_benefit,fx.total_nmae_benefit,atol=1e-12,rtol=1e-10)
    effect=pd.read_csv(OUT/'EFFECTS.csv');assert len(effect)==785
    assert (effect.groupby('panel').common_draw_sha256.nunique()==1).all()
    raw=pd.read_csv(OUT/'RAW_SCORES.csv')
    for r in effect.to_dict('records'):
        f=raw[(raw.panel==r['panel'])&(raw.kind==r['kind'])]
        f=f[f.condition.str.startswith(('POINT','BURST'))] if r['condition']=='FAULT' else f[f.condition==r['condition']]
        a=f[f.arm==r['new']].nmae.mean();b=f[f.arm==r['baseline']].nmae.mean()
        np.testing.assert_allclose([a,b,100*(1-a/b)],[r['new_nmae'],r['baseline_nmae'],r['gain_pct']],atol=1e-10,rtol=1e-10)
    matched=read(OUT/'MATCHED_SELECTION_AUDIT.json');assert len(matched['pairs'])==6
    assert all(x['same_LR_and_step'] and x['C3_lr']==x['RECENCY_lr'] and x['C3_step']==x['RECENCY_step'] for x in matched['pairs'])
    cost=read(OUT/'COST.json');assert cost['new_fits']==cost['optimizer_updates']==cost['unapproved_gpu_samples']==0
    status=read(OUT/'status.json');assert status['execution']=='COMPLETE' and status['reports_complete'] and not status['automatic_successor']
    cpu=read(OUT/'CPU_CHECKS.json');assert cpu['passed'] and cpu['tests']==5
    links=0
    for name in ['REPORT.md','FINAL_DECISION.md','PAPER_REVISION.md']:
        for link in re.findall(r'\]\(([^)]+)\)',(OUT/name).read_text()):
            if link.startswith(('https://','http://')) or link=='FINAL_ARTIFACT_AUDIT.json':continue
            assert (OUT/link).is_file(),(name,link);links+=1
    paths=sorted(p for base in [OUT,EXP] for p in base.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='FINAL_ARTIFACT_AUDIT.json')
    assert all(p.stat().st_size<100*2**20 for p in paths)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    save(OUT/'FINAL_ARTIFACT_AUDIT.json',dict(status='VERIFIED',at_utc=datetime.now(timezone.utc).isoformat(),new_fits=0,optimizer_updates=0,prediction_hashes_verified=111,model_checks_verified=105,origin_rows_crc_checked=n,effects_replayed=785,factorial_rows_replayed=42,matched_LR_step_pairs=6,common_bootstrap_draws_per_panel=True,sealed_files_unchanged=len(seal['hashes']),local_report_links_checked=links,all_publish_files_under_100MiB=True,artifact_sha256=hashes,scope='Post-report integrity and arithmetic check, not an independent full inference rerun. Predictions, weights and raw inputs remain local; sealed cache required for full replay.'))
    print('FINAL_ARTIFACT_VERIFIED',len(hashes),flush=True)
if __name__=='__main__':main()
