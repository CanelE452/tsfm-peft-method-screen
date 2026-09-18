"""Independent output accounting and arithmetic replay, with no GPU calls."""
import pandas as pd
from .common import *

def audit():
    check_seal();v=read(OUT/'VERIFICATION.json');assert v['status']=='VERIFIED'
    new=read(OUT/'PREDICTIONS.json');checks=read(OUT/'MODEL_CHECKS.json');assert len(new)==len(checks)==36
    assert {(r['panel'],r['kind'],r['seed'],r['arm']) for r in new.values()}=={(p,k,s,a) for p in PANELS for k in ['standard','shape'] for s in SEEDS for a in ['C3_W_MAG_G','MAG_W_C3_G']}
    for r in list(new.values())+records():assert sha(ROOT/r['path'])==r['sha256']
    for r in models():assert sha(ROOT/r['checkpoint'])==r['sha256']
    f=pd.read_csv(OUT/'DECOMPOSITION.csv');valid=f[f.status=='SCORED'];np.testing.assert_allclose(valid.total,valid.gate+valid.weights,rtol=1e-9,atol=1e-12)
    # Replay ALL from separately serialized per-seed/origin A/B/C/D errors.
    o=pd.read_csv(OUT/'ORIGIN_ERRORS.csv.gz');n=0
    for (p,k,c),g in o.groupby(['panel','kind','condition']):
        means=g[['A','B','C','D']].mean().to_numpy();a,b,cc,d=means;want=[a-d,((a-b)+(cc-d))/2,((a-cc)+(b-d))/2,a-b-cc+d]
        row=f[(f.panel==p)&(f.kind==k)&(f.condition==c)&(f.family=='ALL')].iloc[0];np.testing.assert_allclose(row[['A','B','C','D']].to_numpy(float),means,atol=1e-12,rtol=1e-10);np.testing.assert_allclose(row[['total','gate','weights','interaction']].to_numpy(float),want,atol=1e-12,rtol=1e-9);n+=1
        for family in ['gate','magnitude','run','position']:
            part=valid[(valid.panel==p)&(valid.kind==k)&(valid.condition==c)&(valid.family==family)];assert part.contexts.sum()==row.contexts;np.testing.assert_allclose(part[['weighted_total','weighted_gate','weighted_weights']].sum().to_numpy(),want[:3],rtol=1e-9,atol=1e-12)
    features=pd.read_csv(OUT/'INPUT_FEATURES.csv.gz');assert len(features)==89088;assert not features.duplicated(['panel','kind','condition','draw','origin','channel']).any();np.testing.assert_allclose(features.gap_mass,features.gap_mass_short_runs,rtol=0,atol=0)
    channels=pd.read_csv(OUT/'CHANNEL_CONTRIBUTIONS.csv')
    for (p,k,c),g in channels.groupby(['panel','kind','condition']):
        row=f[(f.panel==p)&(f.kind==k)&(f.condition==c)&(f.family=='ALL')].iloc[0];np.testing.assert_allclose(g.contribution.sum(),row.total,rtol=1e-9,atol=1e-12)
    events=[json.loads(l) for l in (OUT/'gpu_diagnostic.jsonl').read_text().splitlines()];bad=sum(any(not a['own'] and a['name']!='/usr/share/rustdesk/rustdesk' for a in r['apps']) for r in events);assert bad==0
    selection=[]
    for r in models():selection.append({k:r[k] for k in ['source','arm','seed','lr','step','objective','checkpoint','sha256']})
    csvwrite(OUT/'CHECKPOINT_CONTEXT.csv',selection)
    save(OUT/'INDEPENDENT_AUDIT.json',dict(status='VERIFIED',new_fits=0,optimizer_updates=0,new_prediction_hashes=36,reused_prediction_hashes=36,checkpoint_hashes=12,all_condition_arithmetic_rows=n,partitions_and_channels_reconstruct_all=True,input_contexts=89088,all_gate_mass_in_first_7_of_run=True,parent_files_unchanged=True,minimum_gpu_free_mib=min(r['free_mib'] for r in events),unapproved_external_compute_samples=bad,automatic_successor=False))
    print('INDEPENDENT_AUDIT_VERIFIED',n,flush=True)
if __name__=='__main__':audit()
