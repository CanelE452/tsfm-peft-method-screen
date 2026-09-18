import itertools
import pandas as pd
from .common import *

def audit():
 check_seal();counts=read(OUT/'COUNTS.json');assert counts['autograd_calls']==1196 and counts['optimizer_updates']==0 and counts['new_fits']==0
 rules=pd.read_csv(OUT/'CHAIN_RULE_CHECKS.csv');assert len(rules)==140 and rules.relative_l2.max()<=1e-4 and rules.max_abs.max()<=2e-6
 features=pd.read_csv(OUT/'FEATURE_GEOMETRY.csv');assert features.B0_input_embedding_identical.all() and (features.contexts==1024).all()
 z=np.load(CACHE/'initial_gradients.npz');table=pd.read_csv(OUT/'INITIAL_GRADIENT_DECOMPOSITION.csv')
 for r in table.itertuples():
  get=lambda b,v:z[f'{r.source}__{r.arm}__i{r.init}__j{b}__v{v}'];a,b,c,d=[get(j,v) for j,v in [(81551,81551),(81551,81552),(81552,81551),(81552,81552)]];J=(c-a+d-b)/2;V=(b-a+d-c)/2
  np.testing.assert_allclose([np.linalg.norm(d-a),np.linalg.norm(J),np.linalg.norm(V)],[r.total_difference_norm,r.Jacobian_component_norm,r.output_signal_component_norm],rtol=1e-10,atol=1e-12);np.testing.assert_allclose(J+V,d-a,rtol=1e-10,atol=1e-12)
 v=np.load(CACHE/'trajectory_gradients.npz');t=pd.read_csv(OUT/'TRAJECTORY.csv');assert len(t)==160
 for r in t.itertuples():
  g=v[r.fit+f'__{r.step}'];np.testing.assert_allclose([np.linalg.norm(g),np.linalg.norm(g[:4104]),np.linalg.norm(g[4104:])],[r.gradient_norm,r.down_gradient_norm,r.up_gradient_norm],rtol=1e-10,atol=1e-12)
  if r.step==0:assert r.down_gradient_norm==0 and r.up_parameter_norm==0 and r.residual_relative_norm==0
 raw=pd.read_csv(OUT/'RAW_SCORES.csv');effects=pd.read_csv(OUT/'PAIRING_EFFECTS.csv');assert len(effects)==120;checked=0
 for r in effects.itertuples():
  q=raw[(raw.panel==r.panel)&(raw.kind==r.kind)&(raw.condition==r.condition)&(raw.arm==r.arm)];match=q[q.trained_b==q.received_b].nmae.mean();swap=q[q.trained_b!=q.received_b].nmae.mean();assert len(q)==16
  np.testing.assert_allclose([match,swap,swap-match,100*(swap-match)/match],[r.matched_nmae,r.swapped_nmae,r.pairing_penalty,r.penalty_pct],rtol=1e-8,atol=1e-12);checked+=1
 # Independently bootstrap cluster sums, rather than the scorer's origin-weight matrix.
 from experiments.persistence_evidence_extension_20260918 import common as ext
 origins=pd.read_csv(OUT/'PAIRING_BY_ORIGIN.csv.gz'); interval_checks=0
 for r in effects.itertuples():
  q=origins[(origins.panel==r.panel)&(origins.kind==r.kind)&(origins.condition==r.condition)&(origins.arm==r.arm)]
  period=24 if r.panel.startswith('electricity') else 96
  all_orig=np.load(ext.data_path(r.panel)/'E_DISCOVERY_inputs.npz')['origins'];blocks=np.unique(all_orig//(7*period))
  sums=np.zeros(len(blocks));sizes=np.zeros(len(blocks))
  for row in q.itertuples():
   k=np.searchsorted(blocks,row.origin//(7*period));sums[k]+=row.pairing_penalty;sizes[k]+=1
  draws=np.random.default_rng(90418+['electricity','electricity_transfer','ettm1'].index(r.panel)).integers(0,len(blocks),size=(2000,len(blocks)))
  boots=sums[draws].sum(1)/sizes[draws].sum(1)
  np.testing.assert_allclose(np.quantile(boots,[.025,.975,.0125,.9875]),[r.ci_low,r.ci_high,r.bonferroni2_low,r.bonferroni2_high],rtol=1e-8,atol=1e-12);interval_checks+=1
 assert len(raw)==1920 and np.isfinite(raw[['nmae','pinball']].to_numpy()).all()
 manifest=read(OUT/'PREDICTIONS.json');assert len(manifest)==192 and sum(not r['reused_prediction'] for r in manifest.values())==96
 for r in manifest.values():assert sha(ROOT/r['path'])==r['sha256']
 assert counts['E_small_forward_calls']==288
 gpu=[json.loads(l) for l in open(OUT/'gpu_mechanism.jsonl')];external=sum(any(not a['own'] and a['name']!='/usr/share/rustdesk/rustdesk' for a in row['apps']) for row in gpu)
 assert external==0
 artifacts={str(p.relative_to(ROOT)):sha(p) for p in CACHE.glob('*.npz')}
 save(OUT/'INDEPENDENT_AUDIT.json',dict(status='VERIFIED',new_fits=0,optimizer_updates=0,autograd_calls=1196,chain_rule_cases=140,initial_contexts=2048,checkpoint_probes=160,gradient_decompositions=8,pairing_effects_replayed=checked,bootstrap_intervals_replayed=interval_checks,raw_score_rows=1920,new_E_views=96,reused_E_views=96,E_small_forward_calls=288,original_inputs_weights_code_unchanged=True,GPU_min_free_mib=min(r['free_mib'] for r in gpu),external_training_samples=external,gradient_cache_hashes=artifacts))
 print('INDEPENDENT_AUDIT_VERIFIED',flush=True)
if __name__=='__main__':audit()
