"""Read existing evidence only: no training, inference, or new resampling."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def run():
 checked={}
 for name in ['manuscript_v2_20260918','training_factorial_v1_20260918','internal_mechanism_v1_20260918']:
  folder=ROOT/'papers/persistence_adaptation'/name;a=read(folder/'AUDIT.json')
  for relative,h in a['artifact_hashes'].items():
   path=folder/relative
   assert sha(path)==h,(path,'artifact changed');checked[str(path.relative_to(ROOT))]=h
  hs=a.get('source_hashes',{})
  if (folder/'EVIDENCE_MANIFEST.json').exists():hs=read(folder/'EVIDENCE_MANIFEST.json')['evidence_hashes']
  for relative,h in hs.items():
   path=ROOT/relative;assert sha(path)==h,(path,'source changed');checked[str(path.relative_to(ROOT))]=h
 def frame(relative):
  p=ROOT/relative;checked[relative]=sha(p);return pd.read_csv(p)
 controls=frame('results/c3_weakness_controls_20260918/PRIMARY_RESULTS.csv')
 c=controls[(controls.panel=='electricity_transfer')&(controls.baseline=='MAG_ONLY')&(controls.ci_type=='time')].iloc[0]
 np.testing.assert_allclose(c.gain_pct,100*(1-c.new_nmae/c.baseline_nmae),atol=1e-10)
 gates=frame('papers/persistence_adaptation/manuscript_v2_20260918/tables/T3_decomposition.csv')
 g=gates[gates.panel=='electricity_transfer'].iloc[0]
 np.testing.assert_allclose([g.total,g.gate,g.weights],[g.A-g.D,((g.A-g.B)+(g.C-g.D))/2,((g.A-g.C)+(g.B-g.D))/2],atol=1e-12)
 fac=frame('results/c3_training_factorial_20260918/FACTORIAL_EFFECTS.csv')
 q=fac[(fac.panel=='electricity_transfer')&(fac.kind=='standard')&(fac.condition=='SHIFT8')]
 top={}
 for stage in ['fixed1024','selected']:
  f=q[q.stage==stage];r=f.loc[f.effect_nmae.abs().idxmax()]
  top[stage]=dict(factor=r.factor,effect_nmae=float(r.effect_nmae),effect_pct=float(r.effect_pct_common_C3))
 pairing=frame('results/c3_internal_mechanism_20260918/PAIRING_EFFECTS.csv')
 r=pairing[(pairing.panel=='electricity_transfer')&(pairing.kind=='standard')&(pairing.condition=='SHIFT8')]
 np.testing.assert_allclose(r.penalty_pct,100*(r.swapped_nmae/r.matched_nmae-1),atol=1e-10)
 assert len(r)==2
 manuscript=(ROOT/'papers/persistence_adaptation/manuscript_v2_20260918/MANUSCRIPT_KO.md').read_text()
 stale='후자의 분리에는 각 요인을 독립적으로 통제한 학습 비교가 필요하며 본 원고에서는 수행하지 않았다.'
 assert stale in manuscript
 data=dict(status='REVIEW_EVIDENCE_VERIFIED',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),new_fits=0,new_inference=0,new_bootstrap=0,unique_published_files_verified=len(checked),source_hashes=checked,core_numbers=dict(C3_vs_MAG_selected_gain_pct=float(c.gain_pct),gate_cross=dict(A=float(g.A),B=float(g.B),C=float(g.C),D=float(g.D),gate=float(g.gate),weights=float(g.weights)),factorial_largest_absolute_point_estimate=top,B0_swap_SHIFT8=r[['arm','matched_nmae','swapped_nmae','penalty_pct']].to_dict('records')),manuscript_integration_required=True,stale_sentence=stale,scientific_novelty_verified=False,submission_ready=False)
 (OUT/'AUDIT.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 print('REVIEW_EVIDENCE_VERIFIED',len(checked),'published files; no model calls')
if __name__=='__main__':run()
