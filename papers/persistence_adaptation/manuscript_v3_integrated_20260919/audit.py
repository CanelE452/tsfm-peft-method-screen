"""Audit integrated claims, source preservation, numerical contrasts, and rendering."""
from pathlib import Path
import hashlib,json,re,subprocess,zipfile,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def run():
 m=read(P/'EVIDENCE_MANIFEST.json')
 for f,h in m['evidence_hashes'].items():assert sha(ROOT/f)==h,('SOURCE_CHANGED',f)
 for dst,src in m['figure_sources'].items():assert sha(P/dst)==sha(ROOT/src)
 old=P.parent/'manuscript_v2_20260918'
 for p in (old/'tables').glob('*.csv'):assert sha(p)==sha(P/'tables'/p.name)
 md=(P/'MANUSCRIPT_KO.md').read_text();text=(old/'MANUSCRIPT_KO.md').read_text()
 def tables(t):
  blocks=[];current=[]
  for line in t.splitlines()+['']:
   if line.startswith('|'):current.append([v.strip() for v in line.strip('|').split('|')])
   elif current:blocks.append(current);current=[]
  return blocks
 mt,ot=tables(md),tables(text);assert len(mt)==9 and mt[:5]==ot
 assert [len(x)-2 for x in mt]==[6,9,4,3,5,14,2,6,4]
 assert '본 원고에서는 수행하지 않았다' not in md and '@@' not in md
 for phrase in ['배치 구성도 변경','재사용 개발자료','완전한 인과','새 학습·모델 추론·새 통계 추정','preprint','81553','부록 C.']:assert phrase in md,phrase
 assert len(re.findall(r'!\[',md))==8
 links=[]
 for target in re.findall(r'\]\(([^)]+)\)',md):
  if not target.startswith(('https://','http://','#')):assert (P/target).exists(),target;links.append(target)
 f=pd.read_csv(P/'tables/T6_factorial.csv');cells=pd.read_csv(ROOT/'results/c3_training_factorial_20260918/CELL_EFFECTS.csv');checks=0
 for r in f.itertuples():
  q=cells[(cells.panel==r.panel)&(cells.kind==r.kind)&(cells.condition==r.condition)&(cells.stage==r.stage)];assert len(q)==8
  sign=np.ones(8)
  for factor in r.factor.split(':'):sign*=np.where(q[{'B0':'b','INIT':'i','ORDER':'o'}[factor]].to_numpy()==81552,1,-1)
  effect=2*np.mean((q.C3-q.MAG).to_numpy()*sign)
  np.testing.assert_allclose([effect,100*effect/q.C3.mean()],[r.effect_nmae,r.effect_pct_common_C3],atol=1e-10)
  line=mt[5][2+checks];assert line[0]==r.stage and line[1]==r.factor.replace(':','×')
  np.testing.assert_allclose([float(line[2]),float(line[3])],[effect,100*effect/q.C3.mean()],atol=5.01e-5);checks+=1
 assert f[f.stage=='fixed1024'].set_index('factor').effect_nmae.abs().idxmax()=='B0:INIT'
 assert f[f.stage=='selected'].set_index('factor').effect_nmae.abs().idxmax()=='B0'
 grad=pd.read_csv(ROOT/'results/c3_internal_mechanism_20260918/INITIAL_GRADIENT_DECOMPOSITION.csv');g=pd.read_csv(P/'tables/T7_gradient.csv')
 for r in g.itertuples():
  q=grad[grad.source==r.source];np.testing.assert_allclose((q.output_signal_component_norm/q.Jacobian_component_norm).mean(),r.output_J_ratio_mean,atol=1e-12)
 b=pd.read_csv(P/'tables/T8_pairing.csv');raw=pd.read_csv(ROOT/'results/c3_internal_mechanism_20260918/RAW_SCORES.csv')
 for r in b.itertuples():
  q=raw[(raw.panel==r.panel)&(raw.kind==r.kind)&(raw.condition==r.condition)&(raw.arm==r.arm)];assert len(q)==16
  a=q[q.trained_b==q.received_b].nmae.mean();z=q[q.trained_b!=q.received_b].nmae.mean()
  np.testing.assert_allclose([a,z,100*(z/a-1)],[r.matched_nmae,r.swapped_nmae,r.penalty_pct],atol=1e-10)
  line=mt[7][2+r.Index];assert line[1]==r.arm
  np.testing.assert_allclose([float(v) for v in line[2:5]],[a,z,100*(z/a-1)],atol=5.01e-5)
 chain=pd.read_csv(ROOT/'results/c3_internal_mechanism_20260918/CHAIN_RULE_CHECKS.csv');assert len(chain)==140 and f'{chain.relative_l2.max():.2e}'=='8.70e-08'
 w=read(ROOT/'results/c3_weakness_controls_20260918/VERIFICATION.json');fac=read(ROOT/'results/c3_training_factorial_20260918/INDEPENDENT_AUDIT.json');internal=read(ROOT/'results/c3_internal_mechanism_20260918/INDEPENDENT_AUDIT.json');cross=read(ROOT/'results/c3_magnitude_diagnostic_20260918/INDEPENDENT_AUDIT.json')
 assert (w['main_fits'],w['main_updates'],w['new_prediction_views'],w['smoke_updates'])==(30,30720,54,12)
 assert (fac['new_fits'],fac['reused_fits'],fac['new_main_updates'],fac['new_prediction_views'],fac['reused_prediction_views'],fac['smoke_updates'])==(24,8,24576,129,63,12)
 assert (cross['new_fits'],cross['new_prediction_hashes'],cross['reused_prediction_hashes'])==(0,36,36)
 assert (internal['new_fits'],internal['autograd_calls'],internal['checkpoint_probes'],internal['new_E_views'],internal['reused_E_views'])==(0,1196,160,96,96)
 refs=re.findall(r'^\[(\d+)\] ',md,re.M);assert refs==[str(i) for i in range(1,13)]
 with zipfile.ZipFile(P/'MANUSCRIPT_KO.docx') as z:
  assert len([n for n in z.namelist() if n.startswith('word/media/')])==8
  tree=ET.fromstring(z.read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'};assert len(tree.findall('.//w:tbl',ns))==9
  doc=''.join(tree.itertext())
 html=(P/'MANUSCRIPT_KO.html').read_text();assert html.count('data:image/png;base64,')==8
 pdf=subprocess.check_output(['pdftotext',str(P/'MANUSCRIPT_KO.pdf'),'-'],text=True)
 for term in ['초록','2.399%','9.7476%','10.1990%','5.7','5.8','5.9','참고문헌','부록 C.','24,576']:
  assert term in pdf and term in doc,term
 assert '\ufffd' not in pdf
 bbox=subprocess.check_output(['pdftotext','-bbox',str(P/'MANUSCRIPT_KO.pdf'),'-'],text=True)
 tree=ET.fromstring(bbox);ns={'x':'http://www.w3.org/1999/xhtml'};bounded=tree.findall('.//x:page',ns)
 for page in bounded:
  width,height=float(page.attrib['width']),float(page.attrib['height'])
  for word in page.findall('.//x:word',ns):
   v=word.attrib;assert 0<=float(v['xMin'])<=float(v['xMax'])<=width and 0<=float(v['yMin'])<=float(v['yMax'])<=height

 info=subprocess.check_output(['pdfinfo',str(P/'MANUSCRIPT_KO.pdf')],text=True);pages=int(re.search(r'Pages:\s+(\d+)',info)[1]);assert pages>=12
 artifacts={str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file() and p.name!='AUDIT.json' and '__pycache__' not in p.parts}
 result=dict(status='INTEGRATION_VERIFIED',new_fits=0,new_model_inference=0,new_bootstrap=0,pages=pages,figures=8,tables=9,references=12,preserved_source_files=len(m['evidence_hashes']),old_five_tables_unchanged=True,factorial_contrasts_recomputed=checks,gradient_source_summaries_recomputed=2,B0_pairing_effects_reaggregated=6,historical_budgets_verified=True,local_links_checked=len(links),PDF_text_boundaries_checked_pages=len(bounded),manual_visual_samples=[1,10,12,15],stale_nonexecution_claim_removed=True,development_exposure_and_negative_results_preserved=True,full_new_independent_test=False,submission_ready=False,artifact_hashes=artifacts)
 (P/'AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('INTEGRATION_VERIFIED',pages,'pages, 8 figures, 9 tables, 12 references')
if __name__=='__main__':run()
