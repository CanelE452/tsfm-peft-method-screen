"""Render an integrated manuscript from fixed published evidence; no model calls."""
from pathlib import Path
import json,hashlib,shutil,subprocess
import pandas as pd
P=Path(__file__).resolve().parent;ROOT=P.parents[2];OLD=P.parent/'manuscript_v2_20260918'
F=ROOT/'results/c3_training_factorial_20260918';D=ROOT/'results/c3_internal_mechanism_20260918'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build():
 baseline=json.loads((ROOT/'research/paper_viability_review_20260919/AUDIT.json').read_text())['source_hashes'];seen=dict(baseline)
 for name,h in baseline.items():assert sha(ROOT/name)==h,('PREVIOUS_EVIDENCE_CHANGED',name)
 def source(p):seen[str(p.relative_to(ROOT))]=sha(p);return p
 def csv(p):return pd.read_csv(source(p))
 for folder in [F,D]:
  for name in ['PROTOCOL.md','REPORT.md','FINAL_DECISION.md','INDEPENDENT_AUDIT.json','SCORE_VERIFICATION.json']:source(folder/name)
 source(ROOT/'research/paper_viability_review_20260919/REVIEW_KO.md')
 for folder in [OLD,P.parent/'training_factorial_v1_20260918',P.parent/'internal_mechanism_v1_20260918']:
  for p in folder.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:source(p)
 (P/'tables').mkdir(exist_ok=True);(P/'figures').mkdir(exist_ok=True)
 for p in (OLD/'tables').glob('*.csv'):shutil.copyfile(source(p),P/'tables'/p.name)
 def md(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
 fact=csv(F/'FACTORIAL_EFFECTS.csv');f=fact[(fact.panel=='electricity_transfer')&(fact.kind=='standard')&(fact.condition=='SHIFT8')];f.to_csv(P/'tables/T6_factorial.csv',index=False)
 blocks={};rows=[]
 for r in f.itertuples():rows.append([r.stage,r.factor.replace(':','×'),f'{r.effect_nmae:+.6f}',f'{r.effect_pct_common_C3:+.4f}',f'[{r.bonferroni7_low:+.6f}, {r.bonferroni7_high:+.6f}]'])
 blocks['FACTORIAL_TABLE']=md(['checkpoint','요인','nMAE contrast','공통 분모(%)','Bonferroni7 구간'],rows)
 g=csv(D/'MECHANISM_SUMMARY.csv');g.to_csv(P/'tables/T7_gradient.csv',index=False)
 blocks['GRADIENT_TABLE']=md(['source','B0 변경 cosine 범위','출력/J norm 비 평균','gate 변경 cosine 범위'],[[r.source,f'{r.B0_gradient_cos_min:.3f}–{r.B0_gradient_cos_max:.3f}',f'{r.output_J_ratio_mean:.3f}',f'{r.gate_cos_min:.3f}–{r.gate_cos_max:.3f}'] for r in g.itertuples()])
 pairing=csv(D/'PAIRING_EFFECTS.csv');b=pairing[(pairing.kind=='standard')&(pairing.condition=='SHIFT8')];b.to_csv(P/'tables/T8_pairing.csv',index=False)
 labels={'electricity':'전력4','electricity_transfer':'전력 전이16','ettm1':'ETTm1'}
 blocks['PAIRING_TABLE']=md(['패널','방법','원래 nMAE','교환 nMAE','교환 손해(%)','차이의 Bonferroni2 구간'],[[labels[r.panel],r.arm,f'{r.matched_nmae:.6f}',f'{r.swapped_nmae:.6f}',f'{r.penalty_pct:+.4f}',f'[{r.bonferroni2_low:+.6f}, {r.bonferroni2_high:+.6f}]'] for r in b.itertuples()])
 for folder,names in [(F,['CELL_EFFECTS.csv','RAW_SCORES.csv','RESOURCE_SUMMARY.json']),(D,['RAW_SCORES.csv','INITIAL_GRADIENT_DECOMPOSITION.csv','CHAIN_RULE_CHECKS.csv','TRAJECTORY.csv','ORDER_FIRST_BATCH.csv','ORDER_FINAL_MOMENTS.csv','COUNTS.json','RESOURCE_SUMMARY.json'])]:
  for name in names:source(folder/name)
 figures={}
 for stem in ['F1_controls','F2_decomposition','F3_tradeoffs','FA1_shapes']:
  for suffix in ['png','pdf','svg']:
   a=source(OLD/'figures'/f'{stem}.{suffix}');z=P/'figures'/a.name;shutil.copyfile(a,z);figures[str(z.relative_to(P))]=str(a.relative_to(ROOT))
 for stem,folder,oldname in [('F4_factorial',F,'01_factorial'),('F5_initial_gradient',D,'initial_gradient'),('FA2_trajectory',D,'trajectory'),('FA3_B0_pairing',D,'B0_pairing')]:
  for suffix in ['png','pdf','svg']:
   a=source(folder/'figures'/f'{oldname}.{suffix}');z=P/'figures'/f'{stem}.{suffix}';shutil.copyfile(a,z);figures[str(z.relative_to(P))]=str(a.relative_to(ROOT))
 text=(P/'MANUSCRIPT_TEMPLATE_KO.md').read_text()
 for key,value in blocks.items():assert '@@'+key+'@@' in text;text=text.replace('@@'+key+'@@',value)
 assert '@@' not in text;(P/'MANUSCRIPT_KO.md').write_text(text)
 source(OLD/'MANUSCRIPT_KO.md')
 (P/'EVIDENCE_MANIFEST.json').write_text(json.dumps(dict(base_commit='7098e40cd8442a998c9ac976266a2375abd6e89d',new_fits=0,new_model_inference=0,new_bootstrap=0,evidence_hashes=seen,figure_sources=figures,old_versions_preserved=True,tables_source={'T1_to_T5':'unchanged v2 tables and text','T6':str((F/'FACTORIAL_EFFECTS.csv').relative_to(ROOT)),'T7':str((D/'MECHANISM_SUMMARY.csv').relative_to(ROOT)),'T8':str((D/'PAIRING_EFFECTS.csv').relative_to(ROOT)),'T9':'historical execution records checked in audit.py'}),ensure_ascii=False,indent=2)+'\n')
 pandoc='/home/minjae/anaconda3/bin/pandoc'
 subprocess.run([pandoc,'MANUSCRIPT_KO.md','--standalone','--embed-resources','--css=paper.css','--metadata=lang:ko','--metadata=pagetitle:시계열 추가 PEFT의 성능과 관측 gate 효과','-o','MANUSCRIPT_KO.html'],cwd=P,check=True)
 subprocess.run([pandoc,'MANUSCRIPT_KO.md','--resource-path=.','--metadata=lang:ko','-o','MANUSCRIPT_KO.docx'],cwd=P,check=True)
 h=P/'MANUSCRIPT_KO.html';h.write_text('\n'.join(l.rstrip() for l in h.read_text().splitlines())+'\n')
 subprocess.run(['/usr/bin/google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-pdf-header-footer','--user-data-dir=/tmp/c3-integrated-v3-chrome','--print-to-pdf='+str(P/'MANUSCRIPT_KO.pdf'),'file://'+str(P/'MANUSCRIPT_KO.html')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 print('INTEGRATED_MANUSCRIPT_RENDERED',len(seen),'fixed evidence files; 0 new fits/inference',flush=True)
if __name__=='__main__':build()
