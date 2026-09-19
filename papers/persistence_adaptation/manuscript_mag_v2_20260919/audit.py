"""Check manuscript provenance, numeric tables and rendered artifacts."""
from pathlib import Path
import hashlib,json,re,subprocess,zipfile
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 manifest=json.loads((P/'EVIDENCE_MANIFEST.json').read_text())
 for name,value in manifest['evidence_hashes'].items():assert sha(ROOT/name)==value,name
 for local,original in manifest['figure_sources'].items():assert sha(P/local)==sha(ROOT/original),local
 raw=pd.read_csv(ROOT/'results/petsa_cell_comparison_20260919/RAW_SCORES.csv')
 copied=pd.read_csv(P/'tables/all_raw_scores.csv')
 assert list(raw.columns)==list(copied.columns) and len(raw)==len(copied)
 for col in ['nmae','mae','pinball']:np.testing.assert_allclose(raw[col],copied[col],rtol=1e-12,atol=1e-12)
 f=raw[(raw.stage=='selected')&(raw.kind=='standard')&(raw.condition=='SHIFT8')].groupby(['panel','arm']).nmae.mean().unstack()
 exported=pd.read_csv(P/'tables/shift8_means.csv',index_col=0)
 np.testing.assert_allclose(f.to_numpy(),exported.loc[f.index,f.columns].to_numpy(),rtol=1e-12,atol=1e-12)
 primary=pd.read_csv(P/'tables/primary.csv');assert len(primary)==4
 for row in primary.itertuples():
  gain=100*(1-f.loc[row.panel,'MAG_ONLY']/f.loc[row.panel,row.baseline])
  np.testing.assert_allclose(gain,row.gain_pct,rtol=1e-12,atol=1e-12)
  for seed,value in json.loads(row.seed_gains).items():
   z=raw[(raw.panel==row.panel)&(raw.stage=='selected')&(raw.kind=='standard')&(raw.condition=='SHIFT8')&(raw.seed==int(seed))].set_index('arm').nmae
   np.testing.assert_allclose(100*(1-z['MAG_ONLY']/z[row.baseline]),value,rtol=1e-12,atol=1e-12)
 md=(P/'MANUSCRIPT_KO.md').read_text();assert '@@' not in md
 links=[]
 for link in re.findall(r'\]\(([^)]+)\)',md):
  if '://' not in link:
   assert (P/link.split('#')[0]).exists(),link;links.append(link)
 pdftext=subprocess.check_output(['pdftotext','-layout',str(P/'MANUSCRIPT_KO.pdf'),'-'],text=True)
 compact=re.sub(r'\s+','',pdftext)
 for token in ['3.794%','1.941%','2.780','1.719','STEP12_D63','참고문헌','8fits','충분한방법론신규성을확정하지않는다']:
  assert token in compact,token
 assert '\ufffd' not in pdftext
 info=subprocess.check_output(['pdfinfo',str(P/'MANUSCRIPT_KO.pdf')],text=True)
 pages=int(re.search(r'Pages:\s+(\d+)',info).group(1));assert pages>=7
 with zipfile.ZipFile(P/'MANUSCRIPT_KO.docx') as z:
  xml=z.read('word/document.xml').decode();assert '초록' in xml and '참고문헌' in xml
  assert '<m:oMath' in xml and len([n for n in z.namelist() if n.startswith('word/media/')])==4
 pending=ROOT/'results/petsa_cell_comparison_20260919'
 assert json.loads((pending/'AUDIT.json').read_text())['status']=='VERIFIED'
 assert len((pending/'UPDATE_LEDGER.jsonl').read_text().splitlines())==8192
 tr=pd.read_csv(P/'tables/petsa_tradeoffs.csv');assert len(tr)==20
 full=pd.read_csv(pending/'EFFECTS.csv')
 for row in tr.itertuples():
  rr=full[(full.panel==row.panel)&(full.proposed==row.proposed)&(full.baseline==row.baseline)&(full.stage==row.stage)&(full.kind==row.kind)&(full.condition==row.condition)]
  assert len(rr)==1
  np.testing.assert_allclose([row.gain_pct,row.ci_low,row.ci_high],rr[['gain_pct','ci_low','ci_high']].to_numpy()[0],rtol=1e-12,atol=1e-12)
 for local in manifest['generated_figures']:assert (P/local).stat().st_size>10000
 pp=pd.read_csv(P/'tables/petsa_primary.csv');assert len(pp)==2
 for row in pp.itertuples():
  np.testing.assert_allclose(100*(1-f.loc[row.panel,'MAG_ONLY']/f.loc[row.panel,row.baseline]),row.gain_pct,rtol=1e-12,atol=1e-12)
  for seed,value in json.loads(row.seed_gains).items():
   z=raw[(raw.panel==row.panel)&(raw.stage=='selected')&(raw.kind=='standard')&(raw.condition=='SHIFT8')&(raw.seed==int(seed))].set_index('arm').nmae
   np.testing.assert_allclose(100*(1-z['MAG_ONLY']/z[row.baseline]),value,rtol=1e-12,atol=1e-12)
 files=[p for p in P.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='AUDIT.json']
 result=dict(status='VERIFIED_METHOD_MANUSCRIPT_DRAFT_NOT_SUBMISSION_READY',pdf_pages=pages,source_files=len(manifest['evidence_hashes']),raw_rows_verified=len(raw),primary_gains_and_seed_gains_recomputed=True,relative_links_verified=len(links),figures_verified=4,docx_has_editable_math=True,new_fits=0,new_model_inferences=0,new_bootstrap=0,petsa_included_fits=8,petsa_main_updates=8192,petsa_smoke_updates=4,petsa_primary_and_seed_gains_verified=True,methodology_goal_complete=False,artifact_hashes={str(p.relative_to(P)):sha(p) for p in files})
 (P/'AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='artifact_hashes'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
