"""Check source preservation, numerical tables and rendered manuscript artifacts."""
from pathlib import Path
import hashlib,json,re,subprocess,zipfile,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
m=json.loads((P/'EVIDENCE_MANIFEST.json').read_text())
for p,h in m['evidence_hashes'].items():assert sha(ROOT/p)==h,p
for dst,src in m['figure_sources'].items():assert sha(P/dst)==sha(ROOT/src)
# Check the new CPU audit against its sealed inputs, including training/cache hashes.
a=ROOT/'research/c3_influence_audit_20260918';d=json.loads((a/'AUDIT.json').read_text())
for p,h in d['input_hashes'].items():assert sha(ROOT/p)==h,p
v=pd.read_csv(a/'VARIANCE_COMPONENTS.csv');s=pd.read_csv(a/'SUMMARY.csv');assert len(s)==60 and len(v)==420
np.testing.assert_allclose(v.groupby(['panel','kind','condition']).share_pct.sum().values,100,atol=1e-6)
t=pd.read_csv(P/'tables/T3_decomposition.csv');r=t[t.panel=='electricity_transfer'].iloc[0]
np.testing.assert_allclose(r.total,r.A-r.D,atol=1e-12);np.testing.assert_allclose(r.total,r.gate+r.weights,atol=1e-12)
z=s[(s.panel=='electricity_transfer')&(s.condition=='SHIFT8')].iloc[0];np.testing.assert_allclose(z.gain_pct,100*(r.A-r.D)/r.A,atol=1e-10)
f=pd.read_csv(P/'tables/T4_seed_decomposition.csv');f=f[f.panel=='electricity_transfer'];share=100*f[f.seed==81552].total.iloc[0]/f.total.sum();assert round(share,1)==85.4
assert round(100*r.weights/r.A,3)==.644 and round(100*r.gate/r.A,3)==-.393
md=(P/'MANUSCRIPT_KO.md').read_text();assert '@@' not in md and '85.4%' in md and '41.17%' in md
assert len(re.findall(r'!\[',md))==4
for target in re.findall(r'\]\(([^)]+)\)',md):
 if not target.startswith(('https://','http://','#')):assert (P/target).exists(),target
with zipfile.ZipFile(P/'MANUSCRIPT_KO.docx') as z:
 images=[n for n in z.namelist() if n.startswith('word/media/')];assert len(images)==4
 doc=ET.fromstring(z.read('word/document.xml')); ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'};assert len(doc.findall('.//w:tbl',ns))==5
 text=''.join(doc.itertext());assert '초록' in text and '41.17%' in text and 'B0' in text
pdftext=subprocess.check_output(['pdftotext',str(P/'MANUSCRIPT_KO.pdf'),'-'],text=True)
for term in ['초록','85.4%','41.17%','0.2374','MAG_ONLY','참고문헌']:assert term in pdftext,term
info=subprocess.check_output(['pdfinfo',str(P/'MANUSCRIPT_KO.pdf')],text=True);pages=int(re.search(r'Pages:\s+(\d+)',info)[1]);assert pages>=10
html=(P/'MANUSCRIPT_KO.html').read_text();assert html.count('data:image/png;base64,')==4
artifacts={str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file() and p.name!='AUDIT.json' and '__pycache__' not in p.parts}
(P/'AUDIT.json').write_text(json.dumps(dict(status='VERIFIED',new_fits=0,new_model_inference=0,new_bootstrap=0,pages=pages,rendered_figures=4,rendered_tables=5,source_files_unchanged=len(m['evidence_hashes']),influence_audit_inputs_unchanged=len(d['input_hashes']),numeric_checks=True,local_links=True,docx_and_pdf_korean_content=True,html_self_contained_images=True,artifact_hashes=artifacts),indent=2,ensure_ascii=False)+'\n')
print('VERIFIED',pages,'pages, 5 tables, 4 figures,',len(artifacts),'artifacts')
