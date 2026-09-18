"""Build a Korean paper addendum, preserving the earlier manuscript unchanged."""
from pathlib import Path
import hashlib,json,subprocess,shutil,re,zipfile
from .common import ROOT,OUT,sha,save

def publish():
 p=ROOT/'papers/persistence_adaptation/training_factorial_v1_20260918';p.mkdir(parents=True,exist_ok=True);(p/'figures').mkdir(exist_ok=True)
 files=['REPORT.md','FINAL_DECISION.md','PAPER_CLAIM_UPDATE_KO.md','INDEPENDENT_AUDIT.json','SCORE_VERIFICATION.json','FACTORIAL_EFFECTS.csv','CELL_EFFECTS.csv','RAW_SCORES.csv','RESOURCE_SUMMARY.json','PROTOCOL.md','SEAL.json','ENVIRONMENT.json','IMPLEMENTATION_NOTES.md']
 hashes={str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in files}
 for image in (OUT/'figures').glob('*'):
  shutil.copyfile(image,p/'figures'/image.name);hashes[str(image.relative_to(ROOT))]=sha(image)
 text=(OUT/'PAPER_CLAIM_UPDATE_KO.md').read_text()+'\n\n---\n\n'+(OUT/'REPORT.md').read_text()+'\n\n---\n\n'+(OUT/'FINAL_DECISION.md').read_text()
 (p/'PAPER_ADDENDUM_KO.md').write_text(text)
 css=(ROOT/'papers/persistence_adaptation/manuscript_v2_20260918/paper.css').read_text()+'\ntable {break-inside:auto;} tr {break-inside:avoid;}\n';(p/'paper.css').write_text(css)
 pandoc='/home/minjae/anaconda3/bin/pandoc'
 subprocess.run([pandoc,'PAPER_ADDENDUM_KO.md','--standalone','--embed-resources','--css=paper.css','--metadata=lang:ko','--metadata=pagetitle:C3와 MAG의 학습 요인 분리','-o','PAPER_ADDENDUM_KO.html'],cwd=p,check=True)
 subprocess.run([pandoc,'PAPER_ADDENDUM_KO.md','--resource-path=.','--metadata=lang:ko','-o','PAPER_ADDENDUM_KO.docx'],cwd=p,check=True)
 subprocess.run(['/usr/bin/google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-pdf-header-footer','--user-data-dir=/tmp/c3-factorial-paper-chrome','--print-to-pdf='+str(p/'PAPER_ADDENDUM_KO.pdf'),'file://'+str(p/'PAPER_ADDENDUM_KO.html')],check=True)
 h=p/'PAPER_ADDENDUM_KO.html';h.write_text('\n'.join(s.rstrip() for s in h.read_text().splitlines())+'\n')
 pdftext=subprocess.check_output(['pdftotext',str(p/'PAPER_ADDENDUM_KO.pdf'),'-'],text=True)
 for term in ['24,576','초기','MAG','1,024']:assert term in pdftext,term
 with zipfile.ZipFile(p/'PAPER_ADDENDUM_KO.docx') as z:assert len([n for n in z.namelist() if n.startswith('word/media/')])==2
 for f,digest in hashes.items():assert sha(ROOT/f)==digest
 info=subprocess.check_output(['pdfinfo',str(p/'PAPER_ADDENDUM_KO.pdf')],text=True);pages=int(re.search(r'Pages:\s+(\d+)',info)[1])
 save(p/'AUDIT.json',dict(status='VERIFIED',source_hashes=hashes,pages=pages,figures=2,new_training_in_rendering=0,artifact_hashes={str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file() and f.name!='AUDIT.json'}))
 print('PAPER_ADDENDUM_VERIFIED',pages,'pages',flush=True)
if __name__=='__main__':publish()
