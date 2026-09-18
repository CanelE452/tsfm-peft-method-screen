"""Render the new evidence without overwriting the historical manuscript."""
from pathlib import Path
import json,subprocess,shutil,re,zipfile
from .common import ROOT,OUT,sha,save,check_seal

def publish():
 check_seal();p=ROOT/'papers/persistence_adaptation/internal_mechanism_v1_20260918';p.mkdir(parents=True,exist_ok=True);(p/'figures').mkdir(exist_ok=True)
 names=['REPORT.md','FINAL_DECISION.md','PAPER_CLAIM_UPDATE_KO.md','INDEPENDENT_AUDIT.json','SCORE_VERIFICATION.json','INITIAL_GRADIENT_DECOMPOSITION.csv','TRAJECTORY.csv','PAIRING_EFFECTS.csv','RAW_SCORES.csv','RESOURCE_SUMMARY.json','PROTOCOL.md','SEAL.json']
 hashes={str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in names}
 for f in (OUT/'figures').glob('*'):shutil.copyfile(f,p/'figures'/f.name);hashes[str(f.relative_to(ROOT))]=sha(f)
 body=(OUT/'PAPER_CLAIM_UPDATE_KO.md').read_text()+'\n\n---\n\n'+(OUT/'REPORT.md').read_text()+'\n\n---\n\n'+(OUT/'FINAL_DECISION.md').read_text()
 (p/'PAPER_ADDENDUM_KO.md').write_text(body)
 css=(ROOT/'papers/persistence_adaptation/manuscript_v2_20260918/paper.css').read_text()+'\ntable {break-inside:auto;} tr {break-inside:avoid;} th, td {font-size: 8pt;}\n';(p/'paper.css').write_text(css)
 pandoc='/home/minjae/anaconda3/bin/pandoc'
 subprocess.run([pandoc,'PAPER_ADDENDUM_KO.md','--standalone','--embed-resources','--css=paper.css','--metadata=lang:ko','--metadata=pagetitle:C3 내부 gradient와 B0 조합 진단','-o','PAPER_ADDENDUM_KO.html'],cwd=p,check=True)
 subprocess.run([pandoc,'PAPER_ADDENDUM_KO.md','--resource-path=.','--metadata=lang:ko','-o','PAPER_ADDENDUM_KO.docx'],cwd=p,check=True)
 subprocess.run(['/usr/bin/google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-pdf-header-footer','--user-data-dir=/tmp/c3-internal-paper-chrome','--print-to-pdf='+str(p/'PAPER_ADDENDUM_KO.pdf'),'file://'+str(p/'PAPER_ADDENDUM_KO.html')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 h=p/'PAPER_ADDENDUM_KO.html';h.write_text('\n'.join(s.rstrip() for s in h.read_text().splitlines())+'\n')
 txt=subprocess.check_output(['pdftotext',str(p/'PAPER_ADDENDUM_KO.pdf'),'-'],text=True)
 for term in ['1,196','gradient','B0','교환','배치']:assert term in txt,term
 assert '\ufffd' not in txt
 with zipfile.ZipFile(p/'PAPER_ADDENDUM_KO.docx') as z:assert len([n for n in z.namelist() if n.startswith('word/media/')])==3
 for f,d in hashes.items():assert sha(ROOT/f)==d
 info=subprocess.check_output(['pdfinfo',str(p/'PAPER_ADDENDUM_KO.pdf')],text=True);pages=int(re.search(r'Pages:\s+(\d+)',info)[1])
 (p/'README.md').write_text('''# C3 내부 기전 — 한국어 논문 보충

- [읽기용 PDF](PAPER_ADDENDUM_KO.pdf) · [편집용 DOCX](PAPER_ADDENDUM_KO.docx) · [원문 Markdown](PAPER_ADDENDUM_KO.md)
- [실험 보고서](../../../results/c3_internal_mechanism_20260918/REPORT.md) · [전체 조건](../../../results/c3_internal_mechanism_20260918/ALL_CONDITIONS.md) · [최종 판단](../../../results/c3_internal_mechanism_20260918/FINAL_DECISION.md)
- [독립 검산](../../../results/c3_internal_mechanism_20260918/INDEPENDENT_AUDIT.json) · [문서 검산](AUDIT.json)

새 학습 없이 기존32경로·160 checkpoint의 gradient와 B0 교환192개 E view를 분석했다. 초기 미분 경로를 확인했으나 최종 성능 차이의 완전한 인과 매개 분석은 아니다. 두 source의 재사용 개발자료이며 방법 신규성·범용 우위·논문 PASS를 주장하지 않는다. 기존 원고와 통제 실험 보충을 보존한 추가 결과다. SVG/PDF/PNG 그림3종은 figures에 있다. raw 데이터·가중치·예측·gradient 원벡터는 ignored 로컬 cache에 있고 hash manifest만 공개한다.
''')
 save(p/'AUDIT.json',dict(status='VERIFIED',source_hashes=hashes,pages=pages,figures=3,new_training_in_rendering=0,artifact_hashes={str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file() and f.name!='AUDIT.json'}))
 print('PAPER_ADDENDUM_VERIFIED',pages,'pages',flush=True)
if __name__=='__main__':publish()
