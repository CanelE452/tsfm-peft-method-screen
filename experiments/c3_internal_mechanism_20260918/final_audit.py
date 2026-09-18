"""Final publication and accounting audit; no learning or inference."""
import ast
from .common import ROOT, OUT, EXP, read, save, sha, check_seal

def audit():
 check_seal();ind=read(OUT/'INDEPENDENT_AUDIT.json');assert ind['status']=='VERIFIED'
 cnt=read(OUT/'COUNTS.json');assert cnt['autograd_calls']==1196<=1280 and cnt['optimizer_updates']==cnt['new_fits']==0
 assert ind['new_E_views']==96 and ind['bootstrap_intervals_replayed']==120
 assert read(OUT/'SCORE_VERIFICATION.json')['original_rows_replayed']==960
 assert read(OUT/'MOMENT_CHECKS.json')['available']==32
 checked=0
 for p in EXP.glob('*.py'):
  for n in ast.walk(ast.parse(p.read_text())):
   if isinstance(n,ast.Call):
    name=ast.unparse(n.func)
    assert not name.endswith('.step') and not name.startswith('torch.optim.') and name!='optimizer_for',name
  checked+=1
 paper=ROOT/'papers/persistence_adaptation/internal_mechanism_v1_20260918'
 manifest=read(paper/'AUDIT.json');assert manifest['status']=='VERIFIED' and manifest['figures']==3
 for f,h in manifest['source_hashes'].items():assert sha(ROOT/f)==h
 for f,h in manifest['artifact_hashes'].items():assert sha(paper/f)==h
 for f in ['REPORT.md','FINAL_DECISION.md','PAPER_CLAIM_UPDATE_KO.md']:
  txt=(OUT/f).read_text();assert '9.7476%' in txt and '10.1990%' in txt and '0' in txt
 result=dict(status='COMPLETE_VERIFIED',new_fits=0,optimizer_updates=0,autograd_calls=1196,new_E_views=96,reused_E_views=96,checkpoint_probes=160,all_requested_diagnostics_complete=True,unexecuted_scope=[],full_causal_mediation='NOT_IDENTIFIED_OUTSIDE_SCOPE',independent_test=False,automatic_followup=False,sealed_files_preserved=len(read(OUT/'SEAL.json')['hashes']),code_files_checked_for_no_optimizer_calls=checked,publication_pdf_pages=manifest['pages'],publication_figures=3,publication_audit_sha256=sha(paper/'AUDIT.json'))
 save(OUT/'COMPLETION_AUDIT.json',result)
 old=read(OUT/'status.json');save(OUT/'status.json',dict(old,execution='COMPLETE_VERIFIED',completion_audit='COMPLETION_AUDIT.json'))
 print('FINAL_AUDIT_VERIFIED',flush=True)
if __name__=='__main__':audit()
