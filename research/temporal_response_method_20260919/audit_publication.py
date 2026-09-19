"""Artifact-scope completion audit; does not assert scientific goal completion."""
from pathlib import Path
import hashlib,json,subprocess,time
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/temporal_response_peft_20260919';R=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert read(OUT/'status.json')['execution']=='COMPLETE_COMPUTE'
    assert read(OUT/'TRAINING_AUDIT.json')['status']=='VERIFIED_TRAINING_ONLY'
    marker=read(OUT/'ALL_PREDICTIONS_SAVED.json');assert marker['manifest_sha256']==sha(OUT/'PREDICTIONS.json')
    evaluation=read(OUT/'EVALUATION_SEAL.json');assert evaluation['selection_sha256']==sha(OUT/'MODEL_SELECTION.json') and evaluation['at']<marker['at']
    models=read(OUT/'MODEL_SELECTION.json');pred=read(OUT/'PREDICTIONS.json');assert len(models)==64 and len(pred)==192
    for row in models:assert sha(ROOT/row['checkpoint'])==row['sha256']
    for row in pred.values():assert sha(ROOT/row['path'])==row['sha256']
    assert sum(not x['reused'] for x in pred.values())==87
    for a in ['AUDIT.json','RESPONSE_DIAGNOSTIC_AUDIT.json']:
        d=read(R/'result_evidence'/a)
        for p,h in d.get('input_sha256',d.get('predictions_sha256',{})).items():assert sha(ROOT/p)==h
    assert read(OUT/'VERIFICATION.json')['scalar_metrics']==14592
    e=pd.read_csv(OUT/'EFFECTS.csv');assert len(e)==840 and len(e[e.primary_family])==4
    # One actual old B0 collision found before evaluation must not survive.
    for k in ['ettm1__standard__MAG_ONLY__s81552__selected','ettm1__shape__MAG_ONLY__s81552__selected']:
        row=pred[k];assert row['step']==0 and 'b81552_' in row['path'],row
    events=[json.loads(l) for l in (OUT/'gpu_trp.jsonl').read_text().splitlines()]
    unapproved=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in x['apps']) for x in events);assert unapproved==0
    assert min(x['free_mib'] for x in events)>=1024
    # Track only additions plus the repository's rolling index since this study began.
    diff=subprocess.check_output(['git','diff','--name-only','4f7b573','HEAD'],cwd=ROOT,text=True).splitlines()
    prefixes=('experiments/temporal_response_peft_20260919/','results/temporal_response_peft_20260919/','research/temporal_response_method_20260919/')
    assert all(p=='docs/RESULTS_INDEX.md' or p.startswith(prefixes) for p in diff)
    hashes={}
    for folder in [OUT,R,ROOT/'experiments/temporal_response_peft_20260919']:
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.name!='PUBLICATION_AUDIT.json':hashes[str(p.relative_to(ROOT))]=sha(p)
    for p in [OUT/'REPORT.md',OUT/'FINAL_DECISION.md',R/'result_evidence/effects.pdf',R/'result_evidence/effects.png']:assert p.stat().st_size>100
    status=dict(at=time.time(),execution_scope='COMPLETE_AND_VERIFIED',scientific_goal='NOT_ACHIEVED',paper_pass=False,new_fits=16,reused_fits=4,main_updates=16384,smoke_updates=16,prediction_views=192,new_prediction_views=87,reused_prediction_views=105,scalar_metric_checks=14592,effect_rows=840,unapproved_external_compute_samples=unapproved,gpu_samples=len(events),minimum_free_mib=min(x['free_mib'] for x in events),old_study_files_unchanged_in_committed_diff=True,automatic_new_training_started=False,hashes=hashes)
    (OUT/'PUBLICATION_AUDIT.json').write_text(json.dumps(status,indent=2,sort_keys=True)+'\n');print({k:v for k,v in status.items() if k!='hashes'})
if __name__=='__main__':main()
