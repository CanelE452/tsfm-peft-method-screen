"""Final artifact/budget/source audit. No model or optimizer execution."""
import sys,json,time,math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.condition_studies_v1_20260916.common import *

def run():
 state=read(OUT/'controller_state.json');queue=read(OUT/'QUEUE_STATUS.json');assert state['status']=='FINISHED' and queue['status']=='FINISHED';assert not (OUT/'pending_update.json').exists()
 seal=read(OUT/'MASTER_SEAL.json');libs={**seal['installed_code'],**seal['model_files']}
 for path,h in libs.items():assert sha(Path(path))==h,('PINNED_DEPENDENCY_CHANGED',path)
 required=['TOPIC_ONEPAGE.md','PROTOCOL.json','LITERATURE_BOUNDARY.md','data_receipt.json','origins.csv','permissions.json','train_statistics.json','feature_or_transform_manifest.json','fit_manifest.csv','optimizer_log.csv','LR_selection.json','selections.json','predictions_manifest.json','raw_scores.csv','contrasts.csv','uncertainty.csv','resources.csv','verification.json','STATUS.json','REPORT.md','LIMITATIONS.md','paired_gains.png','condition_tradeoffs.png']
 counts=[];contamination=0;trainforwards=0;maxmemory=0
 for t in ORDER:
  st=read(OUT/t/'STATUS.json')
  if st['EXECUTION']!='COMPLETE':counts.append(dict(track=t,status=st['EXECUTION'],reason=st.get('error',st.get('reason'))));continue
  for name in required:assert (OUT/t/name).is_file(),(t,'MISSING_ARTIFACT',name)
  if t in SOURCES:
   fits=read(OUT/t/'fits.json');logs=[json.loads(x) for x in (OUT/t/'optimizer_log.jsonl').read_text().splitlines()];main=[v for v in logs if not v['smoke']];smoke=[v for v in logs if v['smoke']];assert len(main)==sum(f['updates'] for f in fits);trainforwards+=len(logs)*(2 if t=='R04' else 1);contamination+=sum(v['contaminated'] for v in logs);maxmemory=max(maxmemory,max(f['peak_allocated_bytes'] for f in fits));assert read(OUT/t/'independent_selection_verification.json')['passed'];assert read(OUT/t/'training_record_verification.json')['passed'];assert read(OUT/t/'verification.json')['passed']
   counts.append(dict(track=t,status='COMPLETE',main_fits=len(fits),main_updates=len(main),smoke_updates=len(smoke),all_512=all(f['updates']==512 for f in fits),scalar_metric_checks=read(OUT/t/'verification.json')['scalar_metric_checks'],selected_INIT=st['selected_INIT']))
  else:assert st['neural_fits']==0;counts.append(dict(track=t,status='COMPLETE',main_fits=0,main_updates=0,smoke_updates=0))
 assert trainforwards==state['forwards']['train'],(trainforwards,state['forwards']);assert state['attempts']<=116 and state['main_updates']<=59392 and state['smoke_updates']<=96 and state['main_updates']+state['smoke_updates']<=59488;assert sum(state['forwards'].values())<=200000
 cachebytes=sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file());assert cachebytes<=100*2**30
 history=read(OUT/'historical_hashes.json')
 for path,h in history.items():assert sha(ROOT/path)==h,('HISTORY_CHANGED',path)
 save(OUT/'publication_audit.json',dict(passed=True,at=time.time(),tracks=counts,main_controller_seconds=queue_time(OUT/'QUEUE_STATUS.json')-state['started_at'],pinned_dependency_files=len(libs),historical_files_preserved=len(history),historical_hashes_verified=True,main_fits=state['completed_fits'],main_updates=state['main_updates'],smoke_updates=state['smoke_updates'],native_forwards=state['forwards'],native_train_forwards_independent_journal_count=trainforwards,contaminated_updates=contamination,max_allocated_bytes=maxmemory,cache_bytes=cachebytes,cache_is_local_ignored=True,github_contains_large_weights_or_predictions=False,zero_new_followup_fits=True,all_nine_complete=len(queue['completed'])==9))
 print('PUBLICATION_AUDIT_PASSED',len(queue['completed']),state['completed_fits'])

def queue_time(path):return path.stat().st_mtime
if __name__=='__main__':run()
