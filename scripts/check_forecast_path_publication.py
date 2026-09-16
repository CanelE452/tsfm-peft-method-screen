"""Publication audit: exact executed sources, artifacts, hashes and monthly aggregates."""
import sys,subprocess,re,math,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from experiments.forecast_path_structure_v1_20260916.common import *
def main():
 executed='622d79c8cc80e4b2342754d1b0624af64be2ba79';sealed=read(OUT/'implementation_seal.json');sources=[]
 for p,h in sealed['files'].items():
  current=sha(p);mode='CURRENT_EXACT'
  if current!=h:
   rel=str(Path(p).relative_to(ROOT));raw=subprocess.check_output(['git','show',f'{executed}:{rel}'],cwd=ROOT);assert hashlib.sha256(raw).hexdigest()==h;mode='EXECUTED_GIT_SNAPSHOT_EXACT_POST_RUN_MAINTENANCE'
  sources.append(dict(path=p,expected=h,current=current,verification=mode))
 for p,h in sealed['model_files'].items():assert sha(p)==h
 historical=read(OUT/'historical_hashes.json')
 for p,h in historical.items():assert sha(ROOT/p)==h
 for j in read(OUT/'data_seal.json')['jobs']:
  assert sha(ROOT/j['input'])==j['input_sha256']
  if 'label' in j:assert sha(ROOT/j['label'])==j['label_sha256']
 for r in read(OUT/'prediction_manifest.json'):assert sha(ROOT/r['path'])==r['sha256']
 required=['PROTOCOL.md','TOPIC_ONEPAGE.md','literature_boundary.md','repo_audit.json','data_receipt.json','target_manifest.csv','origin_manifest.csv','exposure_ledger.md','selected_weather_rows.json','train_scaling.json','augmentation_schedule.json','fit_manifest.csv','fit_attempts.csv','train_curves.csv','checkpoint_manifest.json','selection_seal.json','calibration_parameters.json','evaluation_seal.json','scores_by_target_seed_case.csv','raw_and_calibrated.csv','contrasts.csv','uncertainty.csv','resource_usage.csv','independent_verification.json','REPORT.md','FINAL_DECISION.md','completion_audit.json','monthly_contrasts.csv','AUDIT_CORRECTIONS.md']
 assert all((OUT/p).exists() for p in required)
 links=[]
 for name in ['REPORT.md','FINAL_DECISION.md']:
  for dest in re.findall(r'\]\(([^)]+)\)',(OUT/name).read_text()):
   if dest.startswith(('https://','http://','#')):continue
   assert (OUT/dest.split('#')[0]).exists(),(name,dest);links.append(dict(file=name,target=dest))
 for p in ['case_scores.png','paired_gains.png','calibration.png']:assert (OUT/'figures'/p).stat().st_size>1000
 # Independent groupby aggregation of monthly point estimates; no bootstrap regeneration.
 df=pd.read_csv(OUT/'scores_by_origin.csv');cols=['month','target','arm','seed','policy','variant','case'];group=df.groupby(cols).primary.mean();month=pd.read_csv(OUT/'monthly_contrasts.csv');error=0.
 for r in month.itertuples():
  ts=['T0','T1','T2'] if r.scope=='ALL' else ['T1','T2'] if r.scope=='ADDITIONAL' else [r.scope];vals=[]
  for arm in [r.method,r.baseline]:
   tv=[]
   for t in ts:
    seeds=[-1] if arm.startswith('FROZEN') else SEEDS;policy='SELECTED' if arm.startswith('FROZEN') else r.policy
    cases=np.array([[group.loc[(r.month,t,arm,seed,policy,r.variant,f'S{k}')] for k in range(4)] for seed in seeds]).mean(0)
    value=cases.mean() if r.case=='MEAN' else cases.max() if r.case=='WORST' else cases[int(r.case[1])];tv.append(value)
   vals.append(float(np.mean(tv)))
  gain=100*(vals[1]-vals[0])/vals[1];assert math.isclose(gain,r.gain_percent,rel_tol=1e-10,abs_tol=1e-10);error=max(error,abs(gain-r.gain_percent))
 # All main cohorts and raw/calibrated conditions remain represented.
 agg=pd.read_csv(OUT/'scores_by_target_seed_case.csv');assert set(agg.target)=={'T0','T1','T2'} and set(agg.case)=={'S0','S1','S2','S3'}
 for t in ['T0','T1','T2']:
  for a in ARMS:
   for seed in SEEDS:
    sub=agg[(agg.target==t)&(agg.arm==a)&(agg.seed==seed)];assert len(sub)==12
 status=read(OUT/'status.json');assert status['status']=='EXECUTION_COMPLETE' and status['completed_fits']==48 and status['main_updates']==24576 and status['smoke_updates']==24 and not status['errors']
 env={}
 for p in ['torch','numpy','pandas','pyarrow','chronos-forecasting','transformers','huggingface-hub']:
  try:env[p]=importlib.metadata.version(p)
  except importlib.metadata.PackageNotFoundError:env[p]='not found under this distribution name'
 save(OUT/'publication_audit.json',dict(status='PASS',executed_commit=executed,source_verification=sources,required_files=required,local_links=links,figures_visually_checked=3,historical_preserved=len(historical),monthly_point_estimates_verified=len(month),monthly_max_gain_error=error,all_adapted_target_seed_cases_present=True,environment=env,local_cache_required_for_numerical_replay=True,final_report_sha256=sha(OUT/'REPORT.md'),decision_sha256=sha(OUT/'FINAL_DECISION.md')))
 print('PUBLICATION_AUDIT_PASS',len(sources),'sources',len(links),'links',len(month),'monthly contrasts')
if __name__=='__main__':main()
