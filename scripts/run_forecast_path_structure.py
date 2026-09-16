#!/usr/bin/env python3
"""Explicit bounded stages; no automatic research continuation."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.forecast_path_structure_v1_20260916.common import OUT,read,save

def main():
 p=argparse.ArgumentParser(description='Fixed forecast_path_structure_v1_20260916 contract; max48 fits, one GPU, no follow-up')
 p.add_argument('action',choices=['prepare','checks','run','resume','score','verify','report','status']);a=p.parse_args()
 if a.action=='prepare':
  from experiments.forecast_path_structure_v1_20260916.data import prepare;prepare()
 elif a.action=='checks':
  from experiments.forecast_path_structure_v1_20260916.checks import run_checks;run_checks()
 elif a.action=='status':
  print((OUT/'status.json').read_text() if (OUT/'status.json').exists() else 'PREPARED' if (OUT/'data_seal.json').exists() else 'NOT_PREPARED')
 elif a.action in ['run','resume']:
  from experiments.forecast_path_structure_v1_20260916.checks import run_checks
  from experiments.forecast_path_structure_v1_20260916.runtime import Controller
  from experiments.forecast_path_structure_v1_20260916.evaluate import score_all,contrasts,verify
  from experiments.forecast_path_structure_v1_20260916.report import report
  run_checks();c=Controller(resume=a.action=='resume')
  if c.run():
   df=score_all();contrasts(df);verify();s=read(OUT/'status.json');s.update(status='EXECUTION_COMPLETE',phase='DONE');save(OUT/'status.json',s)
  report()
 elif a.action=='score':
  from experiments.forecast_path_structure_v1_20260916.evaluate import score_all,contrasts
  contrasts(score_all())
 elif a.action=='verify':
  from experiments.forecast_path_structure_v1_20260916.evaluate import verify;verify()
 else:
  from experiments.forecast_path_structure_v1_20260916.report import report;report()
if __name__=='__main__':main()
