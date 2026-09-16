#!/usr/bin/env python
"""Single-contract, bounded condition studies. No follow-up search after queue completion."""
import argparse,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from experiments.condition_studies_v1_20260916.common import *
p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare-all','run-all','status','resume-all','verify-all','report']);p.add_argument('--contract',default='/home/minjae/Downloads/tsfm_new_and_prior_cli_master_20260916.txt');args=p.parse_args()
if args.action=='prepare-all':
 from experiments.condition_studies_v1_20260916.prepare import prepare_all
 from experiments.condition_studies_v1_20260916.audit import seal
 if not (OUT/'MASTER_MANIFEST.json').exists():prepare_all(args.contract)
 if not (OUT/'MASTER_SEAL.json').exists():seal()
 else:print('Already sealed; no duplicate preparation',sha(OUT/'MASTER_SEAL.json'))
elif args.action in ['run-all','resume-all']:
 from experiments.condition_studies_v1_20260916.runtime import Controller
 if args.action=='run-all':assert not (OUT/'controller_state.json').exists(),'Use resume-all for existing state'
 Controller(resume=args.action=='resume-all').run()
elif args.action=='status':
 print(json.dumps({t:read(OUT/t/'STATUS.json') for t in ORDER},ensure_ascii=False,indent=2))
 if (OUT/'controller_state.json').exists():print(json.dumps(read(OUT/'controller_state.json'),indent=2))
elif args.action=='verify-all':
 from experiments.condition_studies_v1_20260916.audit import verify_all
 verify_all()
else:
 from experiments.condition_studies_v1_20260916.report import report_track,report_all
 for t in ORDER:report_track(t)
 report_all()
