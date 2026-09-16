#!/usr/bin/env python
"""Single-contract, bounded sampling repair; no automatic follow-up research."""
import argparse,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from experiments.condition_sampling_repair_v1_20260917.common import *
p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare-all','run-all','resume-all','status','verify-all','report']);p.add_argument('--contract',default='/home/minjae/Downloads/condition_sampling_repair_cli_20260917.txt');args=p.parse_args()
if args.action=='prepare-all':
 from experiments.condition_sampling_repair_v1_20260917.setup import prepare_all
 prepare_all(args.contract)
elif args.action in ['run-all','resume-all']:
 from experiments.condition_sampling_repair_v1_20260917.runtime import Controller
 if args.action=='run-all':assert not (OUT/'controller_state.json').exists(),'Use resume-all; do not duplicate'
 Controller(resume=args.action=='resume-all').run()
elif args.action=='verify-all':
 from experiments.condition_sampling_repair_v1_20260917.verify import verify_all
 verify_all()
elif args.action=='report':
 from experiments.condition_sampling_repair_v1_20260917.report import report_all
 report_all()
else:
 print(json.dumps({t:read(OUT/t/'STATUS.json') for t in ORDER},ensure_ascii=False,indent=2))
 if (OUT/'controller_state.json').exists():print(json.dumps(read(OUT/'controller_state.json'),ensure_ascii=False,indent=2))
