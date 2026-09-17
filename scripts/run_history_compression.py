#!/usr/bin/env python
import os,sys,argparse
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run','verify','report','status']);args=a.parse_args()
from experiments.history_compression_v1_20260917.common import *
if args.action=='prepare':
 from experiments.history_compression_v1_20260917.setup import prepare
 prepare()
elif args.action=='run':
 from experiments.history_compression_v1_20260917.runtime import run
 run()
elif args.action in ['verify','report']:
 from experiments.history_compression_v1_20260917.analysis import verify,report
 verify() if args.action=='verify' else report()
else:print((OUT/'state.json').read_text() if (OUT/'state.json').exists() else 'NOT_STARTED')
