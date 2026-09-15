#!/usr/bin/env python3
"""Bounded sequential stages. No automatic retry, replacement or research continuation."""
import sys,argparse,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments/building_coldstart_coverage_v1_20260915'))
from core import *
from engine import prepare,run_fits
from analysis import screen_gate,rules,heldout_summary,verify
from report import report

def status(value=None):
    if value is not None:
        fs=read(OUT/'fit_attempts.json') if (OUT/'fit_attempts.json').exists() else []
        save(OUT/'status.json',dict(tag='확인',status=value,fits_attempted=len(fs),fits_completed=sum(f['status']=='COMPLETE' for f in fs),actual_updates=sum(f['actual_updates'] for f in fs),automatic_followup=False))
    print(json.dumps(read(OUT/'status.json'),indent=2),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','recipe','screen','rules','heldout','verify','status','report','all']);args=p.parse_args();s=args.stage
    if s=='status':status();return
    if s=='verify':verify();return
    if s=='report':report();return
    if s=='all' and (OUT/'status.json').exists() and read(OUT/'status.json')['status'] in ['STOP_NO_COVERAGE_PROBLEM_SIGNAL','STOP_METHOD_SIGNAL_FAIL','STOP_HELDOUT_METHOD_SIGNAL_FAIL','HELDOUT_METHOD_SIGNAL','EXECUTION_INCONCLUSIVE','INSUFFICIENT_ELIGIBLE_BUILDINGS']:
        status();print('Terminal run retained; no duplicate execution');return
    try:
        if s in ['all','prepare']:prepare()
        if (OUT/'status.json').exists() and read(OUT/'status.json')['status']=='INSUFFICIENT_ELIGIBLE_BUILDINGS':return
        if s in ['all','recipe']:run_fits('recipe');status('RECIPE_COMPLETE')
        if s in ['all','screen']:run_fits('screen');a=screen_gate();status('SCREEN_COMPLETE' if a['status']=='COVERAGE_PROBLEM_SIGNAL' else 'STOP_NO_COVERAGE_PROBLEM_SIGNAL')
        if s in ['all','rules']:
            if read(OUT/'stageA_gate.json')['status']!='COVERAGE_PROBLEM_SIGNAL':return
            b=rules();status('RULES_COMPLETE' if b['status']=='METHOD_SIGNAL' else 'STOP_METHOD_SIGNAL_FAIL')
        if s in ['all','heldout']:
            if read(OUT/'stageB_selection.json')['status']!='METHOD_SIGNAL':return
            run_fits('heldout');c=heldout_summary();status(c['status'] if c['status']=='HELDOUT_METHOD_SIGNAL' else 'STOP_HELDOUT_METHOD_SIGNAL_FAIL')
    except BaseException as e:
        save(OUT/'execution_error.json',dict(tag='확인',error=repr(e),traceback=traceback.format_exc(),stage=s));status('EXECUTION_INCONCLUSIVE');raise
    finally:
        if s=='all':
            verify();report()
if __name__=='__main__':main()
