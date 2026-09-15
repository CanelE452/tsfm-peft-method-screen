#!/usr/bin/env python3
"""Verify saved records and produce Korean report; no training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/building_coldstart_coverage_v1_20260915'))
from analysis import verify
from report import report
if __name__=='__main__':verify();report()
