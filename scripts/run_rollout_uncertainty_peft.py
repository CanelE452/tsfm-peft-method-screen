from pathlib import Path
import runpy
import sys

scope=Path(__file__).resolve().parents[1]/'experiments'/'rollout_uncertainty_peft_v1_20260921'
sys.path.insert(0,str(scope))
runpy.run_path(str(scope/'runner.py'),run_name='__main__')
