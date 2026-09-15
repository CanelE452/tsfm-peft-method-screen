"""Entry point for the user's bounded MOMENT comparison contract."""
import runpy
from pathlib import Path
import sys
p=Path(__file__).resolve().parents[1]/'experiments/channel_controlled_improvement_v1_20260916'
sys.path.insert(0,str(p))
runpy.run_path(str(p/'run.py'),run_name='__main__')
