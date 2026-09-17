import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.additive_b0_adapter_v1_20260917.runner import main
if __name__=="__main__":main()
