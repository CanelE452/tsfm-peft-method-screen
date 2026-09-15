"""Unmodified existing LoRA+head factory; only training origins differ."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/peft_rank12_20260915'))
from rank2_model import make as original_make

def make(ids,seed,device='cpu'):
    return original_make('LH',ids,seed,device)
