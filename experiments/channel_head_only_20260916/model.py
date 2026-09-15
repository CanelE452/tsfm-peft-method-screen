"""Head-only ablation: freeze the initial, zero-update LoRA encoder."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/peft_rank12_20260915'))
from rank2_model import make as original_make

def make(ids,seed,device='cpu'):
    m=original_make('LH',ids,seed,device)
    m.encoder.requires_grad_(False)
    return m
