"""Known ReZero control, not a claimed new PEFT method."""
import sys
from pathlib import Path
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/peft_rank12_20260915'))
from rank2_model import make as original_make

class ResidualGate(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden, frequency, channel_ids=None, merged=False):
        return hidden + self.gate * self.base(hidden, frequency, channel_ids, merged)

def make(ids, seed, device='cpu'):
    model = original_make('SHARED_BUDGET', ids, seed, device)
    model.channel_adapter = ResidualGate(model.channel_adapter).to(device)
    return model
