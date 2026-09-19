"""Observed-embedding scalar gate controls; not an official full GateRA reproduction."""
import math
import torch
from torch import nn
from experiments.additive_persistence_validation_v1_20260917.model import ResidualAdapter,ForecastModel

def binary_entropy(g):
    # FP32 saturation is possible. Clamp only entropy evaluation, not forward gate.
    p=g.clamp(torch.finfo(g.dtype).eps,1-torch.finfo(g.dtype).eps)
    return -(p*p.log()+(1-p)*(1-p).log())/math.log(2)

class LearnedAdapter(ResidualAdapter):
    def __init__(self,seed):
        super().__init__(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+200001)
            self.gate_linear=nn.Linear(512,1)
            nn.init.zeros_(self.gate_linear.weight);nn.init.zeros_(self.gate_linear.bias)
        self.last_gate=None
    def forward(self,embeds,ignored_gate):
        g=self.gate_linear(embeds).squeeze(-1).sigmoid();self.last_gate=g
        return super().forward(embeds,g)

class LearnedForecast(ForecastModel):
    def __init__(self,base,arm,seed):
        super().__init__(base,'C2',seed)
        self.adapter=LearnedAdapter(seed);self.control=arm
    def forward(self,observed,sigma,off=False):
        return super().forward(observed,sigma,residual_mode='off' if off else 'normal')
    def entropy(self):return binary_entropy(self.adapter.last_gate).mean()
    def regularizer(self):return self.entropy()*(.01 if self.control=='TOKEN_GATE_ENTROPY' else 0.)
