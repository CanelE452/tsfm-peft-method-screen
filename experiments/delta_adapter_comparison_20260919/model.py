"""Controlled Chronos transfer of the official delta-Adapter XY cell.

Architecture independently expressed; original upstream class is loaded only
for CPU parity checks. Chronos interfaces are documented in PROTOCOL.md.
"""
import torch
from torch import nn

class DeltaNet(nn.Module):
    def __init__(self,length,width,delta=.1):
        super().__init__()
        self.layers=nn.ModuleList([nn.Linear(length,width),nn.Linear(width,width),nn.Linear(width,length)])
        self.norms=nn.ModuleList([nn.InstanceNorm1d(width),nn.InstanceNorm1d(width)])
        self.delta=delta
    def forward(self,x):
        for linear,norm in zip(self.layers[:2],self.norms):
            # The official cell uses 2D InstanceNorm with affine=False.
            # It normalizes each example across hidden features, not batches.
            x=norm(torch.relu(linear(x)))
        return self.layers[2](x).tanh()*self.delta

class DeltaForecast(nn.Module):
    def __init__(self,b0,arm,seed):
        super().__init__();self.b0=b0.requires_grad_(False)
        width={'DELTA_XY_BUDGET':7,'DELTA_XY_DEFAULT':512}[arm]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.xnet=DeltaNet(512,width);self.ynet=DeltaNet(64,width)
        self.arm=arm
    @property
    def base(self):return self.b0.base
    def forward(self,x,s,off=False):
        if off:return self.b0(x,s)
        center=x.mean(-1,keepdim=True).detach();scale=s[:,None]
        xx=x+scale*self.xnet((x-center)/scale)
        p=self.b0(xx,s)
        y_input=((p-center[:,None,:])/scale[:,None,:]).reshape(-1,64)
        correction=self.ynet(y_input).reshape_as(p)
        return p+scale[:,None,:]*correction
