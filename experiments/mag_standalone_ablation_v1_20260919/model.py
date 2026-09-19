"""No-LoRA builds; unchanged historical residual and magnitude formulas."""
import torch
from experiments.additive_persistence_validation_v1_20260917.model import ForecastModel
from experiments.c3_weakness_controls_20260918.model import ControlModel

class PlainForecast(ForecastModel):
    def __init__(self,base,seed):super().__init__(base,'C2',seed)
    def forward(self,observed,sigma,off=False):
        return super().forward(observed,sigma,residual_mode='off' if off else 'normal')

class MagnitudeForecast(ControlModel):
    def __init__(self,base,seed):super().__init__(base,'MAG_ONLY',seed)
    def forward(self,observed,sigma,off=False):
        return super().forward(observed,sigma,residual_mode='off' if off else 'normal')

class FoundationForecast(ForecastModel):
    def __init__(self,base,seed):super().__init__(base,'C0',seed)
    def forward(self,observed,sigma,off=False):
        return super().forward(observed,sigma,residual_mode='off')
