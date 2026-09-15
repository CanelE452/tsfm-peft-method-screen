"""Known pointwise adapter at an algebraically matched LoRA parameter budget."""
import importlib.util,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('_capacity_transport_reference',ROOT/'experiments/channel_phase_transport_20260916/model.py')
reference=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reference;spec.loader.exec_module(reference)
original_make=reference.original_make
Transport=reference.Transport
ARMS=['CAPACITY_MATCHED']

def make(arm,ids,seed,device='cpu'):
    assert arm=='CAPACITY_MATCHED'
    base=original_make('LH',ids,seed,'cpu')
    m=reference.Model(base,'POINTWISE',seed)
    m.adapter=Transport('POINTWISE',d=512,r=168,seed=seed)
    return m.to(device)
