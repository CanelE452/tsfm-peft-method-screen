import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import numpy as np
import torch
from priority12.channel_data import metrics,independent,batch
from priority12.common import preserve_rng

def test_equal_channel_masked_mse_and_raw_units():
    p=np.ones((3,2,4));y=np.zeros_like(p);y[0,0,0]=np.nan;p[:,1]*=3
    m=metrics(p,y,[2,5]);assert m['mse']==5 and m['mae']==2 and m['channel_raw_mse']==[4,225]
    assert abs(independent(p,y)-m['mse'])<1e-10

def test_context_target_alignment_and_missing_input_fill():
    v=np.arange(800*64,dtype=np.float32).reshape(800,64);v[1,1]=np.nan
    x,y=batch(v,[512,536],'cpu');assert x.shape==(2,64,512) and y.shape==(2,64,96)
    assert x[0,1,1]==0 and y[0,0,0]==v[512,0] and x[1,0,-1]==v[535,0]

def test_rng_restoration():
    torch.manual_seed(4);a=torch.get_rng_state().clone()
    with preserve_rng():torch.randn(20)
    assert torch.equal(a,torch.get_rng_state())
