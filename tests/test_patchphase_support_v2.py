from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tsfm_peft_screen.candidates.patchphase_support_v2 import patch_phase,unpatch,prepare_context


def test_all_phases_preserve_values_masks_and_absolute_times():
    x=torch.arange(672,dtype=torch.float32).reshape(2,336);x[0,17]=float('nan')
    m=SimpleNamespace(instance_norm=lambda x:(x,(None,None)),dtype=torch.float32,chronos_config=SimpleNamespace(time_encoding_scale=1.))
    for phase in range(16):
        torch.testing.assert_close(unpatch(patch_phase(x,phase)),x,rtol=0,atol=0,equal_nan=True)
        tokens,valid,_=prepare_context(m,x,None,phase)
        times,values,masks=tokens.split(16,-1)
        def undo(p):return unpatch((p,phase))
        assert torch.equal(undo(masks),torch.isfinite(x).float())
        assert torch.equal(undo(times),torch.arange(-336,0).expand_as(x))
        torch.testing.assert_close(undo(values),torch.nan_to_num(x),rtol=0,atol=0)
        assert int(masks.sum())==671
    with pytest.raises(ValueError):patch_phase(x,16)


def test_training_condition_spans_both_directions():
    phases=np.arange(0,16,2);a=np.stack([np.sin(2*np.pi*phases/16),np.cos(2*np.pi*phases/16)],1)
    assert np.linalg.matrix_rank(a)==2
    assert set(phases).isdisjoint([1,5,9,13]) and set(phases).isdisjoint([3,7,11,15])
