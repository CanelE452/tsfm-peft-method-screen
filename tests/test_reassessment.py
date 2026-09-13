import json
from pathlib import Path
import numpy as np
import torch
from tsfm_peft_screen.candidates.dualclock import EventAdapter, event_features
from tsfm_peft_screen.reassessment import DiagnosticEventAdapter, schedule, choose, objective


def test_normal_event_adapter_preserves_original_outputs_and_gradients():
    torch.manual_seed(112)
    x=torch.randint(0,4,(4,336)).float();state=event_features(x)
    for full in (False,True):
        original=EventAdapter(16,full);diagnostic=DiagnosticEventAdapter(16,full)
        with torch.no_grad():original.up.weight.normal_(0,.1)
        diagnostic.load_state_dict(original.state_dict())
        a=torch.randn(4,3,16,requires_grad=True);b=a.detach().clone().requires_grad_()
        p=original(a,state);q=diagnostic(b,state)
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        p.square().mean().backward();q.square().mean().backward()
        torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
        for left,right in zip(original.parameters(),diagnostic.parameters()):
            torch.testing.assert_close(left.grad,right.grad,rtol=0,atol=0)


def test_schedule_extends_original_without_changing_first_360_batches():
    root=Path(__file__).resolve().parents[1]
    original=json.loads((root/'results/candidate_02/sampling_manifest.json').read_text())
    origins=list(range(336,1400-47,24))
    new=schedule('dualclock',origins,30000,1440)
    assert len(new)==1440
    assert new[:360]==[{k:r[k] for k in ('origins','channels')} for r in original]


def test_ablations_keep_hidden_input_and_drop_exactly():
    torch.manual_seed(13)
    adapter=DiagnosticEventAdapter(16,True)
    with torch.no_grad():adapter.up.weight.normal_(0,.1)
    h=torch.randn(4,3,16);state=event_features(torch.randint(0,4,(4,336)).float())
    saved=h.clone();normal=adapter(h,state)
    adapter.mode='zero_events';zero=adapter(h,state)
    adapter.mode='rotate_events';rotate=adapter(h,state)
    adapter.mode='drop_adapter';drop=adapter(h,state)
    assert torch.equal(h,saved) and torch.equal(drop,h)
    assert not torch.equal(normal,zero) and not torch.equal(normal,rotate)


def test_zero_teacher_displacement_has_no_anchor_penalty_and_selection_ties():
    z=torch.zeros(2,21,48,requires_grad=True);p=z.sinh();y=torch.ones(2,48)
    loc=torch.zeros(2,1);scale=torch.ones(2,1);sc=torch.tensor([1.,2.])
    _,reg=objective('anchor','native_anchor',z,p,y,loc,scale,sc,p.detach())
    assert float(reg.detach())==0
    _,displaced=objective('anchor','raw_anchor',z,p,y,loc,scale,sc,p.detach()+1)
    assert np.isclose(float(displaced.detach()),.075)
    rows=[dict(step=s,lr=lr,metrics={'scaled_2pinball':1.}) for s,lr in [(1440,1e-4),(360,1e-4),(360,3e-5)]]
    assert choose(rows,360)['lr']==3e-5 and choose(rows)['step']==360
