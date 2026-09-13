import copy
import pytest
import torch
from tsfm_peft_screen.forecast_query.model import Head, Side
from tsfm_peft_screen.forecast_query.equal_time import adapted_head, adapted_side, TrainingClock, choose_storage


@pytest.mark.parametrize('arm', ['head','side'])
def test_nonzero_adapter_checkpoint_output_gradient_adam_update(arm):
    torch.manual_seed(55)
    m = Head() if arm=='head' else Side()
    base = torch.randn(2,3,768)
    features = [torch.randn(2,8,768) for _ in range(12)]
    time, group = torch.zeros(2,1,1,8), torch.zeros(8,1,2,2)
    call = lambda cp: adapted_head(m,base,cp) if arm=='head' else adapted_side(m,features,time,group,base,cp)
    opt = torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=0.)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        call(False).square().mean().backward()
        opt.step()
    state, adam = copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict())
    outputs=[]
    for cp in (False,True):
        m.load_state_dict(state)
        opt.load_state_dict(copy.deepcopy(adam))
        opt.zero_grad(set_to_none=True)
        y=call(cp)
        y.square().mean().backward()
        grad=torch.cat([p.grad.flatten() for p in m.parameters()])
        opt.step()
        outputs.append((y.detach(),grad,torch.cat([p.detach().flatten() for p in m.parameters()])))
    for a,b in zip(*outputs):
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
    assert outputs[0][1].norm()>0
    assert all(f.grad is None for f in features)


def test_time_budget_boundary_and_no_extra_update():
    clock=TrainingClock(boundaries=(1.,2.),tolerance=.2)
    assert clock.add(.6) is None
    assert clock.add(.5)==1
    assert clock.add(.9)==2
    assert clock.complete and clock.updates==3
    with pytest.raises(RuntimeError):
        clock.add(.1)
    with pytest.raises(RuntimeError,match='overshoot'):
        TrainingClock(boundaries=(1.,),tolerance=.1).add(1.2)
    with pytest.raises(RuntimeError,match='update cap'):
        TrainingClock(boundaries=(1.,),max_updates=1).add(.1)


def test_storage_choice_allows_fast_standard_off_and_memory_tie():
    rows=[dict(checkpoint=cp,seconds=t,peak_allocated_bytes=mem)
          for cp,t,mem in [(False,1.,200),(True,1.5,100)] for _ in range(3)]
    assert not choose_storage(rows,300)['checkpoint']
    assert choose_storage(rows,150)['checkpoint']
    for row in rows:
        if row['checkpoint']:
            row['seconds']=1.01
    assert choose_storage(rows,300)['checkpoint']


def test_evaluation_requires_complete_intact_selection_seal(tmp_path,monkeypatch):
    import importlib
    from pathlib import Path
    from tsfm_peft_screen.reproducibility import write_json,sha,digest
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=importlib.import_module('run_forecast_query_equal_time')
    out,cache=tmp_path/'out',tmp_path/'cache'
    out.mkdir()
    cache.mkdir()
    monkeypatch.setattr(runner,'OUT',out)
    monkeypatch.setattr(runner,'CACHE',cache)
    with pytest.raises(FileNotFoundError):
        runner.require_selection_seal()
    write_json(out/'contract.json',{'fixture':True})
    (cache/'weights.pt').write_bytes(b'fixture checkpoint')
    selections=[dict(dataset=d,seed=s,arm=a,checkpoint_file='weights.pt',
                     checkpoint_sha256=sha(cache/'weights.pt'))
                for d in runner.CFG['datasets'] for s in runner.CFG['seeds'] for a in runner.CFG['arms']]
    seal=dict(selections=selections,contract_sha256=sha(out/'contract.json'))
    seal['seal_sha256']=digest(seal)
    write_json(out/'selection_seal.json',seal)
    assert len(runner.require_selection_seal()['selections'])==16
    (cache/'weights.pt').write_bytes(b'changed')
    with pytest.raises(AssertionError):
        runner.require_selection_seal()
