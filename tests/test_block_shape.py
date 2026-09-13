import copy
import torch
from tsfm_peft_screen.block_shape.method import (
    SplitAdapter, acceptance, per_origin_loss, snapshot_transaction, rollback_transaction)

def test_identity_order_and_center_shape_separation():
    torch.manual_seed(1)
    h = torch.randn(8, 3, 768)
    f0 = torch.randn(8, 21, 48).sort(1).values
    model = SplitAdapter()
    p, _ = model(h, f0)
    torch.testing.assert_close(p, f0, atol=1e-6, rtol=1e-5)
    with torch.no_grad():
        model.shape.up.weight.normal_(std=.1)
    changed, _ = model(h, f0)
    assert (changed[:, 1:] >= changed[:, :-1]).all()
    torch.testing.assert_close(changed[:, 10], f0[:, 10])
    assert not torch.allclose(changed, f0)
    before_gaps = changed[:, 1:] - changed[:, :-1]
    with torch.no_grad():
        model.center.up.weight.normal_(std=.1)
    shifted, _ = model(h, f0)
    torch.testing.assert_close(shifted[:, 1:] - shifted[:, :-1], before_gaps, atol=1e-6, rtol=1e-5)

def test_native_objective_and_missing_target_gradients():
    from tsfm_peft_screen.backbone import native_loss
    p = torch.randn(8, 21, 48, requires_grad=True)
    y = torch.randn(8, 48)
    y[0, :3] = float('nan')
    a = per_origin_loss(p, y).mean()
    b = native_loss(p.asinh(), y, torch.zeros(8, 1), torch.ones(8, 1))
    torch.testing.assert_close(a, b)
    a.backward()
    assert torch.isfinite(p.grad).all()

def test_block_gate_accepts_coherent_change_rejects_mixed_change():
    ids = torch.arange(8).repeat_interleave(15)
    coherent = torch.full((120,), -.01)
    assert acceptance(coherent, ids)['accept']
    mixed = torch.tensor([-.1, -.1, -.1, -.1, .08, .08, .08, .08]).repeat_interleave(15)
    assert acceptance(mixed, ids, 0.)['accept']
    assert not acceptance(mixed, ids, 1.)['accept']
    # Duplicating windows inside blocks must not invent independent evidence.
    a = acceptance(mixed, ids)
    b = acceptance(mixed.repeat_interleave(3), ids.repeat_interleave(3))
    assert abs(a['criterion'] - b['criterion']) < 1e-12

def test_rejection_restores_adam_moments_and_parameters():
    model = torch.nn.Linear(3, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=.1)
    def step():
        opt.zero_grad()
        model(torch.ones(4, 3)).square().mean().backward()
        opt.step()
    step()
    snapshot = snapshot_transaction(model, opt)
    reference_model = copy.deepcopy(model)
    reference_opt = torch.optim.AdamW(reference_model.parameters(), lr=.1)
    reference_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    step()
    rollback_transaction(model, opt, snapshot)
    for x, y in zip(model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    for state, ref in zip(opt.state.values(), reference_opt.state.values()):
        for key in state:
            torch.testing.assert_close(state[key], ref[key], atol=0, rtol=0)
    step()
    reference_opt.zero_grad()
    reference_model(torch.ones(4, 3)).square().mean().backward()
    reference_opt.step()
    for x, y in zip(model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(x, y, atol=0, rtol=0)

def test_temporal_target_partitions_and_fit_budget():
    import json
    from tsfm_peft_screen.reproducibility import ROOT
    c = json.loads((ROOT/'configs/block_shape_pilot.json').read_text())
    targets = []
    for b in range(c['train_blocks']):
        os = [c['train_block_start']+b*c['train_block_length']+o for o in c['train_within_block_origins']]
        targets.append({t for o in os for t in range(o,o+c['horizon'])})
    assert all(not targets[i].intersection(targets[j]) for i in range(8) for j in range(i))
    assert max(set.union(*targets)) < min(c['validation_origins'])
    assert max(c['validation_origins'])+c['horizon'] <= min(c['evaluation_origins'])
    assert all(b-a >= c['horizon'] for a,b in zip(c['evaluation_origins'],c['evaluation_origins'][1:]))
    assert len(c['datasets'])*len(c['seeds'])*sum(len(c['learning_rates'][a]) for a in c['arms']) == c['fit_cap'] == 48
