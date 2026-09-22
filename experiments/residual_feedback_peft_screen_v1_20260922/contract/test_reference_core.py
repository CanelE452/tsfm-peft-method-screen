import pytest
import torch
from reference_core import (
    a_support_issues, b_support_issues, visible_mask, observed_scale,
    safe_masked_pinball, residual_record, CoefficientGenerator, conditional_lora,
    IssuedLedger, split_ids, break_even,
)

torch.set_num_threads(1)


def fixtures():
    context = torch.linspace(3., 7., 256)
    q = torch.arange(1,10).float().repeat(64,1)
    y = torch.linspace(4.,6.,64)
    return context, q, y


def test_a_all_supports_mature():
    assert [int(visible_mask(i,1000).sum()) for i in a_support_issues(1000)] == [64]*4


def test_b_partial_clock_counts():
    assert [int(visible_mask(i,1000).sum()) for i in b_support_issues(1000)] == [64,32,16,8]


def test_boundary_value_not_visible():
    m = visible_mask(992,1000)
    assert m[7] and not m[8]


def test_current_query_has_no_labels():
    assert not visible_mask(1000,1000).any()


def test_scale_is_finite_for_constant():
    assert observed_scale(torch.zeros(256)) > 0


def test_unseen_labels_do_not_change_record():
    x,q,y=fixtures(); m=visible_mask(992,1000)
    other=y.clone();other[~m]=float('nan')
    r1=residual_record(x,q,y,m,8,observed_scale(x))
    r2=residual_record(x,q,other,m,8,observed_scale(x))
    assert torch.equal(r1,r2)


def test_seen_labels_change_record():
    x,q,y=fixtures();m=visible_mask(992,1000);other=y.clone();other[0]+=1
    assert not torch.equal(residual_record(x,q,y,m,8,observed_scale(x)),residual_record(x,q,other,m,8,observed_scale(x)))


def test_masked_loss_hidden_nan_gradient_finite():
    _,q,y=fixtures();q=q[None].clone().requires_grad_();m=visible_mask(992,1000)[None]
    y=y[None];y[~m]=float('nan')
    loss=safe_masked_pinball(q,y,m,torch.ones(1));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(q.grad).all()
    assert torch.equal(q.grad[:,8:],torch.zeros_like(q.grad[:,8:]))


def test_empty_mask_does_not_count_fake_update():
    _,q,y=fixtures()
    with pytest.raises(ValueError):safe_masked_pinball(q[None],y[None],torch.zeros(1,64,dtype=torch.bool),torch.ones(1))


def test_initial_c_is_one():
    for mode in ['set','time']:
        g=CoefficientGenerator(3,mode=mode)
        assert torch.equal(g(torch.randn(2,4,274)),torch.ones(2,3,8))


def test_size_match_is_one_parameter():
    a=CoefficientGenerator(3,mode='set');b=CoefficientGenerator(3,mode='time')
    assert sum(p.numel() for p in a.parameters())-sum(p.numel() for p in b.parameters()) == 1


def test_set_permutation_invariant_after_learning_nonzero_head():
    torch.manual_seed(3);g=CoefficientGenerator(2,mode='set')
    torch.nn.init.normal_(g.head.weight)
    r=torch.randn(2,4,274)
    assert torch.allclose(g(r),g(r[:,[3,1,0,2]]),atol=1e-6,rtol=1e-6)


def test_time_can_depend_on_order():
    torch.manual_seed(4);g=CoefficientGenerator(2,mode='time');torch.nn.init.normal_(g.head.weight)
    r=torch.randn(2,4,274)
    assert not torch.allclose(g(r),g(r[:,[3,2,1,0]]),atol=1e-6,rtol=1e-6)


def test_c_one_matches_static_lora():
    x=torch.randn(2,5,11);a=torch.randn(8,11);b=torch.randn(7,8)
    actual=conditional_lora(x,a,b,torch.ones(2,8))
    expected=2*torch.nn.functional.linear(torch.nn.functional.linear(x,a),b)
    assert torch.equal(actual,expected)


def test_generator_to_weights_gradient_and_update():
    torch.manual_seed(5)
    g=CoefficientGenerator(1)
    a=torch.nn.Parameter(torch.randn(8,11)*.1)
    b=torch.nn.Parameter(torch.randn(7,8)*.1) # nonzero WARM LoRA, not an all-zero dead bank
    opt=torch.optim.Adam(list(g.parameters())+[a,b],lr=.001)
    x=torch.randn(2,5,11);records=torch.randn(2,4,274);target=torch.randn(2,5,7)
    first=g(records).detach().clone()
    for step in range(2):
        opt.zero_grad();c=g(records)[:,0]
        loss=(conditional_lora(x,a,b,c)-target).square().mean();loss.backward()
        assert g.head.weight.grad.norm() > 0
        if step==1:assert g.encoder[0].weight.grad.norm() > 0
        opt.step()
    assert not torch.equal(first,g(records).detach())


def test_issued_predictions_immutable():
    ledger=IssuedLedger();q=torch.randn(64,9)
    entry=ledger.issue('client',1000,q)
    assert entry.issue==1000 and entry.horizon==64
    with pytest.raises(ValueError):ledger.issue('client',1000,q+1)


def test_series_partitions_disjoint_and_deterministic():
    donor,dev,test=split_ids([str(i) for i in range(100)])
    assert (len(donor),len(dev),len(test))==(32,8,8)
    assert not set(donor)&set(dev) and not set(donor)&set(test) and not set(dev)&set(test)
    assert (donor,dev,test)==split_ids([str(i) for i in reversed(range(100))])


def test_break_even_not_free_pretraining():
    assert break_even(100.,2.,1.) == 100
    assert break_even(100.,1.,2.) is None
