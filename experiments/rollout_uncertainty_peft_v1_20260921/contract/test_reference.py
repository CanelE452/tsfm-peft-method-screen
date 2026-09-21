import numpy as np
import pytest
import torch
from reference_core import (RolloutState, MetadataAdapter, rollout_numpy,
    ordered_quantiles, scaled_pinball, calibration_metrics, affine_calibration,
    role_origins, LEVELS)


def context(batch=2):
    return np.tile(np.sin(np.arange(512)/10), (batch, 1))


def quantiles(batch=2, width=1.):
    return np.broadcast_to(np.arange(9.)[None, None, :] * width,
                           (batch, 64, 9)).copy()


def test_initial_metadata_zero():
    s = RolloutState.initial(context(), np.ones(2))
    for mode in ['STATE','UNCERTAINTY']:
        assert s.metadata(mode).shape == (2,32,4)
        assert np.count_nonzero(s.metadata(mode)) == 0


def test_generated_values_not_truth():
    s = RolloutState.initial(context(), np.ones(2)).append(quantiles())
    np.testing.assert_array_equal(s.values[:, -64:], 4.)
    np.testing.assert_array_equal(s.generated[:, :512], 0.)
    np.testing.assert_array_equal(s.generated[:, -64:], 1.)
    np.testing.assert_array_equal(s.lead[:, -64:], np.tile(np.arange(1,65),(2,1)))
    np.testing.assert_array_equal(s.width[:, -64:], 8.)


def test_context_grows_without_losing_observations():
    s = RolloutState.initial(context(), np.ones(2))
    lengths = []
    for _ in range(4):
        lengths.append(s.values.shape[1]); s = s.append(quantiles())
    assert lengths == [512,576,640,704]
    np.testing.assert_array_equal(s.values[:, :512], context())
    with pytest.raises(ValueError): s.append(quantiles())


def test_uncertainty_feature_changes_not_state_feature():
    a = RolloutState.initial(context(), np.ones(2)).append(quantiles(width=1))
    b = RolloutState.initial(context(), np.ones(2)).append(quantiles(width=2))
    np.testing.assert_array_equal(a.metadata('STATE'), b.metadata('STATE'))
    assert not np.array_equal(a.metadata('UNCERTAINTY'), b.metadata('UNCERTAINTY'))


def test_quantile_rearrangement_and_width():
    q = quantiles()[..., ::-1]
    assert np.all(np.diff(ordered_quantiles(q), axis=-1) >= 0)
    s = RolloutState.initial(context(), np.ones(2)).append(q)
    assert np.all(s.width >= 0)


def test_adapter_count_and_initial_identity():
    a = MetadataAdapter()
    assert sum(p.numel() for p in a.parameters()) == 8744
    h, z = torch.randn(2,36,512), torch.rand(2,36,4)
    assert torch.equal(h, a(h,z))


def test_observed_patch_identity_even_after_training():
    a = MetadataAdapter()
    with torch.no_grad(): a.up.weight.normal_(); a.up.bias.fill_(.1)
    h = torch.randn(2,36,512); z = torch.rand(2,36,4); z[:, :32,0] = 0
    out = a(h,z)
    assert torch.equal(h[:,:32], out[:,:32])
    assert not torch.equal(h[:,32:], out[:,32:])


def test_real_parameter_update_and_no_claim_all_initial_grads_nonzero():
    torch.manual_seed(7)
    a = MetadataAdapter(); opt = torch.optim.AdamW(a.parameters(), lr=1e-3)
    h,z = torch.randn(2,36,512), torch.ones(2,36,4)
    before = {k:v.detach().clone() for k,v in a.state_dict().items()}
    for _ in range(2):
        opt.zero_grad(); a(h,z).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())
        opt.step()
    assert not torch.equal(before['up.weight'], a.up.weight)
    assert not torch.equal(before['down.weight'], a.down.weight)


def test_pinball_scalar_agreement():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(2,2,3)); q=np.sort(rng.normal(size=(2,2,3,9)),axis=-1)
    scale=np.array([1.,2.]); vals=[]
    for o in range(2):
        for s in range(2):
            for h in range(3):
                for j,tau in enumerate(LEVELS):
                    e=y[o,s,h]-q[o,s,h,j]
                    vals.append(2*(tau*e if e>=0 else (tau-1)*e)/scale[s])
    assert abs(scaled_pinball(y,q,scale)-np.mean(vals))<1e-12


def test_wider_is_not_automatically_better_score():
    y=np.zeros((2,1,256)); base=np.broadcast_to((LEVELS-.5)[None,None,None,:],(2,1,256,9)).copy()
    wide=base*100
    assert scaled_pinball(y,wide,np.ones(1)) > scaled_pinball(y,base,np.ones(1))
    assert calibration_metrics(y,wide,np.ones(1))['width80_train_scaled'] > calibration_metrics(y,base,np.ones(1))['width80_train_scaled']


def test_affine_identity_and_order():
    q=np.broadcast_to(LEVELS,(2,2,256,9)).copy()
    np.testing.assert_array_equal(affine_calibration(q,np.ones(2),np.ones(4),np.zeros(4)),q)
    out=affine_calibration(q,np.ones(2),np.full(4,2.),np.full(4,.5))
    assert np.all(np.diff(out,axis=-1)>=0)


def test_no_future_argument_and_deterministic_toy_rollout():
    seen=[]
    def toy(x, z):
        seen.append((x.shape[1],z.shape[1]))
        return quantiles(batch=len(x))
    x=context(); preserved=x.copy()
    q=rollout_numpy(toy,x,np.ones(2))
    assert q.shape==(2,256,9)
    assert seen==[(512,32),(576,36),(640,40),(704,44)]
    np.testing.assert_array_equal(x,preserved)
    # No target is passed into this function at all; a separate trainer must
    # additionally verify data-loader isolation from its target arrays.


def test_role_targets_stay_in_role():
    T=17420; roles=role_origins(T); edges=[0,int(.6*T),int(.7*T),int(.8*T),T]
    for (name,a),lo,hi in zip(roles.items(),edges[:-1],edges[1:]):
        assert len(a)>0 and np.all(np.diff(a)==24)
        assert np.all(a>=max(lo,512)) and np.all(a+256<=hi)
    assert len(set(np.concatenate(list(roles.values()))))==sum(map(len,roles.values()))
