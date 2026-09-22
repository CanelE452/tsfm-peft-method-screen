import numpy as np
import torch
from model import estimate,eigbasis,pinball
from evaluation import metrics,calibrate
from data import Data

def test_covariance_matches_independent_pairs():
    z=np.random.default_rng(1).normal(size=(4,16,12));c0,c1=estimate(z);zz=z-z.mean(1,keepdims=True)
    expected=sum(np.outer(row[t],row[t+1])+np.outer(row[t+1],row[t]) for row in zz for t in range(15))/(4*15*2)
    np.testing.assert_allclose(c1,expected,atol=1e-12)
    np.testing.assert_allclose(c0,np.einsum('bti,btj->ij',zz,zz)/64,atol=1e-12)
def test_shuffle_changes_temporal_operator_not_variance():
    z=np.random.default_rng(2).normal(size=(4,16,12));a,b=estimate(z);c,d=estimate(z,True)
    np.testing.assert_array_equal(a,c);assert not np.allclose(b,d);np.testing.assert_array_equal(estimate(z,True)[1],d)
def test_eigenbasis_orthogonal_and_signed():
    z=np.random.default_rng(3).normal(size=(4,16,12));b,_=eigbasis(estimate(z)[0]);np.testing.assert_allclose(b@b.T,np.eye(8),atol=1e-6)
    assert (b[np.arange(8),np.abs(b).argmax(1)]>0).all()
def test_metric_matches_torch_training_loss():
    rng=np.random.default_rng(4);q=rng.normal(size=(2,64,9));y=rng.normal(size=(2,64));s=np.array([1.,2.])
    actual=pinball(torch.tensor(q),torch.tensor(y),torch.tensor(s));np.testing.assert_allclose(actual,metrics(q,y,s)['pinball'].mean(),rtol=1e-7)
def test_calibration_identity_tie():
    p=dict(q=np.zeros((2,64,9)),y=np.zeros((2,64)),sigma=np.ones(2));c=calibrate(p);assert c['alpha']==1 and c['beta']==0
def test_data_all_targets_in_role_and_basis_disjoint():
    for source in ['Electricity','ETTh1']:
        d=Data(source)
        for role,lo,hi in zip(['TRAIN','CAL','VAL','TEST'],d.edges[:-1],d.edges[1:]):
            p=d.pairs[role];assert (p[:,1]>=max(lo,256)).all() and (p[:,1]+64<=hi).all()
        for s in range(len(d.ids)):assert (np.diff(d.basis_pairs[d.basis_pairs[:,0]==s,1])>=256).all()
        np.testing.assert_array_equal(d.schedule(92341),d.schedule(92341))
