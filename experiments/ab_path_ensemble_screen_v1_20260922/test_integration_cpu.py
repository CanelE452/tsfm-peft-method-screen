import numpy as np
import torch
from model import fast_medoids,medoids3,weighted_crps
from evaluation import metric,calibrate,apply_cal,score
from references import cpu_predict,features
from A.data import Solar

def test_medoid_vectorization_matches_exact_reference():
    torch.manual_seed(10);paths=torch.randn(3,9,64)
    for a,b in zip(fast_medoids(paths),medoids3(paths)):torch.testing.assert_close(a,b)

def test_numpy_metric_matches_independent_pairwise_crps():
    rng=np.random.default_rng(3);z=rng.normal(size=(2,8,27));p=rng.uniform(size=z.shape);p/=p.sum(-1,keepdims=True)
    y=rng.normal(size=(2,8));s=np.array([2.,3.])
    oracle=(p*np.abs(z-y[...,None])).sum(-1)-.5*(p[...,None]*p[...,None,:]*np.abs(z[...,None]-z[...,None,:])).sum((-1,-2))
    np.testing.assert_allclose(metric(z,p,y,s)['crps'],oracle/s[:,None],atol=1e-12)

def test_common_cal_grid_recomputes_score_and_identity_tie():
    z=np.zeros((2,8,9));p=np.ones_like(z)/9;y=np.zeros((2,8));s=np.ones(2)
    pred=dict(z=z,p=p,y=y,sigma=s)
    fitted=calibrate(pred,'B');assert all(row['alpha']==1 and row['beta']==0 for row in fitted)
    assert score(pred,'B',fitted)==0

def test_seasonal_naive_never_requires_issue_time_observation():
    x=np.arange(128,dtype=float)[None]
    packet=dict(x=x,y=np.zeros((1,8)),sigma=np.ones(1),pairs=np.array([[0,0]]))
    z=cpu_predict('NAIVE',packet)['z'][0,:,0]
    np.testing.assert_array_equal(z,[121,122,123,124,125,126,127,120])

def test_a_all_hourly_train_origins_and_disjoint_role_targets():
    d=Solar();origins=d.pairs['TRAIN'][d.pairs['TRAIN'][:,0]==0,1]
    assert origins[0]==512 and origins[-1]+128==5256 and np.all(np.diff(origins)==1)
    for role,(lo,hi) in {'CAL':(5256,6132),'VAL':(6132,7008),'TEST':(7008,8760)}.items():
        o=d.pairs[role][:,1];assert o.min()>=lo and o.max()+128<=hi
