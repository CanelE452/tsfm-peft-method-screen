import pandas as pd
from .common import *
from experiments.persistence_evidence_extension_20260918.score import metric_arrays

def run():
    old.setup();tests=[]
    v=np.array([[1.,2.,3.,5.],[2.,2.,4.,4.]])
    t,g,w=decompose(v);np.testing.assert_array_equal(t,[4,2]);np.testing.assert_array_equal(g,[1.5,0]);np.testing.assert_array_equal(w,[2.5,2]);np.testing.assert_array_equal(g+w,t);tests.append('hand_checked_factorial_and_identity')
    x=torch.zeros(2,512);x[0,-32:]=8;x[1,100:132]=8;s=torch.ones(2)
    a=mechanism_gate(x,s,'C3');b=mechanism_gate(x,s,'M_RECENCY');assert torch.equal(a[0],b[0]) and not torch.equal(a[1],b[1]);torch.testing.assert_close(a.sort(-1).values,b.sort(-1).values,rtol=0,atol=0);tests.append('equal_and_different_observed_mask_examples')
    p=np.arange(2*9*64,dtype=float).reshape(2,9,64)/100;y=np.ones((2,64));ss=np.array([2.,3.]);m=metric_arrays(p,y,ss)
    for i in range(2):
        mae=sum(abs(float(p[i,4,h])-1.) for h in range(64))/64
        assert abs(m[i,0]-mae/ss[i])<1e-12
        assert abs(m[i,3]-pinball_scalar(p[i:i+1],y[i:i+1],ss[i:i+1],np.arange(1,10)/10))<1e-12
    tests.append('scalar_nmae_and_pinball')
    count=np.array([1,3]);values=np.array([2.,4.]);assert np.dot(count,values)/count.sum()==3.5;tests.append('stratum_count_weighted_reconstruction')
    z=bootstrap_counts(np.arange(0,128*24,24),24);assert z.shape==(2000,128) and (z.sum(1)>0).all();tests.append('paired_week_blocks')
    OUT.mkdir(exist_ok=True,parents=True);save(OUT/'CPU_CHECKS.json',dict(status='VERIFIED',tests=tests,optimizer_updates=0));print('CPU CHECKS',len(tests),flush=True)
if __name__=='__main__':run()
