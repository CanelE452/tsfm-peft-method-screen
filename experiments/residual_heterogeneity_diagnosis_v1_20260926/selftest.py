import numpy as np
from run import corr,fro,agg,ks_uniform
R=corr(np.array([[0,0],[1,1],[2,2],[3,3]],float),.1)
assert np.linalg.eigvalsh(R).min()>0
assert fro(R,R)==0
assert agg(np.eye(4))==1.0
assert 0<=ks_uniform(np.linspace(.01,.99,100))<.03
print("SELFTEST_PASS")
