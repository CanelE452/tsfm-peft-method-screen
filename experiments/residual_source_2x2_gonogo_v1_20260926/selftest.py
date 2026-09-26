import numpy as np
from run import corr, crps, gain
R=corr(np.array([[0,0],[1,1],[2,2]],float),.1)
assert np.linalg.eigvalsh(R).min()>0
assert abs(crps(np.ones(20),1.0))<1e-12
assert abs(gain(10,8)-20)<1e-12
print("SELFTEST_PASS")
