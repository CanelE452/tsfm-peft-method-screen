import itertools
import numpy as np
COMBINATIONS=list(itertools.product([81551,81552],repeat=3))
SIGNS=np.array([[-1 if x==81551 else 1 for x in row] for row in COMBINATIONS])
AXES=[(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)]
NAMES=['B0','INIT','ORDER','B0:INIT','B0:ORDER','INIT:ORDER','B0:INIT:ORDER']
DESIGN=np.stack([SIGNS[:,a].prod(1) for a in AXES],axis=1)
def contrasts(values):return 2*np.asarray(values)@DESIGN/8
