import numpy as np
from B import data as source
from common import *

class Wind:
    def __init__(self):
        audit=read(RESULTS/'B/DATA_AUDIT.json');assert audit['status']=='PASS'
        assert sha(audit['prepared_npz']['path'])==audit['prepared_npz']['sha256']
        self.raw=source.load_data();self.dim=self.raw['weather'].shape[-1]*8
        with np.load(audit['prepared_npz']['path']) as z:
            self.weather_raw=z['weather_raw'];self.control_raw=z['control_raw'];self.weather_mean=z['weather_mean'];self.weather_std=z['weather_std']
        self.indices={r:np.flatnonzero(self.raw['role']==r) for r in ['TRAIN','CAL','VAL','TEST']}
    def batch(self,indices):
        b=source.batch(self.raw,indices);idx=b.pop('indices');b['pairs']=np.stack([np.zeros(len(idx),dtype=int),idx],1);return b
    def packet(self,role):return self.batch(self.indices[role])
    def schedule(self,seed):return source.schedule(seed)
