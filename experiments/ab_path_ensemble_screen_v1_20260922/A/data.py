import gzip
import numpy as np
from common import *
IDS=[36,10,44,119,62,104,116,118]
RANGES={'TRAIN':(0,5256),'CAL':(5256,6132),'VAL':(6132,7008),'TEST':(7008,8760)}

class Solar:
    def __init__(self):
        prior=read(ROOT/'results/continuation_lora_v1_20260922/DATA_AUDIT.json')['source']
        raw=ROOT/prior['raw_path'];assert sha(raw)==prior['raw_sha256']
        values=np.loadtxt(raw,delimiter=',',dtype=np.float64)
        assert values.shape==(52560,137) and np.isfinite(values).all()
        self.values=values.reshape(8760,6,137).mean(1)[:,IDS].astype(np.float32)
        self.sigma=self.values[:5256].astype(float).std(axis=0)
        self.pairs={}
        for role,(lo,hi) in RANGES.items():
            first=max(512,lo);stride=1 if role=='TRAIN' else 24
            if stride==24:first=((first+23)//24)*24
            origins=range(first,hi-128+1,stride)
            self.pairs[role]=np.array([(s,o) for s in range(8) for o in origins],dtype=np.int64)
        prepared=ROOT/prior['prepared_npz']['path']
        assert sha(prepared)==prior['prepared_npz']['sha256']
        with np.load(prepared) as z:np.testing.assert_array_equal(self.values,z['values'])
        self.provenance=dict(raw_path=raw,raw_sha256=sha(raw),source=prior['pinned_raw_url'],old_hourly_cache_sha256=sha(prepared))

    def batch(self,pairs):
        pairs=np.asarray(pairs,dtype=int)
        x=np.stack([self.values[o-512:o,s] for s,o in pairs]);y=np.stack([self.values[o:o+128,s] for s,o in pairs])
        return dict(x=x,y=y,sigma=self.sigma[pairs[:,0]].astype(np.float32),pairs=pairs)

    def schedule(self,seed):
        rng=np.random.default_rng(seed+100000);out=[]
        origins=self.pairs['TRAIN'][self.pairs['TRAIN'][:,0]==0,1]
        for _ in range(1024):out.append((rng.integers(0,8),rng.choice(origins)))
        return np.asarray(out,dtype=np.int64).reshape(256,4,2)

def audit():
    d=Solar();r=RESULTS/'A'
    save(r/'DATA_AUDIT.json',dict(status='PASS',**d.provenance,raw_shape=[52560,137],hourly_shape=[8760,8],
         selected_original_zero_based_columns=IDS,aggregation='six consecutive 10-minute rows mean; float64 aggregation then float32 storage',
         sigma_train_only=d.sigma,missing=0,calendar_timezone_invented=False,prior_exposure='Reused development source, not independent confirmation',
         roles={k:dict(range=RANGES[k],samples=len(v),origins=len(v)//8,first_origin=int(v[0,1]),last_origin=int(v[-1,1])) for k,v in d.pairs.items()}))
    for seed in SEEDS:
        p=CACHE/'A'/f'schedule_{seed}.npz';np.savez_compressed(p,pairs=d.schedule(seed));save(p.with_suffix('.json'),dict(sha256=sha(p)))
    save(r/'INFORMATION_CONTRACT.json',dict(context=512,horizon=128,first64='frozen F0 raw quantile-labelled paths',training_origins='all hourly legal origins; uniform series/origin',future_y_routing=False,scale='TRAIN only',baseline_selection='VAL after CAL; prior to TEST',teacher='same-seed newly trained FULL9 only'))
    return d

if __name__=='__main__':audit()
