"""Bounded paper evidence checks; immutable parent studies are read-only."""
from experiments.persistence_evidence_extension_20260918.common import *
from experiments.persistence_evidence_extension_20260918 import common as ext
from experiments.additive_persistence_validation_v1_20260917.evaluate import checkpoint_row
NAME='paper_readiness_20260918'
OUT=ROOT/'results'/NAME;CACHE=ROOT/'.cache'/NAME;EXP=ROOT/'experiments'/NAME
class Watch(BaseWatch):
    def __init__(self):super().__init__(OUT,'evaluation',wall_cap=10800)
    def sample(self):
        r=super().sample()
        for a in r['apps']:a['allowed_desktop']=a['name']=='/usr/share/rustdesk/rustdesk'
        r['busy']=any(not a['own'] and not a['allowed_desktop'] for a in r['apps']) or r['free_mib']<1024
        return r
    def limits(self):
        import shutil
        super().limits()
        if shutil.disk_usage(ROOT).free<10*2**30:raise ResourceError('DISK_BELOW_10GIB')
def check_seal():
    d=read(OUT/'SEAL.json')
    for p,h in d['hashes'].items():assert sha(ROOT/p)==h,('SEALED_CHANGED',p)
    return d

def naive(x,period):
    assert x.shape[-1]==512 and period in [24,96]
    last=np.repeat(x[:,-1:],64,axis=1)
    seasonal=x[:,-period:][:,np.arange(64)%period]
    return last,seasonal
