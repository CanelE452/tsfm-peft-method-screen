"""Frozen-method temporal replication and input-only conditional diagnostics."""
from experiments.persistence_evidence_extension_20260918.common import *
from experiments.persistence_evidence_extension_20260918 import common as ext
from experiments.paper_readiness_20260918 import common as ready
from experiments.additive_persistence_validation_v1_20260917.model import mechanism_gate
NAME='c3_identifiability_temporal_20260918'
OUT=ROOT/'results'/NAME;CACHE=ROOT/'.cache'/NAME;EXP=ROOT/'experiments'/NAME
NEW='neso_2026_h1'
DIAG_PANELS=['electricity','electricity_transfer','ettm1','neso_2025',NEW]
class Watch(ready.Watch):
    def __init__(self):BaseWatch.__init__(self,OUT,'evaluation',wall_cap=10800)
def check_seal():
    d=read(OUT/'SEAL.json')
    for p,h in d['hashes'].items():assert sha(ROOT/p)==h,('SEALED_CHANGED',p)
    return d
def panel_path(panel,kind):
    return CACHE/('conditions' if kind=='standard' else 'shapes')/panel if panel==NEW else ext.panel_path(panel,kind)
def data_path(panel):return CACHE/'data'/panel if panel==NEW else ext.data_path(panel)
def inputs(panel,kind):
    p=panel_path(panel,kind)
    return np.load(p/('E_DISCOVERY_x.npy' if kind=='standard' else 'x.npy'),mmap_mode='r'),np.load(p/('E_DISCOVERY_sigma.npy' if kind=='standard' else 'sigma.npy'),mmap_mode='r')
def metadata(panel,kind):
    d=np.load(data_path(panel)/'E_DISCOVERY_inputs.npz');nc=len(d['sigma'])
    if kind=='standard':names=STATES;ids=np.arange(128);offset=np.load(panel_path(panel,kind)/'E_DISCOVERY_offset.npy')
    else:
        m=read(panel_path(panel,kind)/'manifest.json');names=m['states'];ids=np.array(m['origin_indices']);offset=np.load(panel_path(panel,kind)/'offset.npy')
    return d,names,ids,nc,offset

def bootstrap_counts(origins,period):
    blocks=np.unique(origins//(7*period));rng=np.random.default_rng(90401)
    cnt=np.stack([np.bincount(rng.integers(0,len(blocks),len(blocks)),minlength=len(blocks)) for _ in range(2000)])
    return cnt[:,np.searchsorted(blocks,origins//(7*period))].astype(float)
def decompose(v):
    A,B,C,D=np.moveaxis(np.asarray(v),-1,0)
    return D-A,.5*((B-A)+(D-C)),.5*((C-A)+(D-B))
