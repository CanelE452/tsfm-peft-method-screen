"""Posthoc diagnosis of frozen C3/MAG functions; no optimizer exists here."""
from experiments.c3_weakness_controls_20260918.common import *
from experiments.c3_weakness_controls_20260918 import common as parent
from experiments.persistence_evidence_extension_20260918 import common as ext
from experiments.c3_identifiability_temporal_20260918.common import metadata
from experiments.additive_persistence_validation_v1_20260917.model import correction_gate
NAME='c3_magnitude_diagnostic_20260918'
OUT=ROOT/'results'/NAME; CACHE=ROOT/'.cache'/NAME; EXP=ROOT/'experiments'/NAME
PANELS=['electricity','electricity_transfer','ettm1']; SEEDS=[81551,81552,81553]
class Watch(ext.Watch):
    def __init__(self):ext.BaseWatch.__init__(self,OUT,'diagnostic',wall_cap=7200)
def check_seal():
    d=read(OUT/'SEAL.json')
    for p,h in d['hashes'].items():assert sha(ROOT/p)==h,('SEALED_CHANGED',p)
    return d

def models():
    return [r for r in read(old.OUT/'MODEL_SELECTION.json') if r['source'] in ['electricity','ettm1'] and r['arm']=='C3']+[r for r in read(parent.OUT/'MODEL_SELECTION.json') if r['arm']=='MAG_ONLY']

def records():
    a=[r for r in read(old.OUT/'PREDICTIONS_MANIFEST.json').values() if r.get('stage')=='selected' and r['panel'] in PANELS and r['arm']=='C3' and r['kind'] in ['standard','shape']]
    b=[r for r in read(parent.OUT/'PREDICTIONS.json').values() if r['arm']=='MAG_ONLY']
    assert len(a)==18 and len(b)==18
    return a+b

def load(row,gate):
    m=parent.build('MAG_ONLY',row['seed'],row['source'])
    assert sha(ROOT/row['checkpoint'])==row['sha256']
    restore(m,torch.load(ROOT/row['checkpoint'],weights_only=True,map_location='cpu'))
    if gate=='C3':m.gate=lambda x,s:correction_gate(x,s)
    else:assert gate=='MAG_ONLY'
    m.requires_grad_(False);return m.eval()

def decomposition(v):
    # A=C3 weights,C3 gate; B=C3 weights,MAG gate;
    # C=MAG weights,C3 gate; D=MAG weights,MAG gate. Error, smaller is better.
    A,B,C,D=np.moveaxis(np.asarray(v),-1,0)
    return A-D,.5*((A-B)+(C-D)),.5*((A-C)+(B-D)),A-B-C+D

def bootstrap(origins,period,panel):
    allorig=np.load(ext.data_path(panel)/'E_DISCOVERY_inputs.npz')['origins']
    blocks=np.unique(allorig//(7*period));rng=np.random.default_rng(918301+PANELS.index(panel))
    counts=np.stack([np.bincount(rng.integers(0,len(blocks),len(blocks)),minlength=len(blocks)) for _ in range(2000)])
    return counts[:,np.searchsorted(blocks,origins//(7*period))].astype(float)
