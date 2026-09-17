from pathlib import Path
import json,time,hashlib,os
import numpy as np
import torch
from experiments.additive_persistence_validation_v1_20260917 import common as old
from experiments.additive_persistence_validation_v1_20260917.evaluate import load_model
from experiments.additive_persistence_validation_v1_20260917.train import predict
from experiments.additive_persistence_validation_v1_20260917.prepare import STATES,SHAPES
from experiments.outlier_signal_peft_v1_20260917.reference_core import select_days_reference,pinball_scalar
from priority12.common import read,save,sha,csvwrite,tensor_hash,Watch as BaseWatch,ResourceError
ROOT=old.ROOT
NAME='persistence_evidence_extension_20260918'
OUT=ROOT/'results'/NAME;CACHE=ROOT/'.cache'/NAME;EXP=ROOT/'experiments'/NAME
PRIOR=old.OUT;PC=old.CACHE
PANELS=['electricity','electricity_transfer','ettm1','ettm2','neso_2025']
SWAPS={'C3_W_RECENCY_G':('C3','M_RECENCY'),'RECENCY_W_C3_G':('M_RECENCY','C3')}
ARMS=['C0','C1','C2','C3']+old.CONTROLS
class Watch(BaseWatch):
    def __init__(self):super().__init__(OUT,'inference',wall_cap=10800)
    def sample(self):
        r=super().sample()
        for a in r['apps']:a['allowed_desktop']=a['name']=='/usr/share/rustdesk/rustdesk'
        r['busy']=any(not a['own'] and not a['allowed_desktop'] for a in r['apps']) or r['free_mib']<1024
        return r
    def limits(self):
        import shutil
        super().limits()
        if shutil.disk_usage(ROOT).free<10*2**30:raise ResourceError('DISK_BELOW_10GIB')
def panel_path(panel,kind):return (CACHE if panel=='neso_2025' else PC)/('conditions' if kind=='standard' else 'shapes')/panel
def data_path(panel):return (CACHE if panel=='neso_2025' else PC)/'data'/panel
def inputs(panel,kind):
    f=panel_path(panel,kind)
    return np.load(f/('E_DISCOVERY_x.npy' if kind=='standard' else 'x.npy'),mmap_mode='r'),np.load(f/('E_DISCOVERY_sigma.npy' if kind=='standard' else 'sigma.npy'),mmap_mode='r')
def check_seal():
    d=read(OUT/'SEAL.json')
    for p,h in d['hashes'].items():assert sha(ROOT/p)==h,('SEALED_FILE_CHANGED',p)
    return d
