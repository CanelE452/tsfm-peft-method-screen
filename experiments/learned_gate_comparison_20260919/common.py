from experiments.additive_persistence_validation_v1_20260917.common import *
from experiments.additive_persistence_validation_v1_20260917 import common as parent
from experiments.additive_persistence_validation_v1_20260917.checks import arrays,batch
from experiments.additive_persistence_validation_v1_20260917.train import validation,predict
from experiments.outlier_signal_peft_v1_20260917.model import loss_2pinball
NAME='learned_gate_comparison_20260919';OUT=ROOT/'results'/NAME;EXP=ROOT/'experiments'/NAME;CACHE=ROOT/'.cache'/NAME
SOURCES=['electricity','ettm1'];ARMS=['TOKEN_GATE','TOKEN_GATE_ENTROPY'];SEEDS=[81551,81552];LRS=[1e-4,3e-4]
MAIN_CAP=16384;SMOKE_CAP=8
class Watch(parent.Watch):
    def __init__(self):
        self.last_disk=0;OriginalWatch.__init__(self,OUT,'gate',wall_cap=10800)
    def limits(self):
        import shutil
        OriginalWatch.limits(self)
        if shutil.disk_usage(ROOT).free<10*2**30:raise ResourceError('DISK_BELOW_10GIB')
def build(arm,seed,source,device='cuda'):
    from .model import LearnedForecast
    assert arm in ARMS
    b0=parent.build('C0',seed,source,device='cpu')
    model=LearnedForecast(b0.base,arm,seed).eval().to(device)
    assert sum(v.numel() for v in parameters(model).values())==9225
    return model
def fit_id(source,arm,seed,lr):return f'{source}_{arm}_s{seed}_lr{lr:g}'
def check_seal():
    for p,h in read(OUT/'SEAL.json')['hashes'].items():assert sha(ROOT/p)==h,('SEALED_FILE_CHANGED',p)
def status(**kw):
    receipts=[read(p) for p in (OUT/'fits').glob('*/receipt.json')]
    n=sum(1 for _ in open(OUT/'UPDATE_LEDGER.jsonl')) if (OUT/'UPDATE_LEDGER.jsonl').exists() else 0
    save(OUT/'status.json',dict(at=time.time(),completed_fits=len(receipts),main_updates=n,main_cap=MAIN_CAP,automatic_successor=False,**kw))

from experiments.persistence_evidence_extension_20260918 import common as ext
NEW='neso_2026_jul_aug'
STATES=ext.STATES
SHAPES=ext.SHAPES
def data_path(panel):return CACHE/'data'/panel if panel==NEW else ext.data_path(panel)
def panel_path(panel,kind):return CACHE/('conditions' if kind=='standard' else 'shapes')/panel if panel==NEW else ext.panel_path(panel,kind)
def inputs(panel,kind):
    f=panel_path(panel,kind)
    return np.load(f/('E_DISCOVERY_x.npy' if kind=='standard' else 'x.npy'),mmap_mode='r'),np.load(f/('E_DISCOVERY_sigma.npy' if kind=='standard' else 'sigma.npy'),mmap_mode='r')
