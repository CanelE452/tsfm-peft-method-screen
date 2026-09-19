from experiments.additive_persistence_validation_v1_20260917.common import *
from experiments.additive_persistence_validation_v1_20260917 import common as parent
from experiments.additive_persistence_validation_v1_20260917.checks import arrays,batch
from experiments.additive_persistence_validation_v1_20260917.train import validation,predict
from experiments.outlier_signal_peft_v1_20260917.model import loss_2pinball
NAME='petsa_cell_comparison_20260919';OUT=ROOT/'results'/NAME;EXP=ROOT/'experiments'/NAME;CACHE=ROOT/'.cache'/NAME
SOURCES=['electricity','ettm1'];ARMS=['PETSA_XY_OFFLINE'];SEEDS=[81551,81552];LRS=[1e-4,3e-4]
MAIN_CAP=8192;SMOKE_CAP=4
class Watch(parent.Watch):
    def __init__(self):
        self.last_disk=0;OriginalWatch.__init__(self,OUT,'petsa',wall_cap=10800)
    def limits(self):
        import shutil
        OriginalWatch.limits(self)
        if shutil.disk_usage(ROOT).free<10*2**30:raise ResourceError('DISK_BELOW_10GIB')
def build(arm,seed,source,device='cuda'):
    from .model import PetsaForecast
    b0=parent.build('C0',seed,source,device='cpu')
    assert arm in ARMS
    model=PetsaForecast(b0,seed).eval().to(device)
    assert sum(p.numel() for p in parameters(model).values())==19010
    return model
def fit_id(source,arm,seed,lr):return f'{source}_{arm}_s{seed}_lr{lr:g}'
def check_seal():
    for p,h in read(OUT/'SEAL.json')['hashes'].items():assert sha(ROOT/p)==h,('SEALED_FILE_CHANGED',p)
def status(**kw):
    receipts=[read(p) for p in (OUT/'fits').glob('*/receipt.json')]
    n=sum(1 for _ in open(OUT/'UPDATE_LEDGER.jsonl')) if (OUT/'UPDATE_LEDGER.jsonl').exists() else 0
    save(OUT/'status.json',dict(at=time.time(),completed_fits=len(receipts),main_updates=n,main_cap=MAIN_CAP,automatic_successor=False,**kw))

from experiments.learned_gate_comparison_20260919.common import NEW,STATES,SHAPES,data_path,panel_path,inputs

def require_authorization():
    p=OUT/'AUTHORIZATION.json'
    if not p.exists():
        raise PermissionError('No user authorization for the additional 8 fits / 8192+4 updates')
    a=read(p)
    assert a['scope']=='PETSA_XY_OFFLINE_8_FITS' and a['main_cap']==MAIN_CAP and a['smoke_cap']==SMOKE_CAP
    assert a['user_message'] and a['protocol_sha256']==sha(EXP/'PROTOCOL.md')
