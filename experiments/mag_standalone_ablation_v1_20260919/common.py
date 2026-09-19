from experiments.additive_persistence_validation_v1_20260917.common import *
from experiments.additive_persistence_validation_v1_20260917 import common as parent
from experiments.additive_persistence_validation_v1_20260917.checks import arrays,batch
from experiments.additive_persistence_validation_v1_20260917.train import validation,predict
from experiments.outlier_signal_peft_v1_20260917.model import loss_2pinball
from experiments.learned_gate_comparison_20260919.common import NEW,STATES,SHAPES,data_path,panel_path,inputs
NAME='mag_standalone_ablation_v1_20260919';OUT=ROOT/'results'/NAME;EXP=ROOT/'experiments'/NAME;CACHE=ROOT/'.cache'/NAME
SOURCES=['electricity','ettm1'];ARMS=['F0_PLAIN','F0_MAG'];SEEDS=[81551,81552];LRS=[1e-4,3e-4]
MAIN_CAP=16384;SMOKE_CAP=8
RENAMES={'B0':'B0','PLAIN':'B0_PLAIN','MAG_ONLY':'B0_MAG'}
ALL_ARMS=['F0','F0_PLAIN','F0_MAG','B0','B0_PLAIN','B0_MAG']
PANELS=['electricity','ettm1','electricity_transfer',NEW]
class Watch(parent.Watch):
    def __init__(self):
        self.last_disk=0;OriginalWatch.__init__(self,OUT,'standalone',wall_cap=28800)
    def limits(self):
        import shutil
        OriginalWatch.limits(self)
        if shutil.disk_usage(ROOT).free<10*2**30:raise ResourceError('DISK_BELOW_10GIB')

def snapshot():return read(ROOT/'results/outlier_signal_peft_v1_20260917/download_receipts.json')['amazon/chronos-bolt-small']
def audit_model(model,arm):
    train=parameters(model);names=list(train)
    modules=[(n,type(m).__name__) for n,m in model.named_modules()]
    badmodules=[(n,c) for n,c in modules if 'lora' in (n+' '+c).lower()]
    badnames=[n for n,_ in model.named_parameters() if 'lora' in n.lower() or n.endswith(('.a','.b'))]
    assert not badmodules and not badnames
    assert not any(p.requires_grad for p in model.base.parameters())
    assert not any(b.requires_grad for b in model.base.buffers())
    assert all(n.startswith('adapter.') for n in names)
    count=sum(p.numel() for p in train.values());assert count==(0 if arm=='F0' else 8712)
    return dict(arm=arm,trainable_names=names,trainable_count=count,lora_parameter_names=badnames,lora_modules=badmodules,base_parameter_keys=list(dict(model.base.named_parameters())),module_types=sorted(set(c for n,c in modules)),foundation_frozen=True,buffers_frozen=True)

def build(arm,seed,source,device='cuda'):
    from chronos import ChronosBoltPipeline
    from .model import PlainForecast,MagnitudeForecast,FoundationForecast
    assert arm in ARMS+['F0'] and source in SOURCES
    receipt=snapshot()
    base=ChronosBoltPipeline.from_pretrained(str(ROOT/receipt['snapshot']),device_map='cpu',torch_dtype=torch.float32,local_files_only=True).model
    base.eval().requires_grad_(False)
    for m in base.modules():
        if isinstance(m,torch.nn.Dropout):m.p=0.
        elif isinstance(getattr(m,'dropout',None),float):m.dropout=0.
    cls={'F0':FoundationForecast,'F0_PLAIN':PlainForecast,'F0_MAG':MagnitudeForecast}[arm]
    model=cls(base,seed).eval().to(device);audit_model(model,arm);return model

def fit_id(source,arm,seed,lr):return f'{source}_{arm}_s{seed}_lr{lr:g}'
def check_seal():
    for p,h in read(OUT/'SEAL.json')['hashes'].items():assert sha(ROOT/p)==h,('SEALED_FILE_CHANGED',p)
def require_authorization():
    a=read(OUT/'AUTHORIZATION.json');assert a['experiment']==NAME and a['main_cap']==MAIN_CAP and a['smoke_cap']==SMOKE_CAP
    assert a['contract_sha256']==sha(EXP/'CONTRACT.txt')
def status(**kw):
    receipts=[read(p) for p in (OUT/'fits').glob('*/receipt.json')]
    n=sum(1 for _ in open(OUT/'UPDATE_LEDGER.jsonl')) if (OUT/'UPDATE_LEDGER.jsonl').exists() else 0
    sn=sum(1 for _ in open(OUT/'SMOKE_LEDGER.jsonl')) if (OUT/'SMOKE_LEDGER.jsonl').exists() else 0
    save(OUT/'status.json',dict(at=time.time(),completed_fits=len(receipts),main_updates=n,smoke_updates=sn,main_cap=MAIN_CAP,automatic_successor=False,joint_training=False,**kw))
