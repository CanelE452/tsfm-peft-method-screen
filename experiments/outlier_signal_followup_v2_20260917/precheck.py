"""Zero optimizer eligibility: actual model parity and deterministic input-only gates."""
from .common import *
from .model import preprocess
from .prepare import OLD,OLDC,audit
from experiments.outlier_signal_peft_v1_20260917.common import build as old_build

@torch.no_grad()
def precheck(watch):
    if (OUT/'pretraining_checks.json').exists():return
    audit()
    for info in read(OUT/'download_receipts.json').values():
        for name,h in info['files'].items():assert sha(ROOT/name)==h
    rows=[];parameter={};hashes=[]
    for source in SOURCES:
        f=CACHE/'conditions'/source;x=torch.tensor(np.load(f/'V_SELECT_x.npy',mmap_mode='r')[-8:].copy(),device='cuda');s=torch.tensor(np.load(f/'V_SELECT_sigma.npy',mmap_mode='r')[-8:].copy(),device='cuda')
        for arm in ARMS:
            watch.boundary();model=build(arm,81550);state=cpu_state(model)
            lora={n:v for n,v in state.items() if n!='gate' and not n.startswith('adapter.')};hashes.append(tensor_hash(lora))
            count=sum(v.numel() for v in state.values());parameter[arm]=dict(trainable=count,extra=count-294912,shapes={n:list(v.shape) for n,v in state.items()},gate_initial=state['gate'].tolist() if 'gate' in state else None)
            a=model(x,s);assert torch.isfinite(a).all()
            assert torch.equal(a,model(x,s)) # hypothetical y is outside model signature
            diff=None
            if arm=='B0':
                native=model.base(context=x).quantile_preds
                diff=float(((a-native)/s[:,None,None]).abs().max());assert diff<=1e-5 or torch.allclose(a,native,rtol=1e-4,atol=0)
            if arm in ['B1','B2']:
                old=old_build('A2' if arm=='B1' else 'A5',81550);b=old(x,s)
                diff=float(((a-b)/s[:,None,None]).abs().max());assert torch.equal(a,b)
                if arm=='B2':assert tensor_hash(state)==tensor_hash(cpu_state(old))
                del old
            if arm in ['B3','B4','B5']:
                eff,g,p,e=model.transform(x,s);assert ((g>=0)&(g<=1)).all() and torch.isfinite(eff).all()
            rows.append(dict(source=source,arm=arm,trainable=count,parity_max_normalized_error=diff,input_only=True,optimizer_updates=0))
            del model;cleanup()
    assert len(set(hashes))==1
    save(OUT/'PARAMETER_RECEIPT.json',parameter)
    save(OUT/'pretraining_checks.json',dict(rows=rows,optimizer_updates=0,initial_lora_common=True,all_model_weights_hash_verified=True,old_origins_exact=True))
