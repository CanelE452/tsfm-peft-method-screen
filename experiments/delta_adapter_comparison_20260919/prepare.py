import ast,copy,subprocess,warnings
from .common import *
from .model import DeltaNet

def cpu_check():
    path=CACHE/'prior/exp_online_xy_add.py'
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PostProcessingNet')
    namespace={'torch':torch,'F':torch.nn.functional}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    rows=[]
    for length,width in [(512,7),(64,7),(512,512),(64,512)]:
        torch.manual_seed(91941)
        official=namespace['PostProcessingNet'](length,width,length,.1).double().eval()
        ours=DeltaNet(length,width).double().eval()
        with torch.no_grad():
            for a,b in zip(ours.layers,[official.fc1,official.fc2,official.fc3]):a.load_state_dict(b.state_dict())
        x=torch.randn(3,length,dtype=torch.float64,requires_grad=True)
        a=official(x);b=ours(x);assert torch.equal(a,b)
        ga=torch.autograd.grad(a.square().sum(),x)[0];gb=torch.autograd.grad(b.square().sum(),x)[0];assert torch.equal(ga,gb)
        assert torch.equal(ours(x[[2,0,1]]),b[[2,0,1]])
        assert float(b.abs().max())<=.1
        rows.append(dict(length=length,width=width,output_exact=True,input_gradient_exact=True,batch_permutation_exact=True))
    save(OUT/'CPU_CELL_PARITY.json',dict(reference_commit='0add06ea7b4d2e0a84c364a8be72eef2676a92f2',cases=rows,optimizer_updates=0))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    if (OUT/'SEAL.json').exists():return check_seal()
    assert not (OUT/'UPDATE_LEDGER.jsonl').exists()
    cpu_check();hashes={}
    def add(p):hashes[str(p.relative_to(ROOT))]=sha(p)
    for p in EXP.glob('*'):
        if p.suffix in ['.py','.md']:add(p)
    for folder in ['additive_persistence_validation_v1_20260917','additive_b0_adapter_v1_20260917','outlier_signal_followup_v2_20260917','outlier_signal_peft_v1_20260917','persistence_evidence_extension_20260918']:
        for p in (ROOT/'experiments'/folder).glob('*.py'):add(p)
    add(ROOT/'scripts/priority12/common.py')
    prior=read(CACHE/'prior/receipts.json');save(OUT/'PRIOR_CODE_RECEIPTS.json',prior)
    for r in prior:assert sha(ROOT/r['local'])==r['sha256'];add(ROOT/r['local'])
    add(OUT/'PRIOR_CODE_RECEIPTS.json')
    for source in SOURCES:
        for f in ['train_x.npy','train_y.npy','train_sigma.npy','V_SELECT_x.npy','V_SELECT_y.npy','V_SELECT_sigma.npy']:
            p=parent.CACHE/'conditions'/source/f
            assert sha(p)==sha(ROOT/'.cache/additive_b0_adapter_v1_20260917/conditions'/source/f)
            add(p)
        for seed in [81550]+SEEDS:
            r=parent.baseline_row(source,seed);assert sha(ROOT/r['checkpoint'])==r['sha256'];add(ROOT/r['checkpoint'])
    add(parent.OUT/'BASELINE_MANIFEST.json');add(parent.OUT/'DATA_MANIFEST.json')
    for receipt in read(ROOT/'results/outlier_signal_followup_v2_20260917/download_receipts.json').values():
        if isinstance(receipt,dict):
            for p,h in receipt.get('files',{}).items():assert sha(ROOT/p)==h;add(ROOT/p)
    from experiments.persistence_evidence_extension_20260918 import common as ext
    for panel in ['electricity','electricity_transfer','ettm1']:
        for kind in ['standard','shape']:
            for p in ext.panel_path(panel,kind).iterdir():
                if (p.name.startswith('E_DISCOVERY') or kind=='shape') and p.suffix in ['.npy','.json']:add(p)
        for p in ext.data_path(panel).glob('E_DISCOVERY*.npz'):add(p)
    prior_predictions=ROOT/'results/temporal_response_peft_20260919/PREDICTIONS.json';add(prior_predictions)
    reuse={k:r for k,r in read(prior_predictions).items() if r['arm'] in ['B0','PLAIN','C3','MAG_ONLY']}
    assert len(reuse)==96
    for r in reuse.values():assert sha(ROOT/r['path'])==r['sha256'];assert r['source']==('electricity' if r['panel'].startswith('electricity') else 'ettm1')
    save(OUT/'REUSED_PREDICTIONS.json',reuse);add(OUT/'REUSED_PREDICTIONS.json')
    save(OUT/'SEAL.json',dict(at=time.time(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),hashes=hashes,main_cap=MAIN_CAP,smoke_cap=SMOKE_CAP,old_checkpoint_retrained=False))
    status(execution='SEALED')

def smoke(watch):
    if (OUT/'SMOKE.json').exists():return
    assert not (OUT/'SMOKE_LEDGER.jsonl').exists(),'Partial smoke must not silently repeat'
    rows=[]
    for source in SOURCES:
        a=arrays(source);x,y,s=batch(a,'C2',0,np.arange(32))
        for arm in ARMS:
            watch.boundary();m=build(arm,81550,source);initial=cpu_state(m);before=frozen_hash(m)
            with torch.no_grad():p=m(x,s);b=m(x,s,off=True)
            base=parent.build('C0',81550,source)
            with torch.no_grad():assert torch.equal(b,base(x,s))
            del base;cleanup();opt=torch.optim.AdamW(parameters(m).values(),lr=1e-4,weight_decay=0)
            for step in range(2):
                begin=watch.before();opt.zero_grad(set_to_none=True);loss=loss_2pinball(m(x,s),y,s,m.base.quantiles);loss.backward()
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters(m).values())
                norm=torch.nn.utils.clip_grad_norm_(parameters(m).values(),1.);assert torch.isfinite(norm);opt.step()
                _,bad=watch.after(begin);assert not bad
                append(OUT/'SMOKE_LEDGER.jsonl',dict(source=source,arm=arm,step=step+1,loss=float(loss.detach())))
            assert tensor_hash(initial)!=tensor_hash(cpu_state(m)) and frozen_hash(m)==before
            state=cpu_state(m)
            with torch.no_grad():trained=m(x,s)
            restore(m,initial)
            with torch.no_grad():assert torch.equal(m(x,s),p)
            fresh=build(arm,81550,source);restore(fresh,state)
            with torch.no_grad():assert torch.equal(trained,fresh(x,s))
            rows.append(dict(source=source,arm=arm,updates=2,trainable_parameters=sum(v.numel() for v in initial.values()),frozen_unchanged=True,off_exact_B0=True,restore_exact=True,initial_is_B0=bool(torch.equal(p,b))))
            del fresh,m,opt;cleanup()
    assert sum(r['updates'] for r in rows)==SMOKE_CAP;save(OUT/'SMOKE.json',rows)

if __name__=='__main__':
    setup();warnings.filterwarnings('ignore',message="input's size at dim=0.*");prepare()
