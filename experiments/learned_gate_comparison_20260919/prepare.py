import subprocess,warnings
from .common import *
from .data import prepare_data
from .model import LearnedAdapter,binary_entropy

def cpu_check():
    torch.manual_seed(91943)
    a=LearnedAdapter(81550).double();h=torch.randn(3,32,512,dtype=torch.float64)
    plain=parent.build # no backbone construction in this CPU component check
    from experiments.additive_persistence_validation_v1_20260917.model import ResidualAdapter
    b=ResidualAdapter(81550).double()
    for key,value in b.state_dict().items():assert torch.equal(a.state_dict()[key],value)
    assert torch.equal(a(h,None),h)
    assert torch.equal(a.last_gate,torch.full((3,32),.5,dtype=torch.float64))
    assert torch.equal(binary_entropy(a.last_gate),torch.ones_like(a.last_gate))
    with torch.no_grad():a.up.weight.fill_(.01)
    out=a(h,None);out.square().sum().backward()
    assert a.gate_linear.weight.grad.abs().sum()>0
    assert torch.equal(a(h.flip(0),None).flip(0),a(h,None))
    z=torch.tensor([0.,.1,.5,.9,1.],requires_grad=True);v=binary_entropy(z)
    assert torch.isfinite(v).all() and (v>=0).all() and (v<=1).all()
    v.sum().backward();assert torch.isfinite(z.grad).all()
    assert sum(p.numel() for p in a.parameters())==9225
    save(OUT/'CPU_CHECKS.json',dict(initial_identity=True,matched_residual_initialization=True,gate_initial_half=True,normalized_entropy_bound=[0,1],nonzero_gate_gradient_with_nonzero_residual=True,batch_permutation_exact=True,parameters=9225,optimizer_updates=0,actual_backbone_checked=False))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    if (OUT/'SEAL.json').exists():return check_seal()
    assert not (OUT/'UPDATE_LEDGER.jsonl').exists()
    cpu_check();prepare_data()
    save(OUT/'AUTHORIZATION.json',dict(user_message='자동시작해',scope='Previously proposed MAG fixed / two learned gate controls / maximum16 fits and16384+8updates',no_automatic_successor=True))
    prior=ROOT/'results/temporal_response_peft_20260919'
    reuse={k:r for k,r in read(prior/'PREDICTIONS.json').items() if r['arm'] in ['B0','PLAIN','C3','MAG_ONLY']};assert len(reuse)==96
    models=[r for r in read(prior/'MODEL_SELECTION.json') if r['arm'] in ['B0','PLAIN','C3','MAG_ONLY'] and r['source']=='electricity'];assert len(models)==16
    for r in models:assert sha(ROOT/r['checkpoint'])==r['sha256']
    for r in reuse.values():assert sha(ROOT/r['path'])==r['sha256']
    save(OUT/'REUSED_PREDICTIONS.json',reuse);save(OUT/'REUSED_MODELS.json',models)
    hashes={}
    def add(p):hashes[str(p.relative_to(ROOT))]=sha(p)
    for p,h in read(ROOT/'results/delta_adapter_comparison_20260919/SEAL.json')['hashes'].items():assert sha(ROOT/p)==h;hashes[p]=h
    for folder in [EXP,ROOT/'experiments/c3_training_factorial_20260918',ROOT/'experiments/c3_weakness_controls_20260918',ROOT/'experiments/temporal_response_peft_20260919',ROOT/'experiments/c3_identifiability_temporal_20260918']:
        for p in folder.glob('*'):
            if p.suffix in ['.py','.md']:add(p)
    for p in CACHE.rglob('*'):
        if p.is_file():add(p)
    for r in models:add(ROOT/r['checkpoint'])
    for p in OUT.glob('*'):
        if p.is_file():add(p)
    add(ROOT/'research/gated_method_claim_review_20260919/PERIOD_CAPACITY.json')
    add(ROOT/'research/gated_method_claim_review_20260919/PRIOR_REVIEW.json')
    add(ROOT/'.cache/c3_identifiability_temporal_20260918/raw/neso_2026.csv')
    save(OUT/'SEAL.json',dict(at=time.time(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),hashes=hashes,main_cap=MAIN_CAP,smoke_cap=SMOKE_CAP,old_checkpoint_retrained=False))
    status(execution='SEALED')

def smoke(watch):
    if (OUT/'SMOKE.json').exists():return
    assert not (OUT/'SMOKE_LEDGER.jsonl').exists(),'Partial smoke must not silently repeat'
    rows=[]
    for source in SOURCES:
        a=arrays(source);x,y,s=batch(a,'C2',0,np.arange(32))
        for arm in ARMS:
            watch.boundary();m=build(arm,81550,source);initial=cpu_state(m);before=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()))
            with torch.no_grad():p=m(x,s);b=m(x,s,off=True)
            base=parent.build('C0',81550,source)
            with torch.no_grad():assert torch.equal(b,base(x,s)) and torch.equal(p,b)
            del base;cleanup();opt=torch.optim.AdamW(parameters(m).values(),lr=1e-4,weight_decay=0)
            for step in range(2):
                begin=watch.before();opt.zero_grad(set_to_none=True);loss=loss_2pinball(m(x,s),y,s,m.base.quantiles)+m.regularizer();loss.backward()
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters(m).values())
                norm=torch.nn.utils.clip_grad_norm_(parameters(m).values(),1.);assert torch.isfinite(norm);opt.step()
                _,bad=watch.after(begin);assert not bad
                append(OUT/'SMOKE_LEDGER.jsonl',dict(source=source,arm=arm,step=step+1,loss=float(loss.detach())))
            assert tensor_hash(initial)!=tensor_hash(cpu_state(m)) and frozen_hash(m)==before
            assert buffers==tensor_hash(dict(m.named_buffers()))
            state=cpu_state(m)
            assert not torch.equal(state['adapter.gate_linear.weight'],initial['adapter.gate_linear.weight'])
            with torch.no_grad():trained=m(x,s)
            restore(m,initial)
            with torch.no_grad():assert torch.equal(m(x,s),p)
            fresh=build(arm,81550,source);restore(fresh,state)
            with torch.no_grad():assert torch.equal(trained,fresh(x,s))
            rows.append(dict(source=source,arm=arm,updates=2,trainable_parameters=sum(v.numel() for v in initial.values()),gate_updated=True,buffers_unchanged=True,frozen_unchanged=True,off_exact_B0=True,restore_exact=True,initial_is_B0=bool(torch.equal(p,b))))
            del fresh,m,opt;cleanup()
    assert sum(r['updates'] for r in rows)==SMOKE_CAP;save(OUT/'SMOKE.json',rows)

if __name__=='__main__':
    setup();warnings.filterwarnings('ignore',message="input's size at dim=0.*");prepare()
