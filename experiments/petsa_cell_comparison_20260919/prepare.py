"""Hash-only preparation followed by separately authorized GPU smoke."""
import subprocess
from .common import *

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    if (OUT/'SEAL.json').exists():return check_seal()
    assert not (OUT/'UPDATE_LEDGER.jsonl').exists()
    cpu=read(OUT/'CPU_CHECKS.json');assert cpu['status']=='CPU_ACTUAL_MODEL_WIRING_VERIFIED'
    assert len(cpu['actual_Chronos_cases'])==2 and cpu['optimizer_updates']==0
    previous=ROOT/'results/learned_gate_comparison_20260919'
    assert read(previous/'AUDIT.json')['status']=='VERIFIED'
    hashes={}
    def add(p):hashes[str(p.relative_to(ROOT))]=sha(p)
    for name,h in read(previous/'SEAL.json')['hashes'].items():
        assert sha(ROOT/name)==h,('OLD_SEAL_CHANGED',name)
        hashes[name]=h
    for r in read(ROOT/'research/method_baseline_compatibility_20260919/PETSA_CODE_RECEIPTS.json'):
        assert sha(ROOT/r['local'])==r['sha256'];add(ROOT/r['local'])
    reuse=read(previous/'PREDICTIONS.json');assert len(reuse)==192
    for r in reuse.values():assert sha(ROOT/r['path'])==r['sha256'];add(ROOT/r['path'])
    save(OUT/'REUSED_PREDICTIONS.json',reuse)
    for p in EXP.glob('*'):
        if p.suffix in ['.py','.md']:add(p)
    for p in [OUT/'CPU_CHECKS.json',OUT/'REUSED_PREDICTIONS.json',previous/'AUDIT.json',previous/'RAW_SCORES.csv',previous/'PREDICTIONS.json']:add(p)
    save(OUT/'SEAL.json',dict(at=time.time(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),hashes=hashes,main_cap=MAIN_CAP,smoke_cap=SMOKE_CAP,authorization_required=True,all_evaluation_panels_previously_scored=True))
    status(execution='PREPARED_NOT_AUTHORIZED')

def smoke(watch):
    require_authorization();check_seal()
    if (OUT/'SMOKE.json').exists():
        rows=read(OUT/'SMOKE.json');assert len(rows)==2 and sum(r['updates'] for r in rows)==SMOKE_CAP
        return
    assert not (OUT/'SMOKE_LEDGER.jsonl').exists() and not (OUT/'SMOKE_INTENT.json').exists(),'Partial smoke must not silently repeat'
    rows=[]
    for source in SOURCES:
        a=arrays(source);x,y,s=batch(a,ARMS[0],0,np.arange(32))
        watch.boundary();m=build(ARMS[0],81550,source);initial=cpu_state(m);before=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()))
        with torch.no_grad():p=m(x,s);b=m(x,s,off=True)
        assert torch.equal(p,b)
        opt=torch.optim.AdamW(parameters(m).values(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        for step in range(2):
            begin=watch.before();opt.zero_grad(set_to_none=True);loss=loss_2pinball(m(x,s),y,s,m.base.quantiles)
            assert torch.isfinite(loss);loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters(m).values())
            norm=torch.nn.utils.clip_grad_norm_(parameters(m).values(),1.);assert torch.isfinite(norm)
            save(OUT/'SMOKE_INTENT.json',dict(source=source,step=step+1,status='BEFORE_OPTIMIZER'))
            opt.step();sync()
            append(OUT/'SMOKE_LEDGER.jsonl',dict(source=source,arm=ARMS[0],step=step+1,loss=float(loss.detach())))
            save(OUT/'SMOKE_INTENT.json',dict(source=source,step=step+1,status='JOURNALED'))
            _,bad=watch.after(begin);assert not bad
        state=cpu_state(m)
        assert tensor_hash(initial)!=tensor_hash(state) and frozen_hash(m)==before
        assert buffers==tensor_hash(dict(m.named_buffers()))
        for name in ['in_cali.lora_A','in_cali.lora_B','in_cali.gating','out_cali.lora_A','out_cali.lora_B','out_cali.gating']:
            assert not torch.equal(initial[name],state[name]),('UNUPDATED_CELL_PARAMETER',name)
        with torch.no_grad():trained=m(x,s)
        fresh=build(ARMS[0],81550,source);restore(fresh,state)
        with torch.no_grad():assert torch.equal(trained,fresh(x,s))
        rows.append(dict(source=source,updates=2,initial_B0_exact=True,trainable_parameters=19010,frozen_unchanged=True,buffers_unchanged=True,fresh_restore_exact=True,gates_updated=True))
        del m,fresh,opt;cleanup()
    assert sum(r['updates'] for r in rows)==SMOKE_CAP;save(OUT/'SMOKE.json',rows)

if __name__=='__main__':prepare()
