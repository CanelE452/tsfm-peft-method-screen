"""Zero-update local backward diagnosis with a fixed conditional pilot gate."""
import fcntl,gc,json,subprocess,time
from contextlib import nullcontext,ExitStack
import numpy as np
import torch
from tsfm_peft_screen.memory.common import build,batch,forward,gradient,params,zero,ROOT,CACHE as WARM
from tsfm_peft_screen.memory.local_backward import local_backward
from tsfm_peft_screen.memory.compression import checkpoint_blocks
from tsfm_peft_screen.backbone import native_loss
from tsfm_peft_screen.lora import restore,snapshot
from tsfm_peft_screen.reproducibility import sha,write_json,source_hashes,guard
OUT=ROOT/'results/local_backward_feasibility';CACHE=ROOT/'.cache/local_backward_feasibility'
CFG=json.loads((ROOT/'configs/local_backward_feasibility.json').read_text());METHODS=CFG['methods']

def compare(actual,reference,label):
    names=[n for n in reference if label=='all' or n.endswith('lora_'+label)]
    a=torch.cat([actual[n].flatten().double() for n in names]);r=torch.cat([reference[n].flatten().double() for n in names])
    assert r.norm()>0 and torch.isfinite(a).all()
    return dict(relative_l2=float((a-r).norm()/r.norm()),cosine=float(torch.nn.functional.cosine_similarity(a,r,dim=0)),reference_norm=float(r.norm()))

def activate(stack,m,method,seed):
    if method in ('checkpoint','amp_checkpoint'):stack.enter_context(checkpoint_blocks(m))
    if method.startswith('amp'):stack.enter_context(torch.autocast('cuda',dtype=torch.bfloat16))
    if method in ['local_fp16','care','prac','residual','all_details']:
        return stack.enter_context(local_backward(m,'fp16' if method=='local_fp16' else method,seed))
    return None

def run(m,x,y,g,method,seed,ref=None,expected=None,keep=False):
    zero(m);x.grad=None;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
    with ExitStack() as stack:
        codecs=activate(stack,m,method,seed)
        z,p,l,s=forward(m,x,g);loss=native_loss(z,y,l,s);loss.backward()
    torch.cuda.synchronize();elapsed=time.monotonic()-tick;peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    actual=gradient(m);assert all(torch.isfinite(v).all() for v in actual.values())
    out=dict(method=method,seed=seed,seconds=elapsed,peak_allocated_bytes=peak,peak_reserved_bytes=reserved,loss=float(loss.detach()),compression=codecs.stats if codecs else None)
    if ref is not None:
        out['gradient']={label:compare(actual,ref,label) for label in ['all','A','B']}
        out['forward_max_abs_error']=float((z.detach().cpu()-expected).abs().max())
        if not method.startswith('amp'):
            assert out['forward_max_abs_error']==0,(method,'forward')
            assert out['gradient']['B']['relative_l2']<=1e-5,(method,'B parity',out['gradient']['B'])
        if method in ['standard','checkpoint','all_details']:assert out['gradient']['A']['relative_l2']<=1e-5
    return out,actual,z.detach().cpu(),x.grad.detach().cpu().clone() if x.grad is not None else None

lock=open(ROOT/'.cache/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not OUT.exists(),'Immutable diagnostic directory'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'Commit before running'
idle=None
while True:
    output=subprocess.check_output([str(ROOT/'scripts/with_cuda.sh'),'nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if output.strip():idle=None;print('WAIT external GPU job',flush=True)
    else:
        if idle is None:idle=time.monotonic()
        if time.monotonic()-idle>=30:break
    time.sleep(10)
OUT.mkdir();CACHE.mkdir();start=time.monotonic();rows=[];audits=[];completed_passes=0
contract=dict(execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hashes=source_hashes(),config=CFG,config_sha256=sha(ROOT/'configs/local_backward_feasibility.json'),warm_hashes={n:sha(WARM/f'{n}_warm.pt') for n in CFG['datasets']},data_hashes={n:sha(ROOT/'data/processed'/n/'fit.npz') for n in CFG['datasets']},data_scope='fit.npz only; no V/E arrays',optimizer_states='FP32 AdamW exp_avg and exp_avg_sq zero-initialized and resident for every arm, including AMP; no optimizer steps; measured forward/backward peaks, not optimizer-step peak',amp='autocast BF16 with FP32 parameters and optimizer state; compare approximation against FP32 reference, not exact parity',storage='QKV inputs share codecs when backing storage and views match. CARE depends on A and does not share decoder. Payload sums are cumulative unique encoding costs, not peak; CARE Z may alias exact B saved Z.',no_full_reconstructed_x='residual,CARE,PRAC form grad_A directly; local_fp16 reconstructs its local input only',timing='three rotated-order repeats, stochastic arms16 independent codec seeds per case total; all costs included; CPU metric/replay-cache writes excluded')
write_json(OUT/'contract.json',contract)
try:
    for name in CFG['datasets']:
        m=build();restore(torch.load(WARM/f'{name}_warm.pt',weights_only=True),m);initial=snapshot(m)
        opt=torch.optim.AdamW(params(m),lr=1e-4,weight_decay=0)
        for p in params(m):opt.state[p]={'step':torch.tensor(0.),'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.zeros_like(p)}
        # Independent integration parity pass with input gradients enabled.
        x,y,g=batch(name,1024);x.requires_grad_(True)
        _,reference,expected,dx=run(m,x,y,g,'standard',52000);completed_passes+=1
        for method in ['all_details','residual','care','prac','local_fp16']:
            row,_,_,actual_dx=run(m,x,y,g,method,52000,reference,expected);completed_passes+=1
            error=float((actual_dx.double()-dx.double()).norm()/dx.double().norm());assert error<=1e-5
            audits.append(dict(dataset=name,method=method,input_gradient_relative_l2=error,A_relative_l2=row['gradient']['A']['relative_l2'],B_relative_l2=row['gradient']['B']['relative_l2'],forward_max_abs_error=row['forward_max_abs_error']))
        write_json(OUT/'integration_parity.json',audits)
        for length in CFG['contexts']:
            x,y,g=batch(name,length)
            _,reference,expected,_=run(m,x,y,g,'standard',52000);completed_passes+=1
            torch.save(reference,CACHE/f'{name}_{length}_reference.pt')
            case={method:[] for method in METHODS}
            for repeat in range(16):
                order=METHODS[repeat:]+METHODS[:repeat] if repeat<3 else ['prac','residual']
                for method in order:
                    seed=53000+repeat
                    row,actual,_,_=run(m,x,y,g,method,seed,reference,expected);completed_passes+=1
                    case[method].append(row)
                    if method in ['prac','residual']:
                        torch.save({n:v for n,v in actual.items() if n.endswith('lora_A')},CACHE/f'{name}_{length}_{method}_{seed}.pt')
                    elif repeat==0:torch.save(actual,CACHE/f'{name}_{length}_{method}.pt')
                    guard(start)
                    if repeat<3 or repeat==15:print('LOCAL',name,length,repeat,method,'MiB',round(row['peak_allocated_bytes']/2**20,1),'Aerr',round(row['gradient']['A']['relative_l2'],5),flush=True)
            for method,runs in case.items():
                record=dict(dataset=name,context=length,method=method,peak_allocated_bytes=int(np.median([v['peak_allocated_bytes'] for v in runs[:3]])),median_seconds=float(np.median([v['seconds'] for v in runs[:3]])),A_error_rms=float(np.sqrt(np.mean([v['gradient']['A']['relative_l2']**2 for v in runs]))),repeats=runs)
                if method in ['prac','residual']:
                    names=[n for n in reference if n.endswith('lora_A')];r=torch.cat([reference[n].flatten().double() for n in names]);mean=torch.zeros_like(r);norms=[]
                    for item in runs:
                        saved=torch.load(CACHE/f"{name}_{length}_{method}_{item['seed']}.pt",weights_only=True)
                        v=torch.cat([saved[n].flatten().double() for n in names]);mean+=v/len(runs);norms.append(float(v.square().sum()))
                    record['mean_A_relative_error']=float((mean-r).norm()/r.norm())
                    record['A_centered_variance_relative']=max(0.,(np.mean(norms)-float(mean.square().sum()))/float(r.square().sum()))
                rows.append(record)
            write_json(OUT/'metrics.json',rows)
        after=snapshot(m);assert all(torch.equal(initial[n],after[n]) for n in initial)
        del m,opt,x,y,g,initial,after;gc.collect();torch.cuda.empty_cache()
    assert source_hashes()==contract['source_hashes']
    decisions=[]
    for name in CFG['datasets']:
        r={v['method']:v for v in rows if v['dataset']==name and v['context']==4096};v=r['residual'];standard=r['standard']
        reduction=1-v['peak_allocated_bytes']/standard['peak_allocated_bytes'];ratio=v['median_seconds']/standard['median_seconds'];errorratio=v['A_error_rms']/r['local_fp16']['A_error_rms']
        dominated=any(r[k]['peak_allocated_bytes']<=v['peak_allocated_bytes'] and r[k]['median_seconds']<=v['median_seconds'] for k in ['checkpoint','amp_checkpoint'])
        passed=reduction>=.05 and ratio<=1.15 and errorratio<=1 and not dominated
        decisions.append(dict(dataset=name,peak_reduction=reduction,time_ratio=ratio,A_rms_error_ratio_vs_local_fp16=errorratio,dominated_by_checkpoint=dominated,pass_gate=passed))
    write_json(OUT/'status.json',dict(status='COMPLETE',verdict='PILOT_ELIGIBLE' if all(d['pass_gate'] for d in decisions) else 'STOP',decisions=decisions,completed_backward_passes=completed_passes,optimizer_updates=0,parameter_state_unchanged=True,e_access=False,wall_seconds=time.monotonic()-start))
except Exception as e:
    write_json(OUT/'status.json',dict(status='IMPLEMENTATION_BLOCKED',error=str(e),completed_backward_passes=completed_passes,completed_cases=len(rows),optimizer_updates=0));raise
