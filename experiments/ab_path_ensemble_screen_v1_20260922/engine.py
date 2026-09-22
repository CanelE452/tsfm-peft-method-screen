import csv,random,time
import numpy as np
import torch
from common import *
from model import weighted_crps,cdf_l2,tensor_hash

class Ledger:
    def __init__(self,candidate):
        self.c=candidate;self.path=RESULTS/candidate/'BUDGET_STATE.json'
        self.state=read(self.path) if self.path.exists() else dict(main_updates=0,smoke_updates=0,main_fits=0,fits={})
    def log(self,kind,**kw):
        with (RESULTS/self.c/'UPDATE_LEDGER.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(utc=time.time(),kind=kind,**kw))+'\n');f.flush()
        save(self.path,self.state)
    def start(self,key,phase,steps):
        assert key not in self.state['fits'],'Existing fit cannot be replayed'
        if phase=='main':
            assert self.state['main_fits']<12;self.state['main_fits']+=1
        self.state['fits'][key]=dict(phase=phase,planned=steps,intent=0,commit=0,status='RUNNING')
        self.log('fit_start',key=key,phase=phase,steps=steps)
    def intent(self,key):
        f=self.state['fits'][key];assert f['intent']==f['commit'] and f['intent']<f['planned']
        counter=f['phase']+'_updates';cap=3072 if f['phase']=='main' else 12
        assert self.state[counter]<cap;self.state[counter]+=1;f['intent']+=1
        self.log('update_intent',key=key,step=f['intent'])
    def commit(self,key):
        f=self.state['fits'][key];assert f['intent']==f['commit']+1;f['commit']+=1
        self.log('update_commit',key=key,step=f['commit'])
    def finish(self,key):
        f=self.state['fits'][key];assert f['commit']==f['planned'];f['status']='COMPLETE';self.log('fit_complete',key=key)

def tensors(packet):return {k:torch.as_tensor(v,device='cuda',dtype=torch.float32) for k,v in packet.items() if k in ['x','y','sigma','weather','control','first','teacher_z','teacher_p']}

def distribution(model,b,c):
    return model.distribution(b['x'],b.get('first')) if c=='A' else model.distribution(b['x'],b.get('weather'),b.get('control'),b['sigma'])

def objective(model,b,c):
    z,p,_=distribution(model,b,c)
    if c=='A':
        z=z[:,64:];p=p[:,64:];y=b['y'][:,64:]
    else:y=b['y']
    loss=(weighted_crps(z,p,y)/b['sigma'][:,None]).mean()
    if 'teacher_z' in b:
        loss=loss+.5*(cdf_l2(z,p,b['teacher_z'].detach(),b['teacher_p'].detach())/b['sigma'][:,None]).mean()
    return loss

def optimizer(model):
    base=[];module=[]
    for n,p in model.named_parameters():
        if p.requires_grad:(base if n.startswith('network.') else module).append(p)
    return torch.optim.AdamW([{'params':base,'lr':1e-4},{'params':module,'lr':3e-4}],betas=(.9,.999),eps=1e-8,weight_decay=0.)

def fit(model,c,key,packets,phase='main',validation=None):
    book=Ledger(c);book.start(key,phase,len(packets));folder=CACHE/c/'fits'/key;folder.mkdir(parents=True,exist_ok=False)
    opt=optimizer(model);initial=model.learned();frozen=model.frozen_hash();curves={};cps={0:initial};trace=[]
    if validation is not None:curves[0]=validation(model)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();fit_seconds=0
    for step,packet in enumerate(packets,1):
        tick=time.perf_counter();model.eval();opt.zero_grad(set_to_none=True);b=tensors(packet);loss=objective(model,b,c)
        assert torch.isfinite(loss);loss.backward()
        gradients={n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.named_parameters() if p.requires_grad}
        assert all(v is not None and np.isfinite(v) for v in gradients.values())
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        assert torch.isfinite(norm) and norm>0
        book.intent(key);opt.step();book.commit(key)
        torch.cuda.synchronize();fit_seconds+=time.perf_counter()-tick
        trace.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm)))
        if step in [len(packets)//2,len(packets)]:
            cps[step]=model.learned()
            torch.save(dict(learned=cps[step],optimizer=opt.state_dict(),step=step,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate()),folder/f'checkpoint_{step}.pt')
            if validation is not None:curves[step]=validation(model)
        if step%64==0:event('progress',candidate=c,key=key,step=step)
    assert frozen==model.frozen_hash();assert tensor_hash(initial.items())!=tensor_hash(model.learned().items())
    report=dict(key=key,phase=phase,steps=len(packets),fit_seconds=fit_seconds,wall_seconds=time.perf_counter()-start,curves=curves,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],initial_sha=tensor_hash(initial.items()),final_sha=tensor_hash(model.learned().items()),
        frozen_sha=frozen,frozen_unchanged=True,buffer_scope='all named_buffers, persistent and nonpersistent',last_gradients=gradients,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),trace=trace)
    torch.save(cps,folder/'states.pt');save(RESULTS/c/'fits'/f'{key}.json',report);book.finish(key)
    event('fit_complete',candidate=c,key=key,phase=phase,seconds=report['wall_seconds'])
    return cps,report
