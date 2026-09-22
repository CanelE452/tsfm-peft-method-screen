from model import *
from contract.reference_core import residual_record

def scale(x):return float(max(np.std(x,dtype=np.float64),1e-6*np.mean(np.abs(x),dtype=np.float64),1e-6))
def issues(t,track):return [t-a for a in ([256,192,128,64] if track=='A' else [64,32,16,8])]
def episode(prefix,t,track):
    assert len(prefix)==t and np.isfinite(prefix).all()
    ss=issues(t,track);x=np.stack([prefix[s-256:s] for s in ss]);y=np.full((4,64),np.nan,dtype=np.float32);mask=np.zeros((4,64),dtype=bool)
    for i,s in enumerate(ss):
        n=min(64,t-s);y[i,:n]=prefix[s:s+n];mask[i,:n]=True
    return dict(x=prefix[t-256:t].copy(),support=x,y=y,mask=mask,issues=ss,scale=scale(prefix[t-256:t]))
def records(ep,qref,arm):
    out=[];s=tensor(ep['scale'])
    for i,issue in enumerate(ep['issues']):
        mask=torch.as_tensor(ep['mask'][i],device='cuda')
        if arm=='B_FULLGEN' and not bool(mask.all()):mask=torch.zeros_like(mask)
        r=residual_record(tensor(ep['support'][i]),tensor(qref[i]),tensor(ep['y'][i]),mask,ep['now']-issue,s,no_error=arm=='A_NOERROR')
        if arm=='B_FULLGEN':r[-1]=float(ep['mask'][i].mean())
        out.append(r)
    return torch.stack(out)
def make_episode(prefix,t,track):
    e=episode(prefix,t,track);e['now']=t;return e
def metrics(q,y,s):
    e=y[...,None].astype(float)-q.astype(float);levels=np.arange(1,10)/10
    pb=(2*np.maximum(levels*e,(levels-1)*e)).mean(-1)
    return dict(pinball=pb/s,raw_pinball=pb,nmae=np.abs(e[...,4])/s,raw_mae=np.abs(e[...,4]),coverage=((y>=q[...,0])&(y<=q[...,-1])).astype(float),width=(q[...,-1]-q[...,0])/s,crossing=(np.diff(q,axis=-1)<0).mean(-1))
def affine(qsupport,ep):
    q=tensor(qsupport);y=tensor(ep['y']);mask=torch.as_tensor(ep['mask'],device='cuda');s=tensor(np.full(4,ep['scale']));choices=[]
    with torch.no_grad():
        for a in [.5,.75,1,1.25,1.5,2]:
            for b in [-.5,-.25,0,.25,.5]:
                v=q[:,:,4,None]+a*(q-q[:,:,4,None])+b*ep['scale'];lossval=float(safe_masked_pinball(v,y,mask,s));choices.append((lossval,(a-1)**2+b*b,a,abs(b),b))
    z=min(choices);return dict(alpha=z[2],beta=z[4])
def apply_affine(q,c,s):return q[:,4,None]+c['alpha']*(q-q[:,4,None])+c['beta']*s
def cosa_forward(cell,q,x):
    scales=tensor(np.array([scale(v) for v in x]));xt=tensor(x)
    context=torch.stack([xt.mean(1),xt.std(1,unbiased=False),xt.min(1).values,xt.max(1).values,xt[:,-1]],1)/scales[:,None]
    return cell(q/scales[:,None,None],context)*scales[:,None,None]

class Reference:
    def __init__(self,seed,model):
        self.seed=seed;self.model=model;self.bank_hash=hash_tensors(model.bank().items());self.directory=CACHE/'reference'/str(seed);self.directory.mkdir(parents=True,exist_ok=True);self.calls=0;self.seconds=0
    def get(self,series,issue,context):
        assert len(context)==256 and issue>=read(RESULTS/'TIME_SPLIT.json')['edges'][1]
        digest=hashlib.sha256(context.tobytes()).hexdigest();p=self.directory/f'{series}_{issue}.npz'
        if p.exists():
            z=np.load(p);assert str(z['context_hash'])==digest and str(z['bank_hash'])==self.bank_hash;return z['q'].copy()
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.no_grad():q=self.model(tensor(context[None])).cpu().numpy()[0]
        torch.cuda.synchronize();seconds=time.perf_counter()-start;self.calls+=1;self.seconds+=seconds
        np.savez_compressed(p,q=q,context_hash=digest,bank_hash=self.bank_hash)
        append(RESULTS/'REFERENCE_FORECASTS.jsonl',dict(seed=self.seed,series=int(series),issue=int(issue),context_sha=digest,bank_sha=self.bank_hash,file=p.relative_to(ROOT).as_posix(),sha256=sha(p),seconds=seconds,bytes=p.stat().st_size))
        return q
    def support(self,series,ep):return np.stack([self.get(series,s,x) for s,x in zip(ep['issues'],ep['support'])])

class Issuer:
    def __init__(self,track,role,arm,seed,tag):
        self.track=track;self.role=role;self.arm=arm;self.seed=seed;self.tag=str(tag);self.folder=CACHE/track/'predictions'/role/arm/f'{seed}_{tag}';self.folder.mkdir(parents=True,exist_ok=True)
    def put(self,series,t,q,**extra):
        p=self.folder/f'{series}_{t}.npz';assert not p.exists(),'Issued forecasts cannot be overwritten';assert q.shape==(64,9) and np.isfinite(q).all()
        np.savez_compressed(p,q=q,series=series,issue=t)
        append(RESULTS/self.track/'ISSUED_FORECASTS.jsonl',dict(role=self.role,arm=self.arm,seed=self.seed,tag=self.tag,series=int(series),issue=int(t),target_start=int(t),visible_before=int(t),file=p.relative_to(ROOT).as_posix(),sha256=sha(p),**extra))
def read_predictions(track,role,arm,seed,tag,pairs):
    folder=CACHE/track/'predictions'/role/arm/f'{seed}_{tag}'
    return np.stack([np.load(folder/f'{s}_{t}.npz')['q'] for s,t in pairs])
