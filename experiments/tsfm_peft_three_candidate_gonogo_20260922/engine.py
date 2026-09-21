from common import *
from model import *
from metrics import metric_arrays,summarize


class Panel:
    def __init__(self,values,sigma,offset=0):
        self.values=values
        self.sigma=sigma
        self.offset=offset

    def batch(self,tuples,targets=True):
        local=tuples.copy();local[:,1]-=self.offset
        assert (local[:,1]>=512).all()
        assert (local[:,1]+(64 if targets else 0)<=len(self.values)).all()
        return batch(self.values,local,targets)

    def scales(self,tuples):
        return torch.tensor(self.sigma[tuples[:,0]],device='cuda',dtype=torch.float32)


@torch.no_grad()
def forecast(net,panel,origins,series,head=None,batch_size=64):
    tuples=np.array([(s,o) for o in origins for s in series],dtype=np.int64)
    outputs=[]
    for i in range(0,len(tuples),batch_size):
        t=tuples[i:i+batch_size]
        q=predict(net,panel.batch(t,False))
        if head is not None:q=head(q,torch.tensor(t[:,1],device='cuda'),panel.scales(t))
        outputs.append(q.cpu().numpy())
    return np.concatenate(outputs).reshape(len(origins),len(series),64,9)


def score(q,panel,origins,series):
    y=np.stack([panel.values[o-panel.offset:o-panel.offset+64,series].T for o in origins])
    return summarize(metric_arrays(q,y,panel.sigma[series]))


def grad_norm(grads):
    return float(torch.sqrt(sum(g.detach().float().square().sum() for g in grads if g is not None)))


def optimizer(net,lr=1e-4):
    return torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=lr,
                            betas=(.9,.999),eps=1e-8,weight_decay=0)


def save_state(path,state):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    torch.save(state,tmp);tmp.replace(path)


def check_seal(candidate):
    sealed=read(RESULTS/f'{candidate}_SOURCE_SEAL.json')
    for relative,digest in sealed['source'].items():assert sha(ROOT/relative)==digest,relative


def fit(net,panel,schedule,origins,arm,seed,pseudo=None):
    check_seal(arm[0])
    run=f'{arm}_{seed}'; folder=CACHE/'fits'/run; out=RESULTS/'fits'/run
    budget=Budget()
    assert read(budget.path)['runs'].get(run,0)==0, 'Spent run cannot silently restart'
    assert not (out/'FIT.json').exists()
    initial=audit(net);initial_state=train_state(net)
    write(out/'INITIAL_AUDIT.json',initial)
    opt=optimizer(net); params=[p for p in net.parameters() if p.requires_grad]
    selected=float('inf');selected_step=None;history=[];start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(257):
        if step in [0,128,256]:
            q=forecast(net,panel,origins,list(range(16)))
            summary=score(q,panel,origins,list(range(16)))
            value=summary['macro']['pinball']
            save_state(folder/f'step{step}.pt',train_state(net))
            np.save(folder/f'val_step{step}.npy',q)
            history.append({'step':step,'score':summary,'checkpoint_sha256':sha(folder/f'step{step}.pt')})
            if value<selected:selected=value;selected_step=step
            event('validation',run=run,step=step,pinball=value)
        if step==256:break
        tuples=schedule[step];x,y=panel.batch(tuples);sigma=panel.scales(tuples)
        opt.zero_grad(set_to_none=True)
        q=predict(net,x);true=loss(q,y,sigma)
        pseudo_loss=None
        if pseudo is not None:
            target=torch.from_numpy(pseudo[step]).to('cuda')
            pseudo_loss=((q.sort(-1).values-target).abs()/sigma[:,None,None]).mean()
            norm_true=grad_norm(torch.autograd.grad(true,params,retain_graph=True,allow_unused=True))
            norm_pseudo=grad_norm(torch.autograd.grad(pseudo_loss,params,retain_graph=True,allow_unused=True))
            total=true+.5*pseudo_loss
        else:
            total=true;norm_true=None;norm_pseudo=None
        assert torch.isfinite(total)
        total.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.0))
        if norm_true is None:norm_true=norm
        budget.step(opt,run)
        record={'step':step+1,'true_loss':float(true.detach()),'pseudo_loss':float(pseudo_loss.detach()) if pseudo_loss is not None else 0,
                'true_gradient_norm':norm_true,'pseudo_gradient_norm':norm_pseudo,'total_gradient_norm':norm}
        out.mkdir(parents=True,exist_ok=True)
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if (step+1)%32==0:event('train',run=run,step=step+1,loss=record['true_loss'])
    final=train_state(net)
    assert frozen_hash(net)==initial['frozen_hash']
    changed=[n for n in final if not torch.equal(final[n],initial_state[n])]
    assert changed
    report={'run':run,'arm':arm,'seed':seed,'selected_step':selected_step,'validation':history,
            'seconds':time.perf_counter()-start,'peak_allocated':torch.cuda.max_memory_allocated(),
            'peak_reserved':torch.cuda.max_memory_reserved(),'updates':256,'changed_names':changed,
            'frozen_unchanged':True,'trainable_count':initial['count'],
            'selected_checkpoint_sha256':sha(folder/f'step{selected_step}.pt')}
    write(out/'FIT.json',report)
    return report


def smoke(net,panel,schedule,run,parity=False,pseudo=False):
    initial=train_state(net);before=frozen_hash(net);opt=optimizer(net)
    x,y=panel.batch(schedule[0]);sigma=panel.scales(schedule[0])
    with torch.no_grad():
        q=predict(net,x)
        if parity:
            with net.disable_adapter():base=predict(net,x)
            assert torch.equal(q,base)
    # Target substitution is checked while reusing precisely the same input.
    assert torch.equal(x,panel.batch(schedule[0],False))
    norms=[]
    for i in range(2):
        opt.zero_grad(set_to_none=True)
        x,y=panel.batch(schedule[i]);q=predict(net,x)
        objective=loss(q,y,panel.scales(schedule[i]))
        if pseudo:objective=objective+.5*(q-q.detach().sort(-1).values).abs().mean()/sigma.mean()
        objective.backward()
        params=[p for p in net.parameters() if p.requires_grad]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        norms.append(float(torch.nn.utils.clip_grad_norm_(params,1)))
        Budget().step(opt,run,smoke=True)
    final=train_state(net)
    assert frozen_hash(net)==before
    assert any(not torch.equal(final[n],initial[n]) for n in initial)
    path=CACHE/'smoke'/f'{run}.pt';save_state(path,final)
    with torch.no_grad():after=predict(net,x).cpu()
    restore(net,initial);restore(net,torch.load(path,weights_only=True))
    with torch.no_grad():reloaded=predict(net,x).cpu()
    assert torch.equal(after,reloaded)
    report={'status':'PASS','updates':2,'initial_hash':tensor_hash(initial.items()),'gradient_norms':norms,
            'frozen_hash':before,'save_restore_exact':True,'zero_init_parity':parity,
            'trainable_count':sum(p.numel() for p in net.parameters() if p.requires_grad)}
    write(RESULTS/'smoke'/f'{run}.json',report)
    return initial
