from common import *
from model import *
from engine import *
import copy

ARMS=['F_LOCAL','F_SHARED','F_AFFINE','F_HEAD','F_PERIODIC']


def make(seed,local=False):
    _,base=load_base()
    return attach(base,seed,ffa=not local)


class Client:
    def __init__(self,client,values,sigma,schedule,origins,arm):
        self.client=client
        self.panel=Panel(values.copy(),np.asarray([sigma]))
        self.schedule=schedule.copy();self.schedule[:,:,0]=0
        self.origins=origins.copy()
        self.arm=arm
        self.head=PrivateHead(arm)
        self.private_optimizer_state=None
        self.local_state=None
        self.seconds=0
        self.peak=0

    def train(self,net,start,steps,run,smoke=False):
        params=[p for p in net.parameters() if p.requires_grad]
        groups=[{'params':params,'lr':3e-4}]
        if self.head.coeff.numel():groups.append({'params':[self.head.coeff],'lr':1e-3})
        opt=torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8,weight_decay=0)
        if self.private_optimizer_state is not None:
            opt.state[self.head.coeff]=copy.deepcopy(self.private_optimizer_state)
        train_params=params+([self.head.coeff] if self.head.coeff.numel() else [])
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        logs=[]
        for index in range(start,start+steps):
            tuples=self.schedule[index];x,y=self.panel.batch(tuples);s=self.panel.scales(tuples)
            opt.zero_grad(set_to_none=True)
            q=self.head(predict(net,x),torch.tensor(tuples[:,1],device='cuda'),s)
            value=loss(q,y,s)
            assert torch.isfinite(value)
            value.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in train_params)
            norm=float(torch.nn.utils.clip_grad_norm_(train_params,1))
            Budget().step(opt,run,smoke)
            logs.append({'client':self.client,'local_step':index+1,'true_loss':float(value.detach()),'gradient_norm':norm})
        if self.head.coeff.numel():self.private_optimizer_state=copy.deepcopy(opt.state[self.head.coeff])
        if steps:
            shared_step=float(opt.state[params[0]]['step'])
            assert shared_step==steps
            logs[-1]['shared_optimizer_step_after_reset']=shared_step
            if self.head.coeff.numel():
                logs[-1]['private_optimizer_cumulative_step']=float(self.private_optimizer_state['step'])
                assert float(self.private_optimizer_state['step'])==start+steps
        self.seconds+=time.perf_counter()-started
        self.peak=max(self.peak,torch.cuda.max_memory_allocated())
        return train_state(net),logs

    def evaluate(self,net,folder,round_index):
        q=forecast(net,self.panel,self.origins,[0],self.head)
        summary=score(q,self.panel,self.origins,[0])
        folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/f'client{self.client}_val_round{round_index}.npy',q)
        write(folder/f'client{self.client}_val_round{round_index}.json',summary)
        # Server only receives this scalar; predictions/scale stay in client files.
        return summary['macro']['pinball']


def aggregate(payloads,net):
    keys=list(payloads[0])
    assert all('.lora_B.' in n for n in keys)
    assert all(list(p)==keys for p in payloads)
    mean={k:torch.stack([p[k].float() for p in payloads]).mean(0) for k in keys}
    maximum=0.0
    modules=dict(net.named_modules())
    for name in keys:
        module_name=name.split('.lora_B.')[0]
        a=modules[module_name].lora_A['default'].weight.detach().float().cpu()
        left=mean[name]@a
        right=sum((p[name].float()@a for p in payloads))/len(payloads)
        error=float((left-right).abs().max());maximum=max(maximum,error)
        assert torch.allclose(left,right,rtol=1e-4,atol=1e-7)
        independent=np.mean(np.stack([p[name].numpy() for p in payloads]),axis=0)
        assert np.allclose(mean[name].numpy(),independent,rtol=1e-6,atol=1e-8)
    return mean,{'uploaded_keys':keys,'uploaded_bytes':sum(p[k].numel()*p[k].element_size() for p in payloads for k in keys),
                 'ba_equivalence_max_abs':maximum,'numpy_mean_verified':True,
                 'raw_sigma_private_uploaded':False}


def get_clients(arm,seed):
    import data
    d=data.load()
    end=read(RESULTS/'DATA_AND_SPLIT_AUDIT.json')['split']['q_val_end']
    clients=[Client(c,d['values'][:end,c:c+1],d['sigma'][c],data.schedule('Q_TRAIN',seed,client=c),
                    d['origins']['Q_VAL'],arm) for c in range(4)]
    return clients


def preflight():
    records=[]
    # Two distinct clients each spend one update per workflow (eight total).
    for arm in ['F_LOCAL','F_AFFINE','F_HEAD','F_PERIODIC']:
        net=make(92201,arm=='F_LOCAL');clients=get_clients(arm,92201)[:2]
        initial=train_state(net);base_hash=frozen_hash(net);payload=[]
        for client in clients:
            restore(net,initial)
            x,_=client.panel.batch(client.schedule[0]);s=client.panel.scales(client.schedule[0])
            with torch.no_grad():
                q=predict(net,x)
                assert torch.equal(q,client.head(q,torch.tensor(client.schedule[0,:,1],device='cuda'),s))
                with net.disable_adapter():assert torch.equal(q,predict(net,x))
            state,logs=client.train(net,0,1,'F_smoke_'+arm,True)
            assert any(not torch.equal(state[k],initial[k]) for k in initial)
            assert frozen_hash(net)==base_hash
            payload.append(state)
            if client.head.coeff.numel():
                old_step=float(client.private_optimizer_state['step'])
                client.train(net,1,0,'F_no_optimizer_step',True)
                assert float(client.private_optimizer_state['step'])==old_step
        if arm!='F_LOCAL':
            avg,communication=aggregate(payload,net)
            restore(net,avg)
        else:communication={'shared':False}
        # A separate private tensor object for every client, with independent state.
        assert clients[0].head.coeff is not clients[1].head.coeff
        states=[c.head.coeff.detach().clone() for c in clients]
        if states[0].numel():
            with torch.no_grad():
                clients[0].head.coeff[0].add_(1)
                assert torch.equal(clients[1].head.coeff,states[1])
                clients[0].head.coeff.copy_(states[0])
        path=CACHE/'smoke'/f'{arm}_state.pt'
        save_state(path,{'model':train_state(net),'heads':[c.head.state_dict() for c in clients]})
        with torch.no_grad():
            reference=clients[0].head(predict(net,x),torch.tensor(clients[1].schedule[0,:,1],device='cuda'),s).cpu()
        saved=torch.load(path,weights_only=True)
        restore(net,saved['model'])
        for c,h in zip(clients,saved['heads']):c.head.load_state_dict(h)
        with torch.no_grad():
            replay=clients[0].head(predict(net,x),torch.tensor(clients[1].schedule[0,:,1],device='cuda'),s).cpu()
        assert torch.equal(reference,replay)
        report={'arm':arm,'updates':2,'communication':communication,'frozen_hash':base_hash,'status':'PASS',
                'private_parameter_count':clients[0].head.coeff.numel(),'client_states_disjoint':True,
                'initial_parity':True,'save_restore_exact':True,'trainable':audit(net)['trainable']}
        write(RESULTS/'smoke'/f'{arm}.json',report)
        del clients,net;clear()


def workflow(arm,seed):
    check_seal('F')
    run=f'{arm}_{seed}';out=RESULTS/'fits'/run;folder=CACHE/'fits'/run
    budget=Budget();assert read(budget.path)['runs'].get(run,0)==0
    local=arm=='F_LOCAL';net=make(seed,local);clients=get_clients(arm,seed)
    initial=train_state(net);frozen=frozen_hash(net)
    assert all(torch.count_nonzero(p)==0 for n,p in initial.items() if '.lora_B.' in n)
    write(out/'INITIAL_AUDIT.json',audit(net))
    global_state=initial
    for c in clients:c.local_state=copy.deepcopy(initial)
    best=float('inf');chosen=None;history=[];communication=[];start=time.perf_counter()
    for round_index in range(17):
        if round_index in [0,8,16]:
            scalars=[]
            for c in clients:
                restore(net,c.local_state if local else global_state)
                scalars.append(c.evaluate(net,folder,round_index))
            value=float(np.mean(scalars))
            state={'shared':global_state if not local else None,'local':[c.local_state for c in clients] if local else None,
                   'heads':[c.head.state_dict() for c in clients]}
            save_state(folder/f'round{round_index}.pt',state)
            history.append({'step':round_index,'client_scalars':scalars,'pinball':value,
                            'checkpoint_sha256':sha(folder/f'round{round_index}.pt')})
            if value<best:best=value;chosen=round_index
            event('validation',run=run,round=round_index,pinball=value)
        if round_index==16:break
        payloads=[];all_logs=[]
        for c in clients:
            restore(net,c.local_state if local else global_state)
            state,logs=c.train(net,round_index*4,4,run)
            if local:c.local_state=state
            else:payloads.append(state)
            all_logs.extend(logs)
        if not local:
            global_state,info=aggregate(payloads,net)
            info['round']=round_index+1
            info['downloaded_bytes']=info['uploaded_bytes']
            communication.append(info)
        with (out/'training.jsonl').open('a') as f:
            for log in all_logs:f.write(json.dumps(log)+'\n')
        event('federated_round',run=run,round=round_index+1)
    assert frozen_hash(net)==frozen
    bytes_state=sum(t.numel()*t.element_size() for t in initial.values())
    report={'run':run,'arm':arm,'seed':seed,'selected_step':chosen,'validation':history,'updates':256,
            'unit':'4 independent local fits' if local else 'federated workflow with 4 clients',
            'seconds':time.perf_counter()-start,'frozen_unchanged':True,
            'selected_checkpoint_sha256':sha(folder/f'round{chosen}.pt'),
            'clients':[{'client':c.client,'local_updates':64,'seconds':c.seconds,'peak_allocated':c.peak,
                        'private_parameter_bytes':c.head.coeff.numel()*c.head.coeff.element_size(),
                        'private_optimizer_tensor_bytes':sum(t.numel()*t.element_size() for t in (c.private_optimizer_state or {}).values() if isinstance(t,torch.Tensor)),
                        'adapter_tensor_bytes':bytes_state} for c in clients],
            'uploaded_bytes':sum(r['uploaded_bytes'] for r in communication),
            'downloaded_bytes':sum(r['downloaded_bytes'] for r in communication),
            'server_adapter_state_bytes':0 if local else bytes_state,
            'initial_trainable_hash':tensor_hash(initial.items()),
            'frozen_A_bytes_per_client':sum(p.numel()*p.element_size() for n,p in net.named_parameters() if '.lora_A.' in n) if not local else 0,
            'communication_scope':'round B transfer only; common pretrained backbone and seed-defined frozen A provisioning excluded',
            'private_state_upload':False,'server_validation_payload':'four scalars per validation round',
            'communication':communication}
    write(out/'FIT.json',report)
    del net,clients;clear()


def main():
    for seed in [92201,92202]:
        for arm in ARMS:
            if not (RESULTS/'fits'/f'{arm}_{seed}'/'FIT.json').exists():workflow(arm,seed)
    event('F_training_complete')


if __name__=='__main__':
    if sys.argv[1]=='smoke':preflight()
    elif sys.argv[1]=='train':main()
