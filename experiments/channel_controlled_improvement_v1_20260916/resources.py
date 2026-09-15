"""Discarded resource-only options; main training architecture never changes."""
import types
from runtime import *

def fast_forward(self,x_enc,input_mask=None):
    b,c,t=x_enc.shape;assert t==96
    if input_mask is None:input_mask=torch.ones(b,t,device=x_enc.device,dtype=torch.bool)
    assert input_mask.all()
    with torch.no_grad():
        x=self.normalizer(x=x_enc,mask=input_mask,mode='norm');x=torch.nan_to_num(x,nan=0.,posinf=0.,neginf=0.)
        patches=self.patch_embedding(self.tokenizer(x),mask=input_mask);b,c,n,d=patches.shape;assert(n,d)==(12,512)
        mask=attention.up.Masking.convert_seq_to_patch_view(input_mask,self.patch_embedding.patch_len).repeat_interleave(c,dim=0)
        encoded=self.encoder(inputs_embeds=patches.reshape(b*c,n,d),attention_mask=mask,output_hidden_states=True)
    states=encoded.hidden_states[1:];assert len(states)==8
    side=torch.zeros_like(states[0])
    for layer,h in zip(self.side,states):side=layer(h,side,None)
    hidden=(encoded.last_hidden_state+side).reshape(b,c,n,d)
    return attention.up.TimeseriesOutputs(input_mask=input_mask,forecast=self.normalizer(x=self.head(hidden),mode='denorm'))

def variant(m,name):
    if name=='SIDE-fast':
        for h in m._handles:h.remove()
        m._handles=[];m._snapshots=[];m.forward=types.MethodType(fast_forward,m)
    elif name=='LH-native-checkpoint':
        m.encoder.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})
        assert m.encoder.base_model.model.gradient_checkpointing
    return m

def measure(w):
    c=contract();assert state_json()['status']=='EVALUATION_COMPLETE';status(status='RESOURCES')
    selected=read(OUT/'selection_seal.json')['selections'];canonical={};parity=[];results=[]
    def applied():
        s=state_json();assert s['resource_updates']<120;status(resource_updates=s['resource_updates']+1)
    for d,dc in c['data'].items():
        for arm in ARMS:
            sel=next(r for r in selected if r['dataset']==d and r['arm']==arm and r['seed']==41000 and r['policy']=='P_MAIN')
            m=make(arm,dc['channel_ids'],41000,'cuda');restore(m,torch.load(ROOT/sel['checkpoint_path'],weights_only=True,map_location='cpu'));opt=optimizer(m,sel['lr'])
            # Old trajectories do not contain Adam. Canonical resource state deliberately starts
            # with empty Adam for every arm/option, then performs the specified two warmup steps.
            state=dict(parameters=cpu_state(m),optimizer=opt.state_dict(),rng=rng_state(),lr=sel['lr'],selection=sel)
            path=CACHE/f'resource_canonical_{d}_{arm}.pt';atomic_torch(path,state);canonical[(d,arm)]=path
            del m,opt;cleanup()
    def build(d,arm,name):
        st=torch.load(canonical[(d,arm)],map_location='cpu',weights_only=False)
        m=variant(make(arm,c['data'][d]['channel_ids'],41000,'cuda'),name);restore(m,st['parameters']);opt=optimizer(m,st['lr']);opt.load_state_dict(st['optimizer']);rng_restore(st['rng']);m.train();return m,opt
    accepted={}
    for d,dc in c['data'].items():
        vals,_=load(DATA,d,'train');oo=dc['panels']['TRAIN'][:8]
        for arm,option in [('SIDE','SIDE-fast'),('LH','LH-native-checkpoint')]:
            snapshots=[];reason=None
            try:
                for name in [arm+'-current',option]:
                    m,opt=build(d,arm,name);out=[]
                    handle=m.register_forward_hook(lambda mod,args,result:out.append(result.forecast.detach().float().cpu()))
                    for _ in range(2):step(m,opt,vals,oo,8,w,applied)
                    snapshots.append(dict(outputs=out,grads={n:p.grad.detach().cpu() for n,p in parameters(m).items()},params=cpu_state(m)))
                    handle.remove();del m,opt;cleanup()
                exact=True;max_abs=0.
                for category in ['outputs','grads','params']:
                    aa=snapshots[0][category];bb=snapshots[1][category]
                    pairs=zip(aa,bb) if isinstance(aa,list) else ((aa[k],bb[k]) for k in aa)
                    for a,b in pairs:
                        exact &= torch.equal(a,b);max_abs=max(max_abs,float((a-b).abs().max()))
                        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
                accepted[(d,option)]=True
                parity.append(dict(dataset=d,option=option,status='PASS',bitwise_equal=exact,max_abs=max_abs,updates=4))
            except (AssertionError,AttributeError,NotImplementedError) as e:
                accepted[(d,option)]=False;parity.append(dict(dataset=d,option=option,status='NOT_MEASURED',reason=repr(e)))
            save(OUT/'resource_equivalence.json',parity)
    options=[(d,name) for d in c['data'] for name in ['LH-current','LH-native-checkpoint','SIDE-fast','PRIOR-current']]
    options=[options[i] for i in np.random.default_rng(66016).permutation(len(options))]
    for d,name in options:
        if (d,name) in accepted and not accepted[(d,name)]:
            results.append(dict(dataset=d,option=name,status='NOT_MEASURED'));continue
        arm=name.split('-')[0];m,opt=build(d,arm,name);vals,_=load(DATA,d,'train');oo=c['data'][d]['panels']['TRAIN'][:8];records=[]
        for k in range(11):records.append(dict(index=k,block=-1 if k<2 else (k-2)//3,**step(m,opt,vals,oo,8,w,applied)))
        timed=records[2:]
        results.append(dict(dataset=d,option=name,status='MEASURED',steps=records,peak_allocated=max(r['peak_allocated'] for r in timed),peak_reserved=max(r['peak_reserved'] for r in timed),median_seconds=float(np.median([r['seconds'] for r in timed])),mean_seconds=float(np.mean([r['seconds'] for r in timed])),contaminated=any(r['external_compute_contaminated'] for r in records),canonical_path=str(canonical[(d,arm)].relative_to(ROOT)),canonical_sha256=sha(canonical[(d,arm)]),optimizer='identical empty Adam canonical; two warmup updates; discarded',updates=11))
        save(OUT/'resources.json',results);del m,opt;cleanup()
    save(OUT/'resources.json',results);csvwrite(OUT/'resources.csv',[{k:v for k,v in r.items() if k!='steps'} for r in results]);status(status='RESOURCES_COMPLETE')
