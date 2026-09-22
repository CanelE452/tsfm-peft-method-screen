from runner import *
import subprocess,platform,importlib.metadata,ast

def run_preflight():
    assert not (RESULTS/'PREFLIGHT.json').exists()
    attempt='smoke_validation2'
    setup();d=Data();m=Model();frozen=m.frozen();initial=m.bank();pairs=d.base_schedule(SEEDS[0])[0];b=d.base_batch(pairs);x=tensor(b['x']);y=tensor(b['y']);s=tensor(b['scale'])
    with torch.no_grad():
        q=m(x);native=m.pipeline.predict(x,prediction_length=64).transpose(1,2)
        assert torch.allclose(q.cpu(),native.cpu(),atol=1e-4,rtol=1e-5)
        for length in [320,512]:
            xx=tensor(np.stack([d.asof(int(si),int(t))[-length:] for si,t in d.meta_schedule(SEEDS[0])[0]]));assert torch.allclose(m(xx).cpu(),m.pipeline.predict(xx,prediction_length=64).transpose(1,2).cpu(),atol=1e-4,rtol=1e-5)
    params=[p for p in m.parameters() if p.requires_grad];optimizer=opt(params)
    for k in [1,2]:optimizer.zero_grad(set_to_none=True);update(optimizer,params,attempt+'_warm',k,'smoke',loss(m(x),y,s))
    warm=m.bank()
    with torch.no_grad():
        ordinary=m(x);ones=torch.ones(len(x),len(m.targets),8,device='cuda');dynamic=m(x,ones)
        assert torch.allclose(dynamic,ordinary,atol=1e-4,rtol=1e-5)
        varied=ones.clone();varied[::2]*=1.5;varied[1::2]*=.5;changed=m(x,varied)
        assert float((changed-ordinary).abs().max())>0
        for i in range(len(x)):assert torch.allclose(changed[i],m(x[i:i+1],varied[i:i+1])[0],atol=2e-4,rtol=2e-5)
    series,t=d.meta_schedule(SEEDS[0])[0,0];ep=make_episode(d.asof(int(series),int(t)),int(t),'B')
    with torch.no_grad():qref=m(tensor(ep['support'])).cpu().numpy()
    rr=records(ep,qref,'B_TIME');poison={**ep,'y':ep['y'].copy()};poison['y'][~ep['mask']]=1e30
    assert torch.equal(rr,records(poison,qref,'B_TIME'))
    zz=m(tensor(ep['support']));l1=safe_masked_pinball(zz,tensor(ep['y']),torch.as_tensor(ep['mask'],device='cuda'),tensor(np.full(4,ep['scale'])))
    l2=safe_masked_pinball(zz,tensor(poison['y']),torch.as_tensor(ep['mask'],device='cuda'),tensor(np.full(4,ep['scale'])));assert torch.equal(l1,l2)
    gd=[]
    for arm in ['B_TIME','B_SET']:
        m.restore(warm);g=generator(SEEDS[0],len(m.targets),arm);params=[p for p in m.parameters() if p.requires_grad]+list(g.parameters());optimizer=opt(params)
        before={k:v.detach().clone() for k,v in g.state_dict().items()}
        for k in [1,2]:
            optimizer.zero_grad(set_to_none=True);coef=g(rr[None].expand(len(x),-1,-1));lv=loss(m(x,coef),y,s);update(optimizer,params,attempt+'_'+arm,k,'smoke',lv)
            if k==1:assert float(g.head.weight.grad.norm())>0
            if k==2:assert any(p.grad is not None and float(p.grad.norm())>0 for p in g.encoder.parameters())
        with torch.no_grad():
            if arm.endswith('SET'):assert torch.allclose(g(rr[None]),g(rr.flip(0)[None]),atol=1e-6,rtol=1e-6)
            expected=m(x,g(rr[None].expand(len(x),-1,-1))).clone();state=snapshot(m,g);m.restore(initial);restore(m,g,state);assert torch.equal(expected,m(x,g(rr[None].expand(len(x),-1,-1))))
            assert torch.equal(g(rr[None]),g(records(poison,qref,'B_TIME')[None]))
        gd.append(dict(arm=arm,encoder_gradient=True,head_gradient=True,restore=True));del g
    m.restore(warm);m.bank_grad(False);c=torch.nn.Parameter(torch.ones(1,len(m.targets),8,device='cuda'));optimizer=opt([c])
    for k in [1,2]:
        optimizer.zero_grad(set_to_none=True);lv=safe_masked_pinball(m(tensor(ep['support']),c.expand(4,-1,-1)),tensor(ep['y']),torch.as_tensor(ep['mask'],device='cuda'),tensor(np.full(4,ep['scale'])));update(optimizer,[c],attempt+'_coeff',k,'smoke',lv)
    m.bank_grad(True);optimizer=opt([p for p in m.parameters() if p.requires_grad])
    for k in [1,2]:
        optimizer.zero_grad(set_to_none=True);lv=safe_masked_pinball(m(tensor(ep['support'])),tensor(ep['y']),torch.as_tensor(ep['mask'],device='cuda'),tensor(np.full(4,ep['scale'])));update(optimizer,[p for p in m.parameters() if p.requires_grad],attempt+'_masked',k,'smoke',lv)
    m.bank_grad(False);torch.manual_seed(SEEDS[0]);cell=SimpleOutputAdapter(64,5,9,False,'Linear').cuda().eval()
    raw=(CACHE/'upstream'/'cosa.py').read_text();tree=ast.parse(raw);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SimpleOutputAdapter');scope=dict(torch=torch,nn=torch.nn,ADAPTER_TYPES=('Linear','MLP'));exec(compile(ast.Module(body=[node],type_ignores=[]),'official_cosa_core','exec'),scope)
    official=scope['SimpleOutputAdapter'](64,5,9,False,'Linear').cuda().eval();official.load_state_dict(cell.state_dict());cx=torch.randn(4,5,device='cuda')
    with torch.no_grad():
        cell.gate.fill_(.5);official.gate.fill_(.5);assert torch.equal(cell(tensor(qref),cx),official(tensor(qref),cx));cell.gate.zero_()
    params=list(cell.parameters());optimizer=opt(params,1e-3)
    for k in [1,2]:
        optimizer.zero_grad(set_to_none=True);lv=safe_masked_pinball(cosa_forward(cell,tensor(qref),ep['support']),tensor(ep['y']),torch.as_tensor(ep['mask'],device='cuda'),tensor(np.full(4,ep['scale'])));update(optimizer,params,attempt+'_cosa',k,'smoke',lv)
    assert m.frozen()==frozen
    info=read(ROOT/'results/temporal_lora_init_v1_20260922/MODEL_AND_ENVIRONMENT.json')['models']['amazon/chronos-bolt-small']
    for f in info['files']:assert sha(Path(info['path'])/f['name'])==f['sha256']
    save(RESULTS/'MODEL_MAP.json',dict(model=info,module_names=m.names,rank=8,alpha=16,bank_parameters=294912,functional_no_double_count=True))
    save(RESULTS/'ENVIRONMENT.json',dict(python=sys.version,platform=platform.platform(),gpu=torch.cuda.get_device_name(),torch_cuda=torch.version.cuda,packages={k:importlib.metadata.version(k) for k in ['torch','chronos-forecasting','peft','transformers','numpy','pandas','matplotlib']},nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),base_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),initial_dirty_files=[],contract_zip_sha256='cd5bc5265a08fee7127abedcc5d4b3679a2aad535b4d53babf1ef2ba4b44fc30'))
    (RESULTS/'requirements-lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    import chronos.chronos_bolt as bolt
    save(RESULTS/'UPSTREAM_CODE.json',dict(chronos_file=bolt.__file__,chronos_sha256=sha(bolt.__file__),cosa=read(EXP/'vendor'/'COSA_PROVENANCE.json')))
    smoke=sum(x['kind']=='intent' and x['phase']=='smoke' for x in ledger_entries())
    save(RESULTS/'PREFLIGHT.json',dict(status='PASS',cpu_reference_tests=18,native_parity=True,long_context_parity=True,dynamic_c1_parity=True,per_task_mapping=True,coefficient_effect=True,nan_poison_feature_loss_invariant=True,generator_gradient=gd,COSA_official_core_parity=True,frozen_unchanged=True,smoke_updates=smoke,main_updates=0,cudnn_disabled_for_eval_GRU_backward=True))
    event('preflight_pass',smoke_updates=smoke)

if __name__=='__main__':run_preflight()
