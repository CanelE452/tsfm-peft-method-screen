from common import *
from model import *
import ast
from safetensors import safe_open
from peft import replace_lora_weights_loftq
import data


def qera_factors(residual, scale, rank):
    scale=scale.to(residual).clamp_min(1e-8)
    u,s,v=torch.linalg.svd(residual*scale[None,:],full_matrices=False)
    root=s[:rank].sqrt()
    return root[:,None]*v[:rank]/scale[None,:],u[:,:rank]*root[None,:]


def formula_check():
    source=(CACHE/'upstream/initialization.py').read_text()
    tree=ast.parse(source)
    names=['_low_rank_decomposition_qera','init_lora_qera_4bit']
    selected=ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names],type_ignores=[])
    scope={'torch':torch,'bnb':bnb}
    exec(compile(selected,'pinned_qera_functions','exec'),scope)
    torch.manual_seed(92200)
    w=torch.randn(32,64,device='cuda')
    qw=bnb.nn.Params4bit(w.cpu(),requires_grad=False,compress_statistics=True,quant_type='nf4').to('cuda')
    scale=torch.rand(64,device='cuda')+.1
    a,b,_,_=scope['init_lora_qera_4bit'](qw,w,scale,4,'cuda')
    residual=w-bnb.functional.dequantize_4bit(qw.data,qw.quant_state).float()
    aa,bb=qera_factors(residual,scale,4)
    diff=(b@a-bb@aa).abs().max().item()
    assert torch.allclose(b@a,bb@aa,atol=1e-5,rtol=1e-4)
    return {'official_function_product_max_abs':diff,'source_sha256':sha(CACHE/'upstream/initialization.py'),
            'differences':['elementwise diagonal multiplication/division instead of dense diagonal solve',
                          'FP32 square/FP64 accumulation for activation RMS','zero RMS floor 1e-8',
                          'no claim of full QERA benchmark reproduction']}


@torch.no_grad()
def prepare():
    assert not (RESULTS/'Q_ALLOCATION.json').exists(), 'Do not repeat sensitivity budget'
    start=time.perf_counter()
    check=formula_check()
    d=data.load();pairs=data.probe();x=batch(d['values'],pairs,False)
    sigma=torch.tensor(d['sigma'][pairs[:,0]],device='cuda',dtype=torch.float32)
    pipe,fp=load_base()
    sums={};counts={};handles=[]
    def hook(name):
        def record(mod,args,out):
            z=args[0].detach().float().reshape(-1,args[0].shape[-1])
            val=z.square().double().sum(0)
            sums[name]=sums.get(name,0)+val
            counts[name]=counts.get(name,0)+len(z)
        return record
    for n,m in fp.named_modules():
        if isinstance(m,nn.Linear):handles.append(m.register_forward_hook(hook(n)))
    target=predict(fp,x)
    for h in handles:h.remove()
    scales={n:(s/counts[n]).sqrt().float().cpu() for n,s in sums.items()}
    torch.save(scales,CACHE/'qera_scales.pt')
    fp.save_pretrained(CACHE/'deployment_bf16',safe_serialization=True)
    qp,q=load_base(quant=True)
    qmodules=dict(q.named_modules());fpmodules=dict(fp.named_modules())
    def distance(output):return float(((output-target)/sigma[:,None,None]).square().mean())
    reference=distance(predict(q,x)); sensitivities={};costs={g:0 for g in GROUPS}
    for n,m in fp.named_modules():
        if isinstance(m,nn.Linear):costs[group(n)]+=m.in_features+m.out_features
    for g in GROUPS:
        names=[n for n,m in qmodules.items() if isinstance(m,bnb.nn.Linear4bit) and group(n)==g]
        if not names:continue
        for n in names:
            parent,leaf=n.rsplit('.',1);setattr(q.get_submodule(parent),leaf,fpmodules[n])
        restored=distance(predict(q,x))
        for n in names:
            parent,leaf=n.rsplit('.',1);setattr(q.get_submodule(parent),leaf,qmodules[n])
        sensitivities[g]={'names':names,'restored_discrepancy':restored,'sensitivity':max(0,reference-restored)}
    q.save_pretrained(CACHE/'deployment_nf4',safe_serialization=True)
    ranks={g:1 for g in GROUPS if costs[g]}
    budget=4*sum(costs.values())
    if not any(s['sensitivity']>0 for s in sensitivities.values()):ranks={g:4 for g in ranks}
    else:
        while True:
            spent=sum(ranks[g]*costs[g] for g in ranks)
            options=[g for g in ranks if ranks[g]<16 and spent+costs[g]<=budget]
            if not options:break
            g=max(options,key=lambda g:sensitivities.get(g,{}).get('sensitivity',0)*np.log((ranks[g]+2)/(ranks[g]+1))/costs[g])
            ranks[g]+=1
    spent=sum(ranks[g]*costs[g] for g in ranks)
    write(RESULTS/'Q_ALLOCATION.json',{'groups':GROUPS,'costs_per_rank':costs,'ranks':ranks,
        'uniform_budget':budget,'candidate_parameters':spent,'unused_fraction':1-spent/budget,
        'resource_mismatch':1-spent/budget>=.05,'nf4_discrepancy':reference,'sensitivities':sensitivities,
        'sensitivity_example_forwards':32*(2+len(sensitivities)),
        'activation_stat_example_forwards':32,'activation_collection_shared_with_bf16_reference':True,
        'probe_sha256':sha(CACHE/'data/probe_seed92200.npy'),'future_target_access':False,
        'qera_formula_check':check,'seconds':time.perf_counter()-start})
    del fp,pipe,q,qp,qmodules,fpmodules,target,x
    clear()
    # Backend IO exceptions are added to its own defaults, never replacing the defaults.
    ip,io=load_base(quant=True,io16=True)
    io.save_pretrained(CACHE/'deployment_io16',safe_serialization=True)
    write(RESULTS/'MODEL_IO16.json',audit(io))
    del ip,io
    clear()


@torch.no_grad()
def initialize(net,arm):
    if arm=='Q_LOFTQ':
        replace_lora_weights_loftq(net,model_path=read(RESULTS/'UPSTREAM.json')['models']['small']['path'])
    if arm in ['Q_QERA','Q_IO16','Q_FORECAST']:
        scales=torch.load(CACHE/'qera_scales.pt',weights_only=True)
        weight_path=Path(read(RESULTS/'UPSTREAM.json')['models']['small']['path'])/'model.safetensors'
        with safe_open(str(weight_path),framework='pt',device='cpu') as weights:
            for n,m in net.named_modules():
                if not hasattr(m,'lora_A') or not isinstance(m.base_layer,bnb.nn.Linear4bit):continue
                name=n.removeprefix('base_model.model.')
                w=weights.get_tensor(name+'.weight').to('cuda').float()
                residual=w-bnb.functional.dequantize_4bit(m.weight.data,m.weight.quant_state).float()
                a,b=qera_factors(residual,scales[name],m.r['default'])
                m.lora_A['default'].weight.copy_(a)
                m.lora_B['default'].weight.copy_(b/m.scaling['default'])


def make_q(arm,seed,init=True):
    p,b=load_base(quant=arm!='Q_FP',io16=arm=='Q_IO16')
    ranks=read(RESULTS/'Q_ALLOCATION.json')['ranks'] if arm=='Q_FORECAST' else None
    net=attach(b,seed,'all',ranks)
    if init:initialize(net,arm)
    return net


if __name__=='__main__':prepare()
