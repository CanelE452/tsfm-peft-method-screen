from common import *
from model import *
from engine import Panel
import statistics


def folder_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def load_selected(arm,seed,client=0,deployed=False):
    head=None
    if arm in ['A0','N0']:
        _,net=load_base('small' if arm=='A0' else 'base')
        return net,head,None
    result=read(RESULTS/'fits'/f'{arm}_{seed}'/'FIT.json')
    prefix='round' if arm.startswith('F_') else 'step'
    path=CACHE/'fits'/f'{arm}_{seed}'/f"{prefix}{result['selected_step']}.pt"
    state=torch.load(path,weights_only=True)
    if arm.startswith('Q_'):
        if deployed:
            base_dir=CACHE/({'Q_FP':'deployment_bf16','Q_IO16':'deployment_io16'}.get(arm,'deployment_nf4'))
            pipeline=ChronosBoltPipeline.from_pretrained(str(base_dir),device_map='cuda',dtype=torch.bfloat16,local_files_only=True)
            base=pipeline.model.eval();base.requires_grad_(False)
            ranks=read(RESULTS/'Q_ALLOCATION.json')['ranks'] if arm=='Q_FORECAST' else None
            net=attach(base,seed,'all',ranks)
        else:
            from quantization import make_q
            net=make_q(arm,seed,init=False)
        restore(net,state)
    elif arm.startswith('T_'):
        from run_t import build_model
        net=build_model('small' if arm=='T_A1' else 'base',seed);restore(net,state)
    else:
        from run_f import make
        net=make(seed,arm=='F_LOCAL')
        restore(net,state['local'][client] if arm=='F_LOCAL' else state['shared'])
        head=PrivateHead(arm);head.load_state_dict(state['heads'][client])
    return net,head,path


@torch.no_grad()
def measure(arm,seed):
    import data
    start=time.perf_counter()
    net,head,checkpoint=load_selected(arm,seed,deployed=True)
    packed={n:{'dtype':str(m.weight.dtype),'nested':m.weight.quant_state.nested,
               'quant_type':m.weight.quant_state.quant_type} for n,m in net.named_modules() if isinstance(m,bnb.nn.Linear4bit)}
    if arm.startswith('Q_') and arm!='Q_FP':
        assert len(packed)==(84 if arm=='Q_IO16' else 90)
        assert all(v['dtype']=='torch.uint8' and v['nested'] and v['quant_type']=='nf4' for v in packed.values())
    assert all(p.device.type=='cuda' for p in net.parameters())
    d=data.load();pairs=data.schedule('Q_TRAIN',92201)[0]
    if arm.startswith('F_'):pairs[:,0]=0
    panel=Panel(d['values'],d['sigma'])
    load_seconds=time.perf_counter()-start
    clear()
    samples=[]
    for count in [1,8]:
        t=pairs[:count];x=panel.batch(t,False);sigma=panel.scales(t)
        o=torch.tensor(t[:,1],device='cuda')
        def forward():
            q=predict(net,x)
            return head(q,o,sigma) if head is not None else q
        for _ in range(3):forward()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        timings=[]
        for _ in range(20):
            begin=time.perf_counter();q=forward();torch.cuda.synchronize()
            timings.append(time.perf_counter()-begin)
        samples.append({'batch':count,'median_seconds':statistics.median(timings),
                        'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
                        'finite':bool(torch.isfinite(q).all()),'repetitions':20,'warmup':3})
    if arm.startswith('Q_'):
        base_dir=CACHE/({'Q_FP':'deployment_bf16','Q_IO16':'deployment_io16'}.get(arm,'deployment_nf4'))
    elif arm.startswith('T_') and arm!='T_A1' or arm=='N0':
        base_dir=CACHE/'deployment_base'
        if not base_dir.exists():
            # Export pretrained parameters only; adapter is serialized separately below.
            # This directory is prepared by the caller before measurement runs.
            raise RuntimeError('Missing isolated base deployment artifact')
    else:base_dir=CACHE/'deployment_bf16'
    base_bytes=folder_bytes(base_dir);adapter_bytes=checkpoint.stat().st_size if checkpoint else 0
    reference=folder_bytes(CACHE/'deployment_bf16')
    report={'arm':arm,'seed':seed,'fresh_process_pid':os.getpid(),'load_seconds':load_seconds,
            'measurements':samples,'base_artifact_bytes':base_bytes,'adapter_artifact_bytes':adapter_bytes,
            'total_artifact_bytes':base_bytes+adapter_bytes,'bf16_small_reference_bytes':reference,
            'storage_ratio':(base_bytes+adapter_bytes)/reference,'storage_cap':.6,
            'teacher_or_fp_copy_present':False,'inference_client':0 if arm.startswith('F_') else None,
            'actual_serialized_packed_artifact_loaded':arm.startswith('Q_') and arm!='Q_FP',
            'packed_modules':packed,'all_model_parameters_on_cuda':True,
            'artifact_scope':'workflow state (all four clients) plus common backbone' if arm.startswith('F_') else 'single deployed model plus adapter',
            'checkpoint_sha256':sha(checkpoint) if checkpoint else None,
            'deployment_files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in base_dir.iterdir() if p.is_file()},
            'model_trainable_count':sum(p.numel() for p in net.parameters() if p.requires_grad),
            'buffer_hash_inference':tensor_hash(net.named_buffers())}
    write(RESULTS/'resources'/f'{arm}_{seed}.json',report)


if __name__=='__main__':measure(sys.argv[1],int(sys.argv[2]))
