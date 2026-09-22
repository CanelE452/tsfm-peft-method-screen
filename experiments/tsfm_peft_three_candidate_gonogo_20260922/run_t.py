from common import *
from model import *
from engine import *
import subprocess

ARMS=['T_RECENT','T_KD','T_BLEND','T_DELTA']


def build_model(size,seed):
    p,b=load_base(size)
    return attach(b,seed)


def preflight():
    import data
    d=data.load();p=Panel(d['values'],d['sigma'])
    net=build_model('small',92201)
    smoke(net,p,data.schedule('T_OLD_TRAIN',92201),'T_smoke_small',True)
    write(RESULTS/'smoke'/'T_small_MODULES.json',audit(net))
    del net,p,d;clear()
    subprocess.run([sys.executable,'-u',str(EXP/'run_t.py'),'smoke_base'],check=True,cwd=ROOT)


def smoke_base():
    from bridge_data import load_bridge
    reads,denied=guard_old_access()
    packet=load_bridge(92201)
    p=Panel(packet['values'],packet['sigma'],packet['offset'])
    net=build_model('base',92201)
    smoke(net,p,packet['schedule'],'T_smoke_base',True,True)
    write(RESULTS/'smoke'/'T_base_MODULES.json',audit(net))
    with torch.no_grad():
        x=p.batch(packet['schedule'][:8].reshape(-1,2),False)
        assert torch.isfinite(predict(net,x)).all()
    del net;clear()
    # Native64 parity for the actual new backbone, separate from zero-LoRA parity.
    pipeline,base=load_base('base')
    x=p.batch(packet['schedule'][0],False)
    with torch.no_grad():
        direct=predict(base,x).cpu()
        official=pipeline.predict(x,prediction_length=64).transpose(1,2).float().cpu()
    assert torch.equal(direct,official)
    write(RESULTS/'T_NATIVE_PARITY.json',{'model':'base','max_abs':float((direct-official).abs().max()),'input_sha256':tensor_hash([('x',x)])})
    del pipeline,base;clear()


@torch.no_grad()
def tuple_forecast(net,panel,tuples):
    return np.concatenate([predict(net,panel.batch(tuples[i:i+64],False)).cpu().numpy()
                           for i in range(0,len(tuples),64)])


def teachers():
    import data
    from bridge_data import load_bridge
    d=data.load();old=Panel(d['values'],d['sigma'])
    for seed in [92201,92202]:
        run=f'T_A1_{seed}'
        if not (RESULTS/'fits'/run/'FIT.json').exists():
            net=build_model('small',seed)
            fit(net,old,data.schedule('T_OLD_TRAIN',seed),d['origins']['T_OLD_VAL'],'T_A1',seed)
            del net;clear()
    del old,d
    for seed in [92201,92202]:
        packet=load_bridge(seed)
        panel=Panel(packet['values'],packet['sigma'],packet['offset'])
        tuples=packet['schedule'].reshape(-1,2)
        vorigin=packet['origins']
        for name,size in [('A0','small'),('A1','small'),('N0','base')]:
            output=CACHE/'teacher'/f'{name}_{seed}.npz'
            if output.exists():
                assert (RESULTS/'teacher'/f'{name}_{seed}.json').exists()
                continue
            output.parent.mkdir(exist_ok=True)
            start=time.perf_counter()
            if name=='A1':
                net=build_model(size,seed)
                chosen=read(RESULTS/'fits'/f'T_A1_{seed}'/'FIT.json')['selected_step']
                checkpoint=CACHE/'fits'/f'T_A1_{seed}'/f'step{chosen}.pt'
                restore(net,torch.load(checkpoint,weights_only=True))
                teacher_hash=sha(checkpoint)
            else:
                _,net=load_base(size);teacher_hash=None
            train=tuple_forecast(net,panel,tuples).reshape(256,8,64,9)
            val=forecast(net,panel,vorigin,list(range(16)))
            np.savez(output,train=train,val=val)
            write(RESULTS/'teacher'/f'{name}_{seed}.json',{
                'model':read(RESULTS/'UPSTREAM.json')['models'][size], 'seed':seed,
                'teacher_checkpoint_sha256':teacher_hash,'schedule_bytes_sha256':hashlib.sha256(tuples.tobytes()).hexdigest(),
                'bridge_values_sha256':hashlib.sha256(packet['values'].tobytes()).hexdigest(),
                'seconds':time.perf_counter()-start,'example_queries':len(tuples)+len(vorigin)*16,
                'batch_calls':int(np.ceil(len(tuples)/64)+np.ceil(len(vorigin)*16/64)),
                'cache_bytes':output.stat().st_size,'cache_sha256':sha(output),
                'bridge_validation':score(val,panel,vorigin,list(range(16)))})
            del net;clear()
            event('teacher_cache_complete',name=name,seed=seed)


def guard_old_access():
    reads=[]
    denied=(CACHE/'data').resolve()
    def access_hook(event_name,args):
        if event_name!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(denied):raise RuntimeError('Student attempted OLD/full raw access: '+str(path))
        if path.is_relative_to(CACHE) and path.suffix in ['.npz','.npy','.pt','.json']:reads.append(str(path.relative_to(CACHE)))
    sys.addaudithook(access_hook)
    return reads,denied


def student(arm,seed):
    from bridge_data import load_bridge
    reads,denied=guard_old_access()
    packet=load_bridge(seed)
    panel=Panel(packet['values'],packet['sigma'],packet['offset'])
    pseudo=None
    if arm!='T_RECENT':
        a1=np.load(CACHE/'teacher'/f'A1_{seed}.npz')['train']
        if arm=='T_KD':pseudo=np.sort(a1,axis=-1)
        else:
            n0=np.load(CACHE/'teacher'/f'N0_{seed}.npz')['train']
            if arm=='T_BLEND':pseudo=np.sort(.5*n0+.5*a1,axis=-1)
            else:
                a0=np.load(CACHE/'teacher'/f'A0_{seed}.npz')['train']
                pseudo=np.sort(n0+a1-a0,axis=-1)
    net=build_model('base',seed)
    report=fit(net,panel,packet['schedule'],packet['origins'],arm,seed,pseudo)
    checkpoint=CACHE/'fits'/f'{arm}_{seed}'/f"step{report['selected_step']}.pt"
    keys=list(torch.load(checkpoint,weights_only=True))
    assert all('.lora_A.' in k or '.lora_B.' in k for k in keys)
    write(RESULTS/'fits'/f'{arm}_{seed}'/'ACCESS.json',{'raw_old_directory_blocked':str(denied),
          'cache_reads':sorted(set(reads)),'student_keys':keys,'teacher_modules_in_student':False,
          'bridge_offset':packet['offset'],'contexts_min_start':int(packet['schedule'][:,:,1].min()-512)})


def main():
    teachers()
    for seed in [92201,92202]:
        for arm in ARMS:
            if (RESULTS/'fits'/f'{arm}_{seed}'/'FIT.json').exists():continue
            subprocess.run([sys.executable,'-u',str(EXP/'run_t.py'),'student',arm,str(seed)],check=True,cwd=ROOT)
    event('T_training_complete')


if __name__=='__main__':
    action=sys.argv[1]
    if action=='smoke':preflight()
    elif action=='smoke_base':smoke_base()
    elif action=='train':main()
    elif action=='student':student(sys.argv[2],int(sys.argv[3]))
