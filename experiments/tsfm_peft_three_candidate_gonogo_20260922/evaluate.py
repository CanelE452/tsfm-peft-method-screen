from common import *
from model import *
from engine import Panel,forecast,score
from resource import load_selected
import subprocess

Q=['Q_FP','Q_STD','Q_LOFTQ','Q_QERA','Q_IO16','Q_FORECAST']
T=['T_A1','T_RECENT','T_KD','T_BLEND','T_DELTA']
F=['F_LOCAL','F_SHARED','F_AFFINE','F_HEAD','F_PERIODIC']


def resources():
    # Export untouched new-model base in a process that exits before resource measurements.
    if not (CACHE/'deployment_base').exists():
        subprocess.run([sys.executable,'-u',str(EXP/'evaluate.py'),'export_base'],check=True)
    for arm in Q+T+F+['A0','N0']:
        for seed in ([0] if arm in ['A0','N0'] else [92201,92202]):
            if (RESULTS/'resources'/f'{arm}_{seed}.json').exists():continue
            subprocess.run([sys.executable,'-u',str(EXP/'resource.py'),arm,str(seed)],check=True)
            event('resource_complete',arm=arm,seed=seed)


def validation(arm,seed):
    if arm=='N0':return read(RESULTS/'teacher'/f'N0_{seed}.json')['bridge_validation']['macro']['pinball']
    report=read(RESULTS/'fits'/f'{arm}_{seed}'/'FIT.json')
    item=next(x for x in report['validation'] if x['step']==report['selected_step'])
    return item['pinball'] if arm.startswith('F_') else item['score']['macro']['pinball']


def select():
    assert not (RESULTS/'SELECTIONS.json').exists()
    result={}
    for category,arms in [('Q',Q[1:-1]),('T',['N0','T_RECENT','T_KD','T_BLEND']),('F',F[:-1])]:
        result[category]={}
        for seed in [92201,92202]:
            allowed=[]
            for arm in arms:
                if category=='Q':
                    r=read(RESULTS/'resources'/f'{arm}_{seed}.json')
                    if r['storage_ratio']>.6:continue
                allowed.append((arm,validation(arm,seed)))
            assert allowed,'No allowed baseline; requires resource block decision'
            winner=min(allowed,key=lambda pair:pair[1])[0]
            result[category][str(seed)]={'baseline':winner,'validation_scores':dict(allowed),
                                        'selection_uses_test':False}
    result['checkpoints']={f'{arm}_{seed}':read(RESULTS/'fits'/f'{arm}_{seed}'/'FIT.json')['selected_checkpoint_sha256']
                           for arm in Q+T+F for seed in [92201,92202]}
    result['time']=time.time()
    write(RESULTS/'SELECTIONS.json',result)
    event('all_selections_sealed',sha256=sha(RESULTS/'SELECTIONS.json'))


@torch.no_grad()
def predict_all():
    import data
    assert (RESULTS/'SELECTIONS.json').exists()
    assert not (RESULTS/'TEST_SCORES.json').exists()
    d=data.load();panel=Panel(d['values'],d['sigma']);origins=d['origins']['Q_TEST']
    assert np.array_equal(origins,d['origins']['T_TEST'])
    folder=CACHE/'test_predictions';folder.mkdir(exist_ok=True)
    manifest={}
    for arm in Q+T+F+['A0','N0']:
        for seed in ([0] if arm in ['A0','N0'] else [92201,92202]):
            key=f'{arm}_{seed}';path=folder/f'{key}.npy';started=time.perf_counter()
            if not path.exists():
                if arm.startswith('F_'):
                    net,head,_=load_selected(arm,seed,0)
                    report=read(RESULTS/'fits'/key/'FIT.json')
                    state=torch.load(CACHE/'fits'/key/f"round{report['selected_step']}.pt",weights_only=True)
                    outputs=[]
                    for client in range(4):
                        restore(net,state['local'][client] if arm=='F_LOCAL' else state['shared'])
                        head.load_state_dict(state['heads'][client])
                        outputs.append(forecast(net,panel,origins,[client],head))
                    q=np.concatenate(outputs,axis=1)
                else:
                    net,head,_=load_selected(arm,seed)
                    q=forecast(net,panel,origins,list(range(16)))
                assert np.isfinite(q).all()
                np.save(path,q)
                del net,head,q;clear()
                event('test_predictions_saved',arm=arm,seed=seed,seconds=time.perf_counter()-started)
            manifest[key]={'sha256':sha(path),'bytes':path.stat().st_size,'shape':list(np.load(path,mmap_mode='r').shape),
                           'path':str(path.relative_to(ROOT))}
    write(RESULTS/'TEST_PREDICTIONS_SEAL.json',{'selection_sha256':sha(RESULTS/'SELECTIONS.json'),
          'predictions':manifest,'all_saved_before_scoring':True,'time':time.time(),
          'origins_sha256':hashlib.sha256(origins.tobytes()).hexdigest()})


if __name__=='__main__':
    action=sys.argv[1]
    if action=='export_base':
        _,net=load_base('base');net.save_pretrained(CACHE/'deployment_base',safe_serialization=True)
    elif action=='resources':resources()
    elif action=='select':select()
    elif action=='predict':predict_all()
