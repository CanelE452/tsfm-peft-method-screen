from common import *
from model import *
from engine import *
from quantization import make_q
import data

ARMS=['Q_FP','Q_STD','Q_LOFTQ','Q_QERA','Q_IO16','Q_FORECAST']


def preflight():
    d=data.load();panel=Panel(d['values'],d['sigma'])
    schedule=data.schedule('Q_TRAIN',92201)
    for arm in ARMS:
        net=make_q(arm,92201)
        initial=smoke(net,panel,schedule,'Q_smoke_'+arm,parity=arm in ['Q_FP','Q_STD'])
        save_state(CACHE/'initial'/f'{arm}_92201.pt',initial)
        write(RESULTS/'smoke'/f'{arm}_MODULES.json',audit(net))
        event('smoke_complete',arm=arm)
        del net;clear()
    # Verify base supports eval batch64 using actual contexts, without optimizer calls.
    net=make_q('Q_FORECAST',92201,init=False)
    with torch.no_grad():
        q=predict(net,panel.batch(schedule[:8].reshape(-1,2),False))
    assert q.shape==(64,64,9) and torch.isfinite(q).all()
    del net;clear()


def main():
    d=data.load();panel=Panel(d['values'],d['sigma'])
    for seed in [92201,92202]:
        schedule=data.schedule('Q_TRAIN',seed)
        for arm in ARMS:
            run=f'{arm}_{seed}'
            if (RESULTS/'fits'/run/'FIT.json').exists():continue
            initial=CACHE/'initial'/f'{run}.pt'
            net=make_q(arm,seed,init=not initial.exists())
            if initial.exists():restore(net,torch.load(initial,weights_only=True))
            fit(net,panel,schedule,d['origins']['Q_VAL'],arm,seed)
            del net;clear()
    event('Q_training_complete')


if __name__=='__main__':
    if sys.argv[1]=='smoke':preflight()
    elif sys.argv[1]=='train':main()
