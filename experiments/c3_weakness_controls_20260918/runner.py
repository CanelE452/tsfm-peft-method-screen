import signal,traceback
from .common import *
from .prepare import prepare
from . import train
from .checks import smoke
from .evaluate import evaluate

def run():
    setup();prepare();check_seal();signal.signal(signal.SIGTERM,train.request_stop);signal.signal(signal.SIGINT,train.request_stop);watch=Watch()
    try:
        watch.boundary(startup=True);smoke(watch);models=[];choices={}
        for source in SOURCES:
            choices[source]={}
            for arm in ARMS:
                for lr in [.0001,.0003]:train.fit(source,arm,81550,lr,watch)
                c=train.choice(source,arm,81550);choices[source][arm]=c;save(OUT/'LR_SELECTION.json',choices)
                for seed in [81551,81552,81553]:
                    r=train.fit(source,arm,seed,c['lr'],watch);models.append(dict(source=source,arm=arm,seed=seed,fit=r['fit'],lr=r['lr'],**r['selected']));save(OUT/'MODEL_SELECTION.json',models)
        assert len(models)==18;check_seal();evaluate(watch)
        from .score import score
        score();status(execution_result='COMPLETE_COMPUTE',automatic_successor=False)
    except BaseException as e:
        save(OUT/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));status(execution_result='RESOURCE_BLOCK' if isinstance(e,ResourceError) else 'IMPLEMENTATION_ERROR',error=str(e));raise
    finally:watch.close()
if __name__=='__main__':run()
