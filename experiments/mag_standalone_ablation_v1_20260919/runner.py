import signal,traceback,warnings
from .common import *
from .prepare import prepare,smoke
from . import train

def run():
    setup();warnings.filterwarnings('ignore',message="input's size at dim=0.*");prepare();check_seal();require_authorization()
    signal.signal(signal.SIGTERM,train.request_stop);signal.signal(signal.SIGINT,train.request_stop)
    watch=Watch()
    try:
        watch.boundary(startup=True);smoke(watch);train.train_all(watch)
        from .evaluate import evaluate
        evaluate(watch)
        from .score import score
        score();status(execution='COMPLETE_COMPUTE')
    except BaseException as e:
        save(OUT/'ERROR.json',dict(at=time.time(),error=str(e),traceback=traceback.format_exc()));status(execution='RESOURCE_BLOCKED' if isinstance(e,ResourceError) else 'IMPLEMENTATION_ERROR',error=str(e));raise
    finally:watch.close()
    from .finalize import finalize
    finalize();status(execution='COMPLETE_VERIFIED')
if __name__=='__main__':run()
