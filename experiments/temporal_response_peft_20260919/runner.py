import signal,traceback
from .common import *
from .prepare import prepare,cpu_checks
from .checks import smoke
from . import train

def run():
    setup();prepare();cpu_checks();check_seal();signal.signal(signal.SIGTERM,train.request_stop);signal.signal(signal.SIGINT,train.request_stop);watch=Watch()
    try:
        watch.boundary(startup=True);smoke(watch)
        for row in read(OUT/'GRID.json'):train.fit(row['source'],row['arm'],row['seed'],watch)
        from .evaluate import evaluate
        evaluate(watch)
        from .score import score
        score();status(execution='COMPLETE_COMPUTE')
    except BaseException as e:
        save(OUT/'ERROR.json',dict(at=time.time(),error=str(e),traceback=traceback.format_exc()));status(execution='RESOURCE_BLOCK' if isinstance(e,ResourceError) else 'IMPLEMENTATION_ERROR',error=str(e));raise
    finally:watch.close()
if __name__=='__main__':run()
