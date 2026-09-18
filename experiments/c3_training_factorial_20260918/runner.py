import signal,traceback
from .common import *
from .prepare import prepare
from .checks import cpu_checks,smoke
from . import train
from .evaluate import evaluate

def run():
 setup();prepare();check_seal();cpu_checks();signal.signal(signal.SIGTERM,train.request_stop);signal.signal(signal.SIGINT,train.request_stop);watch=Watch()
 try:
  watch.boundary(startup=True);smoke(watch);models=[]
  for cell in read(OUT/'GRID.json'):
   r=train.fit(cell['source'],cell['arm'],cell['seed'],cell['lr'],watch)
   for stage,checkpoint in [('fixed1024',next(c for c in r['checkpoints'] if c['step']==1024)),('selected',r['selected'])]:models.append(dict(source=cell['source'],arm=cell['arm'],seed=cell['seed'],fit=cell['fit'],lr=cell['lr'],stage=stage,**checkpoint))
   save(OUT/'MODEL_SELECTION.json',models)
  assert len(models)==64;evaluate(watch)
  from .score import score
  score();status(execution_result='COMPLETE_COMPUTE')
 except BaseException as e:
  save(OUT/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));status(execution_result='RESOURCE_BLOCK' if isinstance(e,ResourceError) else 'IMPLEMENTATION_ERROR',error=str(e));raise
 finally:watch.close()
if __name__=='__main__':run()
