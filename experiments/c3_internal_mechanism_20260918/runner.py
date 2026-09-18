import traceback
from .common import *
from .prepare import prepare
from .gradients import initial,order_first_batch,trajectory
from .evaluate import evaluate
from .score import score
from .cpu_checks import check as cpu_check

def run():
 setup();prepare();check_seal();cpu_check();watch=Watch()
 try:
  watch.boundary(startup=True)
  # No silent automatic retries of partially completed gradient probes.
  assert not (OUT/'COUNTS.json').exists(),'PARTIAL_RUN_REQUIRES_EXPLICIT_ACCOUNTING'
  initial(watch);order_first_batch(watch);trajectory(watch)
  from .moments import analyze
  analyze();save_counts();assert COUNTS['autograd_calls']==1196 and COUNTS['optimizer_updates']==0
  evaluate(watch);score();save(OUT/'status.json',dict(execution='COMPLETE_COMPUTE',**COUNTS,new_E_views=96,reused_E_views=96,automatic_followup=False))
 except BaseException as e:
  save_counts();save(OUT/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));save(OUT/'status.json',dict(execution='RESOURCE_BLOCK' if isinstance(e,ResourceError) else 'IMPLEMENTATION_ERROR',error=str(e),**COUNTS));raise
 finally:watch.close()
if __name__=='__main__':run()
