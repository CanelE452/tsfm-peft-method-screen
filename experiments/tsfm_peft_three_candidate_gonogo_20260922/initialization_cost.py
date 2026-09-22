from common import *
from model import *
from quantization import make_q,initialize


if __name__=='__main__':
    arm=sys.argv[1]
    model=make_q(arm,92201,init=False)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    initialize(model,arm)
    torch.cuda.synchronize()
    write(RESULTS/'resources'/f'{arm}_initialization.json',{'arm':arm,'seconds':time.perf_counter()-started,
          'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
          'scope':'separate runtime-only replay of fixed initialization using already sealed statistics; no optimizer or forecast probe',
          'original_fit_initialization_time_was_not_individually_instrumented':True,'optimizer_updates':0})
