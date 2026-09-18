from types import SimpleNamespace
from .common import *

def check():
 torch.manual_seed(90418);h=torch.randn(4,3,5,dtype=torch.float64);g=torch.rand(4,3,dtype=torch.float64);r=torch.randn_like(h)
 down=torch.nn.Linear(5,2,dtype=torch.float64);up=torch.nn.Linear(2,5,dtype=torch.float64);torch.nn.init.zeros_(up.weight);torch.nn.init.zeros_(up.bias);adapter=SimpleNamespace(down=down,up=up)
 u=h+capof(h)*g[...,None]*torch.tanh(up(torch.nn.functional.gelu(down(h))));loss=(u*r).sum();gd,gb,gu,gub=torch.autograd.grad(loss,[down.weight,down.bias,up.weight,up.bias]);actual,_=formula(h,g,r,adapter);expected=torch.cat([gu.flatten(),gub]);torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12);assert torch.count_nonzero(gd)==0 and torch.count_nonzero(gb)==0
 a=torch.randn(10,dtype=torch.float64);b=torch.randn(10,dtype=torch.float64);c=torch.randn(10,dtype=torch.float64);d=torch.randn(10,dtype=torch.float64);J=.5*(c-a+d-b);V=.5*(b-a+d-c);torch.testing.assert_close(J+V,d-a,rtol=1e-12,atol=1e-12)
 meta=[]
 for source in SOURCES:
  generator=read(old.CACHE/'conditions'/source/'train_generator_audit.json');lookup={(r['epoch'],r['example']):r for r in generator};chosen=[lookup[e['epoch'],i] for e in read(OUT/'PROBE.json') for i in e['indices']];counts={s:sum(r['state']==s for r in chosen) for s in ['REFERENCE','POINT','BURST','SHIFT']};assert set(counts.values())=={32};shifts={(r['amplitude'],r['duration']) for r in chosen if r['state']=='SHIFT'};assert shifts=={(4,24),(8,24),(4,48),(8,48)};meta.append(dict(source=source,probe_contexts=128,state_counts=counts,shift_magnitude_duration=sorted(shifts)))
 save(OUT/'CPU_CHECKS.json',dict(status='VERIFIED',synthetic_chain_rule=True,zero_down_gradient=True,Jacobian_signal_identity=True,probe=meta,optimizer_updates=0));print('CPU_VERIFIED',flush=True)
if __name__=='__main__':check()
