import gc,itertools
import torch
from tsfm_peft_screen.memory.local_backward import LocalA,LocalCodecs,LocalPayload

def trial(mode,group=False):
    torch.manual_seed(17)
    x=torch.randn(2,12,9,dtype=torch.float64)
    if group:x=x.transpose(0,1)
    x=x.detach().requires_grad_();a=torch.randn(3,9,dtype=x.dtype,requires_grad=True);b=torch.randn(7,3,dtype=x.dtype,requires_grad=True)
    incoming=torch.randn(*x.shape[:-1],7,dtype=x.dtype)
    def compute(custom):
        z=LocalA.apply(x,a,LocalCodecs(mode),group) if custom else torch.nn.functional.linear(x,a)
        y=2*torch.nn.functional.linear(z,b)
        return y,torch.autograd.grad(y,(x,a,b),incoming)
    expected,eg=compute(False);actual,ag=compute(True)
    assert torch.equal(actual,expected)
    torch.testing.assert_close(ag[0],eg[0],rtol=1e-12,atol=1e-12)
    assert torch.equal(ag[2],eg[2])
    if mode in ('exact','all_details'):torch.testing.assert_close(ag[1],eg[1],rtol=1e-12,atol=1e-12)
    return ag,eg

def test_all_codecs_exact_propagation():
    for mode in ['exact','fp16','care','prac','residual','all_details']:
        for group in [False,True]:trial(mode,group)

def test_exhaustive_sampling_expectation():
    torch.manual_seed(5);x=torch.randn(1,10,5,dtype=torch.float64);u=torch.randn(1,10,3,dtype=torch.float64)
    mean=(x[:,:-4:2]+x[:,1:-4:2])/2;delta=x[:,:-4:2]-x[:,1:-4:2]
    p=torch.tensor([.2,.3,.5],dtype=x.dtype);expected=u.reshape(-1,3).T@x.reshape(-1,5);average=torch.zeros_like(expected)
    for choices in itertools.product(range(3),repeat=2):
        idx=torch.tensor([choices]);detail=delta[:,idx[0]]
        payload=LocalPayload('residual',(mean,x[:,-4:],detail,idx,1/p[idx]),tuple(x.shape),False)
        average+=p[idx].prod()*payload.grad(u)
    torch.testing.assert_close(average,expected,rtol=1e-12,atol=1e-12)

def test_payload_sharing_and_no_original_retention():
    x=torch.randn(2,12,9);a=torch.randn(3,9);z=x@a.T
    for mode in ['fp16','prac','residual']:
        c=LocalCodecs(mode);first=c.pack(x,z,False);second=c.pack(x,z,False)
        assert first is second and c.stats['shared_hits']==1
        assert all(v.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr() for v in first.parts if isinstance(v,torch.Tensor))
    weak=first.weak;del x;gc.collect();assert weak.expired()
