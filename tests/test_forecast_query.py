import torch
from tsfm_peft_screen.forecast_query.model import Head,Side,SideLayer

def test_budget_and_initial_head_output():
    torch.manual_seed(3);h=torch.randn(2,3,768);head=Head();side=Side()
    assert sum(p.numel() for p in head.parameters())==1179648
    assert sum(p.numel() for p in side.parameters())==1179648
    assert torch.equal(head(h),h)
    head(h).square().mean().backward();assert head.b.weight.grad.abs().sum()>0

def test_side_zero_decoder_and_gradient_flow():
    torch.manual_seed(4);side=Side();features=[torch.randn(2,8,768) for _ in range(12)];base=torch.randn(2,3,768)
    time=torch.zeros(2,1,1,8);group=torch.zeros(8,1,2,2)
    y=side(features,time,group,base);assert torch.equal(y,base)
    y.square().mean().backward();assert side.out.weight.grad.abs().sum()>0
    assert all(f.grad is None for f in features)

def test_side_attention_honors_key_mask():
    torch.manual_seed(5);layer=SideLayer();h=torch.randn(2,8,64);mask=torch.zeros(2,1,1,8);mask[:,:,:,-1]=-torch.inf
    a=layer.attend(h,mask);h[:,-1]+=100;b=layer.attend(h,mask)
    torch.testing.assert_close(a[:,:-1],b[:,:-1],rtol=0,atol=0)
