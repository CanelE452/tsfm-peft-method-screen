import gc
import torch
from torch import nn
from tsfm_peft_screen.memory.compression import CompressedSaved,checkpoint_blocks

def fixture(mode):
    m=nn.Linear(2,2)
    return CompressedSaved(m,mode,2,12)

def test_fp16_alias_and_original_storage_release():
    c=fixture('fp16');x=torch.randn(2,12,768);view=x.transpose(0,1)
    a=c.pack(x);b=c.pack(view)
    assert a[0] is b[0] and c.stats['alias_hits']==1
    assert a[0].data.dtype==torch.float16
    torch.testing.assert_close(c.unpack(b),view,rtol=1e-3,atol=1e-3)
    weak=a[0].weak
    del x,view;gc.collect()
    assert weak.expired(),'Compressed payload must not retain original allocation'
    del a,b;gc.collect();assert len(c.cache)==0

def test_zero_sign_preserved_for_relu_backward():
    c=fixture('fp16');x=torch.zeros(2,12,768);x.flatten()[:4]=torch.tensor([1e-20,-1e-20,0,1.])
    y=c.unpack(c.pack(x))
    assert torch.equal(x>0,y>0) and torch.equal(x<0,y<0)

def test_matched_bytes_and_exact_special_tokens():
    x=torch.randn(2,12,768)
    a=fixture('temporal').pack(x);b=fixture('int8').pack(x)
    assert a[0].nbytes()==b[0].nbytes()
    for p in [a,b]:assert torch.equal(CompressedSaved.unpack(p)[:,-4:],x[:,-4:])
    y=CompressedSaved.unpack(a)
    expected=x[:,:8].reshape(2,4,2,768).mean(2).half().float().repeat_interleave(2,1)
    assert torch.equal(y[:,:8],expected)

def test_group_attention_physical_layout():
    x=torch.randn(2,12,768);tbc=x.transpose(0,1).contiguous()
    c=fixture('temporal');c.stack=['encoder.block.0.layer.1.self_attention.q']
    p=c.pack(tbc.reshape(24,768))
    assert p[0].layout=='tbc'
    restored=c.unpack(p).reshape(12,2,768).transpose(0,1)
    assert torch.equal(restored[:,-4:],x[:,-4:])
    c=fixture('temporal');p=c.pack(x.transpose(0,1))
    assert p[0].layout=='btc'
    assert torch.equal(c.unpack(p)[-4:],x.transpose(0,1)[-4:])

def test_checkpoint_gradients_and_restore():
    m=nn.Module();m.encoder=nn.Module();m.encoder.block=nn.ModuleList([nn.Linear(3,3),nn.Linear(3,3)])
    x=torch.randn(2,3)
    def run():
        y=x
        for block in m.encoder.block:y=block(y).relu()
        y.sum().backward()
        return [p.grad.clone() for p in m.parameters()]
    expected=run();m.zero_grad()
    originals=[b.forward for b in m.encoder.block]
    with checkpoint_blocks(m):actual=run()
    for a,b in zip(actual,expected):assert torch.equal(a,b)
    assert [b.forward for b in m.encoder.block]==originals
