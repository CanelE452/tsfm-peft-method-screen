from functools import partial
import copy
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2EncoderBlock
from tsfm_peft_screen.forecast_query.checkpoint_diagnostic import query_block


def fixture():
    torch.manual_seed(123)
    cfg = Chronos2CoreConfig(d_model=16, d_kv=8, d_ff=32, num_heads=2,
                            num_layers=2, dropout_rate=0.)
    layers = nn.ModuleList([Chronos2EncoderBlock(cfg) for _ in range(2)])
    h = torch.randn(2, 4, 16)  # frozen input: requires_grad=False
    positions = torch.arange(10).reshape(1, 10).expand(2, -1)
    time = torch.zeros(2, 1, 1, 10)
    time[:, :, :, 1] = -torch.inf
    group = torch.zeros(4, 1, 2, 2)
    caches = [(torch.randn(2, 6, 16), torch.randn(2, 6, 16)) for _ in layers]
    return layers, h, positions, time, group, caches


def test_checkpoint_frozen_inputs_multilayer_output_gradient_update():
    layers, h, pos, time, group, caches = fixture()
    initial = copy.deepcopy(layers.state_dict())
    results = []
    for enabled in (False, True):
        layers.load_state_dict(initial)
        opt = torch.optim.AdamW(layers.parameters(), lr=1e-3, weight_decay=0.)
        out = h
        for layer, (k, v) in zip(layers, caches):
            fn = partial(query_block, layer)
            args = (out, k, v, pos, time, group)
            out = checkpoint(fn, *args, use_reentrant=False) if enabled else fn(*args)
        out.square().mean().backward()
        grads = torch.cat([p.grad.flatten() for p in layers.parameters()])
        opt.step()
        results.append((out.detach(), grads, torch.cat([p.detach().flatten() for p in layers.parameters()])))
        opt.zero_grad(set_to_none=True)
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    assert results[0][1].abs().sum() > 0
    assert all(k.grad is None and v.grad is None for k, v in caches)


def test_query_block_masks_past_key_and_group():
    layers, h, pos, time, group, caches = fixture()
    k, v = caches[0]
    fn = partial(query_block, layers[0])
    a = fn(h, k, v, pos, time, group)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:, 1] += 1000
    changed_v[:, 1] += 1000
    torch.testing.assert_close(a, fn(h, changed_k, changed_v, pos, time, group), rtol=0, atol=0)
    group[:, :, 0, 1] = -torch.inf
    group[:, :, 1, 0] = -torch.inf
    a = fn(h, k, v, pos, time, group)
    changed_h = h.clone()
    changed_h[1] += 100
    b = fn(changed_h, k, v, pos, time, group)
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
