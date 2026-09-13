"""Storage-only variants of the historical forecast-query computation.

The historical model remains unchanged. Every recomputation receives immutable
tensor inputs and a bound layer, never a mutable cache or a loop-variable closure.
"""
from contextlib import contextmanager, nullcontext
from functools import partial
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from chronos.chronos2.layers import Chronos2RotaryEmbedding
from .model import ForecastModel
from ..memory.compression import checkpoint_blocks


def query_block(block, h, past_k, past_v, positions, time_mask, group_mask):
    layer = block.layer[0]
    a = layer.self_attention
    norm = layer.layer_norm(h)
    def shape(v):
        return v.reshape(v.shape[0], v.shape[1], a.n_heads, a.kv_proj_dim).transpose(1, 2)
    q = shape(a.q(norm))
    k = shape(torch.cat((past_k, a.k(norm)), dim=1))
    v = shape(torch.cat((past_v, a.v(norm)), dim=1))
    cos, sin = a.rope_embed(v, positions)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    n = h.shape[1]
    q = q * cos[:, :, -n:] + Chronos2RotaryEmbedding.rotate_half(q) * sin[:, :, -n:]
    k = k * cos + Chronos2RotaryEmbedding.rotate_half(k) * sin
    out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=time_mask, scale=1.)
    h = h + a.o(out.transpose(1, 2).reshape(h.shape[0], n, h.shape[2]))
    h = block.layer[1](h, attention_mask=group_mask)[0]
    return block.layer[2](h)


class DiagnosticModel(ForecastModel):
    def __init__(self, arm, seed=30000):
        assert arm in ('standard', 'query')
        super().__init__(arm, seed)
        self.checkpoint_enabled = False
        self.observer = None

    def phase(self, name):
        return self.observer.phase(name) if self.observer else nullcontext()

    def query_hidden(self, c):
        h = c['initial']
        assert len(c['k']) == len(c['v']) == 12
        for i, block in enumerate(self.base.encoder.block):
            args = (h, c['k'][i], c['v'][i], c['positions'],
                    c['time_mask'], c['group_mask'][-4:])
            fn = partial(query_block, block)
            # partial binds this exact block; all cache values are explicit inputs.
            h = checkpoint(fn, *args, use_reentrant=False) if self.checkpoint_enabled else fn(*args)
        return self.base.encoder.final_layer_norm(h)[:, -3:]

    def forward(self, x, g):
        if self.arm == 'standard':
            ctx = checkpoint_blocks(self.base) if self.checkpoint_enabled else nullcontext()
            with self.phase('standard_encoder'), ctx:
                enc, (loc, scale), _, n = self.base.encode(context=x, group_ids=g, num_output_patches=3)
                assert n == (x.shape[-1] + 15) // 16
                h = enc.last_hidden_state[:, -3:]
        else:
            with self.phase('frozen_encoder_and_cache'):
                c, loc, scale = self.encode_frozen(x, g)
            if self.observer:
                self.observer.cache(c)
            with self.phase('query_branch'):
                h = self.query_hidden(c)
        with self.phase('output_head'):
            z = self.base.output_patch_embedding(h).reshape(len(x), 3, 21, 16).permute(0, 2, 1, 3).reshape(len(x), 21, 48).float()
            p = z.sinh() * scale[:, None, :] + loc[:, None, :]
        return z, p, loc, scale


class StorageObserver:
    """Forward saved-storage union, deduplicated by live storage identity.

    Weak storage identities avoid pointer-reuse collisions without retaining GPU
    allocations. This is a union across pack calls, NOT simultaneous peak memory.
    Nested checkpoint hooks hide their own internal saves; allocator peaks are
    measured separately. No instrumentation is used in the 24 timing trials.
    """
    def __init__(self, model):
        self.model = model
        self.rows = {}
        self.weak = []
        self.phases = []
        self.phase_name = 'outside'
        self.cache_summary = None
        self.parameters = {p.untyped_storage()._cdata for p in list(model.parameters()) + list(model.buffers())}

    @contextmanager
    def phase(self, name):
        previous = self.phase_name
        self.phase_name = name
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.phases.append(dict(phase=name, start_allocated_bytes=start,
                                    end_allocated_bytes=torch.cuda.memory_allocated(),
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                    peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            self.phase_name = previous

    def cache(self, c):
        def union(tensors):
            return sum({t.untyped_storage()._cdata: t.untyped_storage().nbytes() for t in tensors}.values())
        kv = c['k'] + c['v']
        all_tensors = kv + [v for v in c.values() if isinstance(v, torch.Tensor)]
        self.cache_summary = dict(kv_unique_bytes=union(kv), all_unique_bytes=union(all_tensors),
                                  kv_tensors=len(kv), kv_requires_grad=any(t.requires_grad for t in kv))

    def pack(self, t):
        from torch.multiprocessing.reductions import StorageWeakRef
        s = t.untyped_storage()
        key = s._cdata
        if key not in self.rows:
            self.weak.append(StorageWeakRef(s))
            self.rows[key] = dict(bytes=s.nbytes(), parameter_storage=key in self.parameters,
                                  device=str(t.device), views=set(), owners=set(), calls=0)
        row = self.rows[key]
        row['calls'] += 1
        row['views'].add((tuple(t.shape), tuple(t.stride()), t.storage_offset(), str(t.dtype)))
        row['owners'].add(self.phase_name)
        return t.detach()

    def result(self):
        rows = []
        for row in self.rows.values():
            rows.append(dict(**{k: v for k, v in row.items() if k not in ('views', 'owners')},
                             views=[dict(shape=v[0], stride=v[1], offset=v[2], dtype=v[3]) for v in sorted(row['views'])],
                             owners=sorted(row['owners'])))
        return dict(phases=self.phases, cache=self.cache_summary, storages=rows,
                    saved_unique_cuda_nonparameter_bytes=sum(r['bytes'] for r in rows if not r['parameter_storage'] and r['device'].startswith('cuda')),
                    scope='Union of outer saved-tensor pack calls; not peak residency, not all temporaries, checkpoint internals may be hidden.')
