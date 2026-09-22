import hashlib
from types import MethodType
import torch
from torch import nn
from chronos import ChronosBoltPipeline, Chronos2Pipeline
from peft import LoraConfig, get_peft_model
from common import RESULTS, read

def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def tensor_hash(items):
    h = hashlib.sha256()
    for name, value in sorted(items):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def embeddings(self):
    return self.shared

def empirical_crps(z, y):
    m = z.shape[-1]
    ordered = z.sort(dim=-1).values
    coeff = 2 * torch.arange(1, m + 1, device=z.device, dtype=z.dtype) - m - 1
    return (z - y[..., None]).abs().mean(-1) - (ordered * coeff).sum(-1) / (m * m)

def pinball(q, y):
    tau = torch.arange(1, 10, device=q.device, dtype=q.dtype) / 10
    e = y[..., None] - q
    return (2 * torch.maximum(tau * e, (tau - 1) * e)).mean(-1)

def continuation_score(components, y, arm):
    # components: B,branch,H,quantile; each branch is a nine-atom measure.
    if arm == 'COMPONENT':
        return empirical_crps(components, y[:, None, :]).mean(1)
    if arm == 'MIXTURE':
        return empirical_crps(components.permute(0, 2, 1, 3).flatten(2), y)
    raise ValueError(arm)

def branch_context(x, first):
    # first B,Q,64 is not sorted: match the official quantile-labelled paths.
    return torch.cat((x[:, None].expand(-1, 9, -1), first.detach()), -1).flatten(0, 1)

class BranchModel(nn.Module):
    def __init__(self, seed=92231, lora=True):
        super().__init__()
        info = read(RESULTS / 'SOURCE_MANIFEST.json')['models']['amazon/chronos-bolt-small']
        self.pipeline = ChronosBoltPipeline.from_pretrained(info['path'], device_map='cuda', torch_dtype=torch.float32, local_files_only=True)
        base = self.pipeline.model
        base.requires_grad_(False)
        base.get_input_embeddings = MethodType(embeddings, base)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.network = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0., target_modules=['q','v'], bias='none')) if lora else base
        self.pipeline.model = self.network.get_base_model() if lora else base
        self.eval()

    def conditional(self, x):
        return self.network(context=x).quantile_preds

    def learned(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}

    def load_learned(self, state):
        params = dict(self.named_parameters())
        assert set(state) == {n for n,p in params.items() if p.requires_grad}
        with torch.no_grad():
            for n,v in state.items(): params[n].copy_(v)

    def frozen_hash(self):
        return tensor_hash([(n,p) for n,p in self.named_parameters() if not p.requires_grad] + list(self.named_buffers()))

    def forecast(self, x):
        first = self.conditional(x)
        ctx = branch_context(x, first)
        second = self.conditional(ctx).reshape(len(x), 9, 9, 64)
        atoms = second.reshape(len(x), 81, 64)
        levels = torch.tensor(self.pipeline.quantiles, device=x.device, dtype=x.dtype)
        reduced = torch.quantile(atoms, levels, dim=1).transpose(0, 1)
        q = torch.cat((first, reduced), -1).transpose(1, 2)
        return q, second.permute(0, 1, 3, 2)

def update(model, optimizer, x, y, sigma, arm):
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    first = model.conditional(x)
    ctx = branch_context(x, first)
    assert not ctx.requires_grad
    loss1 = .5 * (pinball(first.transpose(1, 2), y[:, :64]) / sigma[:, None]).mean()
    loss1.backward()
    second = model.conditional(ctx).reshape(len(x), 9, 9, 64).permute(0, 1, 3, 2)
    loss2 = .5 * (continuation_score(second, y[:, 64:], arm) / sigma[:, None]).mean()
    loss2.backward()
    pars = [p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pars)
    norm = torch.nn.utils.clip_grad_norm_(pars, 1.)
    assert torch.isfinite(norm)
    optimizer.step()
    return dict(loss=float(loss1.detach()+loss2.detach()), first_loss=float(loss1.detach()), continuation_loss=float(loss2.detach()), grad_norm=float(norm))

def load_direct():
    info = read(RESULTS / 'SOURCE_MANIFEST.json')['models']['amazon/chronos-2']
    return Chronos2Pipeline.from_pretrained(info['path'], device_map='cuda', torch_dtype=torch.float32, local_files_only=True)

@torch.no_grad()
def direct(model, x):
    qs, _ = model.predict_quantiles([r[None].cpu() for r in x], prediction_length=128, quantile_levels=[i/10 for i in range(1,10)], batch_size=len(x), context_length=512, cross_learning=False)
    return torch.cat(qs, dim=0)
