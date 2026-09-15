"""Post-seal descriptive probe on eight training inputs; CPU FP32, no fits."""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from model import ROOT, make
from common import read, save, sha
from rank2_data import load, batch

OUT = ROOT / 'results/channel_attention_prior_20260916'
DATA = ROOT / '.cache/channel_sharing_screen_v1_20260915'


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(9019)
    seal = read(OUT / 'seal.json')
    for p, h in seal['source_hashes'].items():
        assert sha(ROOT / p) == h
    records = []
    start = time.monotonic()
    for source, data in seal['data'].items():
        origins = data['origins']['train'][:8]
        values, _ = load(DATA, source, 'development')
        x, _ = batch(values, origins, 'cpu')
        model = make('PRIOR', data['channel_ids'], 41000, 'cpu').eval()
        hooks = []
        def hook(i):
            def capture(module, args):
                logp = args[2].detach().double()
                p = logp.exp()
                assert torch.isfinite(p).all()
                row_sum_error = float((p.sum(-1) - 1).abs().max())
                entropy = -(p * logp).sum(-1)
                kl = (p * (logp + math.log(p.shape[-1]))).sum(-1)
                l1 = (p - 1 / p.shape[-1]).abs().sum(-1)
                records.append(dict(
                    dataset=source, layer=i, origins=origins,
                    rows=p.shape[0] * p.shape[1],
                    max_probability_sum_error=row_sum_error,
                    mean_normalized_entropy=float(entropy.mean() / math.log(p.shape[-1])),
                    mean_KL_to_uniform=float(kl.mean()),
                    mean_L1_to_uniform=float(l1.mean()),
                    median_max_probability=float(p.max(-1).values.median()),
                    mean_self_probability=float(p.diagonal(dim1=-2, dim2=-1).mean())))
            return capture
        for i, layer in enumerate(model.side):
            hooks.append(layer.register_forward_pre_hook(hook(i)))
        with torch.no_grad():
            model(x)
        for h in hooks:
            h.remove()
        del model, x, values
    assert len(records) == 16
    save(OUT / 'prior_probe.json', dict(
        description='Post-seal descriptive mechanism probe; not a new selection or success rule.',
        precision='CPU FP32 backbone, statistics accumulated in FP64. '
                  'Not claimed bitwise equal to BF16 training attention.',
        scope='First eight existing training origins per source, all 32 channels; '
              'no V/E prediction or target scoring.',
        new_fits=0, optimizer_updates=0, source_sha256=sha(Path(__file__)),
        seconds=time.monotonic()-start, records=records))
    print(json.dumps(records, ensure_ascii=False))


if __name__ == '__main__':
    main()
