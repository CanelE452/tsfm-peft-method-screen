"""Finite temporal-response matching, not a claim of new Jacobian matching."""
import numpy as np
import torch
from experiments.outlier_signal_peft_v1_20260917.reference_core import rng

ARMS = ['PLAIN', 'ANCHOR', 'SHUFFLE', 'IDEAL', 'TRP']

def probes(source, epoch, indices, sigma, device='cpu'):
    """Independent of labels, clean histories, state/mask and model outputs.

    Suffix and shuffled views have exactly the same perturbation histogram.
    Draws keyed by epoch and example, independent of arm/seed/batch order.
    """
    suffix=[]; shuffled=[]; amplitude=[]; durations=[]
    for idx, s in zip(indices, np.asarray(sigma)):
        g=rng(91901, source, int(epoch), int(idx))
        d=int(g.choice([16,32,64,128]));a=np.float32(g.choice([-1,1])*g.uniform(2,8)*float(s))
        v=np.zeros(512,dtype=np.float32);v[-d:]=a
        suffix.append(v);shuffled.append(v[g.permutation(512)]);amplitude.append(a);durations.append(d)
    return (torch.as_tensor(np.stack(suffix),device=device),torch.as_tensor(np.stack(shuffled),device=device),
            torch.as_tensor(np.asarray(amplitude),device=device),np.asarray(durations))

def penalty(arm, p, pt, b, bt, sigma, amplitude):
    """Teacher outputs detached. All nine quantiles and 64 horizons included."""
    s=sigma[:,None,None]
    if arm=='PLAIN':return p.sum()*0
    if arm=='ANCHOR':return .5*((p-b.detach()).abs()/s+(pt-bt.detach()).abs()/s).mean()
    if arm in ['TRP','SHUFFLE']:
        return (((pt-p)-(bt.detach()-b.detach())).abs()/s).mean()
    if arm=='IDEAL':return ((pt-p-amplitude[:,None,None]).abs()/s).mean()
    raise ValueError(arm)
