"""Locally authored from the user's sole contract, authorized 2026-09-17.

Not recovered attachment code. Pure input-only transforms; labels stay outside.
"""
import hashlib
import numpy as np

STATES = ['REFERENCE','POINT4','POINT8','POINT16','BURST4','BURST8','BURST16','SHIFT4','SHIFT8','SHIFT_POINT']
SOURCES = ['electricity','ettm1']

def rng(*key):
    seed = int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:8], 'little')
    return np.random.default_rng(seed)

def scale(x, sigma):
    return max(1.4826*np.median(np.abs(x-np.median(x))), .1*float(sigma))

def transform(x0, sigma, state, random, training=False):
    """Return only observed past and a label offset, without consulting future."""
    x = np.array(x0, dtype=np.float64, copy=True)
    r = scale(x,sigma)
    delta = 0.
    audit = dict(state=state, r0=float(r), positions=[], shift_length=0, delta=0.)
    if state.startswith('SHIFT'):
        duration = int(random.choice([24,48])) if training else 32
        amp = 4 if training or state in ['SHIFT4','SHIFT_POINT'] else 8
        delta = float(random.choice([-1,1])*amp*r)
        x[-duration:] += delta
        audit.update(shift_length=duration,delta=delta)
    if state.startswith('POINT') or state == 'SHIFT_POINT':
        count = int(random.choice([1,2,4])) if training else 2
        amp = 8 if training or state == 'SHIFT_POINT' else int(state[5:])
        positions = random.choice(len(x),count,replace=False)
        signed = random.choice([-1,1],count)*amp*r
        x[positions] += signed
        audit.update(positions=positions.tolist(),fault_delta=signed.tolist())
    if state.startswith('BURST'):
        duration = int(random.choice([2,4])) if training else 8
        amp = 8 if training else int(state[5:])
        start = int(random.integers(len(x)-duration+1))
        positions = np.arange(start,start+duration)
        signed = float(random.choice([-1,1])*amp*r)
        x[positions] += signed
        audit.update(positions=positions.tolist(),fault_delta=[signed]*duration)
    return x.astype(np.float32), delta, audit

def select_days_reference(legal, period, count, seed):
    available = set(map(int,legal))
    days = sorted(d for d in {o//period for o in available}
                  if all(d*period+p in available for p in range(period)))
    if len(days)<count: raise ValueError('INSUFFICIENT_FULL_DAYS')
    picked = [days[int(np.floor((k+.5)*len(days)/count))] for k in range(count)]
    phase = np.array([k*period//count for k in range(count)])
    np.random.default_rng(seed).shuffle(phase)
    return np.array([d*period+p for d,p in zip(picked,phase)])

def pinball_scalar(pred,y,sigma,q):
    total = 0.
    for b in range(len(y)):
        for j,quantile in enumerate(q):
            for h in range(y.shape[-1]):
                e = y[b,h]-pred[b,j,h]
                total += 2*max(quantile*e,(quantile-1)*e)/sigma[b]
    return total/pred.size
