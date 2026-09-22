import numpy as np

TAU = np.arange(1,10,dtype=np.float64)/10
ALPHAS = (.5,.75,1.,1.25,1.5,2.,3.)
BETAS = (-.5,-.25,0.,.25,.5)

def point_loss(y,q,sigma):
    e = y.astype(np.float64)[...,None]-q.astype(np.float64)
    return (2*np.maximum(TAU*e,(TAU-1)*e)).mean(-1)/sigma[None,:,None]

def empirical(z,y):
    m = z.shape[-1]
    z = np.asarray(z,dtype=np.float64)
    return np.abs(z-y[...,None]).mean(-1)-(np.sort(z,axis=-1)*(2*np.arange(1,m+1)-m-1)).sum(-1)/(m*m)

def calibrate(y,q,sigma):
    rows = []
    for block in range(2):
        sl = slice(block*64,(block+1)*64)
        base = q[:,:,sl].astype(np.float64)
        med = base[...,4:5]
        trials = []
        for a in ALPHAS:
            for b in BETAS:
                adjusted = med+b*sigma[None,:,None,None]+a*(base-med)
                trials.append(dict(alpha=a,beta=b,score=float(point_loss(y[:,:,sl],adjusted,sigma).mean())))
        best = min(trials,key=lambda r:(r['score'],(r['alpha']-1)**2+r['beta']**2,r['alpha'],abs(r['beta']),r['beta']))
        rows.append(dict(**best,grid=trials))
    return rows

def apply(q,sigma,params):
    out = q.astype(np.float64).copy()
    for block,p in enumerate(params):
        sl = slice(block*64,(block+1)*64)
        base = q[:,:,sl].astype(np.float64)
        out[:,:,sl] = base[...,4:5]+p['beta']*sigma[None,:,None,None]+p['alpha']*(base-base[...,4:5])
    return out

def summarize(y,q,sigma):
    diff = y.astype(np.float64)-q[...,4]
    return dict(scaled_pinball=float(point_loss(y,q,sigma).mean()), raw_mae=float(np.abs(diff).mean()),
                scaled_rmse=float(np.sqrt(((diff/sigma[None,:,None])**2).mean(axis=(0,2))).mean()),
                coverage80=float(((y>=q[...,0])&(y<=q[...,8])).mean()),
                scaled_width80=float(((q[...,8]-q[...,0])/sigma[None,:,None]).mean()),
                pce=float(np.abs((y[...,None]<=q).mean(axis=(0,2))-TAU).mean()),
                crossing=float((np.diff(q,axis=-1)<0).mean()))

def bootstrap(a,b):
    # Equal seed mean first, then paired resampling of daily origins.
    a,b = np.asarray(a).mean(0),np.asarray(b).mean(0)
    n = len(a)
    assert n >= 7
    rng = np.random.default_rng(92239)
    values = []
    for _ in range(2000):
        starts = rng.integers(0,n-7+1,size=(n+6)//7)
        idx = (starts[:,None]+np.arange(7)).ravel()[:n]
        values.append((a[idx]-b[idx]).mean())
    return dict(effect=float((a-b).mean()),relative_percent=float(100*(a.mean()-b.mean())/a.mean()),
                ci95_low=float(np.quantile(values,.025)),ci95_high=float(np.quantile(values,.975)),
                bootstrap_block=7,draws=2000)
