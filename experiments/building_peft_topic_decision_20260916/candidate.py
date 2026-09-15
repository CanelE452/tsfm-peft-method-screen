"""One hypothesis: residual-consistency allocation of functional anchoring geometry."""
from runtime import *
EPS=1e-6

def geometry(residual,dimension):
    """History-only float64 statistics; no evaluation target. Two orthogonal subspaces."""
    r=np.asarray(residual,dtype=np.float64);assert r.ndim==2 and r.shape[0]>=2 and np.isfinite(r).all();n=len(r)
    b=r.mean(axis=1);s=r-b[:,None];mb=float(b.mean());ms=s.mean(axis=0)
    signal_level=mb*mb;noise_level=float(np.sum((b-mb)**2)/(n*(n-1)))
    signal_shape=float(np.mean(ms*ms));noise_shape=float(np.mean(np.sum((s-ms)**2,axis=0)/(n*(n-1))))
    ulevel=(noise_level+EPS)/(signal_level+noise_level+EPS);ushape=(noise_shape+EPS)/(signal_shape+noise_shape+EPS)
    avg=(ulevel+(dimension-1)*ushape)/dimension;wl=ulevel/avg;ws=ushape/avg
    assert np.isfinite([wl,ws]).all() and wl>0 and ws>0 and abs((wl+(dimension-1)*ws)/dimension-1)<1e-12
    return dict(w_level=wl,w_shape=ws,signal_level=signal_level,noise_level=noise_level,signal_shape=signal_shape,noise_shape=noise_shape,u_level=ulevel,u_shape=ushape,dimension=dimension,n=n)

def penalty(delta,w_level,w_shape):
    mean=delta.mean();return w_level*mean.square()+w_shape*(delta-mean).square().mean()

def build(method,m,h,w):
    assert method in ['SIMPLE','CANDIDATE'];x,y=windows(h);cache={};residual=[];native_losses=[];started=time.monotonic();mi=list(m.chronos_config.quantiles).index(.5)
    with torch.no_grad(),disabled(m):
        for xx,yy in zip(x,y):
            w.boundary();v=native(m,xx,yy);_,(_,scale)=m.instance_norm(tensor(xx).reshape(1,24));sigma=float(scale.reshape(-1)[0]);assert np.isfinite(sigma) and sigma>0
            q0=v.quantile_preds[0,:,:24].detach();cache[np.asarray(xx,dtype=np.float64).tobytes()]=(q0,sigma)
            residual.append((q0[mi].double().cpu().numpy()-yy)/sigma);native_losses.append(float(v.loss))
    residual=np.stack(residual);g=geometry(residual,len(m.chronos_config.quantiles)*24);energy=float(np.mean(residual**2));base_loss=float(np.mean(native_losses));lam=base_loss/(energy+EPS)
    wl,ws=(1.,1.) if method=='SIMPLE' else (g['w_level'],g['w_shape'])
    meta=dict(method=method,history_sha256=hashlib.sha256(np.asarray(h,dtype=np.float64).tobytes()).hexdigest(),geometry=g,effective_w_level=wl,effective_w_shape=ws,lambda_value=lam,f0_native_loss=base_loss,f0_median_residual_energy=energy,epsilon=EPS,calibration_seconds=time.monotonic()-started,extra_trainable=0,information='Exposed history windows only; frozen F0; no future or source series')
    folder=OUT/'calibration';folder.mkdir(exist_ok=True);path=folder/(meta['history_sha256']+'_'+method+'.json')
    if path.exists():
        old=read(path);assert old['geometry']==g and old['lambda_value']==lam
    else:save(path,meta)
    def loss(model,xx,yy):
        v=native(model,xx,yy);q0,sigma=cache[np.asarray(xx,dtype=np.float64).tobytes()];delta=(v.quantile_preds[0,:,:24]-q0)/sigma
        return v.loss+lam*penalty(delta,wl,ws)
    loss.metadata=meta
    return loss,None
