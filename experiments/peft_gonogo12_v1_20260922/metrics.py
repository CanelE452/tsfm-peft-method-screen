import numpy as np


def affine(q, y, weights=None):
    median = np.asarray(q, dtype=np.float64)[..., 4].reshape(-1)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.ones_like(target) if weights is None else np.broadcast_to(np.asarray(weights)[:, None], np.asarray(y).shape).reshape(-1)
    w = w / w.sum()
    xm, ym = np.sum(w * median), np.sum(w * target)
    var = np.sum(w * (median-xm)**2)
    a = float(np.clip(np.sum(w*(median-xm)*(target-ym))/var, .5, 1.5)) if var > 0 else 1.
    return a, float(ym-a*xm)


def point_loss(q, y, sigma):
    ordered = np.sort(np.asarray(q, dtype=np.float64), axis=-1)
    e = np.asarray(y, dtype=np.float64)[..., None] - ordered
    levels = np.arange(1, 10, dtype=np.float64)/10
    return (2*np.maximum(levels*e, (levels-1)*e)).mean(axis=(-1, -2))/np.asarray(sigma)


def mean_score(q, y, sigma):
    return float(point_loss(q, y, sigma).mean())
