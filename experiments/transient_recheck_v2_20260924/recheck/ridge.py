"""Standard multi-output ridge, SVD solver; train-only scaling.

Objective: ||Y_centered - X_standardized B||_F^2 + alpha ||B||_F^2.
This is not a proposed forecasting method. No inverse, no ridge re-tuning on DEV.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .common import require

@dataclass
class RidgeModel:
    mean_x: np.ndarray
    scale_x: np.ndarray
    mean_y: np.ndarray
    coef: np.ndarray
    alpha: float
    n_train: int

    def predict(self, X):
        X=np.asarray(X,dtype=np.float64)
        require(X.ndim==2 and X.shape[1]==len(self.mean_x) and np.isfinite(X).all(), 'Ridge prediction input mismatch')
        pred=((X-self.mean_x)/self.scale_x)@self.coef+self.mean_y
        require(np.isfinite(pred).all(), 'Non-finite ridge predictions')
        return pred

    def save(self,path):
        np.savez(path,mean_x=self.mean_x,scale_x=self.scale_x,mean_y=self.mean_y,
                 coef=self.coef,alpha=np.asarray(self.alpha),n_train=np.asarray(self.n_train))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            require(set(z.files)=={'mean_x','scale_x','mean_y','coef','alpha','n_train'}, 'Ridge state keys mismatch')
            return cls(z['mean_x'].copy(),z['scale_x'].copy(),z['mean_y'].copy(),z['coef'].copy(),float(z['alpha']),int(z['n_train']))


def ridge_path(X,Y,alphas,ledger=None,context=None):
    X=np.asarray(X,np.float64); Y=np.asarray(Y,np.float64)
    require(X.ndim==Y.ndim==2 and len(X)==len(Y) and len(X)>=2,'Invalid ridge training shapes')
    require(np.isfinite(X).all() and np.isfinite(Y).all(),'Non-finite training observations')
    require(len(alphas)>0 and all(float(a)>0 for a in alphas),'Ridge alphas must be positive')
    mx=X.mean(0); sx=X.std(0); sx=np.where(sx<1e-12,1.,sx); my=Y.mean(0)
    Z=(X-mx)/sx; T=Y-my
    if ledger:ledger.begin_svd(**(context or {}),n=len(X),p=X.shape[1],h=Y.shape[1])
    U,s,Vt=np.linalg.svd(Z,full_matrices=False)
    if ledger:ledger.end_svd()
    UTY=U.T@T
    models={}
    for alpha in alphas:
        alpha=float(alpha)
        if ledger:ledger.begin_fit(**(context or {}),alpha=alpha,n=len(X),p=X.shape[1],h=Y.shape[1])
        coef=Vt.T@((s/(s*s+alpha))[:,None]*UTY)
        require(np.isfinite(coef).all(),'Non-finite ridge coefficients')
        models[alpha]=RidgeModel(mx.copy(),sx.copy(),my.copy(),coef,alpha,len(X))
        if ledger:ledger.end_fit()
    return models


def choose_alpha(scores_by_fold,alpha_grid):
    """Scores are per-episode MAE from earlier-prefix fits. No target/horizon selection."""
    require(len(scores_by_fold)>0,'No past-only validation fold for alpha selection')
    means={float(a):float(np.mean([f[float(a)] for f in scores_by_fold])) for a in alpha_grid}
    require(all(np.isfinite(list(means.values()))),'Invalid alpha validation scores')
    return min(means,key=lambda a:(means[a],-a)),means
