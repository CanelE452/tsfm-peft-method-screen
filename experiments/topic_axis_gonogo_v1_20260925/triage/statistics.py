"""TRAIN-fitted ridge, CAL selection, cluster-level descriptive intervals.
No TEST-dependent thresholds or model selection; no iid-window bootstrap.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from .common import require, normalized_features, normalized_error, hash_obj, Blocked

class Ridge:
    def fit(self,X,Y,alpha):
        X=np.asarray(X,float);Y=np.asarray(Y,float)
        require(len(X)>=4 and np.isfinite(X).all() and np.isfinite(Y).all(),'Invalid ridge data')
        self.mx=X.mean(0);self.sx=np.maximum(X.std(0),1e-8);self.my=Y.mean(0)
        Z=(X-self.mx)/self.sx;T=Y-self.my
        # Explicit SVD, same objective ||Y-XB||^2 + alpha ||B||^2.
        U,s,V=np.linalg.svd(Z,full_matrices=False)
        self.B=(V.T*(s/(s*s+alpha)))@(U.T@T);self.alpha=alpha
        return self
    def predict(self,X):return ((np.asarray(X)-self.mx)/self.sx)@self.B+self.my
    def identity(self):return hash_obj([self.mx,self.sx,self.my,self.B,self.alpha])

def ridge_data(cases,structured=False):
    X=[];Y=[]
    for c in cases:
        raw,mu,sd=normalized_features(c)
        X.append(np.r_[raw,c.extra] if structured else raw)
        Y.append(((c.y-mu)/sd).reshape(-1))
    return np.array(X),np.array(Y)

def predict_ridge(model,cases,structured=False):
    X=[]
    for c in cases:
        raw,_,_=normalized_features(c);X.append(np.r_[raw,c.extra] if structured else raw)
    p=model.predict(np.asarray(X));out=[]
    for c,v in zip(cases,p):
        _,mu,sd=normalized_features(c)
        out.append(v.reshape(c.y.shape)*sd+mu)
    return out

def group_mean(values,groups):
    g=np.asarray(groups);v=np.asarray(values,float)
    return {str(k):float(np.mean(v[(g==k)&np.isfinite(v)])) for k in sorted(set(groups)) if np.isfinite(v[g==k]).any()}

def tune_ridge(train,cal,alphas,structured=False):
    valid=[c for c in train if np.isfinite(c.y).all() and (c.mask is None or c.mask.all())]
    if len(valid)<4:raise Blocked('INSUFFICIENT_VALID_TRAIN_TARGETS')
    X,Y=ridge_data(valid,structured);history=[];best=None
    for a in alphas:
        m=Ridge().fit(X,Y,a);p=predict_ridge(m,cal,structured)
        e=[normalized_error(c,q) for c,q in zip(cal,p)]
        score=np.mean(list(group_mean(e,[c.group for c in cal]).values()))
        history.append({'alpha':a,'cal_group_mean_error':float(score)})
        if best is None or score<best[0]:best=(score,m)
    return best[1],history

def bootstrap_mean(values,alpha=.05/3,draws=2000,seed=92531):
    v=np.asarray(values,float);v=v[np.isfinite(v)]
    if not len(v):return {'n_groups':0,'mean':None,'lower':None,'upper':None}
    rng=np.random.default_rng(seed);b=v[rng.integers(0,len(v),(draws,len(v)))].mean(1)
    return {'n_groups':len(v),'mean':float(v.mean()),'lower':float(np.quantile(b,alpha/2)),
            'upper':float(np.quantile(b,1-alpha/2)),'alpha':alpha,'positive_groups':int((v>0).sum()),
            'scope':'exploratory cluster bootstrap; few clusters do not support population proof'}

def correlation(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<5 or np.std(a[m])==0 or np.std(b[m])==0:return None
    return float(spearmanr(a[m],b[m]).statistic)

def strata(scores,cuts):
    q0,q1=cuts
    return np.where(scores<q0,0,np.where(scores>q1,2,1))

def contrasts(values,cases,cuts):
    ss=strata(np.array([c.score for c in cases]),cuts);groups=np.array([c.group for c in cases]);v=np.array(values,float)
    per=[]
    for g in sorted(set(groups)):
        lo=(groups==g)&(ss==0)&np.isfinite(v);hi=(groups==g)&(ss==2)&np.isfinite(v)
        if lo.sum()>=2 and hi.sum()>=2:
            per.append({'group':g,'low_n':int(lo.sum()),'high_n':int(hi.sum()),'difference':float(v[hi].mean()-v[lo].mean())})
    return per

def analyze(train,cal,test,predictions,cal_predictions,best_simple,config):
    a=np.array([c.score for c in train]);finite=a[np.isfinite(a)]
    if len(finite)<16:raise Blocked('INSUFFICIENT_TRAIN_PROXY_SUPPORT')
    cuts=np.quantile(finite,[1/3,2/3]);r=correlation(a,[c.alternate for c in train])
    errors={name:np.array([normalized_error(c,p) for c,p in zip(test,pp)]) for name,pp in predictions.items()}
    report={'cuts_from_train':cuts.tolist(),'estimator_agreement_spearman':r,'models':{},'label':'pilot_proxy_not_validated_complexity'}
    for name,e in errors.items():
        z=contrasts(e,test,cuts)
        report['models'][name]={'group_means':group_mean(e,[c.group for c in test]),
              'high_minus_low':bootstrap_mean([v['difference'] for v in z],config['alpha_family']),
              'contrasts':z,'score_error_spearman_descriptive':correlation([c.score for c in test],e)}
    if 'F0' not in errors:
        report.update(axis_status='BLOCKED_F0_REFERENCE',action='NO_METHOD_DECISION');return report,errors
    # CAL-only nuisance adjustment. It predicts log error, not the future series.
    ce=np.array([normalized_error(c,p) for c,p in zip(cal,cal_predictions['F0'])])
    good=np.isfinite(ce);N=np.array([c.nuisance for c in cal]);Nt=np.array([c.nuisance for c in test])
    if good.sum()<12:raise Blocked('INSUFFICIENT_CAL_VALID_TARGETS')
    adj=Ridge().fit(N[good],np.log1p(ce[good,None]),10.)
    residual=np.log1p(errors['F0'])-adj.predict(Nt).reshape(-1)
    z=contrasts(residual,test,cuts);ci=bootstrap_mean([v['difference'] for v in z],config['alpha_family'])
    report['adjusted_F0_contrast']={'summary':ci,'per_group':z,'adjustment_model_hash':adj.identity(),
                                  'nuisances':'context entropy, scale, variation, trend, topic covariate; CAL-only fit'}
    raw=report['models']['F0']['high_minus_low'];nmin=config['min_contrast_groups']
    if cuts[1]-cuts[0]<=1e-10:
        status='NO_GO_PROXY_COLLAPSED'
    elif r is None or r<config['min_agreement']:
        status='INCONCLUSIVE_MEASUREMENT'
    elif raw['n_groups']<nmin or ci['n_groups']<nmin:
        status='INCONCLUSIVE_STRATUM_SUPPORT'
    elif raw['lower']>0 and ci['lower']>0:
        status='GO_AXIS_PILOT'
    elif raw['upper']<=0 or ci['upper']<=0:
        status='NO_GO_TESTED_AXIS'
    else:
        status='INCONCLUSIVE_EFFECT'
    # Test simple remedy only after it is selected on CAL, never select the TEST winner.
    benefit=errors['F0']-errors[best_simple]
    bg=group_mean(benefit,[c.group for c in test]);bc=bootstrap_mean(list(bg.values()),config['alpha_family'])
    report['selected_simple']=best_simple;report['simple_benefit']=bc
    action='NEXT_LORA_DIAGNOSTIC_ONLY' if status=='GO_AXIS_PILOT' else 'HOLD_NO_NEW_ADAPTER'
    if status=='GO_AXIS_PILOT' and bc['n_groups']>=nmin and bc['lower']>0:action='SIMPLE_BASELINE_FIRST'
    report.update(axis_status=status,action=action,neural_adaptation_tested=False,novelty_established=False,
                  practical_sufficiency='NOT_ESTABLISHED',no_go_scope='only this proxy, task, sample and zero-shot reference')
    return report,errors
