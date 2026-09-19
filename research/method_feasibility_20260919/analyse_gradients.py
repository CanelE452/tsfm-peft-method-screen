"""CPU-only corrected reporting of zero gradient directions; retains all layers."""
from gradient_probe import OUT, ROOT, SOURCES, check_seal, read, sha, moments, top_basis, project, save
import numpy as np
import pandas as pd

def analyse():
    check_seal()
    done = read(OUT/'GRADIENTS_COMPLETE.json')
    rows, aggregate = [], []
    for source in SOURCES:
        record = next(r for r in done['models'] if r['source']==source)
        assert sha(ROOT/record['path'])==record['sha256']
        g = np.load(ROOT/record['path'],mmap_mode='r')
        totals = {a:dict(inner=0.,update_sq=0.,hold_sq=0.,calib_energy=0.)
                  for a in ['RANDOM_ORTHO','MEAN_SVD','SECOND_MOMENT','CROSS_BLOCK','FULL_GRADIENT']}
        for layer_idx,layer in enumerate(record['layers']):
            mean,square,second,cross = moments(g[:12,layer_idx])
            later = g[12:,layer_idx].astype(float).mean(0)
            random,_ = np.linalg.qr(np.random.default_rng(91932+layer_idx).normal(size=(512,8)))
            bases = {'RANDOM_ORTHO':random}
            eig_stats = {}
            for arm,matrix in [('MEAN_SVD',square),('SECOND_MOMENT',second),('CROSS_BLOCK',cross)]:
                basis,eig = top_basis(matrix,8)
                bases[arm]=basis
                eig_stats[arm] = dict(min_eigenvalue=float(eig[0]),eighth_eigenvalue=float(eig[-8]),positive_eigenvalues=int((eig>0).sum()))
            for arm in totals:
                d = mean if arm=='FULL_GRADIENT' else project(mean,bases[arm])
                inner=float((later*d).sum())
                update_sq=float((d*d).sum())
                hold_sq=float((later*later).sum())
                denom=np.sqrt(update_sq*hold_sq)
                assert denom>=0 and np.isfinite(denom)
                energy=float((mean*d).sum())
                rows.append(dict(source=source,layer=layer,arm=arm,rank=512 if arm=='FULL_GRADIENT' else 8,
                                 holdout_direction_inner=inner,holdout_direction_cosine=inner/denom if denom>0 else None,
                                 direction_defined=bool(denom>0),
                                 update_squared_norm=update_sq,holdout_gradient_squared_norm=hold_sq,
                                 calibration_projection_energy=energy,**eig_stats.get(arm,{})))
                for key,val in [('inner',inner),('update_sq',update_sq),('hold_sq',hold_sq),('calib_energy',energy)]:
                    totals[arm][key]+=val
            if layer_idx % 12==11:
                print(f'{source}: CPU subspaces {layer_idx+1}/36',flush=True)
        for arm,t in totals.items():
            aggregate.append(dict(source=source,arm=arm,holdout_direction_inner=t['inner'],
                                  holdout_direction_cosine=t['inner']/np.sqrt(t['update_sq']*t['hold_sq']),
                                  update_norm=np.sqrt(t['update_sq']),calibration_projection_energy=t['calib_energy']))
    f=pd.DataFrame(rows);summary=pd.DataFrame(aggregate)
    assert len(f)==360 and len(summary)==10
    f.to_csv(OUT/'LAYER_RESULTS.csv',index=False)
    summary.to_csv(OUT/'AGGREGATE_RESULTS.csv',index=False)
    findings=[]
    for source in SOURCES:
        a=summary[(summary.source==source)&(summary.arm=='CROSS_BLOCK')].iloc[0]
        b=summary[(summary.source==source)&(summary.arm=='MEAN_SVD')].iloc[0]
        findings.append(dict(source=source,cross_inner=float(a.holdout_direction_inner),mean_inner=float(b.holdout_direction_inner),
                             cross_cosine=float(a.holdout_direction_cosine),mean_cosine=float(b.holdout_direction_cosine),
                             cross_better_on_both=bool(a.holdout_direction_inner>b.holdout_direction_inner and
                                                       a.holdout_direction_cosine>b.holdout_direction_cosine and
                                                       a.holdout_direction_inner>0)))
    save(OUT/'VERIFICATION.json',dict(status='COMPLETE_DIAGNOSTIC',new_fits=0,optimizer_updates=0,
         gradient_calls=48,primary_results=findings,automatic_training=False,novelty_established=False,
         goal_achieved=False,independent_test=False))
    print(summary.to_string(index=False),flush=True)


if __name__ == "__main__":
    analyse()
