"""One fixed CPU retrieval replay for cost accounting, no labels/selection/model calls."""
import sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.condition_studies_v1_20260916.common import *
path=OUT/'N02/retrieval_cost_replay.json'
if path.exists():print('EXISTING_COST_REPLAY');raise SystemExit(0)
r=read(OUT/'N02/data_receipt.json');loadstart=time.perf_counter();a=np.loadtxt(ROOT/r['path'],delimiter=',',usecols=[int(c.split('_')[1]) for c in r['selected_columns']]);load_seconds=time.perf_counter()-loadstart;b1=r['bounds'][1];bank=np.arange(336,b1-48+1,24)
if len(bank)>2048:bank=bank[np.floor(np.linspace(0,len(bank)-1,2048)).astype(int)]
rows=[]
for role in ROLES:
 z=np.load(CACHE/'N02'/f'{role}_inputs.npz')
 for i,o in enumerate(z['origins']):
  start=time.perf_counter();eligible=bank[bank+48<=min(b1,o-360)]
  for c in range(4):
   q=a[o-336:o,c];mu=q.mean();sd=max(q.std(),1e-6);past=np.stack([a[k-336:k,c] for k in eligible]);bm=past.mean(1);bs=np.maximum(past.std(1),1e-6);d=(((past-bm[:,None])/bs[:,None]-(q-mu)/sd)**2).mean(1);ix=np.lexsort((eligible,d))[:2];rr=eligible[ix];pp=(past[ix]-bm[ix,None])/bs[ix,None]*sd+mu;ff=np.stack([(a[k:k+48,c]-bm[j])/bs[j]*sd+mu for k,j in zip(rr,ix)]);assert np.array_equal(rr,z['retrieval_origins'][i,c*2:c*2+2]);assert np.array_equal(pp.astype(np.float32),z['retrieval_past'][i,c*2:c*2+2]);assert np.array_equal(ff.astype(np.float32),z['retrieval_future'][i,c*2:c*2+2])
  rows.append(dict(role=role,origin=int(o),queries=4,bank_candidates=len(eligible),seconds=time.perf_counter()-start))
csvwrite(OUT/'N02/retrieval_cost_by_origin.csv',rows);save(path,dict(at=time.time(),scope='Independent exact replay of all fixed retrievals; original preparation was not timed separately',source_load_seconds=load_seconds,retrieval_seconds=sum(r['seconds'] for r in rows),origins=len(rows),queries=4*len(rows),per_origin_mean_seconds=float(np.mean([r['seconds'] for r in rows])),per_origin_p95_seconds=float(np.quantile([r['seconds'] for r in rows],.95)),all_outputs_exact=True,additional_neural_fits=0,additional_optimizer_updates=0,additional_CPU_parameter_fits=0));print(read(path))
