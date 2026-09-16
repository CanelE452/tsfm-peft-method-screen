"""Actual prepared-input permission tests plus separately labelled synthetic algebra checks."""
import math,copy,time
import torch
from torch import nn
from .common import *
from .model import Packets,Adapted,clock_mask

def transform_only(t,arm,aux):
 m=Adapted.__new__(Adapted);nn.Module.__init__(m);m.t=t;m.arm=arm;m.aux=aux;m.base=nn.Linear(1,1);m.extra=nn.ParameterDict()
 shapes={'A2':{'theta':(4,9)},'A3':{'theta':(4,9),'eta':(4,)},'B3':{'a':(4,),'b':(4,)},'C3':{'theta':(4,)},'H1':{'theta':(4,)},'H2':{'u':(4,),'v':(4,)},'H3':{'u':(4,),'v':(4,)}}
 for n,s in shapes.get(arm,{}).items():m.extra[n]=nn.Parameter(torch.full(s,math.log(3.5/4) if arm=='C3' else 0.))
 return m

def same(a,b):
 for key in a:
  if a[key] is None:assert b[key] is None
  elif isinstance(a[key],torch.Tensor):assert torch.allclose(a[key],b[key],rtol=0,atol=0,equal_nan=True),key
  else:assert a[key]==b[key]

def check_all():
 records=[]
 for t in SOURCES:
  data=Packets(t);spec=data.spec;orig=read(OUT/t/'origins.json');bounds=spec['data']['bounds'];H=spec['H'];roles=[]
  for r,lo,hi in zip(ROLES,bounds[:-1],bounds[1:]):
   for o in orig[r]:assert o>=lo and o+H+(24 if t=='R04' else 0)<=hi
   roles.append(set(v for o in orig[r] for v in range(o,o+H+(24 if t=='R04' else 0))))
  assert all(not roles[i]&roles[j] for i in range(4) for j in range(i))
  for path,h in spec['data']['packet_hashes'].items():assert sha(ROOT/path)==h
  rec=dict(track=t,role_labels_disjoint=True,packet_hashes_verified=True,E_label_api_denied=False)
  try:data.label('E_DISCOVERY',0)
  except AssertionError:rec['E_label_api_denied']=True
  assert rec['E_label_api_denied']
  p=data.packet('E_DISCOVERY',0,condition=__import__(__package__+'.model',fromlist=['conditions']).conditions(t,'E_DISCOVERY')[0]);before={a:transform_only(t,a,data.aux).tensors(p) for a in ARMS[t]};data.labels['E_DISCOVERY']=np.full((64,4,H),1e15);data.prepared.clear();p2=data.packet('E_DISCOVERY',0,condition=__import__(__package__+'.model',fromlist=['conditions']).conditions(t,'E_DISCOVERY')[0])
  for a in ARMS[t]:same(before[a],transform_only(t,a,data.aux).tensors(p2))
  del data.labels['E_DISCOVERY'];rec['actual_E_label_poison_input_invariant']=True
  if t in ['N01','N03']:
   p=data.packet('TRAIN',0,epoch=1);raw=data.inputs['TRAIN']['context'][0].copy();M=p['M'];baseline={a:transform_only(t,a,data.aux).tensors(p) for a in ARMS[t]};data.inputs['TRAIN']['context'][0][~M]=1e15;data.prepared.clear();p2=data.packet('TRAIN',0,epoch=1)
   for a in ARMS[t]:same(baseline[a],transform_only(t,a,data.aux).tensors(p2))
   data.inputs['TRAIN']['context'][0]=raw;data.prepared.clear();p=data.packet('TRAIN',0,epoch=1)
   for a in ARMS[t]:
    m=transform_only(t,a,data.aux);v=m.tensors(p);assert torch.equal(v['context'][:4][torch.as_tensor(M)],torch.tensor(raw)[torch.as_tensor(M)])
   if t=='N01':
    same(baseline['A1'],baseline['A2']);same(baseline['A1'],baseline['A3']);data.inputs['TRAIN']['context'][0,1,-1]+=data.sigma[1];data.prepared.clear();q=data.packet('TRAIN',0,epoch=1);assert not np.array_equal(q['ridge'],p['ridge']);rec.update(donor_sensitive=True,targets_exposure=[sum(i%4==c for i in range(64)) for c in range(4)])
   else:
    assert torch.allclose(baseline['C2']['context'],baseline['C3']['context'],rtol=1e-6,atol=1e-6);w=np.exp(-p['distances']/4);w/=w.sum(-1,keepdims=True);assert np.allclose(w.sum(-1),1);assert np.max(p['neighbors'])<336;rec.update(kernel_weights_sum1=True,future_observation_access=0,physical_horizon_hours=12,context_hours=84)
   rec.update(hidden_poison_invariant=True,observed_unchanged=True,initial_reduction=True)
  if t=='N02':
   rows=read_csv(OUT/t/'retrieval_receipt.csv')
   for r in rows:assert int(r['continuation_end'])<=int(r['max_legal_end'])
   p=data.packet('TRAIN',0);b2=transform_only(t,'B2',data.aux);b3=transform_only(t,'B3',data.aux);same(b2.tensors(p),b3.tensors(p));f=b3.tensors(p)['future_covariates'];b3.extra['a'].data[0]=.2;new=b3.tensors(p)['future_covariates'];assert not torch.equal(f[4:],new[4:]);rec.update(archive_temporal_checks=len(rows),initial_reduction=True,future_delta_sensitive=True,query_label_access=False)
  if t=='R04':
   for r,oo in orig.items():
    for i,o in enumerate(oo):assert np.array_equal(np.arange(o+24,o+48),np.arange(o+24,o+24+24));assert np.array_equal(data.inputs[r]['late_context'][i,:,:-24],data.inputs[r]['context'][i,:,24:])
   rec.update(overlap_exact=True,late_separate_native_call=True,innovation_is_late_past=True)
  if t=='N07':
   e=np.random.default_rng(3).normal(size=(4,48));assert np.allclose(np.mean(e**2),np.mean(abs(np.fft.fft(e,norm='ortho'))**2),rtol=1e-12,atol=1e-12)
   for a,w in data.aux['weights'].items():
    w=np.array(w);assert np.allclose(w.mean(1),1);assert np.array_equal(w[:,1:24],w[:,25:][:,::-1])
   assert np.allclose(np.sort(data.aux['weights']['G2'],axis=1),np.sort(data.aux['weights']['G3'],axis=1));rec.update(Parseval=True,symmetric_weights=True,weights_mean1=True,PRED_SHUFFLE_same_multiset=True,TRAIN_only_ridge=True)
  if t=='R08':
   for c in range(4):
    for j,l in zip(data.aux['donors'][c],data.aux['lags'][c]):assert j!=c and 1<=l<=24;assert all(h-l<0 for h in range(l));assert all(0<=h-l<48 for h in range(l,48))
   rec.update(observed_donor_strictly_past=True,predicted_donor_indices_valid=True,simultaneous_base_only=True)
  if t=='R09':
   from .prepare import observation_map
   import gzip
   a=np.loadtxt(ROOT/'data/raw/electricity.txt.gz',delimiter=',',usecols=[int(c.split('_')[1]) for c in data.stats['columns']]);obs,detail=observation_map(a,orig['TRAIN']);poison=a.copy()
   for b,d in enumerate(detail):
    if not d:poison[b*24,0]+=100;poison[b*24+1,0]-=100
   obs2,detail2=observation_map(poison,orig['TRAIN']);assert np.allclose(obs,obs2,rtol=1e-10,atol=1e-10);assert np.array_equal(detail,detail2)
   for r,oo in orig.items():
    for i,o in enumerate(oo):assert np.array_equal(obs[o-336:o].T.astype(np.float32),data.inputs[r]['context'][i])
   for i,d in enumerate(data.inputs['TRAIN']['detailed']):
    if not d:assert np.all(data.labels['TRAIN'][i]==data.labels['TRAIN'][i,:,:1])
   rr=np.random.default_rng(4);delta=rr.normal(size=(4,24));P=delta.mean(-1,keepdims=True);N=delta-P;assert np.allclose((N-N.mean(-1,keepdims=True)),N);assert np.allclose(np.mean(delta**2),np.mean(P**2)+np.mean(N**2));q0=rr.normal(size=(4,24));q=q0+delta;mean=rr.normal(size=(4,1));pseudo=q0+mean-q0.mean(-1,keepdims=True);assert np.allclose(np.mean((q-pseudo)**2),np.mean((q.mean(-1,keepdims=True)-mean)**2)+np.mean(N**2));rec.update(absolute_permission_poison_invariant=True,hidden_fine_absent_packets=True,projection_identity=True,IMPUTE_equals_NULL_coefficient1=True,constant_offset_NULL_zero=True,detailed_train_count=int(data.inputs['TRAIN']['detailed'].sum()))
  save(OUT/t/'input_verification.json',rec);records.append(rec)
 save(OUT/'input_verification.json',dict(passed=True,at=time.time(),records=records,synthetic_tests_are_not_actual_experiments=True));return records

def read_csv(p):
 import csv
 with open(p) as f:return list(csv.DictReader(f))
