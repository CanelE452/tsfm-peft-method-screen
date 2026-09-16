"""Value-independent day-first sampling with the contract's exact boundary tie rules."""
from .common import *

def nearest_unused(indices,limit):
 used=set();out=[]
 for v in indices:
  v=int(v);j=min((i for i in range(limit) if i not in used),key=lambda i:(abs(i-v),i));used.add(j);out.append(j)
 return out

def pick(candidates,count,phases,period):
 # phases deliberately does not filter legal candidates: day-first contract replaces phase-first.
 a=np.array(sorted(set(map(int,candidates))),dtype=int);days=np.unique(a//period)
 if len(days)<count:raise ValueError(f'BLOCKED_DIVERSITY legal_days={len(days)} required={count}')
 idx=np.rint(np.arange(count)*(len(days)-1)/(count-1)).astype(int)
 if len(set(idx))!=count:idx=np.array(nearest_unused(idx,len(days)))
 chosen=days[idx];assert len(set(chosen))==count and chosen[0]==days[0] and chosen[-1]==days[-1]
 result=[]
 for k,day in enumerate(chosen):
  v=a[a//period==day];phase=v%period;q=k*period//count;distance=np.minimum(abs(phase-q),period-abs(phase-q));j=np.lexsort((v,phase,distance))[0];result.append(int(v[j]))
 return sorted(result)

def legal_candidates(t,a,bounds):
 C=1344 if t=='N02' else 336;H=48+(24 if t=='R04' else 0)
 # Only finite/missingness metadata is used. No magnitudes or prediction errors enter selection.
 valid=np.isfinite(a).all(1);bad=np.r_[0,np.cumsum(~valid)];out={}
 bank=np.arange(336,bounds[1]-48+1,24)
 if len(bank)>2048:bank=bank[np.floor(np.linspace(0,len(bank)-1,2048)).astype(int)]
 for role,lo,hi in zip(ROLES,bounds[:-1],bounds[1:]):
  aidx=np.arange(max(lo,C,2048 if t=='N02' else C),hi-H+1);ok=(bad[aidx]-bad[aidx-C]==0)&(bad[aidx+H]-bad[aidx]==0)
  if t=='N02':ok&=np.searchsorted(bank+48,np.minimum(bounds[1],aidx-360),side='right')>=32
  out[role]=aidx[ok].tolist()
 return out

def dispersion(t,origins,candidates):
 P=96 if t=='N03' else 24;oo=np.array(origins);dd=np.unique(oo//P);cd=np.unique(np.array(candidates)//P);hist=np.bincount(oo%P,minlength=P);gap=np.diff(oo)
 targets=np.concatenate([np.r_[np.arange(o,o+48),np.arange(o+24,o+72)] if t=='R04' else np.arange(o,o+48) for o in oo]);ut,mult=np.unique(targets,return_counts=True);mm,mc=np.unique(mult,return_counts=True);C=1344 if t=='N02' else 336;contexts=np.unique(np.concatenate([np.r_[np.arange(o-C,o),np.arange(o+24-C,o+24)] if t=='R04' else np.arange(o-C,o) for o in oo]))
 return dict(origins=len(oo),distinct_days=len(dd),first_day=int(dd[0]),last_day=int(dd[-1]),selected_day_span=int(dd[-1]-dd[0]),candidate_days=len(cd),candidate_first_day=int(cd[0]),candidate_last_day=int(cd[-1]),candidate_day_span=int(cd[-1]-cd[0]),span_ratio=float((dd[-1]-dd[0])/(cd[-1]-cd[0])),seven_day_blocks=len(np.unique(oo//(7*P))),distinct_phases=int((hist>0).sum()),phase_count_min=int(hist.min()),phase_count_max=int(hist.max()),phase_count_range=int(hist.max()-hist.min()),phase_histogram={str(i):int(v) for i,v in enumerate(hist)},origin_span_slots=int(oo[-1]-oo[0]),origin_span_hours=float((oo[-1]-oo[0])*24/P),gap_slots_quantiles=dict(zip(['min','q25','median','q75','max'],np.quantile(gap,[0,.25,.5,.75,1]).tolist())),target_timestamps_unique=len(ut),target_timestamps_total=len(targets),target_uniqueness_ratio=len(ut)/len(targets),context_timestamps_unique=len(contexts),target_overlap_multiplicity_histogram={str(int(m)):int(c) for m,c in zip(mm,mc)},time_basis='15min index-day/phase (ETTm1 regular dated grid)' if t=='N03' else 'index-day/index-phase, not local calendar time')

def failures(t,role,d):
 n=64 if role in ['TRAIN','E_DISCOVERY'] else 32;minimum_phase=(56 if n==64 else 28) if t=='N03' else (20 if n==64 else 16);checks={'distinct_days_exact':d['distinct_days']==n,'one_origin_per_day':d['distinct_days']==d['origins'],'span_at_least_95pct':d['span_ratio']>=.95,'range_endpoints':d['first_day']==d['candidate_first_day'] and d['last_day']==d['candidate_last_day'],'minimum_week_blocks':d['seven_day_blocks']>=(8 if n==64 else 4),'minimum_phases':d['distinct_phases']>=minimum_phase}
 if t!='N03' and n==64:checks['phase_count_range_at_most_2']=d['phase_count_range']<=2
 return checks,[k for k,v in checks.items() if not v]

def test_selector():
 assert nearest_unused([0,0,3,3],5)==[0,1,3,2]
 assert pick(list(range(100*24)),64,list(range(24)),24)==pick(list(reversed(range(100*24))),64,[],24)
 try:pick(list(range(10*24)),64,[],24)
 except ValueError as e:assert 'BLOCKED_DIVERSITY' in str(e)
 else:raise AssertionError('must not fill duplicate days')
 # phase23 at a last partial day endingphase14 chooses0 by circular distance.
 v=list(range(336,15782-48+1));o=pick(v,64,[],24);d=dispersion('N01',o,v);assert d['distinct_days']==64 and d['phase_count_range']==3
 return dict(round_ties='numpy rint, ties-to-even',nearest_unused_tie='smallest distance then smaller index, deterministic synthetic collision test',boundary_phase_collision_must_block=True,no_same_day_fill=True,reversed_order_identical=True,passed=True)
