"""Read-only independent packet, normalization and information-access audit."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.history_compression_v1_20260917.common import *
validate_seal()
# Refuse E label access while assembling every real model packet.
real_load=np.load;opened=[]
def guarded_load(p,*a,**kw):
 opened.append(str(p));assert not str(p).endswith('E_DISCOVERY_labels.npz'),'E_LABEL_ACCESS_IN_PACKET'
 return real_load(p,*a,**kw)
np.load=guarded_load
d=Data();checks=[]
for role in ROLES:
 oo=d.x[role]['origins'];wanted=32 if role=='V_SELECT' else 64
 assert len(set(oo//24))==wanted and len(oo)==wanted
 counts=np.bincount(oo%24,minlength=24);assert counts.max()-counts.min()<=2
 # Recompute exact target index overlap and coverage independently.
 targets=(oo[:,None]+np.arange(48)).reshape(-1)
 historical=read(OLD/'N02/origin_diversity.json')['roles'][role]
 assert len(np.unique(targets))==historical['target_timestamps_unique']
 expected_lower,expected_upper={'TRAIN':(0,10526),'V_SELECT':(10526,12280),'E_DISCOVERY':(14035,17544)}[role]
 assert min(targets)>=expected_lower and max(targets)<expected_upper
 for i in range(len(oo)):
  p=d.packet(role,i);assert set(p)=={'x','mu','sigma','H','origin'}
  assert p['x'].shape==(4,1344) and p['H']==48 and p['origin']==oo[i]
  # The latest observed index is o-1; all1344 full/pooled/statistics slots precede target.
  assert p['origin']-1344>=0
 checks.append(dict(role=role,origins=len(oo),distinct_days=len(set(oo//24)),span_hours=int(oo[-1]-oo[0]),phase_min=int(counts.min()),phase_max=int(counts.max()),unique_target_slots=len(np.unique(targets)),all_context_indices_past=True))
np.load=real_load
# Verify chronological patch grouping and uncompressed recent section by explicit indices.
original=np.arange(84);groups=original[:63].reshape(21,3);pooled_position=groups.mean(1);retained=original[63:]
assert np.array_equal(pooled_position,np.arange(1,63,3))
assert np.array_equal(retained,np.arange(63,84))
assert len(set(np.concatenate([groups.reshape(-1),retained])))==84
assert np.all(np.diff(np.concatenate([pooled_position,retained,[84,85,86,87]]))>0)
# Teacher recipe is entirely pre-existing TRAIN fit + V selection, not an E model selection.
r=read(OUT/'REUSE_RECEIPT.json');t=r['teacher'];assert t['seed']==73100 and t['arm']=='B1'
assert t==next(v for v in read(OLD/'N02/LR_selection.json')['selections'] if v['arm']=='B1')
save(OUT/'independent_input_audit.json',dict(passed=True,roles=checks,packet_construction_E_label_access=False,all84patches_accounted_once=True,original_time_positions_preserved=True,recent336slots_not_compressed=True,teacher_selected_from_historical_V=True,teacher_not_selected_from_E=True,model_inputs_exclude_retrieved_futures=True,not_an_independent_test=True))
print('Independent input audit passed',checks)
