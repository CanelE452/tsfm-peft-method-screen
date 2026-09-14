import csv,json,math,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results/temporal_transfer_diagnostic_v1'
read=lambda p:json.loads(p.read_text())
rows=list(csv.DictReader((out/'per_origin_scores.csv').open()))
stats={}
for r in rows:
 key=(r['dataset'],int(r['seed']),r['arm'],r['candidate_id'])
 stats.setdefault(key,[]).append(r)
def metric(key,origins):
 rr=[r for r in stats[key] if int(r['origin']) in origins]
 values=[]
 for ch in range(4):
  cr=[r for r in rr if int(r['channel'])==ch]
  num=math.fsum(float(r['scaled_quantile_mean_pinball_numerator']) for r in cr)
  count=sum(int(r['valid_target_count']) for r in cr)
  assert count>0
  values.append(num/count)
 return math.fsum(values)/4
manifest={(m['dataset'],m['seed'],m['arm']):m for m in read(out/'temporal_selection_manifest.json')}
maxerr=0.;checks=0
for r in csv.DictReader((out/'temporal_selection_comparisons.csv').open()):
 key=(r['dataset'],int(r['seed']),r['arm']);m=manifest[key];oo=m['origins'];cid=r['candidate_id']
 for label,indices in [('S',range(8)),('D',range(8,16)),('D_front4',range(8,12)),('D_back4',range(12,16))]:
  loss=metric(key+(cid,),[oo[i] for i in indices]);error=abs(loss-float(r[label+'_loss']));assert error<1e-10;maxerr=max(maxerr,error);checks+=1
  f0=metric(key+('F0',),[oo[i] for i in indices]);g=100*(f0-loss)/f0
  field=label+'_gain_vs_f0_percent' if label in ['S','D'] else label+'_gain_percent'
  assert abs(g-float(r[field]))<1e-10;checks+=1
 if r['rule'] in ['RECENT4','SPREAD4','ALL8']:
  inds=m['splits'][r['rule']];candidates=[]
  inv=list(csv.DictReader((out/'candidate_inventory.csv').open()))
  for c in inv:
   if (c['dataset'],int(c['seed']),c['arm'])==key and c['candidate_id']==c['canonical_id']:
    value=metric(key+(c['candidate_id'],),[oo[i] for i in inds])
    candidates.append((value,int(c['step']),float(c['lr']),c['candidate_id']))
  assert min(candidates)[3]==cid;checks+=1
curves=list(csv.DictReader((out/'correction_scale_curves.csv').open()))
for r in csv.DictReader((out/'correction_scale_selection.csv').open()):
 key=(r['dataset'],r['seed'],r['arm']);pool=[c for c in curves if (c['dataset'],c['seed'],c['arm'])==key]
 choice=min(pool,key=lambda c:(float(c['S_loss']),float(c['alpha'])))
 assert float(choice['alpha'])==float(r['S_selected_alpha']) and float(choice['D_loss'])==float(r['D_loss']);checks+=1
 error=abs(100*(float(r['R2_original_D'])-float(r['D_loss']))/float(r['R2_original_D'])-float(r['D_gain_vs_original_percent']));assert error<1e-10;checks+=1
h=read(out/'source_and_history_hashes.json')
for k in ['historical_and_source_files','predictions_checkpoints_and_development_files']:
 for p,digest in h[k].items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==digest,p
probe=read(out/'gradient_probe_manifest.json');assert len(probe['states'])==12 and all(r['available'] for r in probe['states'])
receipt={'status':'PASS','artifact_numeric_checks':checks,'max_absolute_reaggregation_error':maxerr,'historical_files_unchanged':len(h['historical_and_source_files']),'fixed_D_checkpoints_present':12,'GPU_execution':False}
(out/'artifact_verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
