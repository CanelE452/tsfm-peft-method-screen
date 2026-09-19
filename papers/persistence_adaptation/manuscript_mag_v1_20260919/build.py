"""Render a method manuscript from fixed evidence, without new experiments."""
from pathlib import Path
import hashlib,json,shutil,subprocess
import pandas as pd
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
R=ROOT/'results/learned_gate_comparison_20260919'
M=P.parent/'magnitude_method_20260919'
PANELS=['electricity','electricity_transfer','ettm1','neso_2026_jul_aug']
LABELS=dict(zip(PANELS,['전력4','전력 전이16','ETTm1','NESO 후반']))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(head,rows):
 return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def main():
 seen={}
 def source(p):seen[str(p.relative_to(ROOT))]=sha(p);return p
 def read(p):return json.loads(source(p).read_text())
 def csv(p):return pd.read_csv(source(p))
 audit=read(M/'PAPER_EVIDENCE_AUDIT.json')
 for name,h in audit['source_hashes'].items():assert sha(ROOT/name)==h;source(ROOT/name)
 for name,h in audit['artifact_hashes'].items():assert sha(M/name)==h;source(M/name)
 assert read(R/'AUDIT.json')['status']=='VERIFIED'
 raw=csv(R/'RAW_SCORES.csv');effects=csv(R/'EFFECTS.csv');primary=csv(R/'PRIMARY_COMPARISONS.csv');resource=csv(R/'RESOURCES.csv')
 data=read(ROOT/'results/additive_persistence_validation_v1_20260917/DATA_MANIFEST.json');data.update(read(R/'DATA_MANIFEST.json'))
 origin=csv(R/'ORIGIN_AUDIT.csv')
 for f in ['REPORT.md','FINAL_DECISION.md','MODEL_SELECTION.json','EVALUATION_SEAL.json','ALL_PREDICTIONS_SAVED.json']:source(R/f)
 for p in [ROOT/'experiments/learned_gate_comparison_20260919/PROTOCOL.md',ROOT/'experiments/c3_weakness_controls_20260918/model.py',ROOT/'experiments/additive_persistence_validation_v1_20260917/model.py',ROOT/'experiments/outlier_signal_followup_v2_20260917/prepare.py',ROOT/'experiments/outlier_signal_peft_v1_20260917/reference_core.py',ROOT/'research/method_baseline_compatibility_20260919/BASELINE_COMPATIBILITY_KO.md',ROOT/'research/learned_gate_comparability_20260919/RESOURCE_SCOPE_KO.md',ROOT/'results/petsa_cell_comparison_20260919/status.json']:source(p)
 assert read(ROOT/'results/petsa_cell_comparison_20260919/status.json')['main_updates']==0
 assert not (ROOT/'results/petsa_cell_comparison_20260919/AUTHORIZATION.json').exists(),'This manuscript snapshots the pending baseline; revise its status explicitly if authorized'
 (P/'tables').mkdir(exist_ok=True);(P/'figures').mkdir(exist_ok=True)
 raw.to_csv(P/'tables/all_raw_scores.csv',index=False)
 raw[raw.kind=='shape'].to_csv(P/'tables/all_shape_scores.csv',index=False)
 primary.to_csv(P/'tables/primary.csv',index=False)
 shutil.copyfile(R/'MODEL_SELECTION.json',P/'tables/model_selection.json')
 blocks={};panels=[]
 for panel in PANELS:
  d=data[panel];panels.append([LABELS[panel],len(d['selected_columns']),d['period'],55 if panel.startswith('neso') else 128,'당시 미채점 시간 전이·같은 원천' if panel.startswith('neso') else '재사용 개발 평가'])
 blocks['PANELS']=table(['패널','계열 수','관측/일','E origin days','노출 역할'],panels)
 f=raw[(raw.stage=='selected')&(raw.kind=='standard')&(raw.condition=='SHIFT8')].groupby(['panel','arm']).nmae.mean().unstack()
 f.to_csv(P/'tables/shift8_means.csv')
 arms=['B0','PLAIN','MAG_ONLY','TOKEN_GATE','TOKEN_GATE_ENTROPY']
 blocks['RAW_SHIFT8']=table(['패널','B0','PLAIN','MAG','Gate','Gate+H'],[[LABELS[k],*[f'{f.loc[k,a]:.6f}' for a in arms]] for k in PANELS])
 blocks['PRIMARY']=table(['패널','대조','이득(%)','보정구간','seed81551','seed81552'],[[LABELS[r.panel],r.baseline.replace('TOKEN_GATE_ENTROPY','Gate+H').replace('TOKEN_GATE','Gate'),f'{r.gain_pct:.3f}',f'[{r.bonf4_low:.3f}, {r.bonf4_high:.3f}]',*[f'{json.loads(r.seed_gains)[str(seed)]:.3f}' for seed in [81551,81552]]] for r in primary.itertuples()])
 protect=effects[(effects.proposed=='MAG_ONLY')&effects.panel.isin(['electricity_transfer',PANELS[-1]])&(effects.stage=='selected')&(effects.kind=='standard')&effects.condition.isin(['REFERENCE','FAULT'])&effects.baseline.isin(['B0','PLAIN'])]
 assert len(protect)==8;protect.to_csv(P/'tables/protection.csv',index=False)
 blocks['PROTECTION']=table(['패널','조건','대조','이득(%)','일반95%구간'],[[LABELS[r.panel],r.condition,r.baseline,f'{r.gain_pct:.3f}',f'[{r.ci_low:.3f}, {r.ci_high:.3f}]'] for r in protect.itertuples()])
 bad=effects[(effects.proposed=='MAG_ONLY')&(effects.stage=='selected')&effects.baseline.isin(['B0','PLAIN'])&(((effects.panel=='ettm1')&(effects.condition=='SHIFT8')&(effects.kind=='standard'))|((effects.panel=='electricity_transfer')&(effects.condition=='STEP12_D63')&(effects.kind=='shape')))]
 assert len(bad)==4;bad.to_csv(P/'tables/negative_examples.csv',index=False)
 blocks['NEGATIVE']=table(['패널','조건','대조','MAG nMAE','대조 nMAE','이득(%)'],[[LABELS[r.panel],r.condition,r.baseline,f'{r.proposed_nmae:.6f}',f'{r.baseline_nmae:.6f}',f'{r.gain_pct:.3f}'] for r in bad.itertuples()])
 cost=resource.groupby('arm',as_index=False)[['trainable_parameters','optimizer_seconds','peak_allocated']].mean();cost['peak_MiB']=cost.pop('peak_allocated')/2**20;cost.to_csv(P/'tables/new_gate_resources.csv',index=False)
 blocks['RESOURCES']=table(['방법','추가 parameters','optimizer초 평균','peak allocated MiB 평균'],[['MAG',8712,'동일 timer 비교 없음','동일 조건 비교 없음']]+[[r.arm.replace('TOKEN_GATE_ENTROPY','Gate+H').replace('TOKEN_GATE','Gate'),int(r.trainable_parameters),f'{r.optimizer_seconds:.3f}',f'{r.peak_MiB:.3f}'] for r in cost.itertuples()])
 # Check numeric statements in the prose against authoritative effects.
 def effect(panel,baseline,kind='standard',condition='SHIFT8'):
  z=effects[(effects.panel==panel)&(effects.proposed=='MAG_ONLY')&(effects.baseline==baseline)&(effects.stage=='selected')&(effects.kind==kind)&(effects.condition==condition)];assert len(z)==1;return float(z.iloc[0].gain_pct)
 statements={}
 for panel,baseline,value in [('electricity_transfer','PLAIN',3.794),('electricity_transfer','B0',9.186),('neso_2026_jul_aug','PLAIN',1.941),('neso_2026_jul_aug','B0',4.626)]:
  actual=effect(panel,baseline);assert round(actual,3)==value;statements[f'{panel}/{baseline}/SHIFT8']=actual
 assert round(effect('ettm1','PLAIN'),2)==-.95
 assert round(effect('electricity_transfer','B0','shape','STEP12_D63'),2)==-3.27
 assert round(effect('electricity_transfer','PLAIN','shape','STEP12_D63'),2)==-2.65
 delta=csv(ROOT/'results/delta_adapter_comparison_20260919/EFFECTS.csv')
 for baseline,value in [('DELTA_XY_BUDGET',8.364),('DELTA_XY_DEFAULT',8.730)]:
  z=delta[(delta.panel=='electricity_transfer')&(delta.proposed=='MAG_ONLY')&(delta.baseline==baseline)&(delta.stage=='selected')&(delta.kind=='standard')&(delta.condition=='SHIFT8')];assert len(z)==1 and round(float(z.iloc[0].gain_pct),3)==value
 text=source(P/'MANUSCRIPT_TEMPLATE_KO.md').read_text()
 for key,value in blocks.items():assert '@@'+key+'@@' in text;text=text.replace('@@'+key+'@@',value)
 assert '@@' not in text;(P/'MANUSCRIPT_KO.md').write_text(text)
 figures={}
 for folder,stem in [(M,'method'),(M,'primary_effects'),(R,'comparison')]:
  for ext in ['png','pdf']:
   src=source(folder/f'{stem}.{ext}');dst=P/'figures'/src.name;shutil.copyfile(src,dst);figures[str(dst.relative_to(P))]=str(src.relative_to(ROOT))
 source(P/'build.py');source(P/'paper.css')
 (P/'EVIDENCE_MANIFEST.json').write_text(json.dumps(dict(base_commit='9a808b40ec5d3ac813e63022b22e62f1db35e357',evidence_hashes=seen,figure_sources=figures,checked_prose_values=statements,new_optimizer_updates=0,new_forecast_inferences=0,new_bootstrap=0,petsa_result_included=False,methodology_goal_complete=False),ensure_ascii=False,indent=2)+'\n')
 pandoc='/home/minjae/anaconda3/bin/pandoc'
 subprocess.run([pandoc,'MANUSCRIPT_KO.md','--standalone','--embed-resources','--mathml','--css=paper.css','--metadata=lang:ko','--metadata=pagetitle:관측 진폭으로 추가 잔차를 제한하는 시계열 PEFT','-o','MANUSCRIPT_KO.html'],cwd=P,check=True)
 subprocess.run([pandoc,'MANUSCRIPT_KO.md','--resource-path=.','--metadata=lang:ko','-o','MANUSCRIPT_KO.docx'],cwd=P,check=True)
 h=P/'MANUSCRIPT_KO.html';h.write_text('\n'.join(l.rstrip() for l in h.read_text().splitlines())+'\n')
 subprocess.run(['/usr/bin/google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-pdf-header-footer','--user-data-dir=/tmp/mag-method-v1-chrome','--print-to-pdf='+str(P/'MANUSCRIPT_KO.pdf'),'file://'+str(P/'MANUSCRIPT_KO.html')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 print('METHOD_MANUSCRIPT_RENDERED',len(seen),'fixed evidence files; no new training, inference or bootstrap')
if __name__=='__main__':main()
