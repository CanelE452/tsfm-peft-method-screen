"""Render a method manuscript from fixed evidence, without new experiments."""
from pathlib import Path
import hashlib,json,shutil,subprocess
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
R=ROOT/'results/learned_gate_comparison_20260919'
T=ROOT/'results/petsa_cell_comparison_20260919'
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
 raw=csv(T/'RAW_SCORES.csv');effects=csv(R/'EFFECTS.csv');primary=csv(R/'PRIMARY_COMPARISONS.csv');resource=pd.concat([csv(R/'RESOURCES.csv'),csv(T/'RESOURCES.csv')],ignore_index=True)
 petsa=csv(T/'PRIMARY_COMPARISONS.csv');pe=csv(T/'EFFECTS.csv');pa=read(T/'AUDIT.json');assert pa['status']=='VERIFIED' and pa['new_fits']==8 and pa['main_updates']==8192 and pa['smoke_updates']==4
 assert read(T/'status.json')['execution']=='COMPLETE_VERIFIED'
 for name in ['REPORT.md','FINAL_DECISION.md','MODEL_SELECTION.json','EVALUATION_SEAL.json','ALL_PREDICTIONS_SAVED.json','AUTHORIZATION.json','QUANTILE_CROSSING.csv']:source(T/name)
 for name in ['PRIOR_SCOPE_KO.md','SOURCES.json']:source(ROOT/'research/mag_novelty_extension_20260919'/name)
 data=read(ROOT/'results/additive_persistence_validation_v1_20260917/DATA_MANIFEST.json');data.update(read(R/'DATA_MANIFEST.json'))
 origin=csv(R/'ORIGIN_AUDIT.csv')
 for f in ['REPORT.md','FINAL_DECISION.md','MODEL_SELECTION.json','EVALUATION_SEAL.json','ALL_PREDICTIONS_SAVED.json']:source(R/f)
 for p in [ROOT/'experiments/learned_gate_comparison_20260919/PROTOCOL.md',ROOT/'experiments/c3_weakness_controls_20260918/model.py',ROOT/'experiments/additive_persistence_validation_v1_20260917/model.py',ROOT/'experiments/outlier_signal_followup_v2_20260917/prepare.py',ROOT/'experiments/outlier_signal_peft_v1_20260917/reference_core.py',ROOT/'research/method_baseline_compatibility_20260919/BASELINE_COMPATIBILITY_KO.md',ROOT/'research/learned_gate_comparability_20260919/RESOURCE_SCOPE_KO.md',ROOT/'results/petsa_cell_comparison_20260919/status.json']:source(p)
 (P/'tables').mkdir(exist_ok=True);(P/'figures').mkdir(exist_ok=True)
 raw.to_csv(P/'tables/all_raw_scores.csv',index=False)
 raw[raw.kind=='shape'].to_csv(P/'tables/all_shape_scores.csv',index=False)
 primary.to_csv(P/'tables/primary.csv',index=False)
 shutil.copyfile(R/'MODEL_SELECTION.json',P/'tables/model_selection.json')
 blocks={};panels=[]
 for panel in PANELS:
  d=data[panel];panels.append([LABELS[panel],len(d['selected_columns']),d['period'],55 if panel.startswith('neso') else 128,'gate 비교 당시 미채점·PETSA에서는 재사용' if panel.startswith('neso') else '재사용 개발 평가'])
 blocks['PANELS']=table(['패널','계열 수','관측/일','E origin days','노출 역할'],panels)
 f=raw[(raw.stage=='selected')&(raw.kind=='standard')&(raw.condition=='SHIFT8')].groupby(['panel','arm']).nmae.mean().unstack()
 f.to_csv(P/'tables/shift8_means.csv')
 arms=['B0','PLAIN','MAG_ONLY','TOKEN_GATE','TOKEN_GATE_ENTROPY','PETSA_XY_OFFLINE']
 blocks['RAW_SHIFT8']=table(['패널','B0','PLAIN','MAG','Gate','Gate+H','PETSA cell'],[[LABELS[k],*[f'{f.loc[k,a]:.6f}' for a in arms]] for k in PANELS])
 petsa.to_csv(P/'tables/petsa_primary.csv',index=False)
 shutil.copyfile(T/'MODEL_SELECTION.json',P/'tables/petsa_model_selection.json')
 shutil.copyfile(T/'QUANTILE_CROSSING.csv',P/'tables/quantile_crossing.csv')
 blocks['PETSA_PRIMARY']=table(['패널','MAG nMAE','cell nMAE','이득(%)','family2 구간','seed81551/81552'],[[LABELS[r.panel],f'{r.proposed_nmae:.6f}',f'{r.baseline_nmae:.6f}',f'{r.gain_pct:.3f}',f'[{r.bonf2_low:.3f}, {r.bonf2_high:.3f}]',' / '.join(f'{json.loads(r.seed_gains)[str(seed)]:.3f}' for seed in [81551,81552])] for r in petsa.itertuples()])
 cross=csv(T/'QUANTILE_CROSSING.csv');cx=cross[cross.view.str.contains('PETSA_XY_OFFLINE')];assert len(cx)==32
 blocks['PETSA_CROSSING']=f"{100*cx.fraction.min():.4f}%–{100*cx.fraction.max():.4f}%"
 bypanel=petsa.set_index('panel')
 blocks['PETSA_ABSTRACT']=f"추가 PETSA 공개 cell의 offline 대조에서 SHIFT8 이득은 전력 전이 {bypanel.loc['electricity_transfer','gain_pct']:.3f}%, NESO {bypanel.loc['neso_2026_jul_aug','gain_pct']:.3f}%였으며, 두 패널 동시 기준은 {'충족했다' if pa['limited_additional_baseline_support'] else '충족하지 못했다'}."
 blocks['PETSA_DECISION']=('충족' if pa['limited_additional_baseline_support'] else '미충족')+' ('+pa['decision']+')'
 pp=pe[(pe.proposed=='MAG_ONLY')&(pe.baseline=='PETSA_XY_OFFLINE')&(pe.stage=='selected')&(pe.kind=='standard')&pe.panel.isin(['electricity_transfer',PANELS[-1]])&pe.condition.isin(['REFERENCE','FAULT'])]
 assert len(pp)==4;pp.to_csv(P/'tables/petsa_protection.csv',index=False)
 blocks['PETSA_PROTECTION']=table(['패널','조건','MAG nMAE','cell nMAE','이득(%)'],[[LABELS[r.panel],r.condition,f'{r.proposed_nmae:.6f}',f'{r.baseline_nmae:.6f}',f'{r.gain_pct:.3f}'] for r in pp.itertuples()])
 blocks['PETSA_AUDIT']=f"{pa['prediction_views']} prediction views, {pa['unique_prediction_files']}개 고유 배열, {pa['checkpoints']}개 새 checkpoint, {pa['independent_origin_metrics']}개 원점별 metric"
 blocks['PRIMARY']=table(['패널','대조','이득(%)','보정구간','seed81551','seed81552'],[[LABELS[r.panel],r.baseline.replace('TOKEN_GATE_ENTROPY','Gate+H').replace('TOKEN_GATE','Gate'),f'{r.gain_pct:.3f}',f'[{r.bonf4_low:.3f}, {r.bonf4_high:.3f}]',*[f'{json.loads(r.seed_gains)[str(seed)]:.3f}' for seed in [81551,81552]]] for r in primary.itertuples()])
 protect=effects[(effects.proposed=='MAG_ONLY')&effects.panel.isin(['electricity_transfer',PANELS[-1]])&(effects.stage=='selected')&(effects.kind=='standard')&effects.condition.isin(['REFERENCE','FAULT'])&effects.baseline.isin(['B0','PLAIN'])]
 assert len(protect)==8;protect.to_csv(P/'tables/protection.csv',index=False)
 blocks['PROTECTION']=table(['패널','조건','대조','이득(%)','일반95%구간'],[[LABELS[r.panel],r.condition,r.baseline,f'{r.gain_pct:.3f}',f'[{r.ci_low:.3f}, {r.ci_high:.3f}]'] for r in protect.itertuples()])
 bad=effects[(effects.proposed=='MAG_ONLY')&(effects.stage=='selected')&effects.baseline.isin(['B0','PLAIN'])&(((effects.panel=='ettm1')&(effects.condition=='SHIFT8')&(effects.kind=='standard'))|((effects.panel=='electricity_transfer')&(effects.condition=='STEP12_D63')&(effects.kind=='shape')))]
 assert len(bad)==4;bad.to_csv(P/'tables/negative_examples.csv',index=False)
 blocks['NEGATIVE']=table(['패널','조건','대조','MAG nMAE','대조 nMAE','이득(%)'],[[LABELS[r.panel],r.condition,r.baseline,f'{r.proposed_nmae:.6f}',f'{r.baseline_nmae:.6f}',f'{r.gain_pct:.3f}'] for r in bad.itertuples()])
 cost=resource.groupby('arm',as_index=False)[['trainable_parameters','optimizer_seconds','peak_allocated']].mean();cost['peak_MiB']=cost.pop('peak_allocated')/2**20;cost.to_csv(P/'tables/new_gate_resources.csv',index=False)
 blocks['RESOURCES']=table(['방법','추가 parameters','optimizer초 평균','peak allocated MiB 평균'],[['MAG',8712,'동일 timer 비교 없음','동일 조건 비교 없음']]+[[r.arm.replace('TOKEN_GATE_ENTROPY','Gate+H').replace('TOKEN_GATE','Gate').replace('PETSA_XY_OFFLINE','PETSA cell'),int(r.trainable_parameters),f'{r.optimizer_seconds:.3f}',f'{r.peak_MiB:.3f}'] for r in cost.itertuples()])
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
 # Preserve historical mechanism evidence as a separate estimand.
 dm=csv(ROOT/'results/c3_magnitude_diagnostic_20260918/DECOMPOSITION.csv')
 dm=dm[(dm.panel=='electricity_transfer')&(dm.kind=='standard')&(dm.condition=='SHIFT8')&(dm.family=='ALL')&(dm.stratum=='ALL')];assert len(dm)==1
 d=dm.iloc[0]
 for key,value in [('weights',.6436),('gate',-.3928),('total',.2509)]:
  actual=100*float(d[key])/float(d.A);assert round(actual,4)==value;statements['historical_decomposition_'+key]=actual
 pm=csv(ROOT/'results/c3_internal_mechanism_20260918/PAIRING_EFFECTS.csv')
 for arm,value in [('C3',9.7476),('MAG_ONLY',10.1990)]:
  z=pm[(pm.panel=='electricity_transfer')&(pm.kind=='standard')&(pm.condition=='SHIFT8')&(pm.arm==arm)];assert len(z)==1 and round(float(z.iloc[0].penalty_pct),4)==value
 for folder,names in [('c3_magnitude_diagnostic_20260918',['INTERPRETATION_KO.md','INDEPENDENT_AUDIT.json']),('c3_internal_mechanism_20260918',['REPORT.md','INDEPENDENT_AUDIT.json'])]:
  for name in names:source(ROOT/'results'/folder/name)
 dm.to_csv(P/'tables/historical_gate_weight_decomposition.csv',index=False)
 pm.to_csv(P/'tables/historical_B0_pairing.csv',index=False)
 prior=read(ROOT/'research/mag_novelty_extension_20260919/SOURCES.json')
 assert sha(ROOT/'research/mag_novelty_extension_20260919/PRIOR_SCOPE_KO.md')==prior['report_sha256']
 text=source(P/'MANUSCRIPT_TEMPLATE_KO.md').read_text()
 for key,value in blocks.items():assert '@@'+key+'@@' in text;text=text.replace('@@'+key+'@@',value)
 assert '@@' not in text;(P/'MANUSCRIPT_KO.md').write_text(text)
 figures={}
 for folder,stem in [(M,'method'),(M,'primary_effects'),(T,'comparison')]:
  for ext in ['png','pdf']:
   src=source(folder/f'{stem}.{ext}');dst=P/'figures'/src.name;shutil.copyfile(src,dst);figures[str(dst.relative_to(P))]=str(src.relative_to(ROOT))
 claims=[
 ['구체적 PEFT 방법 명세','B0 동결, 원래 입력 보존, MAD 진폭 gate, 8,712개 추가 잔차','방법 수식·공식 구현·실제 초기 동일성/복원 검사','국소 residual bound를 최종 오차 보장으로 확대하지 않음'],
 ['일반 잔차 대비 추가 효과','전력 전이 SHIFT8 +3.794%; NESO 후반 +1.941%','동일 B0/학습 기회; seed·날짜 점수 보존','ETTm1·원자료·오류·긴 변화 손해 존재'],
 ['학습형 gate 대비 추가 효과','네 주 비교의 family4 구간 하한>0 및 모든 seed 양수','16fits 완료·독립검산','표현/초기 gate/파라미터 수가 달라 trainability만의 효과 아님'],
 ['PETSA 공개 cell 대비 추가 효과',pa['decision'],'8fits/8192+4updates; family2 및 모든 보호조건 확인','offline 이식·모든 E 재사용; 공식 온라인 전체 PETSA 우위 아님'],
 ['기전 해석','고정 규칙 효과와 학습된 가중치 효과는 반대일 수 있음','과거 C3–MAG 교차 분해 및 B0 교환','지속성 식별·최대 원인·완전한 인과 매개 주장 안 함'],
 ['신규성','관측 robust gate를 추가 내부 잔차에 적용한 구체적 조합','GateRA/PETSA/δ/AIRA/온라인 Kalman과 범위 구분','gating/outlier-aware PEFT 최초성·충분한 학술 신규성 미확정'],
 ['자료 일반성','동일 Electricity의 추가 계열 및 NESO 시간 구간','모든 source/노출 이력·ETT 손해 보존','독립 source·실제 센서 사건 해결·범용 우위 미입증'],
 ['자원','추가 MAG8712, Gate9225, PETSA19010 parameters','receipt의 시간·peak memory 공개','B0 학습 비용 존재; timer가 달라 공정한 속도비 미확정']]
 (P/'PAPER_CLAIM_EVIDENCE.md').write_text('# 방법론 주장–근거–한계\n\n'+table(['주장','현재 근거','검증','확대하지 않을 범위'],claims)+'\n\n이번 원고 생성은 신규 학습0회다. 추가 PETSA 실험의8회와 기존 gate16회는 서로 별도 완료 예산이며 전체 연구 누적 비용을24회로 표시하지 않는다.\n')
 (P/'FINAL_DECISION.md').write_text('# 현재 논문 판단\n\nPETSA 추가 대조: '+pa['decision']+'. 실행8/8회·8192 main+4 smoke updates와 평가·검산 완료. 미실행 승인 학습0회.\n\n남길 구현은 기존 B0, PLAIN, 고정 MAG 및 비교군이다. 새 구조로 전환하거나 MAG threshold/LR/seed를 변경하지 않는다. 제한된 방법론 주장으로 원고를 구성할 수 있지만, 이번 문서 생성이나 추가 대조의 국소 기준 충족을 논문 PASS 또는 투고 준비 완료로 선언하지 않는다.\n\n기존 전력·NESO의 SHIFT8 이득을 남기며 원자료·FAULT·ETTm1·긴 변화 손해도 남긴다. C3 지속성 규칙을 핵심 신규 기여로 되살리지 않는다. 가까운 선행의 전체 방법 대비 우위, 충분한 신규성, 독립 source 및 seed 모집단 일반화, 동일 timer의 속도 우위는 미확립이다.\n\n사전 확증 근거와 개발 평가 보강을 구분한다. 후속 학습을 자동 추가하지 않으며 현재 결과와 원고를 scoped commit/push한다.\n')
 # Plot saved intervals only: no additional bootstrap or model evaluation.
 trade=pe[(pe.proposed=='MAG_ONLY')&(pe.baseline=='PETSA_XY_OFFLINE')&(pe.stage=='selected')&(pe.kind=='standard')&pe.condition.isin(['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT'])].copy()
 assert len(trade)==20;trade.to_csv(P/'tables/petsa_tradeoffs.csv',index=False)
 fig,axes=plt.subplots(2,2,figsize=(10,5.5));conditions=['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT']
 for ax,panel in zip(axes.flat,PANELS):
  z=trade[trade.panel==panel].set_index('condition').loc[conditions]
  ax.axvline(0,color='0.5',linewidth=.8)
  for j,(_,row) in enumerate(z.iterrows()):
   color='#146b61' if row.gain_pct>=0 else '#ad4439'
   ax.plot([row.ci_low,row.ci_high],[j,j],color=color,linewidth=1.5)
   ax.scatter([row.gain_pct],[j],color=color,s=20,zorder=3)
  ax.set_yticks(range(5),conditions,fontsize=8);ax.invert_yaxis();ax.set_title(panel,fontsize=9);ax.set_xlabel('MAG gain vs PETSA cell (%)',fontsize=8);ax.grid(axis='x',alpha=.15)
 fig.suptitle('Offline controlled transfer: all reused development panels; marginal 95% intervals',fontsize=10)
 fig.tight_layout();fig.savefig(P/'figures/petsa_tradeoffs.png',dpi=180);fig.savefig(P/'figures/petsa_tradeoffs.pdf');plt.close(fig)
 # Record which fixed criterion failed, without turning uncertainty into a performance defeat.
 selected=read(T/'MODEL_SELECTION.json')
 assert [r['step'] for r in selected if r['source']=='ettm1' and r['stage']=='selected']==[0,0]
 assert float(bypanel.loc['neso_2026_jul_aug','bonf2_low'])<=0<float(bypanel.loc['neso_2026_jul_aug','bonf2_high'])
 protection_all=pe[(pe.proposed=='MAG_ONLY')&pe.panel.isin(['electricity_transfer',PANELS[-1]])&(pe.stage=='selected')&(pe.kind=='standard')&pe.condition.isin(['REFERENCE','FAULT'])]
 verdicts={'두 주 비교 보정구간 하한>0':bool((petsa.bonf2_low>0).all()),'2패널×2seed의 네 이득>0':all(all(v>0 for v in json.loads(x).values()) for x in petsa.seed_gains),'12개 평균 보호조건 악화≤1%':bool((protection_all.gain_pct>=-1).all())}
 detail='\n## 이번 추가 대조의 실제 값\n\n'+blocks['PETSA_PRIMARY']+'\n\n'+table(['봉인한 조건','충족'],[[k,v] for k,v in verdicts.items()])+'\n\nNESO의 구간이0을 포함하는 것이 이번 결합 기준 미충족 사유다. 두seed의 평균 이득이 모두 양수라는 점은 보존한다. 이는 MAG가 항상 졌다는 결과나 전체 robust PEFT의 반증이 아니다. 반대로 불확실한 양의 점추정만으로 PETSA 대비 일반 우위를 주장하지 않는다.\n'
 with (P/'FINAL_DECISION.md').open('a') as fh:fh.write(detail)
 source(P/'build.py');source(P/'paper.css')
 (P/'EVIDENCE_MANIFEST.json').write_text(json.dumps(dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),evidence_hashes=seen,figure_sources=figures,generated_figures={"figures/petsa_tradeoffs.png":"tables/petsa_tradeoffs.csv","figures/petsa_tradeoffs.pdf":"tables/petsa_tradeoffs.csv"},checked_prose_values=statements,new_optimizer_updates=0,new_forecast_inferences=0,new_bootstrap=0,petsa_result_included=True,petsa_main_updates=8192,petsa_smoke_updates=4,methodology_goal_complete=False),ensure_ascii=False,indent=2)+'\n')
 pandoc='/home/minjae/anaconda3/bin/pandoc'
 subprocess.run([pandoc,'MANUSCRIPT_KO.md','--standalone','--embed-resources','--mathml','--css=paper.css','--metadata=lang:ko','--metadata=pagetitle:관측 진폭으로 추가 잔차를 제한하는 시계열 PEFT','-o','MANUSCRIPT_KO.html'],cwd=P,check=True)
 subprocess.run([pandoc,'MANUSCRIPT_KO.md','--resource-path=.','--metadata=lang:ko','-o','MANUSCRIPT_KO.docx'],cwd=P,check=True)
 h=P/'MANUSCRIPT_KO.html';h.write_text('\n'.join(l.rstrip() for l in h.read_text().splitlines())+'\n')
 subprocess.run(['/usr/bin/google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-pdf-header-footer','--user-data-dir=/tmp/mag-method-v2-chrome','--print-to-pdf='+str(P/'MANUSCRIPT_KO.pdf'),'file://'+str(P/'MANUSCRIPT_KO.html')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 print('METHOD_MANUSCRIPT_RENDERED',len(seen),'fixed evidence files; no new training, inference or bootstrap')
if __name__=='__main__':main()
