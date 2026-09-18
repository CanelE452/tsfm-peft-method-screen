"""Build a Korean manuscript from verified evidence; no fitting or inference."""
from pathlib import Path
import hashlib,json,shutil,subprocess
import pandas as pd
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
W=ROOT/'results/c3_weakness_controls_20260918';D=ROOT/'results/c3_magnitude_diagnostic_20260918';T=ROOT/'results/c3_identifiability_temporal_20260918'
seen={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source(p):seen[str(p.relative_to(ROOT))]=sha(p);return p
def readcsv(p):return pd.read_csv(source(p))
def md(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def dump(name,frame):
    (P/'tables').mkdir(exist_ok=True);frame.to_csv(P/'tables'/name,index=False)

labels={'electricity':'Electricity 4계열','electricity_transfer':'Electricity 전이16','ettm1':'ETTm1','ettm2':'ETTm2','neso_2025':'NESO 2025','neso_2026_h1':'NESO 2026 H1'}
h=readcsv(P.parent/'tables/T1_main_SHIFT8.csv');t=readcsv(T/'EFFECTS.csv');t=t[(t.kind=='standard')&(t.condition=='SHIFT8')&(t.new=='C3')];hist=pd.concat([h,t],ignore_index=True);hist=hist[hist.baseline.isin(['C0','C2','M_RECENCY'])];dump('T1_historical.csv',hist)
rows=[]
for panel in labels:
    g=hist[hist.panel==panel];out=[labels[panel]]
    for baseline in ['C0','C2','M_RECENCY']:
        r=g[g.baseline==baseline];out.append(f'{r.iloc[0].gain_pct:+.3f}' if len(r) else '미실행')
    rows.append(out)
blocks={'HISTORICAL_TABLE':md(['평가 패널','B0 대비','C2 대비','RECENCY 대비'],rows)}
f=readcsv(W/'PRIMARY_RESULTS.csv');dump('T2_controls_all_intervals.csv',f);rows=[]
for r in f[f.ci_type=='time'].itertuples():rows.append([labels[r.panel],r.baseline,f'{r.gain_pct:+.3f}',f'[{r.bonferroni3_low_pct:+.3f}, {r.bonferroni3_high_pct:+.3f}]'])
blocks['CONTROL_TABLE']=md(['패널','대조','C3 이득(%)','Bonferroni 구간'],rows)
f=readcsv(D/'DECOMPOSITION.csv');allf=f[f.family=='ALL'];main=allf[(allf.kind=='standard')&(allf.condition=='SHIFT8')];dump('T3_decomposition.csv',main)
r=main[main.panel=='electricity_transfer'].iloc[0];blocks['FACTORIAL_TABLE']=md(['구성','학습 가중치','추론 규칙','nMAE'],[[a,w,g,f'{r[a]:.9f}'] for a,w,g in [('A','C3','C3'),('B','C3','MAG_ONLY'),('C','MAG_ONLY','C3'),('D','MAG_ONLY','MAG_ONLY')]])
blocks['FACTORIAL_TABLE']+='\n\n규칙 항의 절대 nMAE는 '+f"{r.gate:+.6f} (사후95% 구간 [{r.gate_low:+.6f}, {r.gate_high:+.6f}]), 가중치 항은 {r.weights:+.6f} ([{r.weights_low:+.6f}, {r.weights_high:+.6f}])다."
s=readcsv(D/'SEED_DECOMPOSITION.csv');s=s[(s.family=='ALL')&(s.kind=='standard')&(s.condition=='SHIFT8')];dump('T4_seed_decomposition.csv',s);s=s[s.panel=='electricity_transfer'];blocks['SEED_TABLE']=md(['seed','C3 오차 A','MAG 오차 D','total','gate','weights'],[[r.seed,f'{r.A:.6f}',f'{r.D:.6f}',f'{r.total:+.6f}',f'{r.gate:+.6f}',f'{r.weights:+.6f}'] for r in s.itertuples()])
f=allf[(allf.kind=='standard')&allf.condition.isin(['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT'])];dump('T5_tradeoffs.csv',f);blocks['TRADEOFF_TABLE']=md(['조건','C3 오차','MAG 오차','MAG 이득(%)'],[[r.condition,f'{r.A:.6f}',f'{r.D:.6f}',f'{r.MAG_gain_pct:+.4f}'] for r in f[f.panel=='electricity_transfer'].itertuples()])
for folder,name in [(D,'GATE_SUMMARY.csv'),(D,'CHANNEL_CONTRIBUTIONS.csv'),(D,'CHECKPOINT_CONTEXT.csv'),(W,'RESOURCE_REPORT.csv'),(W,'FIT_LEDGER.csv')]:dump(name,readcsv(folder/name))
(P/'figures').mkdir(exist_ok=True);figs={}
for stem,folder,oldname in [('F1_controls',W,'01_primary_controls'),('F2_decomposition',D,'01_gate_weights_decomposition'),('F3_tradeoffs',D,'02_state_tradeoffs'),('FA1_shapes',D,'04_shapes_decomposition')]:
    for suffix in ['png','pdf','svg']:
        src=source(folder/'figures'/f'{oldname}.{suffix}');dst=P/'figures'/f'{stem}.{suffix}';shutil.copyfile(src,dst);assert sha(src)==sha(dst);figs[str(dst.relative_to(P))]=str(src.relative_to(ROOT))
text=(P/'MANUSCRIPT_TEMPLATE_KO.md').read_text()
for k,v in blocks.items():assert '@@'+k+'@@' in text;text=text.replace('@@'+k+'@@',v)
assert '@@' not in text;(P/'MANUSCRIPT_KO.md').write_text(text)
for name in ['REPORT.md','AUDIT.json','TRAINING_PROVENANCE.json','VARIANCE_COMPONENTS.csv','LEAVE_ONE_GROUP_OUT.csv']:source(ROOT/'research/c3_influence_audit_20260918'/name)
for name in ['VERIFICATION.json','INDEPENDENT_AUDIT.json','PROTOCOL.md','SEAL.json']:source(D/name)
for name in ['VERIFICATION.json','PROTOCOL.md','SEAL.json']:source(W/name)
for name in ['EVIDENCE_BRIEF_KO.pdf','EVIDENCE_BRIEF_KO.docx','EVIDENCE_BRIEF_KO.md','EVIDENCE_REPORT.md','CLAIM_EVIDENCE.md','METHODS_AND_PROTOCOL.md']:source(P.parent/name)
(P/'EVIDENCE_MANIFEST.json').write_text(json.dumps(dict(base_commit='962a15a8acdcda260ccf96aaa144b5e47ecf1494',new_fits=0,new_model_inference=0,new_bootstrap=0,evidence_hashes=seen,figure_sources=figs,tables_rendered_from_csv=True,old_manuscripts_preserved=True),ensure_ascii=False,indent=2)+'\n')
pandoc='/home/minjae/anaconda3/bin/pandoc'
subprocess.run([pandoc,'MANUSCRIPT_KO.md','--standalone','--embed-resources','--css=paper.css','--metadata=lang:ko','--metadata=pagetitle:추가 PEFT의 관측 기반 가중과 역할 분해','-o','MANUSCRIPT_KO.html'],cwd=P,check=True)
subprocess.run([pandoc,'MANUSCRIPT_KO.md','--resource-path=.','--metadata=lang:ko','-o','MANUSCRIPT_KO.docx'],cwd=P,check=True)
p=P/'MANUSCRIPT_KO.html';p.write_text('\n'.join(l.rstrip() for l in p.read_text().splitlines())+'\n')
print('MANUSCRIPT_BUILT',len(seen),'evidence files; 0 fits/inference',flush=True)
