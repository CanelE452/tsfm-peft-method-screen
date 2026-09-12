# Presentation only: consumes sealed results; no training or model imports.
import csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
NAMES={1:'Freshness-Gated LoRA',2:'DualClock-PEFT',3:'PatchPhase-PEFT',4:'FR-LoRA',5:'Maturity-PEFT',6:'Conditional Path Adapter',7:'Censor-Preserve LoRA'}
plt.rcParams.update({'figure.dpi':150,'savefig.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def table(rows):
    if not rows:return '[미검증] No fit/E metrics: stopped at an earlier gate.\n'
    keys=[k for k in rows[0] if k not in ['prediction_sha256']]
    return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+''.join('| '+' | '.join(str(row.get(k,'')) for k in keys)+' |\n' for row in rows)
summary=[];totalfits=0;totalstreams=0;wall=0.;updates=0;peak=0;all_integrity={}
for i in range(1,8):
    out=ROOT/'results'/f'candidate_{i:02}';status=read(out/'status.json')
    if status is None:continue
    contract=read(out/'contract.json',{});integ=read(out/'integrity.json',{'status':'NOT_COMPLETED'});resources=read(out/'resource_usage.json',{'fit_count':status.get('fit_count',0),'stream_count':status.get('stream_count',0),'wall_seconds':read(out/'job_exit.json',{}).get('wall_seconds',0)})
    for fn in ['contract.json','integrity.json','resource_usage.json']:
        if not (out/fn).exists():dump(out/fn,{'contract.json':contract,'integrity.json':integ,'resource_usage.json':resources}[fn])
    for fn,header in [('metrics.csv','arm,variant,scaled_2pinball\n'),('selections.csv','arm,lr,step,validation_loss\n'),('trajectories.csv','arm,lr,step,validation_loss\n')]:
        if not (out/fn).exists():(out/fn).write_text(header)
    rows=list(csv.DictReader(open(out/'metrics.csv')));gate=read(out/'round0.json',{})
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,4.8))
    if rows:
        arms=list(dict.fromkeys(row['arm'] for row in rows));scores=[]
        for arm in arms:
            selected=[row for row in rows if row['arm']==arm and not (i==1 and row['variant']=='clean') and not (i==3 and row['variant']=='0')]
            scores.append(np.mean([float(row['scaled_2pinball']) for row in selected]))
        colors=['#157f78' if a==status.get('proposed') else '#7d91ad' for a in arms]
        bars=ax.barh(np.arange(len(arms)),scores,color=colors);ax.set_yticks(np.arange(len(arms)),[a.replace('_',' ') for a in arms]);ax.invert_yaxis();ax.set_xlabel('Scaled 2-pinball (lower is better)')
        ax.set_xlim(0,max(scores)*1.17)
        for b,v in zip(bars,scores):ax.text(v+max(scores)*.012,b.get_y()+b.get_height()/2,f'{v:.6f}',va='center',fontsize=9)
    elif gate.get('loss'):
        ax.bar(list(gate['loss']),list(gate['loss'].values()),color='#7d91ad');ax.set_ylabel('Round 0 F0 scaled 2-pinball');ax.tick_params(axis='x',rotation=25)
    else:
        ax.axis('off');ax.text(.5,.58,status['verdict'],ha='center',va='center',fontsize=21,fontweight='bold',color='#9f3e3e');ax.text(.5,.36,'No GPU fit / no E metric\nSee novelty or failure receipt',ha='center',va='center')
    ax.set_title(f'{i:02}  {NAMES[i]} — {status["verdict"]}',loc='left',pad=14);fig.tight_layout();fig.savefig(figdir/'primary.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8));trajectory=list(csv.DictReader(open(out/'trajectories.csv')))
    if i==1 and rows:
        variants=list(dict.fromkeys(row['variant'] for row in rows));f0={row['variant']:float(row['scaled_2pinball']) for row in rows if row['arm']=='F0'}
        for arm in dict.fromkeys(row['arm'] for row in rows):
            if arm=='F0':continue
            lookup={row['variant']:float(row['scaled_2pinball']) for row in rows if row['arm']==arm};ax.plot(variants,[100*(f0[v]-lookup[v])/f0[v] for v in variants],marker='o',label=arm)
        ax.axhline(0,color='gray',lw=.8);ax.set_ylabel('Gain over F0 (%)');ax.legend(fontsize=7)
    elif i==2 and (out/'series_effects.csv').exists():
        rr=list(csv.DictReader(open(out/'series_effects.csv')));values=np.array([float(v['loss_difference']) for v in rr]);ax.hist(values,bins=30,color='#157f78');ax.axvline(0,color='black',lw=1);ax.set_xlabel('EventSummary loss − DualClock loss');ax.set_ylabel('Series count')
    elif i==5 and (out/'origin_losses.json').exists():
        rr=read(out/'origin_losses.json');
        for a,v in rr.items():ax.plot(np.arange(1,31),np.cumsum(v)/np.arange(1,31),label=a,lw=1.3)
        ax.set_xlabel('Issued origin');ax.set_ylabel('Cumulative prequential loss');ax.legend(fontsize=7)
    elif trajectory:
        keys=list(dict.fromkeys((row['arm'],row.get('lr','')) for row in trajectory))
        for arm,lr in keys:
            rr=[row for row in trajectory if row['arm']==arm and row.get('lr','')==lr]
            if 'validation_loss' in rr[0]:ax.plot([int(row['step']) for row in rr],[float(row['validation_loss']) for row in rr],label=f'{arm} {lr}',lw=1)
        ax.set_xlabel('Optimizer step');ax.set_ylabel('Validation scaled 2-pinball');ax.legend(fontsize=6,ncol=2)
    elif gate.get('degradation_percent'):
        d=gate['degradation_percent'];ax.bar(list(d),list(d.values()),color='#7d91ad');ax.axhline(1,color='gray',ls='--');ax.set_ylabel('F0 degradation (%)');ax.tick_params(axis='x',rotation=25)
    else:
        ax.axis('off');ax.text(.5,.5,'No method-specific experiment executed\nGate evidence is in RESULT.md',ha='center',va='center')
    ax.set_title('Diagnostics / validation only',loc='left');fig.tight_layout();fig.savefig(figdir/'diagnostics.png');plt.close(fig)
    reason=status.get('reason',status.get('error','Fixed PASS threshold evaluation; see diagnostics.'))
    text=f'''# Candidate {i:02}: {NAMES[i]}

[문제]
[확인] Problem gate: {status.get('problem_gate','unknown')}. {json.dumps(gate,ensure_ascii=False)}

[방법]
[확인] The recipe is documented in [CANDIDATE_{i:02}](../../docs/CANDIDATE_{i:02}.md). Proposed: {status.get('proposed',NAMES[i])}.

[강한 단순 baseline]
[확인] Strongest observed simple baseline: {status.get('strongest_baseline','not evaluated')}. All prespecified baseline arms remain in the raw table.

[데이터]
[확인] Dataset: {contract.get('dataset','not evaluated')}. Train/V/E manifests and input hashes are in screening_summary. E opened only after a saved selection seal, when executed.

[학습 파라미터]
[확인] Fits: {status.get('fit_count',0)}; streams: {status.get('stream_count',0)}. Contract: contract.json. Standard attention LoRA rank8/alpha16,96 projections; frozen native head. Candidate-specific additions are recorded in integrity.json.

[무결성]
[확인] {json.dumps(integ,ensure_ascii=False)}

[raw 결과]
{table(rows)}

[relative 결과]
[확인] Gain vs strongest simple baseline (% F0): {status.get('gain_percent_f0','not evaluated')}.
[확인] Diagnostics: {json.dumps(status.get('diagnostics',{}),ensure_ascii=False)}

[성공/실패 판정]
[판정] {status['verdict']}. {reason}

[말할 수 없는 것]
[미검증] A one-seed development screen is not paper-level evidence, cross-dataset robustness or novelty certification. Evaluation origins overlap in time; no independent-sample significance claim is made. Prior training-data overlap of the TSFM is not excluded. Candidate06 is stopped for core-mechanism overlap, not proven exact algebraic identity. Candidate07 recovers synthetically capped recorded M5 sales, not verified real latent demand. TAFAS-like is a scoped adaptation, not full reproduction.

[Round2 추천 여부]
[판정] {'Eligible for review, subject to max-two overall ranking.' if status.get('round2_recommended') else 'Not recommended from this screen.'} Round2 not executed.
'''
    (out/'RESULT.md').write_text(text)
    f=status.get('fit_count') or 0;s=status.get('stream_count') or 0;cost=resources.get('wall_seconds',0);totalfits+=f;totalstreams+=s;wall+=cost
    for u in resources.get('fits',[])+resources.get('streams',[]):updates+=u.get('optimizer_steps',0);peak=max(peak,u.get('gpu_peak_bytes',0))
    all_integrity[f'{i:02}']=integ
    summary.append(dict(candidate=f'{i:02} {NAMES[i]}',problem_gate=status.get('problem_gate'),strongest_baseline=status.get('strongest_baseline'),proposed_primary=status.get('proposed_primary'),proposed_gain=status.get('gain_percent_f0'),method_specificity='Thresholds met' if status['verdict']=='PASS' else 'Not established',robustness=json.dumps(status.get('diagnostics',{})),compute_cost=f'{f} fits + {s} streams; {cost:.1f}s',novelty_risk='collision' if i==6 else 'high' if i==3 else 'medium',major_failure_mode=reason if status['verdict']!='PASS' else 'Single dataset/seed only',verdict=status['verdict']))
summary_dir=ROOT/'results/screening_summary'
with open(summary_dir/'all_candidates.csv','w') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]) if summary else ['candidate']);writer.writeheader();writer.writerows(summary)
assert totalfits<=38 and totalstreams<=5
# Ranking is editorial: order by verdict classes for initial presentation,
# then explicit forecast evidence and novelty/robustness rationale. No score sum.
priority={'PASS':0,'WEAK':1,'FAIL':2,'NO_PROBLEM':3,'INVALID_CONSTRUCT':4,'IMPLEMENTATION_BLOCKED':5,'NOVELTY_COLLISION':6}
ranking=sorted(summary,key=lambda row:(priority[row['verdict']],-(row['proposed_gain'] if row['proposed_gain'] is not None else -1e9)))
lines=['# Round 1 review ranking','', 'Development evidence only. No weighted score sum. PASS requires all candidate-specific conditions; raw gain alone does not justify advancement.','']
for idx,row in enumerate(ranking,1):
    lines.append(f"{idx}. **{row['candidate']} — {row['verdict']}**. Forecast gain: {row['proposed_gain']}% F0; specificity: {row['method_specificity']}; novelty risk: {row['novelty_risk']}. {row['major_failure_mode']}")
(summary_dir/'ranking.md').write_text('\n'.join(lines)+'\n')
recommended=[row['candidate'] for row in ranking if row['verdict']=='PASS'][:2]
dump(summary_dir/'compute_summary.json',dict(fit_count=totalfits,stream_count=totalstreams,optimizer_steps=updates,candidate_wall_seconds=wall,peak_gpu_allocated_bytes=peak,fit_cap=38,stream_cap=5,round2_executed=False))
dump(summary_dir/'integrity_summary.json',all_integrity)
dump(summary_dir/'final_verdict.json',dict(complete=len(summary)==7,ranking=[row['candidate'] for row in ranking],round2_recommendations=recommended,round2_executed=False,reason='Only candidates passing all prespecified criteria are eligible; at most two.',final_paper_evidence=False))
