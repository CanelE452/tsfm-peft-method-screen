"""Korean reports distinguish blocked design, sampling sensitivity, and method evidence."""
import time
import pandas as pd
from .common import *

def table(df):
 if df.empty:return '해당 완료 수치 없음.'
 def fmt(v):return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
 return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def origin_report():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 d=pd.read_csv(OUT/'old_vs_new_origin_dispersion.csv');text='# 날짜 우선 표본 사전 감사\n\n기존 selector는 phase별 quota를 먼저 배정해 각 phase의 첫·중간·마지막 날짜가 겹쳤다. 새 selector는 합법적인 날짜 범위 양 끝을 포함해 날짜를 균등 선택한 다음, 지정 phase roster와 circular distance로 한 날짜당 한 원점만 고른다. 값/정답/성능은 선택에 쓰지 않는다. 전 검사·봉인은 GPU optimizer0회에서 수행한다.\n\n'
 cols=['track','role','version','origins','distinct_days','seven_day_blocks','distinct_phases','phase_count_range','origin_span_hours','span_ratio','target_uniqueness_ratio'];text+=table(d[cols])+'\n\n'
 for t in ORDER:
  old=read(OLD/t/'origins.json');new=read(OUT/t/'origins.json');P=96 if t=='N03' else 24;fig,axs=plt.subplots(2,4,figsize=(16,6),sharey=True)
  for row,(label,oo) in enumerate([('OLD',old),('REPAIRED',new)]):
   for col,r in enumerate(ROLES):
    a=np.array(oo[r]);axs[row,col].scatter(a//P,a%P,s=12);axs[row,col].set_title(f'{label} {r}: {len(set(a//P))} days');axs[row,col].set_xlabel('index-day');axs[row,col].set_ylabel('index-phase')
  fig.suptitle(t+' origin dispersion (panels show their own time range)');fig.tight_layout();fig.savefig(OUT/t/'origin_distribution.png',dpi=130);plt.close(fig)
  div=read(OUT/t/'origin_diversity.json');text+=f'## {t}\n\n'+('PASS' if div['passed'] else 'BLOCKED_DIVERSITY: '+', '.join(div['failures']))+f'。 [모든 day/span/week/phase/overlap 기록]({t}/origin_diversity.json), [old/new 원점]({t}/new_origins.csv).\n\n![{t}]({t}/origin_distribution.png)\n\n'
  if not div['passed']:
   for role,v in div['roles'].items():
    if v['failed']:text+=f'{role}: 날짜 {v["distinct_days"]}개·span ratio {v["span_ratio"]:.3f}·phase {v["distinct_phases"]}개지만 phase 최소/최대 {v["phase_count_min"]}/{v["phase_count_max"]}, 차이 {v["phase_count_range"]}로 한도2를 초과한다. 마지막 부분 날짜에서 지정phase에 가장 가까운 circular phase가0이 되는 경계 효과다. 날짜나 phase를 다시 골라 완화하지 않는다. 해당 트랙 GPU smoke/학습/교차 평가0회.\n\n'
  if t=='N03':
   fig,axs=plt.subplots(1,2,figsize=(11,4))
   for ax,(lab,oo) in zip(axs,[('OLD',old['E_DISCOVERY']),('REPAIRED',new['E_DISCOVERY'])]):
    a=np.array(oo);ax.scatter(a/96,a%96);ax.set_xlabel('continuous index-day');ax.set_ylabel('15-min index-phase');ax.set_title(f'{lab}: {(a[-1]-a[0])*.25:.2f} h span / {len(set(a//96))} days')
   fig.tight_layout();fig.savefig(OUT/'N03_old_vs_repaired_E.png',dpi=140);plt.close(fig)
 text+='## N03 병리 재현과 교정\n\n기존 E23.75시간 집중을 실제 old selector로 재현했고, 새 TRAIN/E64일·V32일과95% 이상 span을 모두 검사했다. 같은 날짜 여러 원점으로 숫자를 채우지 않았다.\n\n![N03 old/new](N03_old_vs_repaired_E.png)\n'
 (OUT/'ORIGIN_REPAIR_AUDIT.md').write_text(text)

def comparisons(t):
 old=pd.read_csv(OLD/t/'contrasts.csv');new=pd.read_csv(OUT/t/'contrasts.csv');condition='REVISION' if t=='R04' else 'PRIMARY';a=old[(old.policy=='selected')&(old.condition==condition)].copy();b=new[(new.policy=='selected')&(new.condition==condition)].copy();cols=['method','baseline','seed','method_score','baseline_score','gain_percent','ci_low','ci_high'];joined=a[cols].merge(b[cols],on=['method','baseline','seed'],suffixes=('_old','_repaired'),how='outer');joined['effect_sign_reversed']=joined.gain_percent_old*joined.gain_percent_repaired<0;joined.to_csv(OUT/t/'old_vs_repaired_contrasts.csv',index=False)
 core=b[(b.baseline==CLOSEST[t])&(b.seed.astype(str)=='MEAN')].iloc[0];seed=b[(b.baseline==CLOSEST[t])&(b.seed.astype(str)!='MEAN')];oldcore=a[(a.baseline==CLOSEST[t])&(a.seed.astype(str)=='MEAN')].iloc[0];g=float(core.gain_percent);tags=[]
 if g<0:
  tag='ROBUST_NEGATIVE' if (seed.gain_percent<0).all() and core.ci_high<0 and oldcore.gain_percent<0 else 'NEGATIVE_UNCERTAIN'
 elif g>0:
  qualifies=(seed.gain_percent>0).all() and core.ci_low>0
  # No shortlist if a prespecified simple control is better in the full primary comparison.
  basics=b[(b.seed.astype(str)=='MEAN')];simple_ok=bool((basics.gain_percent>0).all())
  tag='REOPEN_CANDIDATE' if qualifies and simple_ok else 'POSITIVE_UNCERTAIN'
 else:tag='POSITIVE_UNCERTAIN'
 tags.append(tag)
 if g*float(oldcore.gain_percent)<0:tags.append('SAMPLING_SENSITIVE')
 summary=pd.read_csv(OUT/t/'scores_summary.csv');summary=summary[(summary.policy=='selected')&(summary.condition=='PRIMARY')]
 if t=='N03':
  p=summary.pivot(index='seed',columns='arm',values='score')
  if ((p.C2<p.C3)&(p.C2<p.C0)).all():tags.append('SIMPLE_METHOD_SUFFICIENT')
 selected=read(OUT/t/'selections.json')['selections'];prop=[s for s in selected if s['arm']==PROPOSED[t]]
 if all(s['selected']['step']==0 for s in prop):tags.append('NO_SELECTED_ADAPTATION')
 st=read(OUT/t/'STATUS.json');st.update(EVIDENCE=tag,tags=tags,closest_baseline=CLOSEST[t],old_closest_gain=float(oldcore.gain_percent),repaired_closest_gain=g,ci=[float(core.ci_low),float(core.ci_high)]);save(OUT/t/'STATUS.json',st)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,4));q=joined[joined.baseline==CLOSEST[t]];x=np.arange(len(q));ax.bar(x-.18,q.gain_percent_old,.36,label='old');ax.bar(x+.18,q.gain_percent_repaired,.36,label='repaired');ax.set_xticks(x,q.seed);ax.axhline(0,color='black',lw=.7);ax.set_ylabel('gain (%)');ax.set_title(t+' '+PROPOSED[t]+' vs '+CLOSEST[t]);ax.legend();fig.tight_layout();fig.savefig(OUT/t/'old_vs_repaired_gain.png',dpi=140);plt.close(fig)
 ss=pd.read_csv(OUT/t/'scores_summary.csv');ss=ss[(ss.policy=='selected')&(ss.condition!='PRIMARY')];fig,ax=plt.subplots(figsize=(10,4))
 for (a,s),v in ss.groupby(['arm','seed']):ax.plot(v.condition,v.score,'-o',label=f'{a}/{s}',markersize=3)
 ax.set_ylabel('normalized RMSE');ax.set_title(t+' all conditions / repeat seeds');ax.tick_params(axis='x',rotation=60);ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(OUT/t/'condition_tradeoffs.png',dpi=140);plt.close(fig)
 return joined

def report_all():
 rows=[];budget=[]
 for t in ORDER:
  out=OUT/t
  if not (out/'STATUS.json').exists():continue
  st=read(out/'STATUS.json');fits=read(out/'fits.json') if (out/'fits.json').exists() else [];complete=st['EXECUTION']=='COMPLETE'
  if complete:joined=comparisons(t);st=read(out/'STATUS.json')
  text=f'# {t} {NAMES[t]} 표본 설계 교정\n\n실행: **{st["EXECUTION"]}**. 근거: {st["EVIDENCE"]}. 새 본학습 완료 {sum(f["status"]=="COMPLETE" for f in fits)}경로 / 실제 본업데이트 {sum(f["updates"] for f in fits)}회.\n\n'
  text+='## 날짜·정보·방법\n\n과학적 변경은 원점 선정 하나다. 데이터·split·네 채널·rank8·FP32·loss·LR grid·seed·512updates·체크포인트·metric·핵심 대조를 유지했다. [분산 검사](origin_diversity.json), [old/new 그림](origin_distribution.png), [방법 명세](PROTOCOL.json). E는 재사용 개발 평가다.\n\n'
  if not complete:
   text+='원점 검사 또는 실행이 완료되지 않은 상태이므로 repaired 예측 효과를 판정하지 않는다. 사유: '+str(st.get('reason',st.get('phase','준비')))+'.\n\n'
   if st['EXECUTION']=='BLOCKED_DIVERSITY':
    text+='TRAIN64 distinct days 자체는 충족하지만 경계 날짜의 circular phase 선택으로 phase 수 최소1/최대4, 차이3이다. 계약의 한도2를 넘으므로 GPU smoke·본학습·교차 평가를 실행하지 않았다. 기존 부정적 결과를 ROBUST_NEGATIVE로 승격할 수 없다.\n'
    for name in ['fit_manifest.csv','optimizer_log.csv','raw_scores.csv','contrasts.csv','old_vs_repaired_contrasts.csv','uncertainty.csv','resources.csv']:
     if not (out/name).exists():csvwrite(out/name,[])
    for name in ['LR_selection.json','selections.json','predictions_manifest.json','cross_evaluation_manifest.json','verification.json']:
     if not (out/name).exists():save(out/name,dict(status='BLOCKED_DIVERSITY',reason=st['reason'],neural_fits=0,optimizer_updates=0,performance_measured=False))
  else:
   text+='## 원점수·직접 대조\n\n'+table(pd.read_csv(out/'scores_summary.csv').query("policy == 'selected' and condition == 'PRIMARY'"))+'\n\n'+table(joined)+'\n\n![old/repaired gain](old_vs_repaired_gain.png)\n\n![모든 조건](condition_tradeoffs.png)\n\n판정: '+', '.join(st['tags'])+'. CI가0을 포함하면 동등성이 아닌 불확실성이다. 별도의 목적 점수를 합산하지 않는다.\n\n'
   if (out/'cross_raw_scores.csv').exists():text+='## 무학습 교차 평가\n\n'+table(pd.read_csv(out/'cross_raw_scores.csv'))+'\n\n'+table(pd.read_csv(out/'cross_contrasts.csv'))+'\n\nOLD_MODEL_NEW_E는 평가 원점 변화, NEW_MODEL_NEW_E 대비는 새 TRAIN/V 적응과 선택 효과를 함께 반영한다. NEW_MODEL_OLD_E는 옛 좁은 평가에서의 참고다. 완전한 인과 분해가 아니다. 선택 seed E를 주 결과에 넣지 않았다. 추가 optimizer0회.\n\n'
   text+='## 실측 자원·검산·한계\n\n'+table(pd.read_csv(out/'resources.csv'))+'\n\n[기본 지표·선택 봉인 검산](verification.json), [교차 평가 manifest](cross_evaluation_manifest.json). 모델·head·buffer 보존, finite gradient, 실제 optimizer update, 복원은 smoke와 각fit 기록에 있다. 두 반복 seed와 한 source의 조건부 개발 근거이며 정식 선행 비교·독립 source 검증은 남았다.\n'
  if (out/'INTERPRETATION.md').exists():text+='\n## 결과 해석\n\n'+(out/'INTERPRETATION.md').read_text()+'\n'
  (out/'REPORT.md').write_text(text);rows.append(dict(track=t,execution=st['EXECUTION'],fits=len(fits),evidence=st['EVIDENCE'],tags=','.join(st.get('tags',[]))));budget.append(dict(track=t,main_fits=len(fits),main_updates=sum(f['updates'] for f in fits),smoke_updates=sum(r.get('updates',0) for r in read(out/'smoke.json')) if (out/'smoke.json').exists() else 0))
 csvwrite(OUT/'BUDGET_LEDGER.csv',budget)
 text='# 날짜 다양성 교정 — 전체 보고서\n\n단일 [실행 계약](MASTER_PROTOCOL.md), [저장소 감사](REPOSITORY_AUDIT.md), [GPU 전 원점 감사](ORIGIN_REPAIR_AUDIT.md).\n\n'+table(pd.DataFrame(rows))+'\n\n'+table(pd.DataFrame(budget))+'\n\n'
 text+='본학습 상한96fits/49,152updates, smoke 상한48updates, 총49,200이다. 다양성 미충족 트랙의 예산은 다른 학습에 사용하지 않는다. 실행 완료를 논문 성공으로 해석하지 않는다.\n\n## UNAFFECTED_REFERENCE\n\nR05/N06는 기존 저장 예측 CPU 분석이고 이번 selector와 무관하다. R05의 알려진 혼합 정책은 재사용 E 및 두 모델 비용을 포함한다. N06의 AR1 합계 CRPS 개선은 알려진 단순 결합이며 새 PEFT 근거가 아니다. R09는 기존 E64가64 index-days에 분산되어 대상 밖이다. NULL 대 MIXED의 부정적 결과와 FINE_ONLY의 고유 상세 정답16 대 MIXED64의 정보 차이를 유지한다. 세 결과를 이번 새 증거나 우승 점수로 합산하지 않았다.\n\n'
 text+='## 후보별 보고서\n\n'+'\n'.join(f'- [{t}]({t}/REPORT.md)' for t in ORDER)+'\n\n[최종 결정](FINAL_DECISION.md). 큰 원자료·예측 배열·가중치는 로컬 ignored cache에 보존하며 GitHub만으로 수치 재생이 된다고 주장하지 않는다.\n'
 if (OUT/'MASTER_INTERPRETATION.md').exists():text+='\n'+(OUT/'MASTER_INTERPRETATION.md').read_text()
 (OUT/'MASTER_REPORT.md').write_text(text)
 if not (OUT/'FINAL_DECISION.md').exists():(OUT/'FINAL_DECISION.md').write_text('# 최종 판단 대기\n\n검사를 통과한 비교와 검산이 끝나기 전에는 후속 후보를 확정하지 않는다. 후속 학습 자동 실행 없음.\n')
