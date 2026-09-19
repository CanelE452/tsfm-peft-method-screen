"""Use completed frozen results for paper claims and a primary-family forest plot."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'results/learned_gate_comparison_20260919'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(f):
 return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))
def main():
 audit=json.loads((OUT/'AUDIT.json').read_text());status=json.loads((OUT/'status.json').read_text());assert status['execution']=='COMPLETE_VERIFIED' and audit['status']=='VERIFIED' and audit['prespecified_component_gate_met']
 e=pd.read_csv(OUT/'EFFECTS.csv');raw=pd.read_csv(OUT/'RAW_SCORES.csv');primary=e[e.primary_family].copy();assert len(primary)==4 and (primary.bonf4_low>0).all()
 assert all(all(v>0 for v in json.loads(s).values()) for s in primary.seed_gains)
 keep=e[(e.proposed=='MAG_ONLY')&e.panel.isin(['electricity_transfer','neso_2026_jul_aug'])&(e.stage=='selected')&(e.kind=='standard')]
 protection=keep[keep.condition.isin(['REFERENCE','FAULT'])&keep.baseline.isin(['B0','PLAIN'])];assert len(protection)==8 and (protection.gain_pct>=-1).all()
 counter=protection[protection.ci_low< -1];assert len(counter)==2
 fixed=e[(e.proposed=='MAG_ONLY')&e.panel.isin(['electricity_transfer','neso_2026_jul_aug'])&(e.stage=='fixed1024')&(e.kind=='standard')&(e.condition=='SHIFT8')&e.baseline.isin(['TOKEN_GATE','TOKEN_GATE_ENTROPY'])];assert len(fixed)==4 and (fixed.gain_pct>0).all()
 cols=['panel','baseline','proposed_nmae','baseline_nmae','gain_pct','bonf4_low','bonf4_high','seed_gains'];primary[cols].to_csv(HERE/'PRIMARY_EVIDENCE.csv',index=False)
 protection[['panel','condition','baseline','gain_pct','ci_low','ci_high']].to_csv(HERE/'PROTECTION_EVIDENCE.csv',index=False)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,ax=plt.subplots(figsize=(11.5,5.0));colors=['#28758b','#ab6a36'];ys=np.arange(4)[::-1]
 for y,r in zip(ys,primary.itertuples()):
  ax.errorbar(r.gain_pct,y,xerr=[[r.gain_pct-r.bonf4_low],[r.bonf4_high-r.gain_pct]],fmt='s',color='#26374a',capsize=5,ms=7,label='Gain from seed-mean nMAE; family-4 CI' if y==3 else None)
  for i,(seed,gain) in enumerate(json.loads(r.seed_gains).items()):ax.scatter(gain,y+(-.075 if i==0 else .075),marker='o' if i==0 else '^',s=36,color=colors[i],label=f'Seed {seed}' if y==3 else None,zorder=4)
 labels=[('Electricity development transfer' if r.panel=='electricity_transfer' else 'NESO Jul–Aug temporal check')+'\nvs '+('learned gate + entropy' if r.baseline.endswith('ENTROPY') else 'learned gate') for r in primary.itertuples()]
 ax.set_yticks(ys,labels);ax.axvline(0,color='#888888',ls='--',lw=1);ax.set_xlim(-.3,6.1);ax.set_ylim(-.45,3.8);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Relative SHIFT8 nMAE decrease (%, positive favors MAG)');ax.set_title('Fixed MAG versus learned gates: the four prespecified comparisons',pad=12)
 ax.legend(loc='upper right',fontsize=8,frameon=False);fig.text(.02,.015,'Intervals: 2,000 index-week block resamples, conditional on the two fitted seeds and fixed channels; not independent-source proof.',fontsize=8,color='#555555');fig.tight_layout(rect=[0,.05,1,1])
 for ext in ['pdf','png','svg']:fig.savefig(HERE/f'primary_effects.{ext}',dpi=180)
 plt.close(fig);p=HERE/'primary_effects.svg';p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
 summary=keep[(keep.condition=='SHIFT8')&keep.baseline.isin(['B0','PLAIN'])][['panel','baseline','proposed_nmae','baseline_nmae','gain_pct']]
 report='''# 방법론 주장과 완료된 실험 근거

**현재 판정: 고정 MAG의 제한된 추가 가치 근거가 확인됐다.** 승인된16fits/16384main+8smoke,192prediction views의 저장·채점·독립 검산을 완료했다. 사전에 정한 네 주 비교와 평균 손해 한도를 충족했다. 이는 방법론 논문을 뒷받침하는 한 근거의 완료이며, 사용자가 요청한 Time-PEFT와 같은 방법론 논문의 전체 완성·신규성·채택 가능성을 입증한 상태는 아니다.

## 지금 쓸 수 있는 핵심 주장

> 학습된 B0와 원래 관측을 유지하면서, 전체 관측창의 robust 진폭 통계로 추가 patch 잔차를 제한하는 고정 MAG 설계는 이 실험의 큰 합성 지속 변화(SHIFT8)에서 일반 잔차와 두 학습형 embedding gate보다 낮은 예측 오차를 보였다. 이득은 기존 전력 계열 전이와 사전에 고정한 NESO 후반 기간에서 두 공통 seed에 걸쳐 확인됐으나, 원자료·오류 조건의 소폭 손해와 긴 변화 형태·ETTm1의 한계가 남았다.

이 문장은 구체적인 적응 방법의 비교 결과를 말한다. ‘지속성을 알아냈기 때문에 좋아졌다’거나 ‘학습하지 않는 gate가 일반적으로 우월하다’는 주장으로 바꾸지 않는다. MAG를 후보로 채택한 결정은 기존 E 결과를 본 뒤 이루어졌음을 공개한다. 새로운 평가 기간의 긍정 결과로 과거의 사후 선택을 지우지 않는다.

## 주 비교와 단순 대안

'''+table(primary[cols])+'''

네 보정구간의 하한이 모두0보다 크고, 개별두seed의 이득도모두양수다. 다만 전력전이의entropy대조 대비seed81551 이득은0.123%로작다. 구간은날짜index의7일block을resample한것이며, 훈련seed모집단의불확실성을충분히추정한구간이아니다. 2개seed만으로seed일반화가확립됐다고하지않는다.

![주 비교의 보정구간과 seed별 이득](primary_effects.png)

**단순 잔차와 B0 대비 SHIFT8:**

'''+table(summary)+'''

selected뿐아니라fixed1024에서도네학습gate대조의MAG이득은양수였다. 따라서이자료에서관찰한이득을checkpoint선택하나로만설명하기는어렵다. 그렇다고전체원인중진폭feature·초기gate·학습궤적의기여를분리한것은아니다. [공정성 감사](../../../research/learned_gate_comparability_20260919/COMPARABILITY_KO.md)를따른다.

## 보호 한도: 평균 기준과 비열등성 보장의 차이

'''+table(protection[['panel','condition','baseline','gain_pct','ci_low','ci_high']])+'''

사전1%한도는두seed평균점수의악화에적용됐다. 평균은모두한도안이지만, 새NESO에서B0대비REFERENCE와FAULT의일반95%구간하한은각각약−1.036%와−1.056%다. 따라서이결과로‘95%신뢰에서손해≤1%’라는비열등성을입증했다고하지않는다. 이구간은주family4와다른보조기술구간이며여기서새합격기준을만들지않는다.

## 남겨야 할 부정 결과

- ETTm1의MAG selected checkpoint는두seed에서초기0이며B0와같다. 학습이실패한것이아니라검증선택이추가적응을채택하지않은것이다. SHIFT8에서PLAIN보다약0.95%나쁘다.
- 전력전이의긴STEP12_D63에서MAG는B0보다약3.27%,PLAIN보다약2.65%나쁘다. 큰진폭이라는이유만으로모든지속변화에서유리하다고할수없다.
- 전력전이SHIFT4의PLAIN대비추가가치는거의0이고, 새NESO SHIFT4/SHIFT_POINT는seed방향이혼재한다. SHIFT8의강한이득을모든변화형태로확대하지않는다.
- C3의지속성규칙은여전히MAG대비독립적인추가가치가확립되지않았다. 이번긍정결과를C3지속성기전성공으로옮기지않는다.

## 논문 요건별 현재 증거

| 요건 | 현재 근거 | 판단 |
| --- | --- | --- |
| 실행 가능한 작은 추가 적응 방법 | 고정수식·실제Chronos연결·B0보존·실제update·복원 | 구현 완료 |
| 강한기존B0 위의 추가가치 | PLAIN대MAG 및학습gate직접대조,두seed·두주패널 | SHIFT8에서 지지 |
| 단순방법으로충분한가 | PLAIN,학습gate,기존POS_ONLY/δ보고서 | 현재대조들보다SHIFT8에추가이득;모든형태에우위아님 |
| 평가 재사용 통제 | 과거개발E와새NESO55일분리,모든선택후예측저장·채점 | 시간전이근거,독립source아님 |
| 효율 | MAG8712 vs학습gate9225추가params,동일updates | 파라미터차이확인;순수속도우위미입증 |
| 기존방법과의차이·신규성 | 고정robust-statistic gate와입력보존의구체적조합 | gating자체는알려짐;충분한방법론신규성미확정 |
| 정식선행비교 | δ XY와GateRA원리의통제이식,기존대조 | GateRA/Time-PEFT전체동일설정비교아님 |
| 현실적범위 | 합성fault/shift,한backbone,두학습원천 | 실제사건·독립source·더넓은일반화미검증 |

따라서 방법 절과조건부기여를작성할실험근거는강해졌지만, **전체방법론목표달성으로표시하지않는다.** 공식선행전체비교·더넓은독립검증·신규성검토가남아있다는사실과, 이미확인된좁은양성이득을함께보존한다. 이표의미완료항목을이유로이번실험을실행오류/전면성능실패로바꾸지않는다.

이번승인단위의미실행학습0. 새구조·threshold·LR·seed·dataset·후속학습은자동시작하지않았다. 실행계약을다시열지않고여기서종료한다. 아래원자료와코드로결론을검토할수있다.

[전체 REPORT](../../../results/learned_gate_comparison_20260919/REPORT.md) · [최종 결정](../../../results/learned_gate_comparison_20260919/FINAL_DECISION.md) · [독립 검산](../../../results/learned_gate_comparison_20260919/AUDIT.json) · [모든 seed 원점수](../../../results/learned_gate_comparison_20260919/RAW_SCORES.csv) · [자원 측정 범위](../../../research/learned_gate_comparability_20260919/RESOURCE_SCOPE_KO.md)
'''
 (HERE/'PAPER_CLAIM_EVIDENCE_KO.md').write_text(report)
 files=[OUT/n for n in ['AUDIT.json','RAW_SCORES.csv','EFFECTS.csv','MODEL_SELECTION.json','EVALUATION_SEAL.json','ALL_PREDICTIONS_SAVED.json']]+[Path(__file__)]
 generated=[HERE/n for n in ['PAPER_CLAIM_EVIDENCE_KO.md','PRIMARY_EVIDENCE.csv','PROTECTION_EVIDENCE.csv','primary_effects.png','primary_effects.svg','primary_effects.pdf']]
 record=dict(status='VERIFIED_PAPER_EVIDENCE_NOT_GOAL_COMPLETION',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},artifact_hashes={p.name:sha(p) for p in generated},new_optimizer_updates=0,new_forecast_inferences=0,prespecified_rule_changed=False,goal_achieved=False,primary_positive_rows=4,protection_means_within_one_percent=8,protection_intervals_crossing_minus_one_percent=len(counter))
 (HERE/'PAPER_EVIDENCE_AUDIT.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
