"""Report all registered comparisons; never select a successor model."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def report():
 a=read(OUT/'INDEPENDENT_AUDIT.json');assert a['status']=='VERIFIED';f=pd.read_csv(OUT/'FACTORIAL_EFFECTS.csv');c=pd.read_csv(OUT/'CELL_EFFECTS.csv');raw=pd.read_csv(OUT/'RAW_SCORES.csv');m=read(OUT/'MODEL_SELECTION.json')
 main=f[(f.panel=='electricity_transfer')&(f.condition=='SHIFT8')&(f.stage=='fixed1024')];assert len(main)==7
 top=main.loc[main.effect_nmae.abs().idxmax()];single=main[~main.factor.str.contains(':')];topone=single.loc[single.effect_nmae.abs().idxmax()]
 rows=[[r.factor,f'{r.effect_nmae:+.8f}',f'{r.effect_pct_common_C3:+.4f}',f'[{r.bonferroni7_low:+.8f}, {r.bonferroni7_high:+.8f}]',f'{r.cell_variance_share_pct:.2f}'] for r in main.itertuples()]
 cell=c[(c.panel=='electricity_transfer')&(c.condition=='SHIFT8')&(c.stage=='fixed1024')]
 celltable=table(['B0','初期値','순서','C3 nMAE','MAG nMAE','MAG 이득(%)'],[[r.b,r.i,r.o,f'{r.C3:.8f}',f'{r.MAG:.8f}',f'{r.MAG_gain_pct:+.4f}'] for r in cell.itertuples()]).replace('初期値','초기값')
 triples=[]
 for panel in ['electricity','electricity_transfer','ettm1']:
  for stage in ['fixed1024','selected']:
   for cond in ['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT']:
    x=c[(c.panel==panel)&(c.stage==stage)&(c.condition==cond)];a0=x.C3.mean();d0=x.MAG.mean();triples.append([panel,stage,cond,f'{a0:.7f}',f'{d0:.7f}',f'{100*(a0-d0)/a0:+.4f}'])
 fits=[read(p) for p in (OUT/'fits').glob('*/receipt.json')];new=[r for r in fits if not r.get('reused',False)];wall=read(OUT/'factorial_wall.json');resources=dict(new_optimizer_seconds=sum(r['optimizer_seconds'] for r in new),new_validation_seconds=sum(r['validation_seconds'] for r in new),new_fit_peak_allocated=max(r['peak_allocated'] for r in new),new_inference_seconds=sum(r['inference_seconds'] for r in read(OUT/'PREDICTIONS.json').values()),wall_seconds=wall['seconds'],GPU_min_free_mib=a['GPU_min_free_mib']);save(OUT/'RESOURCE_SUMMARY.json',resources)
 folders=OUT/'figures';folders.mkdir(exist_ok=True)
 fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
 for ax,panel in zip(axes,['electricity','electricity_transfer','ettm1']):
  x=f[(f.panel==panel)&(f.condition=='SHIFT8')&(f.stage=='fixed1024')].copy();x=x.set_index('factor').reindex(main.factor);scale=100/x.common_C3
  y=np.arange(len(x));ax.hlines(y,x.bonferroni7_low*scale,x.bonferroni7_high*scale,color='#346889');ax.scatter(x.effect_pct_common_C3,y,s=24,color='#346889');ax.axvline(0,color='gray',ls='--');ax.set_yticks(y,x.index);ax.set_title(panel);ax.set_xlabel('Contrast / common C3 error (%)')
 fig.suptitle('Fixed 1024 updates: changing B0, init and order independently\nPositive: high levels increase the C3 minus MAG error difference; conditional Bonferroni-7 intervals')
 for suffix in ['png','pdf','svg']:fig.savefig(folders/f'01_factorial.{suffix}',dpi=180)
 plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for ax,stage in zip(axes,['fixed1024','selected']):
  x=c[(c.panel=='electricity_transfer')&(c.condition=='SHIFT8')&(c.stage==stage)];im=x.pivot(index='b',columns=['i','o'],values='MAG_gain_pct');artist=ax.imshow(im.to_numpy(),aspect='auto',cmap='coolwarm');ax.set_yticks(range(2),im.index);ax.set_xticks(range(4),[f'I{i-81550}/O{o-81550}' for i,o in im.columns]);ax.set_title(stage);ax.set_ylabel('B0 seed');fig.colorbar(artist,ax=ax,label='MAG gain vs C3 (%)')
  for row in range(2):
   for col in range(4):ax.text(col,row,f'{im.iloc[row,col]:+.3f}',ha='center',va='center')
 fig.suptitle('Electricity transfer SHIFT8: all eight registered cells')
 for suffix in ['png','pdf','svg']:fig.savefig(folders/f'02_cells.{suffix}',dpi=180)
 plt.close(fig)
 for p in folders.glob('*.svg'):p.write_text('\n'.join(l.rstrip() for l in p.read_text().splitlines())+'\n')
 text=f'''# B0·초기값·학습 순서를 분리한 C3/MAG 통제 실험

**실행 완료:** 32개 경로 중 기존8개를 검증 후 재사용하고 새24개를 모두1024 updates 학습했다. 본학습24,576+smoke12=24,588 updates. 선택64개 모델 view, 평가192개 view 중 신규{a['new_prediction_views']}·재사용{a['reused_prediction_views']}개. 누락된 예정 경로는0개다. 새 후보·LR 검색·후속학습은 없다.

## 질문과 해석 범위

기존 seed 효과에 묶여 있던 B0 가중치, 추가 어댑터 초기값, batch 순서를81551/81552 두 수준에서 교차했다. 동일 조합의 C3/MAG는 다른 조건이 같고 gate만 다르다. 주 분석은 fixed1024, 보조 분석은 기존 V 규칙으로 선택한 checkpoint다. 모든 선택과 전체 예측을 저장한 뒤 E를 채점했다.

이는 기존 결과를 본 뒤 수행한 두 수준의 통제 진단이다.81553을 포함한 과거 결과는 보존했고 원 논문에서 제외하지 않는다. 이번 두 수준의 결과를 초기화 전체 모집단 또는 새로운 독립 시험으로 일반화하지 않는다. B0 수준의 변화는 학습된 checkpoint 전체의 교체이며 B0 내부의 어떤 표현이 원인인지는 별도 문제다.

## 주 분석: 어느 요인의 차등 효과가 컸나

전력 전이 SHIFT8의 d=nMAE(C3)−nMAE(MAG)에서 절대 점추정이 가장 큰 전체 성분은 **{top.factor}**, 단일 요인 주효과 중에는 **{topone.factor}**다. 이는 관찰한 두 수준과 나머지 요인의 동일 가중 평균에 한정된 순위다. 계수의 크기 순위에 대한 별도 통계 검정을 하지 않았으므로 ‘가장 크다는 것이 확정’이라고 쓰지 않는다. 교호 성분이 크면 단일 요인만으로 설명하지 않는다.

contrast는2×mean(d×각 요인의±1 부호곱)이다. 단일 요인에서는 다른 두 요인을 평균한 high−low 차이와 같다. 양수는 해당 요인 수준이 높아지면 MAG 방향의 차이가 커짐을 뜻한다. 이원·삼원 contrast는 동일 척도의 factorial contrast이며 원자료의 단순 두 셀 차이와 다르다. %는8셀의 공통 C3 오차를 분모로 한다. 셀 간 분산 비중은 이8셀의 설명적 분해이며 인과 확률이나 초기화 모집단 분산 비중이 아니다.

'''+table(['요인','d contrast(nMAE)','공통 분모 %','조건부 Bonferroni7 구간','셀 분산 비중%'],rows)+f'''

{celltable}

![세 패널의 원인 분리 contrast](figures/01_factorial.png)

![고정종료와 검증선택의8조합](figures/02_cells.png)

## 원점수와 조건별 손해

아래는 각 조합을 동일 가중한 평균이다. 공식 원래3seed 평균을 대체하는 점수가 아니며 연구 성공 기준으로 새로 선택한 점수도 아니다. 개별8조합·9형태·모든 fault 상태는 CELL_EFFECTS.csv와 RAW_SCORES.csv에 보존했다.

'''+table(['패널','checkpoint','조건','C3','MAG','MAG 이득%'],triples)+f'''

## 검증과 비용

실제 모델의 초기 B0 동일성, residual-off 복원, 학습 중 동결 가중치·buffer 유지, 추가파라미터 갱신, 저장/복원 검사를 통과했다. 독립 감사는 동일 B0 수준의 동결 hash와 동일 초기화 수준의 실제 tensor를 비교했고,840개 contrast를 별도 marginal 합산으로 재계산했다.960개 셀 집계와 기존 selected 대각선120개를 재현했다. 원점별 nMAE와 pinball을 scalar loop로 검산했다. 날짜 bootstrap은 고정 모델에 조건부인2000회7일 block 재추출이고7contrast 내 다중비교 구간을 병기했다. 서로 다른 자료·상태 전체에 대한 familywise 보장을 주장하지 않는다.

새 optimizer 시간은{resources['new_optimizer_seconds']/60:.2f}분, 검증{resources['new_validation_seconds']/60:.2f}분, 새 추론{resources['new_inference_seconds']/60:.2f}분, 전체 runner wall{resources['wall_seconds']/60:.2f}분이다. 새 학습 peak allocated는{resources['new_fit_peak_allocated']/2**20:.1f}MiB, GPU 최소 여유는{a['GPU_min_free_mib']}MiB였다. RustDesk는 기존 승인 예외이고 외부 학습 감지 표본은{a['GPU_external_training_samples']}개다. 원 모델·LoRA 비용을 제외한 추가 모듈의8,712파라미터만으로 전체 배포 비용을 주장하지 않는다.

## 논문에 남길 주장과 한계

이 통제는 앞선 ‘한 seed가 평균 차이에85.4% 기여했다’는 관찰보다 원인을 더 좁힌다. 하지만 직접 통제한 것은 B0·초기값·순서의 선택한 두 수준이다. 내부 gradient나 표현의 작동 원리 전체를 입증한 것은 아니다. 실제 센서 사건 레이블은 없고, E는 반복 사용한 개발 기간이다. 기존 C3의 좁은 양성 결과와 단순 MAG의 우위·손해를 함께 보존한다. 원인 규명이 새 지속성 구성요소의 성능 우위 또는 신규성을 자동으로 만들지는 않는다.

FINAL_DECISION.md에 실제 방향·상호작용·checkpoint 민감도를 해석한다. 새 모델 선택이나 자동 후속학습은 실행하지 않는다. 원 가중치·TRAIN·전체 예측은 로컬 캐시이며 GitHub에는 코드·봉인·집계·검산만 공개한다.
'''
 (OUT/'REPORT.md').write_text(text);save(OUT/'REPORT_FACTS.json',dict(largest_absolute_component=str(top.factor),largest_absolute_main_effect=str(topone.factor),scope='posthoc two levels; fixed1024 electricity_transfer SHIFT8',new_candidates=0,automatic_followup=False))
 print('REPORT_WRITTEN',flush=True)
if __name__=='__main__':report()
