"""Report fixed results; no fitting, selection or posthoc acceptance thresholds."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/c3_weakness_figures')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *
PANELS=['electricity','electricity_transfer','ettm1']
LABELS={'electricity':'Electricity (4)','electricity_transfer':'Electricity transfer (16)','ettm1':'ETTm1'}

def table(f,columns=None):
    if columns:f=f[columns]
    h=['| '+' | '.join(map(str,f.columns))+' |','| '+' | '.join(['---']*len(f.columns))+' |']
    for r in f.itertuples(index=False,name=None):h.append('| '+' | '.join('—' if pd.isna(v) else f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in r)+' |')
    return '\n'.join(h)

def build_report():
    verify=read(OUT/'VERIFICATION.json');assert verify['status']=='VERIFIED';check_seal();e=pd.read_csv(OUT/'MATCHED_CONTRASTS.csv');raw=pd.read_csv(OUT/'RAW_SCORES.csv');seeds=pd.read_csv(OUT/'SEED_EFFECTS.csv');models=pd.DataFrame(read(OUT/'MODEL_SELECTION.json'));cost=read(OUT/'COST.json');dynamics=read(OUT/'PARAMETER_DYNAMICS.json')
    main=e[(e.kind=='standard')&(e.condition=='SHIFT8')&(e.new=='C3')&e.baseline.isin(ARMS)]
    cols=['panel','baseline','new_nmae','baseline_nmae','gain_pct','ci_type','ci_low_pct','ci_high_pct','bonferroni3_low_pct','bonferroni3_high_pct'];main.to_csv(OUT/'PRIMARY_RESULTS.csv',index=False)
    states=['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT'];trade=e[(e.kind=='standard')&e.condition.isin(states)&(e.new=='C3')&e.baseline.isin(ARMS)&(e.ci_type=='time')]
    rawmain=raw[(raw.kind=='standard')&raw.condition.isin(states)&raw.arm.isin(['C0','C2','C3','M_RECENCY','F0']+ARMS)].groupby(['panel','arm','condition']).nmae.mean().unstack().reindex(columns=states).reset_index()
    primaryseeds=seeds[(seeds.kind=='standard')&(seeds.condition=='SHIFT8')&(seeds.new=='C3')&seeds.baseline.isin(ARMS)]
    ft=OUT/'figures';ft.mkdir(exist_ok=True);plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def finish(name):
        plt.tight_layout()
        for extn in ['png','pdf','svg']:plt.savefig(ft/(name+'.'+extn),dpi=180,bbox_inches='tight')
        p=ft/(name+'.svg');p.write_text('\n'.join(v.rstrip() for v in p.read_text().splitlines())+'\n');plt.close()
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,panel in zip(axes,PANELS):
        f=main[(main.panel==panel)&(main.ci_type=='time')].set_index('baseline').loc[ARMS]
        for i,(arm,r) in enumerate(f.iterrows()):
            ax.hlines(i,r.bonferroni3_low_pct,r.bonferroni3_high_pct,color='lightgrey',lw=4);ax.hlines(i,r.ci_low_pct,r.ci_high_pct,color='black');ax.scatter(r.gain_pct,i,color='tab:blue');s=primaryseeds[(primaryseeds.panel==panel)&(primaryseeds.baseline==arm)];ax.scatter(s.gain_pct,np.full(3,i+.12),color='tab:orange',marker='x')
        ax.axvline(0,color='grey',lw=.7);ax.set_yticks(range(3),ARMS);ax.set_ylim(2.4,-.4);ax.set_title(LABELS[panel]);ax.set_xlabel('C3 nMAE reduction vs control (%)')
    fig.suptitle('SHIFT8: fixed three controls / 95% and Bonferroni3 intervals / each seed');finish('01_primary_controls')
    fig,axes=plt.subplots(3,3,figsize=(13,10))
    for row,panel in enumerate(PANELS):
        for col,arm in enumerate(ARMS):
            ax=axes[row,col];f=trade[(trade.panel==panel)&(trade.baseline==arm)].set_index('condition').loc[states];ax.hlines(np.arange(5),f.ci_low_pct,f.ci_high_pct,color='black');ax.scatter(f.gain_pct,np.arange(5));ax.axvline(0,color='grey',lw=.7);ax.set_yticks(range(5),states);ax.invert_yaxis();ax.set_title(LABELS[panel]+' / vs '+arm);ax.set_xlabel('C3 nMAE reduction (%)')
    fig.suptitle('All main states retained: positive favors C3; negative favors control');finish('02_tradeoffs')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,source in zip(axes,SOURCES):
        for d in dynamics:
            if d['source']==source and d['arm']=='POS_ONLY' and d['seed'] in [81551,81552,81553] and d['lr']==read(OUT/'LR_SELECTION.json')[source]['POS_ONLY']['lr']:ax.plot(d['selected_gate'],label=f"seed {d['seed']} / step {d['selected_step']}")
        ax.axhline(.5,color='grey',ls='--',label='initial');ax.set_ylim(0,1);ax.set_xlabel('patch position: old to recent');ax.set_ylabel('selected static gate');ax.set_title(source);ax.legend(fontsize=7)
    fig.suptitle('POS_ONLY: input-independent positions, not oracle event locations');finish('03_selected_position_gates')
    fig,axes=plt.subplots(1,3,figsize=(13,6));sh=e[(e.kind=='shape')&(e.new=='C3')&e.baseline.isin(ARMS)&e.ci_type.eq('time')];names=list(read(old.CACHE/'shapes/electricity/manifest.json')['states']);lim=max(1,float(sh.gain_pct.abs().max()))
    for ax,panel in zip(axes,PANELS):
        z=sh[sh.panel==panel].pivot(index='condition',columns='baseline',values='gain_pct').reindex(index=names,columns=ARMS);im=ax.imshow(z,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto');ax.set_xticks(range(3),ARMS,rotation=45,ha='right');ax.set_yticks(range(9),names);ax.set_title(LABELS[panel])
        for i in range(9):
            for j in range(3):ax.text(j,i,f'{z.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(z.iloc[i,j])>.6*lim else 'black')
    fig.suptitle('All registered shapes: C3 reduction vs control (%), reused development E');finish('04_all_shapes')
    # Resource tables preserve per-seed raw timing and parameter counts; old B0 cost remains separate.
    resource=[]
    for k,r in read(OUT/'RESOURCES.json').items():
        p,kind,arm,seed=k.split('__')
        if kind=='standard':resource.append(dict(panel=p,arm=arm,seed=int(seed),parameters={'POS_ONLY':8744,'MAG_ONLY':8712,'OUTPUT_CONTEXT':4673}[arm],retained_adaptation_parameters=294912+{'POS_ONLY':8744,'MAG_ONLY':8712,'OUTPUT_CONTEXT':4673}[arm],median_seconds=r['median_seconds'],range_seconds=r['range_seconds'],raw_seconds=json.dumps(r['raw_seconds']),inference_peak_allocated_mib=r['peak_allocated']/2**20,profile_series=128,device='RTX3080',precision='FP32',source='new matched-size profile'))
    rr=pd.DataFrame(resource);rr.to_csv(OUT/'RESOURCE_REPORT.csv',index=False);rt=rr.groupby(['panel','arm'])[['parameters','retained_adaptation_parameters','median_seconds','inference_peak_allocated_mib']].mean().reset_index()
    old_resources=pd.read_csv(PRIOR/'RESOURCE_REPORT.csv')
    old_profile=old_resources[(old_resources.kind=='inference_profile')&old_resources.panel.isin(PANELS)&old_resources.arm.isin(['C0','C2','C3','M_RECENCY'])].copy()
    old_profile['source']='reused parent profile; earlier measurement'
    comparison_cols=['panel','arm','seed','median_seconds','range_seconds','profile_series','inference_peak_allocated_mib','source']
    profiles=pd.concat([rr[comparison_cols],old_profile[comparison_cols]],ignore_index=True)
    profiles.to_csv(OUT/'RESOURCE_COMPARISON.csv',index=False)
    profile_summary=profiles.groupby(['panel','arm','source'])[['median_seconds','inference_peak_allocated_mib']].mean().reset_index()
    fit_ledger=pd.read_csv(OUT/'FIT_LEDGER.csv')
    fit_ledger['train_peak_allocated_mib']=fit_ledger.peak_allocated/2**20
    training_summary=fit_ledger.groupby(['source','arm'])[['optimizer_seconds','validation_seconds','train_peak_allocated_mib']].mean().reset_index()
    summary=[]
    for panel in PANELS:
        f=main[(main.panel==panel)&main.ci_type.eq('time')].set_index('baseline');summary.append(panel+': '+', '.join(a+f" 대비 {f.loc[a,'gain_pct']:+.4f}%" for a in ARMS))
    text=f'''# C3 약점 해결 대조 실험 — 한국어 결과

**핵심 결론:** C3의 전력 조건부 이득은 유지되지만, 단순 크기 가중(MAG_ONLY)을 넘어서는 연속성 고유의 추가 가치는 확인하지 못했다. [결과 해석과 논문 주장 수정](INTERPRETATION_KO.md)을 먼저 참고한다.

## 실행 범위와 완료

사용자 다운로드 문서 EXECUTION_CONTRACT.md를 실행 승인에 따라 구현했다. 결과 전 규약/코드 봉인 커밋은396de8d이다. C3/B0는 바꾸지 않고 POS_ONLY/MAG_ONLY/OUTPUT_CONTEXT만 **30/30 fits,30,720 main+12 smoke=30,732 updates**로 실행했다. 반복 가중치18개와 source/전력전이 **54개 새 prediction views**를 전체 저장한 뒤 E를 채점했다. 불리한 성능으로 취소하거나 추가 LR/seed/방법을 만들지 않았다.

각 source×대조는선택seed81550/LR1e-4·3e-4 두경로와선택LR의반복81551/52/53 세경로다.1024updates를모두실행한뒤V로0/256/512/768/1024중선택했다. step0선택은추가효과없음이며실행실패가아니다. C3와기존core/MEAN/ROTATE/RECENCY/F0는부모검증결과를재사용했다.

## 질문과 정보 권한

RECENCY도 입력별 지속성 값들을 정렬해 사용한다. 따라서 RECENCY와 비슷하다고 지속성 정보 전체가 무의미하다고 쓸 수 없다. POS_ONLY는 정적 학습 위치만, MAG_ONLY는 큰 값만 사용하며 OUTPUT_CONTEXT는 동결 B0의예측과관측평균만으로출력을보정한다. 모두원래입력을보존하고미래y/state/clean x/true delta를feature로주지않는다.

POS_ONLY는8712adapter+32logits=8744params, gate LR.01고정이다. MAG_ONLY는8712params. OUTPUT_CONTEXT는72→64 affine+gamma=4673params이며W N(0,.01),b0,gamma0에서시작했다. 초기/off B0 동일성을실제모델에서확인했다. 모든군의TRAIN/V/loss/정답노출/updates는동일하지만파라미터수·optimizer난이도·확률출력표현력까지같다고주장하지않는다. OUTPUT_CONTEXT는quantile간격을보존한다.

OUTPUT_CONTEXT는COSA에서동기를얻은offline구조대조이며TAFAS/COSA의온라인정답도착·갱신을재현하지않았다. [BASELINE_MAPPING.md](BASELINE_MAPPING.md)의공식선행전체와의차이를유지한다. 일반adapter·learnedgate·identity시작자체는신규성이아니다.

## 자료 및 범위

Electricity4계열,동일날짜의transfer16계열,ETTm1 4계열을그대로사용했다. TRAIN256일/V64일/E128일,두E draw,기존9개shape를유지했다. 모두반복사용한개발E다. transfer16은과거노출확인4/불명12이며독립source가아니다. 실제오류/소비변화사건label은없다. 새NESO2026평가,LCL준비/다운로드,ETTm2기전학습은하지않았다. SAME/DIFFERENT의이전분석을새핵심결과로다시계산하지않았다.

## 주 비교: SHIFT8

양수는C3가대조보다낮은nMAE,음수는대조가유리하다.각계열/원점/draw동가중오차의비율이며개별origin백분율평균이아니다.7일paired block2000회,고정seed평균에조건부인구간.transfer는계열+시간구간도보존한다.각 패널 안의 3개 사전 대조에 대한 Bonferroni 구간이다. 전체 패널·조건의 다중성이나 누적 연구 선택 편향을 제거하지 않는다.

{table(main,cols)}

{chr(10).join(summary)}

## 세 seed와 실제 선택

{table(primaryseeds,['panel','baseline','seed','new_nmae','baseline_nmae','gain_pct'])}

{table(models,['source','arm','seed','lr','step'])}

## 원점수와 손익

{table(rawmain)}

개별fault6종,shapes9종,모든seed와기존대조는RAW_SCORES.csv에있다. C3/C2·C3/RECENCY 및각신규대조/C0·C2를MATCHED_CONTRASTS.csv에함께보고한다. 원자료손해를임의허용오차로삭제하지않고평균양수/CI0포함을동등성또는논문PASS로변환하지않는다. shape유리조건을새주조건으로승격하지않았다.

C3와각대조의주요상태차이:

{table(trade,['panel','condition','baseline','gain_pct','ci_low_pct','ci_high_pct'])}

## 학습이 실제로 움직였는가

전체30경로에서동결B0와buffer를보존했다. 파라미터변화·POS_ONLY gate·OUTPUT_CONTEXT gamma 및component gradient를PARAMETER_DYNAMICS.json/UPDATE_LEDGER.jsonl에남겼다. 모든실행경로의추가가중치가실제로변했고,선택step0은학습미실행과구분했다. 초기POSgate .5는입력변형이아니며adapter출력0으로B0와동일하게시작한다. gamma와W를동시에0으로초기화하지않았다.

## 자원과 검산

새training compute {cost['training_compute_seconds']/60:.2f}분,validation {cost['validation_seconds']/60:.2f}분,checkpointIO {cost['checkpoint_io_seconds']:.2f}초,run wall {cost['wall']['seconds']/60:.2f}분. 과거B0학습비용은부모COST_ACCOUNTING.json을별도참조한다. 추가파라미터만으로전체비용이같은비율감소했다고해석하지않는다.

{table(rt)}

본학습 1,024 updates 경로당 평균 시간과 메모리(선택 및 반복 경로 포함):

{table(training_summary)}

기존 C3/C2/B0의 동일 크기 추론 측정과 함께 보기. 기존 결과는 이전 측정 시점의 값을 재사용했으며, 원시 반복 시간은 RESOURCE_COMPARISON.csv 및 부모 RESOURCE_REPORT.csv에 남겼다.

{table(profile_summary)}

같은RTX3080/FP32/micro32/128입력의warmup후3회측정과seed별rawtime을공개했다. 작은시간차이를보편속도우위라고해석하지않는다. 기존C3 등은부모RESOURCE_REPORT의동일크기측정치를참조하며다른측정시점의noise는남는다. OUTPUT_CONTEXT는본체를통과하는추가gradient가필요없는단순출력대조다.

GPU 최소여유 {verify['minimum_free_gpu_mib']}MiB,승인RustDesk외compute0건. CPU부품검사5개와실제smoke6×2updates를구분했다.54예측복원/offidentity/상태불변,150checkpoint hash,update30732상한,선택규칙,전체원점점수재집계,scalar {verify['scalar_rows']}항목×2metric 및주효과재검산을통과했다. Scalar범위는모든새view/조건의첫·마지막원점/채널과두draw이며모든원점을scalar로재생했다고쓰지않는다.

## 미실행·판정 경계

이번승인범위의미실행fit/필수평가는없다. LCL실제사건자료의연결감사,정식온라인TAFAS/COSA전체재현,독립source와다른backbone은미실행이며이번설명대조로대체완료했다고쓰지않는다. 후속새구조·추가학습은자동시작하지않는다.

보고서수나그림수가방법근거가되는것은아니다. 효과·단순대안·신규성·현실타당성을별도로해석한다. 최종근거에맞는결론은FINAL_DECISION.md에있다. 원자료/가중치/전체predictions는ignored로컬cache이며GitHub에는코드·봉인·원점수·검산·그림을공개한다.

## 그림

- [세 주요 대조와seed](figures/01_primary_controls.png)
- [원자료·오류·변화의절충](figures/02_tradeoffs.png)
- [선택된위치gate](figures/03_selected_position_gates.png)
- [전체변화형태](figures/04_all_shapes.png)
'''
    (OUT/'REPORT.md').write_text(text);print('REPORT_AND_FIGURES_READY',flush=True)
if __name__=='__main__':build_report()
