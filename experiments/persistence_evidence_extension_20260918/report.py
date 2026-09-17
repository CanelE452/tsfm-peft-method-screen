"""Read-only analysis and Korean interpretation of the sealed inference experiment."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/persistence_extension_mpl')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,ScalarFormatter
from .common import *
from experiments.outlier_signal_peft_v1_20260917.report import table

def report():
    v=read(OUT/'VERIFICATION.json');assert v['status']=='VERIFIED';check_seal();d=read(OUT/'INTERPRETATION.json');effects=pd.read_csv(OUT/'EFFECTS.csv');raw=pd.read_csv(OUT/'RAW_SCORES.csv');seed=pd.read_csv(OUT/'SEED_EFFECTS.csv');fac=pd.read_csv(OUT/'FACTORIAL_DECOMPOSITION.csv')
    external=effects[(effects.panel=='neso_2025')&(effects.kind=='standard')];columns=['condition','new','baseline','new_nmae','baseline_nmae','gain_pct','ci_low_pct','ci_high_pct']
    ext=external[(external.new=='C3')&external.baseline.isin(['C0','C2','M_RECENCY'])]
    primary=external[(external.condition=='SHIFT8')&external.new.isin(['C2','C3'])]
    primaryseed=seed[(seed.panel=='neso_2025')&(seed.kind=='standard')&(seed.condition=='SHIFT8')&(seed.new=='C3')&seed.baseline.isin(['C0','C2','M_RECENCY'])]
    rawexternal=raw[(raw.panel=='neso_2025')&(raw.kind=='standard')]
    stress=effects[(effects.kind=='shape')&(effects.new=='C3')&effects.baseline.isin(['C2','M_MEAN','M_ROTATE16','M_RECENCY'])&(effects.ci_type=='time')]
    pivot=stress.pivot(index=['panel','condition'],columns='baseline',values='gain_pct').reset_index()
    fstandard=fac[fac.kind=='standard'][['panel','condition','total_nmae_benefit','gate_nmae_benefit','weights_nmae_benefit','interaction']]
    # Post-hoc audit of already published matched schedules; no new model inference or selection.
    choices=read(PRIOR/'MODEL_SELECTION.json');matched=[]
    for source in ['electricity','ettm1']:
        for seedvalue in [81551,81552,81553]:
            a=next(x for x in choices if (x['source'],x['arm'],x['seed'])==(source,'C3',seedvalue))
            b=next(x for x in choices if (x['source'],x['arm'],x['seed'])==(source,'M_RECENCY',seedvalue))
            assert a['lr']==b['lr'] and a['step']==b['step']
            matched.append(dict(source=source,seed=seedvalue,C3_lr=a['lr'],RECENCY_lr=b['lr'],C3_step=a['step'],RECENCY_step=b['step'],same_LR_and_step=True))
    save(OUT/'MATCHED_SELECTION_AUDIT.json',dict(status='VERIFIED',posthoc_receipt_audit=True,new_predictions=0,optimizer_updates=0,pairs=matched,meaning='C3/RECENCY learned-weight contrast is not confounded by different selected LR/step in these actual pairs; inference intervention still not a full account of training dynamics'))
    fixed=pd.read_csv(PRIOR/'PANEL_SCORES.csv');fixed=fixed[(fixed.stage=='fixed1024')&(fixed.kind=='standard')];fx=[]
    for (panel,condition),group in fixed.groupby(['panel','metric_panel']):
        means=group.groupby('arm').nmae.mean()
        for base in ['C0','C2','M_RECENCY']:
            if base not in means:continue
            source='electricity' if panel.startswith('electricity') else panel
            matchedlr=all(next(r for r in choices if (r['source'],r['arm'],r['seed'])==(source,'C3',z))['lr']==next(r for r in choices if (r['source'],r['arm'],r['seed'])==(source,base,z))['lr'] for z in group.seed.unique()) if base!='C0' else None
            fx.append(dict(panel=panel,condition=condition,new='C3',baseline=base,new_nmae=float(means.C3),baseline_nmae=float(means[base]),gain_pct=float(100*(1-means.C3/means[base])),same_LR=matchedlr,additional_adaptation_step=1024,role='previously published fixed1024 supplement; not replacing selected primary'))
    pd.DataFrame(fx).to_csv(OUT/'REUSED_FIXED1024_COMPARISONS.csv',index=False)
    preds=read(OUT/'PREDICTIONS.json');resources=pd.DataFrame([dict(view=k,**{name:r.get(name) for name in ['panel','kind','arm','seed','inference_seconds','peak_allocated','alias_of','forward_models']}) for k,r in preds.items()]);resources['peak_allocated_mib']=resources.peak_allocated/2**20;resources.to_csv(OUT/'RESOURCE_REPORT.csv',index=False)
    wall=read(OUT/'inference_wall.json');samples=[json.loads(l) for l in open(OUT/'gpu_inference.jsonl')];unapproved=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in r['apps']) for r in samples);assert unapproved==0
    costs=dict(optimizer_updates=0,new_fits=0,new_prediction_views=111,model_forward_views=105,alias_views=sum(bool(r.get('alias_of')) for r in preds.values()),postprocess_views=6,wall_seconds=wall['seconds'],inference_seconds_recorded=float(resources.inference_seconds.sum()),min_free_gpu_mib=min(r['free_mib'] for r in samples),unapproved_gpu_samples=unapproved,new_cache_bytes=sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file()),scope='inference timings include new prediction save / alias check, excluding loading/state hashing; wall includes checks/scoring/CI; weights and old cache shared');save(OUT/'COST.json',costs)
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    def finish(name):
        plt.tight_layout()
        for suffix in ['png','pdf']:plt.savefig(figs/f'{name}.{suffix}',dpi=160)
        plt.close()
    conditions=['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT'];fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,base in zip(axes,['C0','C2','M_RECENCY']):
        x=ext[ext.baseline==base].set_index('condition').loc[conditions];ax.hlines(range(5),x.ci_low_pct,x.ci_high_pct);ax.plot(x.gain_pct,range(5),'o');ax.axvline(0,color='grey');ax.set_yticks(range(5),conditions);ax.set_title('NESO2025: C3 vs '+base);ax.set_xlabel('nMAE gain %, time-block conditional 95% CI')
    finish('01_external_tradeoff')
    fig,axes=plt.subplots(2,2,figsize=(12,10));methods=['C2','M_MEAN','M_ROTATE16','M_RECENCY'];names=SHAPES+['PAIRED_SHIFT8_D32'];lim=max(1,float(stress[stress.panel!='ettm2'].gain_pct.abs().max()))
    for ax,panel in zip(axes.flat,['electricity','electricity_transfer','ettm1','neso_2025']):
        x=stress[stress.panel==panel].pivot(index='condition',columns='baseline',values='gain_pct').loc[names,methods];im=ax.imshow(x.to_numpy(),cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto');ax.set_xticks(range(4),methods,rotation=25);ax.set_yticks(range(9),names);ax.set_title(panel)
        for i in range(9):
            for j in range(4):ax.text(j,i,f'{x.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(x.iloc[i,j])>.55*lim else 'black')
    fig.suptitle('C3 vs each trained control: nMAE gain % (positive better)');finish('02_shape_controls')
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,panel in zip(axes,['electricity','electricity_transfer','ettm1']):
        x=fstandard[fstandard.panel==panel].set_index('condition').loc[conditions];y=np.arange(5);ax.barh(y-.15,x.gate_nmae_benefit,.3,label='gate contrast');ax.barh(y+.15,x.weights_nmae_benefit,.3,label='weights contrast');ax.plot(x.total_nmae_benefit,y,'kx',label='total RECENCY - C3');ax.axvline(0,color='grey');ax.set_yticks(y,conditions);ax.set_title(panel);ax.set_xlabel('absolute normalized error reduction');ax.legend(fontsize=7);ax.xaxis.set_major_locator(MaxNLocator(4));fmt=ScalarFormatter(useMathText=True);fmt.set_powerlimits((-3,-3));ax.xaxis.set_major_formatter(fmt)
    finish('03_fixed_weight_decomposition')
    body=f'''# C3 논문 근거 추가 검증

실행 COMPLETE. 새 학습 0회·optimizer update 0회, 추가 예측 111view를 모두 저장하고 채점·검산했다. 기존 58-fit 연구와 C3 설계를 보존했다. 이 후속은 사용자의 2026-09-18 추가 검증 요청에 따라 별도 봉인했다.

{d['summary']}

## 이번에 해결하려 한 질문

이전 연구에서 남은 핵심은 RECENCY 대비 작은 차이, ETT 악화, 반복 노출 자료다. 같은 C3를 계속 튜닝해 양성으로 바꾸는 작업은 하지 않았다. 기존 학습 대조의 변화 형태 평가27view, 가중치 고정 gate교환36view, NESO 외부 원천42모델view+6출력대조를 수행했다. 주 질문과 데이터·코드·checkpoint는 [PROTOCOL.md](PROTOCOL.md), [SEAL.json](SEAL.json)에 채점 전 기록했다.

## 외부 원천의 실제 독립성 범위

NESO2025 National Demand(ND) 한 계열의17,520반시간값을UTC시간별MW평균8,760개로 변환했다. 서머타임의46/50구간과중복/결측을검사했다. target sigma는2025상반기만, E128개 서로다른날짜는하반기전체를균등선정했다. target model fitting·V선택은0회, Electricity의LR/checkpoint/alpha/beta를그대로사용했다. B0는원래전력source에적응한모델이며NESO에잘학습된기준선이라는뜻은아니다. 이번에는NESO전용B0학습과미적응foundation(F0)추가대조를하지않았으므로source-adaptation자체의손익과추가어댑터의손익을완전히분리한외부benchmark라고주장하지않는다. 표본은 [ORIGIN_AUDIT.csv](ORIGIN_AUDIT.csv)에 있다.

현재localweight/config는2024-11-28공개파일과동일하다. 따라서2025의해당관측값은backbone학습뒤시점임을확인했다. 프로젝트의기존추적text4,443파일에서는NESO사용키워드를찾지못했고source PEFT TRAIN은기존Electricity/ETT로보존됐다. 이전연도의같은provider자료노출과외부/기록밖사람의노출까지배제하지는못한다. 새평가원천이지만새로수집한자료가아니며, 국가집계1계열을여러독립meter로세지않는다. 실제오류/사건레이블은없다. 제공자가수정한과거outturn이고당시공개된as-of vintage를복원한것은아니다. [자료감사](EXTERNAL_SOURCE_AUDIT.json), [공식자료](https://www.neso.energy/data-portal/historic-demand-data/historic_demand_data_2025), [품질·수정안내](https://www.neso.energy/data-portal/historic-demand-data).

## 외부 전이 결과

{d['external']}

SHIFT8 원점수·직접대비(세seed평균):

{table(primary[columns+['bonferroni3_low_pct','bonferroni3_high_pct']])}

각 seed의 공동 주 비교:

{table(primaryseed[['seed','new','baseline','new_nmae','baseline_nmae','gain_pct']])}

원자료·오류·다른 변화의 손익:

{table(ext[columns])}

NESO모든조건/군/seed 원점수는 [RAW_SCORES.csv](RAW_SCORES.csv)에 있다. 점수는자체nMAE이며gain은평균오차비율이다. 표의날짜CI는고정3seed와관측기간에조건부이고source모집단·seed모집단구간은아니다. 더낮은오차를보인조건만선별하지않았다.

## 단순 대조는 어디까지 충분한가

{d['controls']}

등록된 모든 형태에서 C3의 각 대조 대비 gain(%):

{table(pivot)}

형태범위의다수비교는탐색적이다. 모든정의된형태를남기고최대이득한칸으로확증을선언하지않는다. PULSE/PAIRED_SHIFT8은같은입력과다른미래이므로이를완벽히구분하는것을요구하지않는다. 합성강도는실제변화발생분포가아니며음수수요같은물리적부적합값을포함할수있는수학적스트레스다.

## 가중치와 gate의 분해

{d['factorial']}

A=C3weights/C3gate,B=C3weights/RECENCYgate,C=RECENCYweights/C3gate,D=RECENCYweights/RECENCYgate. gate효과=((B−A)+(D−C))/2,weight효과=((C−A)+(D−B))/2이며합은D−A다. 아래단위는gain%가아닌절대nMAE감소이고양수가C3쪽유리다. 두교차구성은사후기전검사용이며후보로선택/배포하지않는다. 이미학습된두함수의분해이므로학습과정의순수인과효과가아니다. 실제C3/RECENCY여섯source-seed쌍은선택LR와step이모두같음을추가감사했다(MATCHED_SELECTION_AUDIT.json). 따라서이번쌍의weight항을LR/step차이로설명하면안된다. 같은초기값·입력순서·학습기회에서mask가다른학습경로를만드는것과추론시mask직접효과를구분한다. 다만훈련중어떤표현/gradient가그차이를만들었는지까지이분해가규명한것은아니다.

{table(fstandard)}

전체형태별분해는 [FACTORIAL_DECOMPOSITION.csv](FACTORIAL_DECOMPOSITION.csv)에 있다. 기존동일LR/1,024-update결과는새추론없이재집계했다: [재사용고정예산대조](REUSED_FIXED1024_COMPARISONS.csv), [선택조건감사](MATCHED_SELECTION_AUDIT.json). 주결과는여전히봉인된selected모델이다.

## 검산과 비용

CPU5검사,105개모델view의원래선택forward일치·고정가중치보존·재로드검사,새원점수{v['new_origin_rows']:,}행,독립scalar {v['scalar_rows']:,}행×2지표,효과{v['effect_rows_independently_replayed']}행재계산을통과했다. scalar범위는모든새view/조건의첫·마지막원점과첫·마지막채널및두draw이며모든행scalar검산은아니다. [검산](VERIFICATION.json), [모델검사](MODEL_CHECKS.json).

실행wall {costs['wall_seconds']/60:.2f}분,기록된추론·저장시간합{costs['inference_seconds_recorded']:.2f}초,최소GPU여유{costs['min_free_gpu_mib']}MiB,허용되지않은외부compute표본0개다. alias {costs['alias_views']}view는C1step0/C0동치확인후재사용했다. output6view는원래전력V의값으로만들었으며SHRINK의배포비용은여전히두forward다. 기존가중치/자료의학습비용은이전보고서에분리돼있다. 원래guard의external_compute_samples는승인된RustDesk도포함하므로미승인사용은별도필터로계산했다. [자원](RESOURCE_REPORT.csv), [비용](COST.json).

GitHub에는코드·봉인·원점수·manifest·검산을올리며모델weights·원시CSV·predictioncache는로컬보존한다. 공개저장소만으로로컬cache없이완전한수치재현이가능하다고주장하지않는다.

## 논문으로 남길 부분과 남은 검증

{d['decision']}

{d['next']}

선행확인은 [LITERATURE_UPDATE.md](LITERATURE_UPDATE.md)에있다. Persistence Initialization의persistence는예측초기값이며현재observed-runmask와같은정의가아니다. 일반identity residual/gating은신규기여에서제외한다. COSA/TATO/SOLID의정식전체재현은실행하지않았다. 같은정보권한·동일추가계산예산의비교가남는다. 새방법이나후속학습을자동실행하지않는다.

표의 `nan`은 해당 구간을 계산하지 않았거나 해당 비교군이 없는 항목이며 0점이 아니다. ETTm2의 MEAN·ROTATE16·RECENCY는 기존 학습 가중치가 없어 이번에도 실행하지 않았다. 최종 파일과 보존 검사는 [FINAL_ARTIFACT_AUDIT.json](FINAL_ARTIFACT_AUDIT.json)에 기록한다.

![외부전이](figures/01_external_tradeoff.png)
![변화형태대조](figures/02_shape_controls.png)
![고정가중치분해](figures/03_fixed_weight_decomposition.png)
'''
    (OUT/'REPORT.md').write_text(body)
    (OUT/'FINAL_DECISION.md').write_text('# 최종 판단\n\n'+d['summary']+'\n\n'+d['decision']+'\n\n'+d['next']+'\n\n실행 COMPLETE. 새 학습 0회, optimizer update 0회, 추가 예측 111view. 논문 PASS 선언이나 자동 후속 학습은 없다.\n')
    (OUT/'PAPER_REVISION.md').write_text('# 이전 논문 초안에서 고칠 내용\n\n'+d['paper_revision']+'\n\n'+d['next']+'\n\n이전 58-fit 결과를 보존하고 이번 추가 검증을 후속 표로 연결한다. 전체 원점수와 불리한 조건도 보충자료에 남긴다.\n')
    save(OUT/'status.json',dict(execution='COMPLETE',new_prediction_views=111,new_fits=0,optimizer_updates=0,reports_complete=True,automatic_successor=False))
    print('REPORT_COMPLETE',flush=True)
if __name__=='__main__':report()
