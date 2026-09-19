"""Independent numerical replay and Korean answers to the five contract questions."""
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

def table(f):
    return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))

def finalize():
    assert read(OUT/'status.json')['execution']=='COMPLETE_COMPUTE';check_seal()
    ledger=[json.loads(x) for x in (OUT/'UPDATE_LEDGER.jsonl').read_text().splitlines()]
    assert len(ledger)==MAIN_CAP and len({(x['fit'],x['step']) for x in ledger})==MAIN_CAP
    smoke=read(OUT/'SMOKE.json');assert len(smoke)==4 and len((OUT/'SMOKE_LEDGER.jsonl').read_text().splitlines())==8
    for r in smoke:
        assert r['updates']==2 and all(r[k] for k in ['initial_F0_exact','off_F0_exact','restore_exact','frozen_unchanged','buffers_unchanged','finite_gradients','raw_input_unchanged','target_substitution_prediction_unchanged'])
        assert not r['no_lora']['lora_modules'] and r['no_lora']['trainable_count']==8712
    receipts=[read(p) for p in (OUT/'fits').glob('*/receipt.json')];assert len(receipts)==16
    resources=[];checkpoints=0;initials={}
    for r in receipts:
        z=[v for v in ledger if v['fit']==r['fit']];assert [v['step'] for v in z]==list(range(1,1025))
        assert r['frozen_unchanged'] and r['buffers_unchanged'] and r['trainable_parameters']==8712
        initial=read(OUT/'fits'/r['fit']/'initial_parity.json');assert initial['initial_F0_exact'] and initial['off_F0_exact'] and not initial['lora_modules'] and not initial['lora_parameter_names']
        assert all(n.startswith('adapter.') for n in initial['trainable_names'])
        initials.setdefault((r['source'],r['seed']),set()).add(initial['initial_adapter_sha256'])
        assert min(r['checkpoints'],key=lambda c:(c['objective'],c['step']))==r['selected']
        for c in r['checkpoints']:assert sha(ROOT/c['checkpoint'])==c['sha256'];checkpoints+=1
        np.testing.assert_allclose(math.fsum(v['seconds'] for v in z),r['optimizer_seconds'],rtol=1e-10)
        resources.append({k:r[k] for k in ['fit','source','arm','seed','lr','trainable_parameters','optimizer_seconds','validation_seconds','peak_allocated','peak_reserved']})
    assert all(len(v)==1 for v in initials.values())
    for source,arms in read(OUT/'LR_SELECTION.json').items():
        for arm,c in arms.items():
            z=[r for r in receipts if r['source']==source and r['arm']==arm and r['seed']==81550];assert len(z)==2
            expected=min(z,key=lambda r:(r['selected']['objective'],r['lr'],r['selected']['step']));assert expected['lr']==c['lr'] and expected['fit']==c['selection_fit']
    models=read(OUT/'MODEL_SELECTION.json');preds=read(OUT/'PREDICTIONS_MANIFEST.json');assert len(models)==48 and len(preds)==192
    for r in models:
        assert sha(ROOT/r['checkpoint'])==r['sha256']
        if r['arm'] in ARMS:
            choice=read(OUT/'LR_SELECTION.json')[r['source']][r['arm']]
            receipt=read(OUT/'fits'/fit_id(r['source'],r['arm'],r['seed'],choice['lr'])/'receipt.json');expected=receipt['selected'] if r['stage']=='selected' else receipt['checkpoints'][-1]
            assert r['lr']==choice['lr'] and all(r[k]==expected[k] for k in ['step','checkpoint','sha256','objective'])
    seal=read(OUT/'EVALUATION_SEAL.json');marker=read(OUT/'ALL_PREDICTIONS_SAVED.json')
    assert seal['selection_sha256']==sha(OUT/'MODEL_SELECTION.json') and marker['manifest_sha256']==sha(OUT/'PREDICTIONS_MANIFEST.json') and seal['at']<marker['at']
    origin=pd.read_csv(OUT/'ORIGIN_SCORES.csv.gz');channel=pd.read_csv(OUT/'ORIGIN_CHANNEL_SCORES.csv.gz');raw=pd.read_csv(OUT/'RAW_SCORES.csv');effects=pd.read_csv(OUT/'EFFECTS.csv');inter=pd.read_csv(OUT/'INTERACTION_EFFECTS.csv');se=pd.read_csv(OUT/'SEED_EFFECTS.csv')
    keys=['panel','kind','arm','seed','stage','step','condition'];metrics=['nmae','mae','normalized_mse','pinball','crossing']
    repeated=channel.groupby(keys,as_index=False)[metrics].mean().merge(raw,on=keys,suffixes=('_a','_b'),validate='one_to_one');assert len(repeated)==len(raw)
    for col in metrics:np.testing.assert_allclose(repeated[col+'_a'],repeated[col+'_b'],rtol=1e-10,atol=1e-11)
    rm=channel.groupby(keys+['channel']).normalized_mse.mean().pow(.5).groupby(level=list(range(len(keys)))).mean().rename('replayed_nrmse').reset_index().merge(raw,on=keys,validate='one_to_one')
    np.testing.assert_allclose(rm.replayed_nrmse,rm.nrmse,rtol=1e-10,atol=1e-11)
    # Direct prediction replay independent of score.metric_arrays and target_data.
    metric_count=0
    for key,r in preds.items():
        assert sha(ROOT/r['path'])==r['sha256'];panel,kind=r['panel'],r['kind']
        packet=np.load(data_path(panel)/'E_DISCOVERY_inputs.npz');truth=np.load(data_path(panel)/'E_DISCOVERY_labels.npz')['y'];nc=truth.shape[1]
        if kind=='standard':names=STATES;ids=np.arange(len(packet['origins']));offset=np.load(panel_path(panel,kind)/'E_DISCOVERY_offset.npy')
        else:
            m=read(panel_path(panel,kind)/'manifest.json');names=m['states'];ids=np.array(m['origin_indices']);offset=np.load(panel_path(panel,kind)/'offset.npy')
        n=len(ids);pred=np.load(ROOT/r['path'],mmap_mode='r');assert tuple(pred.shape)==(len(names)*2*n*nc,9,64)
        for ci,name in enumerate(names):
            lo=ci*2*n*nc;hi=lo+2*n*nc;p=np.asarray(pred[lo:hi],float);y=np.tile(truth[ids].reshape(-1,64),(2,1))+offset[lo:hi,None];sigma=np.tile(packet['sigma'],2*n)
            mae=np.abs(p[:,4]-y).mean(1);nmae=mae/sigma;mse=((p[:,4]-y)**2).mean(1)/sigma**2;err=y[:,None,:]-p;q=np.arange(1,10)/10
            pin=(2*np.maximum(q[None,:,None]*err,(q[None,:,None]-1)*err)).mean((1,2))/sigma;cross=(p[:,1:]<p[:,:-1]).mean((1,2))
            val=np.stack([nmae,mae,mse,pin,cross],-1).reshape(2,n,nc,5).mean((0,2))
            saved=origin[(origin.panel==panel)&(origin.kind==kind)&(origin.arm==r['arm'])&(origin.seed==r['seed'])&(origin.stage==r['stage'])&(origin.condition==name)].set_index('origin').loc[packet['origins'][ids]]
            np.testing.assert_allclose(val,saved[metrics].to_numpy(),rtol=1e-10,atol=1e-10);metric_count+=val.size
        model=next(v for v in models if all(v[k]==r[k] for k in ['source','arm','seed','stage']))
        assert model['sha256']==r['checkpoint_sha256']
    for k,v in read(OUT/'REUSED_PREDICTIONS.json').items():assert preds[k]==v
    prior=pd.read_csv(ROOT/'results/learned_gate_comparison_20260919/RAW_SCORES.csv');prior=prior[prior.arm.isin(RENAMES)].copy();prior['arm']=prior.arm.map(RENAMES)
    z=raw[raw.arm.isin(RENAMES.values())].merge(prior,on=keys,suffixes=('_new','_old'),validate='one_to_one');assert len(z)==len(prior)
    for metric in ['nmae','mae','pinball']:np.testing.assert_allclose(z[metric+'_new'],z[metric+'_old'],rtol=1e-10,atol=1e-11)
    for (panel,kind,stage,condition),g in origin.groupby(['panel','kind','stage','condition']):
        a=g.groupby(['origin','arm']).nmae.mean().unstack();period=96 if panel=='ettm1' else 24;blocks=a.index.to_numpy()//(7*period);unique=np.unique(blocks);rng=np.random.default_rng(91942)
        counts=np.array([np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique)) for _ in range(2000)]);weight=counts[:,np.searchsorted(unique,blocks)];denom=weight.sum(1)
        ee=effects[(effects.panel==panel)&(effects.kind==kind)&(effects.stage==stage)&(effects.condition==condition)]
        for e in ee.itertuples():
            av=a[e.proposed].to_numpy();bv=a[e.baseline].to_numpy();boot=100*(1-(weight@av)/(weight@bv));bounds=np.quantile(boot,[.025,.975]);ab=weight@(bv-av)/denom;abl=np.quantile(ab,[.025,.975])
            np.testing.assert_allclose([100*(1-av.mean()/bv.mean()),*bounds,(bv-av).mean(),*abl],[e.gain_pct,e.ci_low,e.ci_high,e.absolute_gain,e.absolute_ci_low,e.absolute_ci_high],rtol=1e-8,atol=1e-9)
            for seed,ss in g.groupby('seed'):
                v=ss.groupby('arm').nmae.mean();row=se[(se.panel==panel)&(se.kind==kind)&(se.stage==stage)&(se.condition==condition)&(se.seed==seed)&(se.proposed==e.proposed)&(se.baseline==e.baseline)].iloc[0]
                np.testing.assert_allclose(row.gain_pct,100*(1-v[e.proposed]/v[e.baseline]),rtol=1e-8,atol=1e-9)
        terms={'plain_gain_F0':a.F0-a.F0_PLAIN,'plain_gain_B0':a.B0-a.B0_PLAIN,'mag_gain_F0':a.F0-a.F0_MAG,'mag_gain_B0':a.B0-a.B0_MAG,'mag_specific_F0':a.F0_PLAIN-a.F0_MAG,'mag_specific_B0':a.B0_PLAIN-a.B0_MAG}
        terms['second_stage_interaction_mag']=terms['mag_gain_B0']-terms['mag_gain_F0'];terms['mag_rule_interaction']=terms['mag_specific_B0']-terms['mag_specific_F0']
        for term,val in terms.items():
            row=inter[(inter.panel==panel)&(inter.kind==kind)&(inter.stage==stage)&(inter.condition==condition)&(inter.term==term)].iloc[0];bounds=np.quantile(weight@val.to_numpy()/denom,[.025,.975])
            np.testing.assert_allclose([val.mean(),*bounds],[row.nmae_difference,row.ci_low,row.ci_high],rtol=1e-8,atol=1e-10)
    gpu=[json.loads(x) for x in (OUT/'gpu_standalone.jsonl').read_text().splitlines()]
    unapproved=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in v['apps']) for v in gpu);assert unapproved==0
    pd.DataFrame(resources).to_csv(OUT/'RESOURCES.csv',index=False)
    selection=pd.DataFrame([{k:r.get(k) for k in ['source','arm','seed','stage','step','lr']} for r in models]);selection.to_csv(OUT/'CHECKPOINT_SELECTION.csv',index=False)
    main=effects[(effects.stage=='selected')&(effects.kind=='standard')&effects.condition.isin(['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT'])]
    answers=[];classifications=[]
    for panel in PANELS:
        source='ettm1' if panel=='ettm1' else 'electricity'
        for condition in ['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT']:
            z=main[(main.panel==panel)&(main.condition==condition)]
            def row(proposed,base):return z[(z.proposed==proposed)&(z.baseline==base)].iloc[0]
            p=row('F0_PLAIN','F0');m=row('F0_MAG','F0');specific=row('F0_MAG','F0_PLAIN');bm=row('B0_MAG','B0');bs=row('B0_MAG','B0_PLAIN')
            def repeated(r):return all(v>0 for v in json.loads(r.seed_gains).values())
            f0steps=selection[(selection.source==source)&(selection.arm=='F0_MAG')&(selection.stage=='selected')].step.tolist();b0steps=selection[(selection.source==source)&(selection.arm=='B0_MAG')&(selection.stage=='selected')].step.tolist()
            standalone=repeated(m) and repeated(specific) and all(v>0 for v in f0steps)
            second=repeated(bm) and repeated(bs) and all(v>0 for v in b0steps)
            decision='BOTH_USES_SUPPORTED' if standalone and second else 'STANDALONE_MAG_SUPPORTED' if standalone else 'SECOND_STAGE_DEPENDENCE_SUPPORTED' if second and m.gain_pct<=0 and specific.gain_pct<=0 else 'SECOND_STAGE_SUPPORTED_STANDALONE_UNCERTAIN' if second else 'MIXED_OR_UNSUPPORTED'
            interaction=inter[(inter.panel==panel)&(inter.kind=='standard')&(inter.stage=='selected')&(inter.condition==condition)&(inter.term=='second_stage_interaction_mag')].iloc[0]
            answers.append(dict(panel=panel,condition=condition,F0_PLAIN_vs_F0_pct=p.gain_pct,F0_MAG_vs_F0_pct=m.gain_pct,F0_MAG_vs_PLAIN_pct=specific.gain_pct,mag_gain_F0=m.absolute_gain,mag_gain_B0=bm.absolute_gain,interaction_B0_minus_F0=interaction.nmae_difference,decision=decision))
            classifications.append(dict(panel=panel,condition=condition,decision=decision,standalone_two_seed_positive=standalone,second_stage_two_seed_positive=second,F0_selected_steps=f0steps,B0_MAG_selected_steps=b0steps,F0_MAG_F0_CI_excludes_zero=bool(m.ci_low>0),F0_MAG_PLAIN_CI_excludes_zero=bool(specific.ci_low>0),F0_MAG_better_than_B0_MAG_two_seeds=repeated(row('F0_MAG','B0_MAG'))))
    answer=pd.DataFrame(answers);answer.to_csv(OUT/'FIVE_QUESTIONS.csv',index=False);save(OUT/'CONDITION_DECISIONS.json',classifications)
    summary=raw[(raw.stage=='selected')&(raw.kind=='standard')&raw.condition.isin(['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT'])].groupby(['panel','condition','arm'],as_index=False).nmae.mean().pivot(index=['panel','condition'],columns='arm',values='nmae').reset_index()
    fig,axes=plt.subplots(1,4,figsize=(17,4))
    for ax,panel in zip(axes,PANELS):
        z=summary[(summary.panel==panel)&(summary.condition=='SHIFT8')].iloc[0]
        ax.bar(range(6),[z[a] for a in ALL_ARMS]);ax.set_xticks(range(6),ALL_ARMS,rotation=45,ha='right');ax.set_title(panel);ax.set_ylabel('SHIFT8 nMAE')
    fig.suptitle('No-LoRA and pre-adapted base comparison / reused development panels');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=170);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(17,4))
    for ax,panel in zip(axes,PANELS):
        z=main[(main.panel==panel)&(main.proposed=='F0_MAG')&(main.baseline=='F0_PLAIN')].set_index('condition').loc[['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT']]
        ax.errorbar(range(5),z.gain_pct,yerr=np.maximum(0,np.vstack([z.gain_pct-z.ci_low,z.ci_high-z.gain_pct])),fmt='o');ax.axhline(0,color='gray');ax.set_xticks(range(5),z.index,rotation=45,ha='right');ax.set_title(panel);ax.set_ylabel('MAG vs PLAIN gain (%)')
    fig.suptitle('Fixed-seed conditional date-block 95% intervals; not population-seed uncertainty');fig.tight_layout();fig.savefig(OUT/'tradeoffs.png',dpi=170);fig.savefig(OUT/'tradeoffs.pdf');plt.close(fig)
    resource=pd.DataFrame(resources).groupby('arm',as_index=False)[['trainable_parameters','optimizer_seconds','validation_seconds','peak_allocated']].mean();resource['peak_MiB']=resource.pop('peak_allocated')/2**20
    inherited=[]
    for source in SOURCES:
        for seed in SEEDS:
            r=parent.baseline_row(source,seed);p=ROOT/r['checkpoint'];receipt=read(ROOT/'results/outlier_signal_followup_v2_20260917/fits'/p.parent.name/'receipt.json')
            inherited.append(dict(source=source,seed=seed,selected_B0_step=r['step'],historical_fit_updates=receipt['updates'],checkpoint_sha256=r['sha256']))
    save(OUT/'HISTORICAL_COST.json',inherited)
    report=f"""# LoRA 선행 여부 직접 검증

## 다섯 질문에 대한 수치 답변

아래는 **selected·두seed 평균**, 양수%는 개선이다. Q1=F0_PLAIN_vs_F0, Q2=F0_MAG_vs_F0, Q3=F0_MAG_vs_PLAIN. Q4는 동일 nMAE 단위의 mag_gain_F0와mag_gain_B0를 비교한다. interaction 양수이면 B0의 절대 감소가 더 크고 음수이면F0가 더 크다. Q5는 각 조건의 반복 방향을 기록한다. 데이터·조건을 합쳐 전역 우승자를 만들지 않는다.

{table(answer)}

BOTH_USES_SUPPORTED는 두 경로 모두 두seed에서각자base와PLAIN보다 개선했다는 개발 근거다. STANDALONE_MAG_SUPPORTED는해당조건의no-LoRA추가가치, SECOND_STAGE_DEPENDENCE_SUPPORTED는F0이득이없고B0이득이남는현재설정의근거다. MIXED_OR_UNSUPPORTED/UNCERTAIN은혼재·미확보다. 이는논문PASS나독립검증판정이아니다. 날짜CI가0을포함하는지와step0선택은CONDITION_DECISIONS.json및아래표에서별도로공개한다. F0_MAG가B0_MAG보다좋은경우도동파일에기록하며LoRA불필요로일반화하지않는다.

## 원점수와 선택

{table(summary)}

{table(selection[selection.stage=='selected'])}

step0은F0또는B0그대로fallback이며추가PEFT의성공으로세지않는다. F0의두seed·두stage는동일모델정렬용표기다. B0 fixed1024는기존선택B0를고정하고추가adapter를1024까지학습한비교이다. B0자체를1024checkpoint로교체하지않았다.

## 직접 대조·불확실성과 손해

{table(main[['panel','condition','proposed','baseline','gain_pct','ci_low','ci_high','seed_gains']])}

2,000회 index7일 paired block bootstrap, 기존 RNG91942, 고정두seed·고정채널조건부95%구간이다. seed·channel·합성draw를독립날짜로세지않으며optimizer-seed모집단불확실성을모두표현하지않는다. 새전역PASS문턱·다중검정family를만들지않았다. 전체standard10조건+FAULT,9shape,selected/fixed1024와모든음성결과는RAW_SCORES/EFFECTS/SEED_EFFECTS.csv에남겼다. 불리한조건을제외하지않았다. INTERACTION_EFFECTS.csv는계약의두차분식을절대nMAE로보여주며완전한인과효과가아니다.

## 실제 실행·검산

16/16새fits,16384 unique main updates,8 smoke updates,미실행승인학습0. 새학습은F0_PLAIN/F0_MAG뿐이며B0/PLAIN/MAG재학습0,joint0,추가seed/LR/후속0. {checkpoints}개checkpoint hash,각fit1..1024순서,LR/V최소화선택,모든선택봉인→전체192prediction views저장→E채점을검사했다. 첫/마지막origin의독립scalar검산과전체예측직접계산 {metric_count}개metric,과거원점수 {len(prior)}행재현,효과·interaction·bootstrap재계산완료.

실제모델no-LoRA name/module검사,trainable8712개,공통초기값,step0/nativeF0/off예측동일,raw입력보존,finite gradient및parameter변경,동결weights/buffers보존,복원동일,배치순서정합을검사했다. RustDesk만외부GPU예외이며미승인compute표본{unapproved}개. 원자료·checkpoint·prediction은로컬캐시이고공개저장소만으로완전수치재현가능하다고하지않는다.

## 파라미터·누적 비용

|경로|이번stage trainable|deployed adaptation params|누적학습stage|반복경로별학습량|
|---|---:|---:|---:|---|
|F0|0|0|0|0|
|F0_PLAIN/F0_MAG|8712|8712|1|새1024 updates;V선택step별도|
|B0|이번0,과거294912|294912|1|과거LoRA1024fit 재사용;선택step별도|
|B0_PLAIN/B0_MAG|이번0,과거추가8712|303624|2|과거LoRA1024+adapter1024fit 재사용|

선택LR탐색비용은위단일경로학습량과별개이며이번총16fits를모두포함한다. B0의과거selectedstep/fit비용은HISTORICAL_COST.json. 두단계전체budget과standalonebudget은같지않다. B0우세를LoRA필수로, F0우세를LoRA불필요로단정하지않는다. compute-matched2048updates나joint는수행하지않았다.

{table(resource)}

측정시간은intent저장등을포함한현재runner기준이고과거시간비로순수속도우위를주장하지않는다. 비용해석에누적LoRA구축비용을제외하지않는다.

## 논문에서 사용할 범위와 종료

기존MAG_ONLY는no-LoRA가아니었고이번에F0로직접검증했다. 발전된PEFT방법론주장을쓸수있는조건은위직접대조의실제반복결과로제한한다. 재사용개발패널4개이며NESO도같은provider의기존노출기간이다. 실제오류/실제변화사건label없음,독립source일반화·정식선행전체우위·신규성완료를주장하지않는다. 이번학습승인만종료하고학습정지로돌아간다. 자동후속학습0.
"""
    (OUT/'REPORT.md').write_text(report)
    (OUT/'FINAL_DECISION.md').write_text('# 최종 판단\n\n실행·검산완료. 논문PASS로선언하지않는다. 각조건의standalone/second-stage판단을아래에분리했다.\n\n'+table(answer[['panel','condition','decision']])+'\n\n두경로누적예산이달라LoRA의필요/불필요를보편적으로판정할수없다. step0은fallback이며방법성공으로세지않는다. 사용가능한조건과음성결과를함께남긴다. 새LR/seed/rank/gate/data/joint/후속학습은0이며기존학습금지로복귀한다.\n')
    audit=dict(status='VERIFIED',new_fits=16,main_updates=MAIN_CAP,smoke_updates=SMOKE_CAP,checkpoints=checkpoints,independent_prediction_metrics=metric_count,reused_score_rows=len(prior),prediction_views=192,unique_prediction_files=len({r['path'] for r in preds.values()}),paper_pass=False,automatic_successor=False,unapproved_gpu_samples=unapproved,training_stopped_after_scope=True,old_hashes_unchanged=True)
    save(OUT/'AUDIT.json',audit);save(OUT/'VERIFICATION.json',dict(read(OUT/'VERIFICATION.json'),**audit))
