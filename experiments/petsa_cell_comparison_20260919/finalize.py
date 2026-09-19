"""Independent direct prediction/selection checks and a bounded Korean report."""
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

def table(f):
    return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))

def finalize():
    assert read(OUT/'status.json')['execution']=='COMPLETE_COMPUTE'
    check_seal()
    ledger=[json.loads(x) for x in (OUT/'UPDATE_LEDGER.jsonl').read_text().splitlines()]
    assert len(ledger)==MAIN_CAP and len({(x['fit'],x['step']) for x in ledger})==MAIN_CAP
    assert len((OUT/'SMOKE_LEDGER.jsonl').read_text().splitlines())==SMOKE_CAP
    receipts=[read(p) for p in (OUT/'fits').glob('*/receipt.json')];assert len(receipts)==8
    smoke=read(OUT/'SMOKE.json');assert len(smoke)==2
    assert all(r['updates']==2 and r['initial_B0_exact'] and r['frozen_unchanged'] and r['buffers_unchanged'] and r['fresh_restore_exact'] and r['gates_updated'] for r in smoke)
    resources=[];checks=0
    for r in receipts:
        z=[v for v in ledger if v['fit']==r['fit']];assert [v['step'] for v in z]==list(range(1,1025))
        assert r['frozen_unchanged'] and r['buffers_unchanged']
        initial=read(OUT/'fits'/r['fit']/'initial_parity.json')
        assert initial['exact'] and initial['B0_sha256']==parent.baseline_row(r['source'],r['seed'])['sha256']
        assert min(r['checkpoints'],key=lambda c:(c['objective'],c['step']))==r['selected']
        for c in r['checkpoints']:assert sha(ROOT/c['checkpoint'])==c['sha256'];checks+=1
        np.testing.assert_allclose(math.fsum(v['seconds'] for v in z),r['optimizer_seconds'],rtol=1e-10)
        resources.append({k:r[k] for k in ['fit','source','arm','seed','lr','trainable_parameters','optimizer_seconds','validation_seconds','peak_allocated','peak_reserved']})
    for source,arms in read(OUT/'LR_SELECTION.json').items():
        for arm,c in arms.items():
            z=[r for r in receipts if r['source']==source and r['arm']==arm and r['seed']==81550];assert len(z)==2
            expected=min(z,key=lambda r:(r['selected']['objective'],r['lr'],r['selected']['step']))
            assert expected['lr']==c['lr'] and expected['fit']==c['selection_fit']
    models=read(OUT/'MODEL_SELECTION.json');preds=read(OUT/'PREDICTIONS.json');assert len(preds)==224
    assert len(models)==8 and len({(r['source'],r['arm'],r['seed'],r['stage']) for r in models})==8
    for row in models:
        choice=read(OUT/'LR_SELECTION.json')[row['source']][row['arm']]
        receipt=read(OUT/'fits'/fit_id(row['source'],row['arm'],row['seed'],choice['lr'])/'receipt.json')
        expected=receipt['selected'] if row['stage']=='selected' else receipt['checkpoints'][-1]
        assert row['lr']==choice['lr'] and all(row[k]==expected[k] for k in ['step','checkpoint','sha256','objective'])
    seal=read(OUT/'EVALUATION_SEAL.json');marker=read(OUT/'ALL_PREDICTIONS_SAVED.json')
    assert seal['selection_sha256']==sha(OUT/'MODEL_SELECTION.json') and marker['manifest_sha256']==sha(OUT/'PREDICTIONS.json') and seal['at']<marker['at']
    origin=pd.read_csv(OUT/'ORIGIN_SCORES.csv.gz');raw=pd.read_csv(OUT/'RAW_SCORES.csv');effects=pd.read_csv(OUT/'EFFECTS.csv')
    keys=['panel','kind','arm','seed','stage','step','condition']
    repeated=origin.groupby(keys,as_index=False)[['nmae','mae','pinball']].mean().merge(raw,on=keys,suffixes=('_a','_b'),validate='one_to_one')
    assert len(repeated)==len(raw)
    for col in ['nmae','mae','pinball']:np.testing.assert_allclose(repeated[col+'_a'],repeated[col+'_b'],rtol=1e-10,atol=1e-11)
    # Independent vector calculation: no calls to the scorer's metric_arrays.
    metrics_checked=0;crossing=[]
    for key,r in preds.items():
        assert sha(ROOT/r['path'])==r['sha256'];panel,kind=r['panel'],r['kind']
        packet=np.load(data_path(panel)/'E_DISCOVERY_inputs.npz');truth=np.load(data_path(panel)/'E_DISCOVERY_labels.npz')['y'];nc=truth.shape[1]
        if kind=='standard':names=STATES;ids=np.arange(len(packet['origins']));offset=np.load(panel_path(panel,kind)/'E_DISCOVERY_offset.npy')
        else:
            m=read(panel_path(panel,kind)/'manifest.json');names=m['states'];ids=np.array(m['origin_indices']);offset=np.load(panel_path(panel,kind)/'offset.npy')
        n=len(ids);pred=np.load(ROOT/r['path'],mmap_mode='r');assert tuple(pred.shape)==(len(names)*2*n*nc,9,64)
        q=np.arange(1,10)/10;out=[];cross_count=0
        for ci,name in enumerate(names):
            lo=ci*2*n*nc;hi=lo+2*n*nc;p=np.asarray(pred[lo:hi],float)
            y=np.tile(truth[ids].reshape(-1,64),(2,1))+offset[lo:hi,None]
            sigma=np.tile(packet['sigma'],2*n);mae=np.abs(p[:,4]-y).mean(1);nmae=mae/sigma
            err=y[:,None,:]-p;pin=(2*np.maximum(q[None,:,None]*err,(q[None,:,None]-1)*err)).mean((1,2))/sigma
            val=np.stack([nmae,mae,pin],-1).reshape(2,n,nc,3).mean((0,2))
            saved=origin[(origin.panel==panel)&(origin.kind==kind)&(origin.arm==r['arm'])&(origin.seed==r['seed'])&(origin.stage==r['stage'])&(origin.condition==name)].set_index('origin').loc[packet['origins'][ids]]
            np.testing.assert_allclose(val,saved[['nmae','mae','pinball']].to_numpy(),rtol=1e-10,atol=1e-10);metrics_checked+=val.size
            cross_count+=int((p[:,1:]<p[:,:-1]).sum())
        crossing.append(dict(view=key,crossing_pairs=cross_count,total_pairs=len(pred)*8*64,fraction=cross_count/(len(pred)*8*64)))
        if r['arm'] in ARMS:
            model=next(v for v in models if all(v[k]==r[k] for k in ['source','arm','seed','stage']))
            assert model['sha256']==r['checkpoint_sha256'] and model['B0_sha256']==r['B0_sha256']
    old=read(OUT/'REUSED_PREDICTIONS.json')
    for k,v in old.items():assert preds[k]==v
    published=pd.read_csv(ROOT/'results/learned_gate_comparison_20260919/RAW_SCORES.csv')
    z=raw[~raw.arm.isin(ARMS)].merge(published,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(z)==len(published)
    for metric in ['nmae','mae','pinball']:np.testing.assert_allclose(z[metric+'_new'],z[metric+'_old'],rtol=1e-10,atol=1e-11)
    # Independent time-block interval replay, fixed two seeds and channels.
    for (panel,kind,stage,condition),g in origin.groupby(['panel','kind','stage','condition']):
        a=g.groupby(['origin','arm']).nmae.mean().unstack();period=96 if panel=='ettm1' else 24;blocks=a.index.to_numpy()//(7*period);unique=np.unique(blocks);random=np.random.default_rng(91942)
        counts=np.array([np.bincount(random.integers(0,len(unique),len(unique)),minlength=len(unique)) for _ in range(2000)])
        weight=counts[:,np.searchsorted(unique,blocks)];denom=weight.sum(1)
        ee=effects[(effects.panel==panel)&(effects.kind==kind)&(effects.stage==stage)&(effects.condition==condition)]
        for e in ee.itertuples():
            av=a[e.proposed].to_numpy();bv=a[e.baseline].to_numpy();boot=100*(1-(weight@av)/(weight@bv));bounds=np.quantile(boot,[.025,.975,.0125,.9875])
            np.testing.assert_allclose([100*(1-av.mean()/bv.mean()),*bounds],[e.gain_pct,e.ci_low,e.ci_high,e.bonf2_low,e.bonf2_high],rtol=1e-8,atol=1e-9)
    gpu=[json.loads(x) for x in (OUT/'gpu_petsa.jsonl').read_text().splitlines()]
    unapproved=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in v['apps']) for v in gpu);assert unapproved==0
    pd.DataFrame(resources).to_csv(OUT/'RESOURCES.csv',index=False)
    pd.DataFrame(crossing).to_csv(OUT/'QUANTILE_CROSSING.csv',index=False)
    primary=effects[effects.primary_family];assert len(primary)==2
    primary.to_csv(OUT/'PRIMARY_COMPARISONS.csv',index=False)
    protection=effects[(effects.proposed=='MAG_ONLY')&effects.panel.isin(['electricity_transfer',NEW])&(effects.stage=='selected')&(effects.kind=='standard')&effects.condition.isin(['REFERENCE','FAULT'])]
    assert len(protection)==12
    supported=bool((primary.bonf2_low>0).all() and all(all(v>0 for v in json.loads(s).values()) for s in primary.seed_gains) and (protection.gain_pct>=-1).all())
    decision='LIMITED_ADDITIONAL_BASELINE_SUPPORT' if supported else 'ADDITIONAL_BASELINE_SUPPORT_NOT_ESTABLISHED'
    summary=raw[(raw.stage=='selected')&(raw.kind=='standard')&raw.condition.isin(['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT'])].groupby(['panel','condition','arm'],as_index=False).nmae.mean().pivot(index=['panel','condition'],columns='arm',values='nmae').reset_index()
    arms=['B0','PLAIN','C3','MAG_ONLY','TOKEN_GATE','TOKEN_GATE_ENTROPY']+ARMS
    fig,axes=plt.subplots(1,4,figsize=(17,4))
    for ax,panel in zip(axes,['electricity','electricity_transfer','ettm1',NEW]):
        z=raw[(raw.panel==panel)&(raw.stage=='selected')&(raw.kind=='standard')&(raw.condition=='SHIFT8')].groupby('arm').nmae.mean().reindex(arms)
        ax.bar(range(7),z.to_numpy());ax.set_xticks(range(7),['B0','Plain','C3','MAG','Gate','Gate+H','PETSA cell'],rotation=45,ha='right');ax.set_title(panel);ax.set_ylabel('SHIFT8 nMAE')
    fig.suptitle('Offline controlled component comparison; reused development panels');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=170);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    resource=pd.DataFrame(resources).groupby('arm',as_index=False)[['trainable_parameters','optimizer_seconds','peak_allocated']].mean();resource['peak_MiB']=resource.pop('peak_allocated')/2**20
    report=f"""# PETSA 공개 보정 부품의 offline 직접 대조

실행·검산 완료. 판정: {decision}. 공식 PETSA 전체 재현이나 논문 PASS를 뜻하지 않는다.

## 실제 실행과 검산

8/8 새 fits, 8192 main +4 smoke updates. 기존192개 prediction views를 hash 재사용하고 새32개를 추가해 총224개를 검산했다. 미실행 승인 학습0. 모델 선택 봉인→새 전체 예측 저장→새 E 채점 순서, {checks}개 checkpoint, 원점별 독립 metric {metrics_checked}개, 기존 점수 {len(published)}행의 재현과 bootstrap 구간 재계산을 검사했다. 실제 초기 B0 동일성, 두 cell의 update, 동결 보존, fresh restore를 확인했다. 미승인 외부 GPU compute 표본 {unapproved}개. raw/weights/predictions는 로컬 캐시이며 공개 저장소만으로 모든 수치가 재생된다고 하지 않는다.

## 방법과 비교 범위

공식 PETSA GCM의 rank16·초기 gate .01·var-wise 수식을 이식했다. 관측 mean/기존 TRAIN sigma 좌표에서 입력·출력 correction을 더하며 출력 cell은9개 quantile에 공유한다. 이 좌표계와 probabilistic output 연결은 로컬 이식 선택이다. 공식 온라인 partial/delayed-label 적응과 복합 loss는 재현하지 않았다. 같은 B0·TRAIN·두LR·선택seed·두 반복seed·1024updates·V nMAE 조건으로 normalized2pinball을 학습했다. MAG는 변경하거나 재튜닝하지 않았다. PETSA cell19010 vsMAG8712 추가params이며 예산이 같은 adapter 대조라고 부르지 않는다. 기존 B0의 LoRA294912와 학습 비용도 따로 존재한다.

## 주 비교와 손해

{table(primary[['panel','baseline','proposed_nmae','baseline_nmae','gain_pct','bonf2_low','bonf2_high','seed_gains']])}

양수는 MAG 이득이다. 두 primary 비교에 한한 Bonferroni family2 구간이며, 고정 두seed/채널에 조건부인 index7일 block bootstrap2000회다. 역사 전체의 후보 탐색이나 seed 모집단의 불확실성을 보정하지 않는다. 제한된 추가 대조 기준 충족: {supported}. REFERENCE/FAULT의1% 손해 한도는 평균점수 기준이지 신뢰구간 비열등성 보장이 아니다. 기존 학습형 gate 비교의 family4와 이번 family2를 합쳐 하나의 사전 확증이라고 주장하지 않는다.

{table(protection[['panel','condition','baseline','gain_pct','ci_low','ci_high']])}

## 원점수와 비용

{table(summary)}

모든seed 원점수는 RAW_SCORES.csv, 날짜별 점수는 ORIGIN_SCORES.csv.gz, 채널별 점수는 CHANNEL_SCORES.csv다. selected/fixed1024, 모든오류·변화형태·ETTm1을 보존했다. QUANTILE_CROSSING.csv에 crossing 빈도를 남기며 출력 정렬로 결과를 바꾸지 않았다.

{table(resource)}

optimizer_seconds는 intent 저장 시간을 포함한다. 과거 MAG의 순수 계산 측정과 다르므로 속도비를 주장하지 않는다. 같은 관측 권한은 동일한 특징 표현·개입 위치를 의미하지 않는다.

## 논문 해석과 종료

모든 E 패널은 이번 대조를 설계하기 전에 이미 채점됐다. NESO 후반 기간도 이제 재사용 개발 평가이며 새 독립 검증이라고 부르지 않는다. 긍정 결과는 가까운 공개 보정 부품보다 특정조건에서 유리하다는 근거를 더할 뿐, 공식 PETSA/Time-PEFT 전체 우위나 gating 최초성을 입증하지 않는다. 부정 결과는 그 추가 근거가 확보되지 않았다는 뜻이며 기존 완료 실험의 좁은 양성 결과를 삭제하지 않는다. C3 지속성 규칙의 성공으로 옮기지 않는다. 자동 후속 학습0.
"""
    (OUT/'REPORT.md').write_text(report)
    (OUT/'FINAL_DECISION.md').write_text(f"# 최종 판단\n\n{decision}. 사전 제한된 추가 대조 기준 충족: {supported}.\n\n기존 MAG/B0와 모든 부정 조건을 보존한다. 정식 PETSA 전체 비교·독립 source 검증·충분한 신규성은 이 비교로 완료되지 않는다. 추가 LR/seed/rank/후속 후보를 시작하지 않는다.\n")
    save(OUT/'AUDIT.json',dict(status='VERIFIED',new_fits=8,main_updates=MAIN_CAP,smoke_updates=SMOKE_CAP,checkpoints=checks,independent_origin_metrics=metrics_checked,reused_score_rows=len(published),prediction_views=224,unique_prediction_files=len({r['path'] for r in preds.values()}),primary_family_size=2,limited_additional_baseline_support=supported,decision=decision,paper_pass=False,automatic_successor=False,unapproved_gpu_samples=unapproved))
