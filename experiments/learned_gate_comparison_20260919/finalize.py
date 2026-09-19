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
    receipts=[read(p) for p in (OUT/'fits').glob('*/receipt.json')];assert len(receipts)==16
    resources=[];gates=[];checks=0
    for r in receipts:
        z=[v for v in ledger if v['fit']==r['fit']];assert [v['step'] for v in z]==list(range(1,1025))
        assert r['frozen_unchanged'] and r['buffers_unchanged']
        assert min(r['checkpoints'],key=lambda c:(c['objective'],c['step']))==r['selected']
        for c in r['checkpoints']:assert sha(ROOT/c['checkpoint'])==c['sha256'];checks+=1
        coefficient=.01 if r['arm']=='TOKEN_GATE_ENTROPY' else 0.
        np.testing.assert_allclose([v['forecast_loss']+coefficient*v['entropy'] for v in z],[v['loss'] for v in z],rtol=2e-7,atol=2e-7)
        assert all(0<=v['entropy']<=1.000001 and 0<=v['gate_mean']<=1 for v in z)
        np.testing.assert_allclose(math.fsum(v['seconds'] for v in z),r['optimizer_seconds'],rtol=1e-10)
        resources.append({k:r[k] for k in ['fit','source','arm','seed','lr','trainable_parameters','optimizer_seconds','validation_seconds','peak_allocated','peak_reserved']})
        gates.append(dict(fit=r['fit'],source=r['source'],arm=r['arm'],seed=r['seed'],selected_step=r['selected']['step'],mean_last_epoch_gate=np.mean([v['gate_mean'] for v in z[-32:]]),mean_last_epoch_entropy=np.mean([v['entropy'] for v in z[-32:]])))
    for source,arms in read(OUT/'LR_SELECTION.json').items():
        for arm,c in arms.items():
            z=[r for r in receipts if r['source']==source and r['arm']==arm and r['seed']==81550];assert len(z)==2
            expected=min(z,key=lambda r:(r['selected']['objective'],r['lr'],r['selected']['step']))
            assert expected['lr']==c['lr'] and expected['fit']==c['selection_fit']
    models=read(OUT/'MODEL_SELECTION.json');preds=read(OUT/'PREDICTIONS.json');assert len(preds)==192
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
        if r['arm'] in ARMS or panel==NEW:
            model=next(v for v in models if all(v[k]==r[k] for k in ['source','arm','seed','stage']))
            assert model['sha256']==r['checkpoint_sha256'] and model['B0_sha256']==r['B0_sha256']
    old=read(OUT/'REUSED_PREDICTIONS.json')
    for k,v in old.items():assert preds[k]==v
    published=pd.read_csv(ROOT/'results/delta_adapter_comparison_20260919/RAW_SCORES.csv')
    z=raw[(raw.panel!=NEW)&raw.arm.isin(['B0','PLAIN','C3','MAG_ONLY'])].merge(published,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(z)==960
    for metric in ['nmae','mae','pinball']:np.testing.assert_allclose(z[metric+'_new'],z[metric+'_old'],rtol=1e-10,atol=1e-11)
    # Independent time-block interval replay, fixed two seeds and channels.
    for (panel,kind,stage,condition),g in origin.groupby(['panel','kind','stage','condition']):
        a=g.groupby(['origin','arm']).nmae.mean().unstack();period=96 if panel=='ettm1' else 24;blocks=a.index.to_numpy()//(7*period);unique=np.unique(blocks);random=np.random.default_rng(91942)
        counts=np.array([np.bincount(random.integers(0,len(unique),len(unique)),minlength=len(unique)) for _ in range(2000)])
        weight=counts[:,np.searchsorted(unique,blocks)];denom=weight.sum(1)
        ee=effects[(effects.panel==panel)&(effects.kind==kind)&(effects.stage==stage)&(effects.condition==condition)]
        for e in ee.itertuples():
            av=a[e.proposed].to_numpy();bv=a[e.baseline].to_numpy();boot=100*(1-(weight@av)/(weight@bv));bounds=np.quantile(boot,[.025,.975,.00625,.99375])
            np.testing.assert_allclose([100*(1-av.mean()/bv.mean()),*bounds],[e.gain_pct,e.ci_low,e.ci_high,e.bonf4_low,e.bonf4_high],rtol=1e-8,atol=1e-9)
    gpu=[json.loads(x) for x in (OUT/'gpu_gate.jsonl').read_text().splitlines()]
    unapproved=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in v['apps']) for v in gpu);assert unapproved==0
    pd.DataFrame(resources).to_csv(OUT/'RESOURCES.csv',index=False);pd.DataFrame(gates).to_csv(OUT/'GATE_TRAINING_SUMMARY.csv',index=False);pd.DataFrame(crossing).to_csv(OUT/'QUANTILE_CROSSING.csv',index=False)
    primary=effects[effects.primary_family];assert len(primary)==4
    primary.to_csv(OUT/'PRIMARY_COMPARISONS.csv',index=False)
    protection=effects[(effects.proposed=='MAG_ONLY')&effects.panel.isin(['electricity_transfer',NEW])&(effects.stage=='selected')&(effects.kind=='standard')&effects.condition.isin(['REFERENCE','FAULT'])&effects.baseline.isin(['B0','PLAIN'])]
    assert len(protection)==8
    supported=bool((primary.bonf4_low>0).all() and all(all(v>0 for v in json.loads(s).values()) for s in primary.seed_gains) and (protection.gain_pct>=-1).all())
    decision='LIMITED_COMPONENT_EVIDENCE' if supported else 'METHOD_CLAIM_NOT_ESTABLISHED'
    summary=raw[(raw.stage=='selected')&(raw.kind=='standard')&raw.condition.isin(['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT'])].groupby(['panel','condition','arm'],as_index=False).nmae.mean().pivot(index=['panel','condition'],columns='arm',values='nmae').reset_index()
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    arms=['B0','PLAIN','C3','MAG_ONLY']+ARMS
    for ax,panel in zip(axes,['electricity','electricity_transfer','ettm1',NEW]):
        z=raw[(raw.panel==panel)&(raw.stage=='selected')&(raw.kind=='standard')&(raw.condition=='SHIFT8')].groupby('arm').nmae.mean().reindex(arms)
        ax.bar(range(6),z.to_numpy());ax.set_xticks(range(6),['B0','Plain','C3','MAG','Gate','Gate+H'],rotation=40,ha='right');ax.set_title(panel);ax.set_ylabel('SHIFT8 nMAE')
    fig.suptitle('Fixed B0, two matched repeat seeds; lower is better');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=170);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,panel in zip(axes,['electricity_transfer',NEW]):
        for arm in arms:
            z=raw[(raw.panel==panel)&(raw.stage=='selected')&(raw.kind=='standard')&(raw.arm==arm)].groupby('condition').nmae.mean()
            ax.scatter(z['FAULT'],z['SHIFT8'],label=arm);ax.annotate(arm,(z['FAULT'],z['SHIFT8']),fontsize=7)
        ax.set_title(panel);ax.set_xlabel('FAULT nMAE');ax.set_ylabel('SHIFT8 nMAE')
    fig.tight_layout();fig.savefig(OUT/'tradeoffs.png',dpi=170);fig.savefig(OUT/'tradeoffs.pdf');plt.close(fig)
    resource=pd.DataFrame(resources).groupby('arm',as_index=False)[['trainable_parameters','optimizer_seconds','peak_allocated']].mean();resource['peak_MiB']=resource.pop('peak_allocated')/2**20
    report=f"""# 고정 MAG의 학습형 gate 대조 및 시간 전이

**실행·선택·192개 예측 view 저장·채점·독립 검산 완료. 판정: {decision}. 논문 PASS를 선언하지 않는다.**

## 실행과 검증

16/16 새 본학습,16384main+8smokeupdates. 미실행 승인 학습0. 기존96views 재사용, 새96views(동일 checkpoint alias 포함). 선택봉인→전체예측저장→E채점 순서를 검증했다. 실제 Chronos 초기 B0/adapter off 동일성, gate update, 동결 parameters/buffers 보존, fresh restore 검사 통과. checkpoint{checks}개 SHA, 모든원점의 독립 계산{metrics_checked}개, 기존 원점수960행, 효과/구간 재계산을 확인했다. GPU 미승인외부compute {unapproved}, 최소여유{min(v['free_mib'] for v in gpu):.0f}MiB. RustDesk만 예외.

기존 B0/PLAIN/C3/MAG는 변경하지 않았다. 두 학습형 gate는 같은 잔차에 부착한 원리 통제이며 GateRA 공식 HiRA/NLP 전체의 재현이 아니다. entropy .01은 사전에 고정한 로컬값이다. 새gate9225 vsMAG8712(+513,5.89%). 초기 gate=.5, up=0; 학습초기 B0동일. 모든 군의 TRAIN/labels/V목적/LR/seed/1024updates 기회를 맞췄다. regularizer포함학습loss와 forecast loss는 UPDATE_LEDGER에 분리했다. gate사용량 평균은 실제 관측학습batch 기준이며 개입 인과기여를 뜻하지 않는다.

## 주 비교 — 양수는 MAG 이득

{table(primary[['panel','baseline','proposed_nmae','baseline_nmae','gain_pct','bonf4_low','bonf4_high','seed_gains']])}

family4는 두패널×두학습gate의selected SHIFT8만이다. index7일block2000회, 두seed/고정채널에 조건부다. 예측horizon이겹치므로56개/55개 독립표본이라고하지않는다. 짧은새기간/약8block/두seed의불확실성과기존E를본후MAG선정은남는다. 이보정은역사전체탐색선택을제거하지않는다. 사전문턱(네구간하한>0,모든seed양수,기존B0/PLAIN대비REFERENCE/FAULT악화≤1%) 충족: {supported}.

## 원점수·부정 조건

{table(summary)}

두 seed별 원점수는 RAW_SCORES.csv, 모든날짜/형태는 ORIGIN_SCORES.csv.gz, 채널은 CHANNEL_SCORES.csv다. fixed1024와selected를모두남겼다. 모든9shape와불리한ETTm1조건을제외하지않았다. C3의 지속성규칙과MAG의진폭마스크를혼동하지않는다. 일반adapter이득=B0대PLAIN,고정MAG추가가치=PLAIN대MAG,지속성추가가치=MAG대C3다. 두학습gate가좋다고모든강건PEFT가반증된것이아니며 MAG가좋아도gate일반론자체는기존방법이다.

## 시간 전이와 노출

새NESO origin은2026-07-01이후완전한55개UTC날짜로평가전에고정했다. 마지막부분날짜는사전제외했다. 표본수128/64를중복날짜로채우지않았다. 기존H1 target과불겹침,512문맥은합법적과거로겹칠수있음. 2025H1 sigma고정. 파일전체는이전다운로드됨. 제한된기록상새평가기간이며같은provider/ND계열이므로독립source라고하지않는다. 실제사건레이블없음:합성fault/shift만검증했다. 이기간손해를전세계센서문제실패로일반화하지않는다.

## 비용·한계

{table(resource)}

시간은이번측정의평균optimizer초,최대메모리평균이다. 전체검증/추론비용은별도receipt에있다. 기존MAG와다른시각에측정했으므로속도우위를확정하지않는다. gate513개/entropy계산추가비용과예측변화를분리한다.

기존δ/POS_ONLY직접비교는완료보고서를참조했으며재학습하지않았다. 정식GateRA전체/Time-PEFT동일backbone공정재현,다양한독립source,실제오류·변화레이블은여전히없다. 논문기여는관측입력의고정진폭gate가동결B0 위의잔차적응을어떤조건에서개선/제한하는지에국한해야한다. sigmoidgate자체신규성·범용PEFT우위·C3지속성기전성공을주장할수없다.

새후속학습은없다. 모델weights/raw/predictionarrays는localcache이며GitHub에는코드·해시·점수·검산·그림을게시한다. GitHub만으로완전수치재생가능하다고하지않는다.
"""
    (OUT/'REPORT.md').write_text(report)
    (OUT/'FINAL_DECISION.md').write_text(f"""# 최종 결정

{decision}. 실행 완료와 논문 성공을 구분한다. 고정된 사전 추가 가치 문턱 충족: {supported}.

- 남길 구현: 변경 없는 MAG와 B0/PLAIN 및 두 학습형gate 통제·해시 검산 코드.
- 중단: C3지속성 규칙을 되살리기 위한 튜닝, 새 gate/LR/seed/λ/rank/dataset 탐색. 자동후속0.
- 이번 데이터가 지지하는 주장은 REPORT의 패널·조건·seed·구간에 한정한다. 새시간기간의 결과와 기존개발기간의 결과를 합쳐 독립반복이라고 하지 않는다.
- 제한된추가가치문턱을넘어도공식선행전체비교·독립source·실제사건평가는완료되지않았으며논문PASS/범용PEFT우위를선언하지않는다. 문턱미충족이면 MAG방법론 주장은 확립되지 않은 상태로 종료한다.
""")
    save(OUT/'AUDIT.json',dict(status='VERIFIED',new_fits=16,main_updates=MAIN_CAP,smoke_updates=SMOKE_CAP,checkpoints=checks,independent_origin_metrics=metrics_checked,reused_score_rows=960,prediction_views=192,unique_prediction_files=len({r['path'] for r in preds.values()}),primary_family_size=4,prespecified_component_gate_met=supported,decision=decision,paper_pass=False,automatic_successor=False,unapproved_gpu_samples=unapproved))
    print('FINALIZED',decision,flush=True)
