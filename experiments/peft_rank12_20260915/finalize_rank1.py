"""Independent stored-tensor/metric verification and Korean report; no fitting."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from tsfm_peft_screen.reproducibility import ROOT,sha,write_json,digest
from tsfm_peft_screen.metrics import independent,score
from tsfm_peft_screen.forecast_query.budget import seal_valid
from numerics import check_parity
RUN='query_budget_repair_v2_20260915';OUT=ROOT/'results'/RUN;CACHE=ROOT/'.cache'/RUN

def read(p):return json.loads(Path(p).read_text())
def save(n,v):write_json(OUT/n,v)
def csvwrite(n,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with open(OUT/n,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)
def gain(b,q):return 100*(b-q)/b if b else None


def verify_numeric(c):
    from run_rank1 import verify_numerical
    verify_numerical()


def finalize():
    from rank1_engine import check_contract
    c=check_contract();s=read(OUT/'status.json');verify_numeric(c);selection=read(OUT/'resource_selection.json') if (OUT/'resource_selection.json').exists() else {}
    for p,h in c['historical_hashes'].items():assert sha(ROOT/p)==h,p
    for d in c['data'].values():
        for p,h in d['raw_sources'].items():assert sha(p)==h
        for p,h in d['processed_sources'].items():assert sha(ROOT/p)==h
    for p,h in c['model_files'].items():assert sha(p)==h
    artifacts=read(OUT/'artifact_index.json') if (OUT/'artifact_index.json').exists() else {}
    for a in artifacts.values():assert sha(ROOT/a['path'])==a['sha256']
    parity=read(OUT/'parity.json') if (OUT/'parity.json').exists() else [];checks=0
    for r in parity:
        key=r['option']+('_fp32' if r['precision']=='fp32' else '_bf16_'+str(r['update']))
        a=torch.load(ROOT/artifacts[key]['path'],weights_only=False,map_location='cpu');b=torch.load(ROOT/artifacts[r['reference']]['path'],weights_only=False,map_location='cpu')
        dataset=r['option'].split('_')[0];p=check_parity(b,a,r['precision'],micro=r['kind']=='microbatch',scale=c['data'][dataset]['scale'])
        assert p=={k:r[k] for k in p},r['option'];checks+=1;del a,b
    fitrows=[];trajectory=[];metrics=[];comparisons=[];perorigin=[];vcount=0;ecount=0;maxerror=0
    if (OUT/'fits.json').exists():
        fits=read(OUT/'fits.json')
        for f in fits:fitrows.append({k:v for k,v in f.items() if k not in ['choice','best']})
        assert len(fits)==s['B_fit_attempts']<=12
    else:fits=[]
    if (OUT/'trajectories.json').exists():
        for r in read(OUT/'trajectories.json'):
            assert sha(ROOT/r['checkpoint_file'])==r['checkpoint_hash']
            assert sha(ROOT/r['prediction_file'])==r['prediction_hash']
            with np.load(ROOT/r['prediction_file']) as z:
                val=independent(z['prediction'],z['target'],z['scale']);cached=score(z['prediction'],z['target'],z['scale'])
            assert cached==r['metrics'];err=abs(val-r['metrics']['scaled_2pinball']);assert err<=1e-10;maxerror=max(maxerror,err);vcount+=1
            trajectory.append(dict(**{k:v for k,v in r.items() if k!='metrics'},**r['metrics']))
    arrays={}
    if s['status']=='B_COMPLETE':
        choices=read(OUT/'selection_seal.json');expected=[(d,z,a) for d in ['ettm2','electricity'] for z in [39000,39001] for a in ['standard','side','query']]
        assert seal_valid(choices,sha(OUT/'contract.json'),expected)
        access=read(OUT/'evaluation_access.json');assert access['selection_hash']==sha(OUT/'selection_seal.json') and access['opened_at']>=choices['sealed_at']
        assert len(read(OUT/'checkpoint_reloads.json'))==12 and all(r['error']==0 for r in read(OUT/'checkpoint_reloads.json'))
        for chosen in choices['selections']:
            candidates=[r for r in trajectory if r['fit']==chosen['fit']]
            best=min(candidates,key=lambda r:(r['scaled_2pinball'],r['nominal_seconds']))
            assert best['checkpoint_hash']==chosen['checkpoint_hash']
        evaluation=read(OUT/'evaluation.json');assert len(evaluation)==26
        for r in evaluation:
            assert sha(ROOT/r['prediction_file'])==r['prediction_hash']
            with np.load(ROOT/r['prediction_file']) as z:p=z['prediction'];y=z['target'];sc=z['scale'];origins=z['origins']
            val=independent(p,y,sc);err=abs(val-r['metrics']['scaled_2pinball']);assert err<=1e-10;maxerror=max(maxerror,err);ecount+=1
            row=dict(dataset=r['dataset'],seed=r['seed'],arm=r['arm'],role=r['role'],**r['metrics']);metrics.append(row)
            values=[]
            for j,o in enumerate(origins):
                value=independent(p[j:j+1],y[j:j+1],sc);values.append(value);perorigin.append(dict(dataset=r['dataset'],seed=r['seed'],arm=r['arm'],role=r['role'],origin=int(o),primary=value))
            arrays[(r['dataset'],r['seed'],r['arm'],r['role'])]=np.array(values)
        indices=np.random.default_rng(59100).integers(0,4,(1000,4));signal=[]
        for d in ['ettm2','electricity']:
            query=np.mean([arrays[(d,z,'query','selected')] for z in [39000,39001]],axis=0)
            f0=arrays[(d,None,'F0','F0')]
            for a in ['standard','side','F0']:
                b=f0 if a=='F0' else np.mean([arrays[(d,z,a,'selected')] for z in [39000,39001]],axis=0)
                qb=query.reshape(4,4).mean(1);bb=b.reshape(4,4).mean(1)
                sample=100*(bb[indices].mean(1)-qb[indices].mean(1))/bb[indices].mean(1)
                comparisons.append(dict(dataset=d,seed='mean',baseline=a,baseline_primary=float(b.mean()),query_primary=float(query.mean()),gain_percent=gain(float(b.mean()),float(query.mean())),CI95_lower=float(np.quantile(sample,.025)),CI95_upper=float(np.quantile(sample,.975))))
                for z in [39000,39001]:
                    b0=f0 if a=='F0' else arrays[(d,z,a,'selected')];q=arrays[(d,z,'query','selected')]
                    comparisons.append(dict(dataset=d,seed=z,baseline=a,baseline_primary=float(b0.mean()),query_primary=float(q.mean()),gain_percent=gain(float(b0.mean()),float(q.mean()))))
            flag=all(next(r['gain_percent'] for r in comparisons if r['dataset']==d and r['seed']=='mean' and r['baseline']==a)>0 for a in ['standard','side']) and all(next(r['gain_percent'] for r in comparisons if r['dataset']==d and r['seed']==z and r['baseline']=='standard')>0 for z in [39000,39001])
            signal.append(dict(dataset=d,prediction_signal=flag))
            for row in metrics:
                if row['dataset']==d:
                    row['loss_over_F0']=row['scaled_2pinball']/float(f0.mean())
                    if row['role']=='selected':row['gain_vs_own_initial_percent']=gain(float(arrays[(d,row['seed'],row['arm'],'initial')].mean()),row['scaled_2pinball'])
        decision='예측 개선 신호' if any(r['prediction_signal'] for r in signal) else '품질·자원 tradeoff 또는 추가 가치 미확보'
        save('forecast_decision.json',dict(decision=decision,source_signals=signal,auto_followup=False))
    else:decision='예측 비교 미실행; '+s['status']
    failures=[r for r in parity if not r['passed']]
    save('independent_verification.json',dict(status='VERIFIED_RECORDED_OUTCOME',historical_files_unchanged=len(c['historical_hashes']),parity_artifact_checks=checks,
        parity_failed=len(failures),V_prediction_replays=vcount,E_prediction_replays=ecount,metric_max_abs_error=maxerror,
        model_data_source_hashes_unchanged=True,new_fits_in_verification=0,A_updates=s['A_updates'],B_updates=s['B_updates'],B_fits=s['B_fits_completed'],
        caveat='Verification confirms the recorded evidence, including failed parity; it does not relabel numerical failure as scientific PASS.'))
    csvwrite('fit_attempts.csv',[dict(phase='Q_numeric_disposable',attempts=s['numeric_update_attempts'],optimizer_updates=s['numeric_updates'],forecasting_fits=0),dict(phase='A_disposable',attempts=s['A_update_attempts'],optimizer_updates=s['A_updates'],forecasting_fits=0)]+[dict(phase='B',**r) for r in fitrows])
    csvwrite('trajectories.csv',trajectory);csvwrite('metrics.csv',metrics);csvwrite('comparisons.csv',comparisons);csvwrite('per_origin_metrics.csv',perorigin)
    resources=[dict(phase=n,**read(OUT/(n+'_wall.json'))) for n in ['A','B'] if (OUT/(n+'_wall.json')).exists()]
    csvwrite('resources.csv',resources)
    # Preserve every measured stage even if integrity stops before timing blocks.
    if (OUT/'A_memory_records.json').exists():
        memory=read(OUT/'A_memory_records.json');csvwrite('resource_all_executions.csv',[{k:v for k,v in r.items() if k!='phase_peaks'} for r in memory])
    lines=['# Query 자원 제약 파일럿 — simulated tensor budget / 기존 평가 구간을 재사용한 개발 비교','',
        f"Q 수치 진단: 폐기용 {s['numeric_updates']} updates. A: 폐기용 optimizer updates {s['A_updates']}회 ({s['A_update_attempts']} attempts). B: {s['B_fits_completed']}/{s['B_fit_attempts']} fits 완료, {s['B_updates']} updates. 종료 상태: **{s['status']}**.",'',
        '## 1. 비교 목적','',
        '동일 context4096, 4채널 origin 그룹, 1,179,648 학습 파라미터에서 Standard/Side/Query의 자원 제약을 비교했다. Standard에는 CP0/3/6/9/12와 origin microbatch1/2를 허용했다. Censor 재튜닝이나 새 어댑터 개발은 하지 않았으며 과거 판정은 보존한다. Query/Side의 frozen encode/cache를 매 forward 비용에 포함한다.','',
        '## 2. 예산과 옵션','',
        '1/2/4/8GiB는 실제 장비 요구가 주어지지 않아 사전에 고정한 tensor allocated 예산 시뮬레이션이다. allocated, reserved, NVML 및 실제 free VRAM은 다른 양이다. 1GiB GPU에서 실행된다는 뜻이 아니다. [전체 budget 표](resource_budget_table.csv), [timing 반복](resource_measurements.csv), [준비·cold/warm/FP32 포함 모든 실행](resource_all_executions.csv)을 구분한다.','',
        '## 3. 수치 무결성과 Standard 대조','',
        f'저장 artifact로 독립 재계산한 parity {checks}개 중 실패 {len(failures)}개. 실패 옵션을 제외해서 Standard를 불가능으로 분류하지 않는다. 수치 무결성을 확보하지 못하면 이번 조합 전체의 A가 판정 불가다.','']
    if failures:
        lines+=['| 옵션 | 검사 | precision | 실패한 주요 양 |','| --- | --- | --- | --- |']
        for r in failures:
            bad={k:v for k,v in r['metrics'].items() if v['relative_l2'] is not None and (v['relative_l2']>(.01 if r['precision']=='bf16' and r['kind']=='microbatch' else 1e-5 if r['precision']=='fp32' else 1e-4) or v['max_absolute']>(1e-5 if r['precision']=='fp32' else .02 if r['kind']=='microbatch' else 1e-4))}
            lines.append(f"| {r['option']} | {r['kind']} | {r['precision']} | {json.dumps(bad)} |")
    if selection.get('choices'):
        lines+=['',f"Block1이 정한 B*: {selection.get('budget_gib')}GiB.",'','| 원천 | arm | CP | micro origins | block1 ms | allocated MiB |','| --- | --- | --- | --- | --- | --- |']
        for r in selection['choices']:lines.append(f"| {r['dataset']} | {r['arm']} | {r['cp']} | {r['micro']} | {r['block1_seconds']*1000:.3f} | {r['peak_allocated']/2**20:.3f} |")
        lines+=['', '선택 뒤 재확인: '+json.dumps(selection.get('speed_validation',[]),ensure_ascii=False)]
    lines+=['','## 4. B 실행 여부와 예측 결과','',decision+'.','']
    if comparisons:
        lines+=['| 원천 | baseline | baseline 원점수 | Query 원점수 | 개선율 % | 기술적 95% CI % |','| --- | --- | --- | --- | --- | --- |']
        for r in comparisons:
            if r['seed']=='mean':lines.append(f"| {r['dataset']} | {r['baseline']} | {r['baseline_primary']:.9f} | {r['query_primary']:.9f} | {r['gain_percent']:+.6f} | [{r['CI95_lower']:+.6f}, {r['CI95_upper']:+.6f}] |")
    else:lines+=['새 E 점수와 최강 예측 대조군은 측정하지 않았다. 과거 점수를 이번 자원 설정의 새 정확도로 가져오지 않는다.']
    lines+=['','## 5. 원점수·반복·비용','',
        '[metrics.csv](metrics.csv)는 새로 평가한 원점수만 담는다. [fit_attempts.csv](fit_attempts.csv), [resources.csv](resources.csv), [trajectories.csv](trajectories.csv)에 실제 시도·시간·V 기회를 구분한다. 기록이 비어 있으면 미실행이며 0 성능을 뜻하지 않는다. Native raw loss는 원천 간 직접 평균하지 않는다. 개선율은 100×(baseline−Query)/baseline이다.','',
        '## 6. 한계','',
        '개발 데이터 재사용, 2 seeds, 길이4096 하나, 제한된 CP/micro 옵션과 단일 LR의 비교다. 4개 시간 블록 bootstrap은 기술적 불확실성 표시이며 재사용 편향을 보정하지 않는다. 실제 소형 VRAM 장비, 다른 길이·원천, 신규성은 검증하지 않았다. F0와 각 arm의 step0 차이를 별도 저장하며 초기 반올림 차이를 모두 학습 이득으로 세지 않는다.','',
        'Side는 [Ladder Side-Tuning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/54801e196796134a2b0ae5e8adef502f-Abstract-Conference.html)의 원리와 관련된 저장소 대조군이며 공식 전체 재현이 아니다. [PyTorch checkpoint 설명](https://pytorch.org/blog/activation-checkpointing-techniques/)과 로컬 PyTorch API를 확인했다. Query의 큰 아이디어가 최초라는 주장을 하지 않는다.','',
        '## 7. 종료와 다음 판단','',
        ('현재 구현의 FP32/BF16 microbatch 및 checkpoint 동등성 문제를 먼저 이해해야 자원 이점이나 새 학습을 판단할 수 있다. 이번 실행 안에서 허용오차를 늘리거나 실패 옵션을 빼고 진행하지 않는다.' if failures else '기록된 자원 신호와 예측 결과를 분리하여 후속 투자를 판단한다.')+' 자동 추가 학습·seed/LR/budget 탐색은 없다.','',
        f"검증: 이전 파일 {len(c['historical_hashes'])}개 해시 보존, scalar metric 최대 차이 {maxerror:.3g}. [independent_verification.json](independent_verification.json)은 기록의 재현성을 확인하며 실패 판정을 PASS로 바꾸지 않는다.",'']
    if s.get('error'):lines+=['종료 근거: `'+s['error']+'`','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    plots(selection,trajectory,metrics)
    print(json.dumps(dict(status=s['status'],A_updates=s['A_updates'],B_fits=s['B_fits_completed'],parity_failed=len(failures),metric_max_abs_error=maxerror)),flush=True)


def plots(selection,trajectory,metrics):
    if not selection.get('all_options'):return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,d in zip(axes,['ettm2','electricity']):
        for a in ['standard','side','query']:
            vv=[]
            for b in [1,2,4,8]:
                rr=[r for r in selection['all_options'] if r['dataset']==d and r['arm']==a and r['peak_allocated']<=.95*b*2**30]
                vv.append(min(r['block1_seconds'] for r in rr)*1000 if rr else np.nan)
            ax.plot([1,2,4,8],vv,'o-',label=a)
        ax.set(title=d,xlabel='Simulated tensor budget (GiB)',ylabel='Fastest feasible block1 step (ms)',xticks=[1,2,4,8]);ax.legend()
    fig.savefig(OUT/'budget_time.png',dpi=160);plt.close(fig)
    if not metrics:return
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,d in zip(axes,['ettm2','electricity']):
        for a in ['standard','side','query']:
            for seed in [39000,39001]:
                rr=[r for r in trajectory if r['dataset']==d and r['arm']==a and r['seed']==seed]
                ax.plot([r['active_seconds'] for r in rr],[r['scaled_2pinball'] for r in rr],'o-',label=f'{a}/{seed}')
        ax.set(title=d,xlabel='Actual active training seconds',ylabel='Validation scaled 2-pinball');ax.legend(fontsize=7)
    fig.savefig(OUT/'error_active_time.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');fits=read(OUT/'fits.json')
    for ax,d in zip(axes,['ettm2','electricity']):
        for a in ['standard','side','query']:
            rr=[r for r in metrics if r['dataset']==d and r['arm']==a and r['role']=='selected'];ff=[f for f in fits if f['dataset']==d and f['arm']==a]
            ax.scatter(np.mean([f['peak_allocated'] for f in ff])/2**20,np.mean([r['scaled_2pinball'] for r in rr]),label=a)
        ax.set(title=d,xlabel='Measured fit peak allocated (MiB)',ylabel='Selected E_dev scaled 2-pinball');ax.legend()
    fig.savefig(OUT/'quality_memory.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true');p.parse_args();finalize()
