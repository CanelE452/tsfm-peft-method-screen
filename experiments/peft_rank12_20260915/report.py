"""Independent scalar replay, fixed comparisons and Korean reports; no training."""
import argparse,csv,json,math,statistics
import numpy as np
from common import ROOT,RESEARCH,R1,R2,C2,read,save,sha,digest,csvwrite,verify_contract,audit_history
from rank2_data import metrics,independent

def gain(b,a):return 100*(b-a)/b if b else None
def verify_rank2():
    c=verify_contract(R2);s=read(R2/'status.json');fits=read(R2/'fits.json') if (R2/'fits.json').exists() else [];tr=read(R2/'trajectories.json') if (R2/'trajectories.json').exists() else [];ev=read(R2/'evaluation.json') if (R2/'evaluation.json').exists() else [];mech=read(R2/'mechanism.json') if (R2/'mechanism.json').exists() else [];replays=[];arrays={}
    for p,h in c['model_files'].items():assert sha(p)==h
    for d in c['data'].values():assert sha(d['raw_path'])==d['raw_sha256']
    for kind,rows in [('V',tr),('reload',[f['reload'] for f in fits if f['status']=='COMPLETE']),('V_mechanism',mech),('E',ev)]:
        for r in rows:
            assert sha(ROOT/r['prediction_path'])==r['prediction_sha256']
            with np.load(ROOT/r['prediction_path']) as z:p=z['prediction'];y=z['target'];std=z['std']
            m=metrics(p,y,std);assert m==r['metrics'];v=independent(p,y);err=abs(v-m['mse']) if v is not None else None;assert err is None or err<=1e-10;replays.append(dict(kind=kind,path=r['prediction_path'],max_abs_error=err))
            if kind=='E':arrays[(r['dataset'],r['seed'],r['arm'],r['role'])]=(p,y)
    for r in tr:assert sha(ROOT/r['checkpoint_path'])==r['checkpoint_sha256']
    for f in fits:
        if f['status']=='COMPLETE':
            best=min([r for r in tr if r['fit']==f['fit']],key=lambda r:(r['metrics']['mse'],r['epoch']));assert f['best']['checkpoint_sha256']==best['checkpoint_sha256'] and f['reload_max_abs']==0
    if ev:
        assert len(fits)==s['fits_completed']==24 and len(ev)==50;seal=read(R2/'selection_seal.json');assert digest({k:v for k,v in seal.items() if k!='seal_hash'})==seal['seal_hash'];access=read(R2/'evaluation_access.json');assert access['opened_at']>=seal['sealed_at'] and access['selection_hash']==sha(R2/'selection_seal.json')
        for control in seal['controls']:
            group=[f for f in fits if f['dataset']==control['dataset'] and f['seed']==control['seed'] and f['arm'] in ['INDIV_BUDGET','SHARED_BUDGET','FACTOR_BUDGET']];best=min(group,key=lambda f:(f['best']['metrics']['mse'],f['arm']));assert control['fit']==best['fit']
    inventory=read(R2/'parameter_inventory.json');params={r['arm']:r['trainable'] for r in inventory};metricrows=[dict(dataset=r['dataset'],seed=r['seed'],arm=r['arm'],role=r['role'],mse=r['metrics']['mse'],mae=r['metrics']['mae']) for r in ev];comparisons=[];decisions=[];channelrows=[]
    for r in ev:
        for i,ch in enumerate(c['data'][r['dataset']]['channel_ids']):channelrows.append(dict(dataset=r['dataset'],seed=r['seed'],arm=r['arm'],role=r['role'],channel=ch,valid_count=r['metrics']['valid_counts'][i],standardized_mse=r['metrics']['channel_mse'][i],standardized_mae=r['metrics']['channel_mae'][i],raw_mae=r['metrics']['channel_raw_mae'][i]))
    for d in sorted({r['dataset'] for r in ev}):
        rr=[r for r in metricrows if r['dataset']==d and r['role']=='selected'];means={a:statistics.mean(r['mse'] for r in rr if r['arm']==a) for a in params};n=len(c['data'][d]['origins']['evaluation']);blocks=np.array_split(np.arange(n),min(8,n));rng=np.random.default_rng(61520);draws=[]
        for _ in range(1000):
            indices=[]
            while len(indices)<n:indices.extend(blocks[int(rng.integers(len(blocks)))].tolist())
            draws.append(indices[:n])
        draws=np.array(draws)
        def boots(key):
            p,y=arrays[key];valid=np.isfinite(y);err=np.where(valid,p.astype(np.float64)-y.astype(np.float64),0);sq=(err**2).sum(-1);counts=valid.sum(-1);den=counts[draws].sum(1);num=sq[draws].sum(1);return np.nanmean(np.divide(num,den,out=np.full_like(num,np.nan),where=den>0),axis=1)
        basisboot=np.mean([boots((d,z,'BASIS_BUDGET','selected')) for z in [41000,41001]],0)
        for baseline in ['INDIV_REF','INDIV_BUDGET','SHARED_BUDGET','FACTOR_BUDGET','LH','V_SELECTED_BUDGET','INIT','SEASONAL_NAIVE']:
            bvals=[];bboots=[];seedg=[]
            for z in [41000,41001]:
                arm=next(r['arm'] for r in seal['controls'] if r['dataset']==d and r['seed']==z) if baseline=='V_SELECTED_BUDGET' else 'BASIS_BUDGET' if baseline=='INIT' else baseline
                role='INIT' if baseline=='INIT' else 'seasonal_naive' if baseline=='SEASONAL_NAIVE' else 'selected';key=(d,None if role=='seasonal_naive' else z,arm,role)
                value=next(r['mse'] for r in metricrows if (r['dataset'],r['seed'],r['arm'],r['role'])==key);candidate=next(r['mse'] for r in rr if r['seed']==z and r['arm']=='BASIS_BUDGET');g=gain(value,candidate);seedg.append(g);bvals.append(value);bboots.append(boots(key));comparisons.append(dict(dataset=d,seed=z,baseline=baseline,selected_control_arm=arm,baseline_mse=value,basis_mse=candidate,gain_percent=g))
            bv=statistics.mean(bvals);gb=100*(np.mean(bboots,0)-basisboot)/np.mean(bboots,0);comparisons.append(dict(dataset=d,seed='mean',baseline=baseline,baseline_mse=bv,basis_mse=means['BASIS_BUDGET'],gain_percent=gain(bv,means['BASIS_BUDGET']),CI95_lower=float(np.quantile(gb,.025)),CI95_upper=float(np.quantile(gb,.975))))
        selected_gain=next(r['gain_percent'] for r in comparisons if r['dataset']==d and r['seed']=='mean' and r['baseline']=='V_SELECTED_BUDGET');wins=all(r['gain_percent']>0 for r in comparisons if r['dataset']==d and r['seed']!='mean' and r['baseline']=='V_SELECTED_BUDGET')
        decisions.append(dict(dataset=d,EXECUTION_VALID=True,BUDGET_SIGNAL=wins and selected_gain>=.5 and means['BASIS_BUDGET']<means['LH'],COMPRESSION_SIGNAL=params['BASIS_BUDGET']<=.8*params['INDIV_REF'] and means['BASIS_BUDGET']<=1.005*means['INDIV_REF'],REFERENCE_EFFECT_REPRODUCED_OR_NOT=means['INDIV_REF']<means['LH'],NOVELTY_STATUS='KNOWN_PARAMETERIZATION',best_arm=min(means,key=means.get),means=means))
    audit_history();csvwrite(R2/'metrics.csv',metricrows);csvwrite(R2/'channel_metrics.csv',channelrows);csvwrite(R2/'comparisons.csv',comparisons);csvwrite(R2/'trajectories.csv',[dict(fit=r['fit'],dataset=r['dataset'],seed=r['seed'],arm=r['arm'],epoch=r['epoch'],updates=r['updates'],mse=r['metrics']['mse'],mae=r['metrics']['mae']) for r in tr]);csvwrite(R2/'resources.csv',[{k:f[k] for k in ['fit','dataset','seed','arm','status','epochs','updates','active_seconds','wall_seconds','peak_allocated','peak_reserved','median_step_seconds','budget_limited','early_stopped'] if k in f} for f in fits]);save(R2/'decisions.json',dict(source_decisions=decisions,automatic_followup=False));save(R2/'independent_verification.json',dict(prediction_replays=len(replays),max_metric_abs_error=max((r['max_abs_error'] for r in replays if r['max_abs_error'] is not None),default=None),records=replays,historical_files_unchanged=len(c['historical_hashes']),new_gpu_forward=0,new_optimizer_updates=0,E_scope='MEASURED' if ev else 'NOT_RUN'))
    lines=['# R2 채널 변환 예산 비교 — 한국어 결과','',f"실행 상태 **{s['status']}**. smoke {s['smoke_updates']}/{s['smoke_attempts']} updates, 본학습 {s['fits_completed']}/{s['fit_attempts']} fits, {s['training_updates']} updates.",'', '## 어떤 파라미터를 공유했나, 왜 필요한가','', 'LH는 LoRA+HEAD 참고군이다. 다섯 channel arm은 같은 LoRA/frequency/HEAD를 사용한다. 독립 up, 완전 공유 up, 채널별 left×공통 right, 공통 affine+3 residual basis를 고정 예산에서 비교했다. 다른 채널 입력을 읽는 새 attention은 없다.','', '## 단순 공유·저랭크와의 차이 및 예산','', 'BASIS는 vectorized affine weights를 공통 bank와 채널 계수로 factorization한 알려진 매개변수화다. FACTOR는 feature-to-hidden 행렬의 일반 저랭크 대조군이다. 폭도 다르므로 basis만의 단일 인과효과 비교가 아니다. [선행 경계](../../research/peft_rank12_20260915/NOVELTY_BOUNDARY.md), [상류 차이](../../research/peft_rank12_20260915/UPSTREAM_DIFF.md).','', '| arm | 채널 블록 | 전체 trainable | 전체 모델 |','| --- | ---: | ---: | ---: |']
    for r in inventory:
        if r['seed']==41000:lines.append(f"| {r['arm']} | {r['groups']['channel_adapter']} | {r['trainable']} | {r['total']} |")
    lines+=['','동일 예산군은 INDIV_BUDGET/SHARED_BUDGET/FACTOR_BUDGET/BASIS_BUDGET이다. LH와 큰 INDIV_REF는 별도 참조군이다. HEAD는 모든 arm에서 같은 shape/초기값으로 학습했다. [실제 목록](PARAMETER_BUDGET.csv).','', '## 원점수와 추가 가치','']
    if ev:
        lines+=['| 원천 | seed | arm/role | MSE | MAE |','| --- | --- | --- | ---: | ---: |']
        for r in metricrows:lines.append(f"| {r['dataset']} | {r['seed']} | {r['arm']}/{r['role']} | {r['mse']:.9g} | {r['mae']:.9g} |")
        lines+=['','MSE/MAE는 Train 표준화 공간이다. [채널별 원단위 MAE·유효 분모](channel_metrics.csv). INIT는 새 예측 head의 초기 상태이며 유효한 pretrained zero-shot이라고 하지 않는다. seasonal-naive는 직전24시간 반복이다.','', '| 원천 | 대조 | 기준 평균 MSE | BASIS 평균 MSE | 개선율(%) | paired 95% CI |','| --- | --- | ---: | ---: | ---: | --- |']
        for r in comparisons:
            if r['seed']=='mean':lines.append(f"| {r['dataset']} | {r['baseline']} | {r['baseline_mse']:.9g} | {r['basis_mse']:.9g} | {r['gain_percent']:+.4f} | [{r['CI95_lower']:+.4f}, {r['CI95_upper']:+.4f}] |")
        lines+=['','두 seed 원점수 평균 후 개선율을 계산했다. [seed별 비교](comparisons.csv). V_SELECTED_BUDGET은 V에서 3개 대조군을 고른 사전 봉인 정책이며 선택 비용은 source/seed당3 fits다. E 최저 모델을 사후 정책 점수로 쓰지 않았다.','']
        for d in decisions:lines.append(f"- {d['dataset']}: EXECUTION_VALID={d['EXECUTION_VALID']}, BUDGET_SIGNAL={d['BUDGET_SIGNAL']}, COMPRESSION_SIGNAL={d['COMPRESSION_SIGNAL']}, REF가 LH보다 나음={d['REFERENCE_EFFECT_REPRODUCED_OR_NOT']}, 최저 MSE={d['best_arm']}, NOVELTY_STATUS={d['NOVELTY_STATUS']}.")
    else:lines+=['E는 NOT_RUN이다. 미완료 cell이나 입력·자원·실행 문제를 예측 FAIL/오차0으로 해석하지 않는다.']
    if fits:
        lines+=['','## 실제 자원과 선택','', '| fit | epochs/updates | active 초 | wall 초 | peak allocated GiB |','| --- | --- | ---: | ---: | ---: |']
        for f in fits:lines.append(f"| {f['fit']} | {f['epochs']}/{f['updates']} | {f.get('active_seconds',0):.3f} | {f.get('wall_seconds',0):.3f} | {f.get('peak_allocated',0)/2**30:.4f} |")
        lines+=['','같은 epoch·노출 상한 비교이며 동일 wall-time 비교가 아니다. [자원 상세](resources.csv). 미완료 fit에서 없는 시간 값은 측정 완료를 뜻하지 않는다.',f"BUDGET_LIMITED {sum(f.get('budget_limited',False) for f in fits)} fits, INIT 선택 {sum(f.get('best',{}).get('epoch')==0 for f in fits)} fits. min_delta는 patience 리셋에만 쓰고 실제 최저 V는 작은 개선도 저장했다.",'']
    if mech:
        lines+=['## V에서만 수행한 계수 개입','', '| 원천 | seed | 개입 | 원래 V MSE | 개입 V MSE |','| --- | --- | --- | ---: | ---: |']
        for r in mech:lines.append(f"| {r['dataset']} | {r['seed']} | {r['role']} | {r['base_validation_mse']:.9g} | {r['metrics']['mse']:.9g} |")
        lines+=['','계수 순열과 채널 평균 대체는 고정 사후 진단이며 selection·threshold를 바꾸지 않았다. 분포 밖 개입일 수 있어 고유한 인과 기여 증명은 아니다.','']
    lines+=['## 검증·신규성·평가 한계','',f"저장 예측 {len(replays)}개를 독립 scalar float64로 다시 계산했다. 최대 허용 절대차1e-10이며 실제 값은 [검산 기록](independent_verification.json)에 있다. 기존 {len(c['historical_hashes'])}개 결과·연구 파일과 모델/소스 해시를 확인했다.",'', 'Time-PEFT-inspired controlled pilot이다. 32채널·길이96·BF16·최대20epochs·고정LR·2seed·이미 본 두 원천의 결과다. 이전64채널 결과 이후의 설계 변경이므로 독립 확증이나 공정한 단일 변수 전후 비교로 포장하지 않는다. 공식 Time-PEFT 전체 성능 재현 및 공식 C-LoRA/MoLA 직접 비교는 수행하지 않았다.','', 'NEXT_ACTION: 이번 예산 비교의 결과와 최강 단순 대조군을 검토해 후속 투자 여부를 결정한다. 후속 학습은 자동 실행하지 않는다.','']
    if s.get('error'):lines+=['실행 종료 근거: `'+s['error']+'`.','']
    (R2/'REPORT.md').write_text('\n'.join(lines).rstrip()+'\n');print('R2 VERIFIED',s['status'],len(replays),flush=True)

def combined():
    q=read(R1/'status.json');r=read(R2/'status.json');audit_history();lines=['# PEFT R1/R2 실행 결과','', '기준9051d03 이후 완료된 Q180-update 수치 진단 및 C24-fit 연구를 보존하고 새 지시문의 차이를 확인했다. [변경 감사](AUDIT.md), [고정 지시문](PROTOCOL.md). 기존 학습을 다시 실행한 것이 아니라 검사 쌍·수치 계약과 별도의32채널 예산 비교를 수행했다.','', '| 트랙 | 상태 | 검사 updates | 자원 updates | 본학습 완료/시도 | 본학습 updates |','| --- | --- | ---: | ---: | ---: | ---: |',f"| R1 | {q['status']} | {q['numeric_updates']}/96 | {q['A_updates']}/480 | {q['B_fits_completed']}/{q['B_fit_attempts']} (상한12) | {q['B_updates']} |",f"| R2 | {r['status']} | {r['smoke_updates']}/24 | dry forward/backward별도 | {r['fits_completed']}/{r['fit_attempts']} (상한24) | {r['training_updates']} |",'', '## R1 — 수치·자원·예측을 분리','', (R1/'REPORT.md').read_text(),'', '## R2 — 구성요소·예산·정확도·신규성','', (R2/'REPORT.md').read_text(),'', '## 종료와 공개 범위','', '두 트랙은 별도 프로세스·환경에서 순차 실행했다. R1의 연구 중단을 R2 중단 조건으로 삼지 않았다. RustDesk만 기존 사용자 승인 예외로 기록했고 다른 compute와 메모리는 감시했다. 추가 후보·Censor 재튜닝·LR/seed 탐색은 없다. 원자료·모델 가중치·대형 예측 cache는 로컬에 남는다. GitHub에는 실행 코드·계약·원점수·해시·검증 기록을 올린다.','']
    # Embedded track-relative links must resolve from the combined report directory.
    qtext=(R1/'REPORT.md').read_text();rtext=(R2/'REPORT.md').read_text()
    import re
    def links(text,folder):
        def repl(m):
            target=m.group(2)
            if '://' in target or target.startswith('#'):return m.group(0)
            import os
            return m.group(1)+'('+os.path.relpath((folder/target).resolve(),RESEARCH)+')'
        return re.sub(r'(!?\[[^\]]*\])\(([^)]+)\)',repl,text)
    text='\n'.join(lines).replace(qtext,links(qtext,R1)).replace(rtext,links(rtext,R2));(RESEARCH/'REPORT.md').write_text(text.rstrip()+'\n');save(RESEARCH/'completion.json',dict(R1=q,R2=r,historical_files_unchanged=len(read(RESEARCH/'historical_hashes.json')),no_automatic_followup=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['rank2','combined']);a=p.parse_args();verify_rank2() if a.stage=='rank2' else combined()
