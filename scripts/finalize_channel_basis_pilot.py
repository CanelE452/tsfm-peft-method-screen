"""Independent cached prediction verification and Korean channel-pilot report; no fits."""
import argparse,json
import numpy as np
from priority12.common import ROOT,read,save,sha,digest,csvwrite
from priority12.channel_data import metrics,independent
from priority12.channel_model import ARMS
OUT=ROOT/'results/channel_basis_pilot_20260915'
def gain(b,a):return 100*(b-a)/b if b else None

def block_draws(n,seed=60100,repeats=1000):
    blocks=[np.arange(i,min(i+4,n)) for i in range(0,n,4)];rng=np.random.default_rng(seed);draws=[]
    for _ in range(repeats):
        picks=[]
        while len(picks)<n:picks.extend(blocks[int(rng.integers(len(blocks)))].tolist())
        draws.append(picks[:n])
    return np.array(draws)

def finalize():
    from run_channel_basis_pilot import contract_check
    c=contract_check();s=read(OUT/'status.json');fits=read(OUT/'fits.json') if (OUT/'fits.json').exists() else []
    tr=read(OUT/'trajectories.json') if (OUT/'trajectories.json').exists() else [];ev=read(OUT/'evaluation.json') if (OUT/'evaluation.json').exists() else []
    verified=[];maxerr=0.;arrays={}
    for kind,rows in [('V',tr),('V_reload',[f['reload'] for f in fits if f['status']=='COMPLETE']),('E',ev)]:
        for row in rows:
            assert sha(ROOT/row['prediction_path'])==row['prediction_sha256']
            with np.load(ROOT/row['prediction_path']) as z:p=z['prediction'];y=z['target'];std=z['std']
            m=metrics(p,y,std);assert m==row['metrics'];err=abs(independent(p,y)-m['mse']);assert err<=1e-10;maxerr=max(maxerr,err)
            if kind=='E' and row['role']=='selected':
                valid=np.isfinite(y);e=np.where(valid,p.astype(np.float64)-y.astype(np.float64),0)
                arrays[(row['dataset'],row['seed'],row['arm'])]=dict(squared=(e**2).sum(-1),counts=valid.sum(-1))
            verified.append(dict(kind=kind,path=row['prediction_path'],metric_abs_error=err))
    for r in tr:assert sha(ROOT/r['checkpoint_path'])==r['checkpoint_sha256']
    for f in fits:
        if f['status']=='COMPLETE':
            assert f['updates']==1024 and f['reload_max_abs']==0
            best=min([r for r in tr if r['fit']==f['fit']],key=lambda r:(r['metrics']['mse'],r['step']))
            assert f['best']['checkpoint_sha256']==best['checkpoint_sha256']
    if ev:
        seal=read(OUT/'selection_seal.json');assert digest({k:v for k,v in seal.items() if k!='seal_hash'})==seal['seal_hash']
        assert seal['contract_hash']==sha(OUT/'contract.json') and seal['planned_attempts_finished']
        access=read(OUT/'evaluation_access.json');assert access['opened_at']>=seal['sealed_at'] and access['selection_hash']==sha(OUT/'selection_seal.json')
        for d in seal['complete_sources']:
            expected={(d,z,a) for z in [40000,40001] for a in ARMS};assert {(r['dataset'],r['seed'],r['arm']) for r in seal['selections'] if r['dataset']==d}==expected
    inventory=read(OUT/'parameter_inventory.json');counts={r['arm']:r['trainable'] for r in inventory};comparisons=[];decisions=[];perorigin=[]
    for (d,z,a),v in arrays.items():
        for j in range(len(v['counts'])):perorigin.append(dict(dataset=d,seed=z,arm=a,origin_index=j,mse=float((v['squared'][j]/v['counts'][j]).mean())))
    for d in sorted({k[0] for k in arrays}):
        source=[r for r in ev if r['dataset']==d and r['role']=='selected'];means={a:float(np.mean([r['metrics']['mse'] for r in source if r['arm']==a])) for a in ARMS};n=len(arrays[(d,40000,'BASIS4')]['counts']);draws=block_draws(n)
        bootstrap={}
        for a in ARMS:
            boot=[]
            for z in [40000,40001]:
                v=arrays[(d,z,a)];boot.append((v['squared'][draws].sum(1)/v['counts'][draws].sum(1)).mean(1))
            bootstrap[a]=np.mean(boot,axis=0)
        for a in ARMS:
            if a=='BASIS4':continue
            gg=100*(bootstrap[a]-bootstrap['BASIS4'])/bootstrap[a]
            comparisons.append(dict(dataset=d,seed='mean',baseline=a,baseline_mse=means[a],basis_mse=means['BASIS4'],gain_percent=gain(means[a],means['BASIS4']),CI95_lower=float(np.quantile(gg,.025)),CI95_upper=float(np.quantile(gg,.975))))
            for z in [40000,40001]:
                b=next(r['metrics']['mse'] for r in source if r['arm']==a and r['seed']==z);q=next(r['metrics']['mse'] for r in source if r['arm']=='BASIS4' and r['seed']==z)
                comparisons.append(dict(dataset=d,seed=z,baseline=a,baseline_mse=b,basis_mse=q,gain_percent=gain(b,q)))
        light=counts['BASIS4']<=.5*counts['SPECIFIC'] and means['BASIS4']<=1.01*means['SPECIFIC']
        coeff=all(next(r['gain_percent'] for r in comparisons if r['dataset']==d and r['seed']=='mean' and r['baseline']==a)>=.5 for a in ['SHARED_WIDE','GROUP4']) and all(next(r['gain_percent'] for r in comparisons if r['dataset']==d and r['seed']==z and r['baseline']==a)>0 for a in ['SHARED_WIDE','GROUP4'] for z in [40000,40001])
        decisions.append(dict(dataset=d,limited_compression_signal=light,coefficient_signal=coeff,best_arm=min(means,key=means.get),means=means,lora_head_smaller_and_more_accurate=counts['LORA_HEAD']<counts['BASIS4'] and means['LORA_HEAD']<means['BASIS4'],specific_gain_vs_lora_head=gain(means['LORA_HEAD'],means['SPECIFIC'])))
    rows=[dict(dataset=r['dataset'],seed=r['seed'],arm=r['arm'],role=r['role'],mse=r['metrics']['mse'],mae=r['metrics']['mae']) for r in ev]
    csvwrite(OUT/'metrics.csv',rows);csvwrite(OUT/'comparisons.csv',comparisons);csvwrite(OUT/'per_origin_metrics.csv',perorigin)
    csvwrite(OUT/'trajectories.csv',[dict(fit=r['fit'],dataset=r['dataset'],seed=r['seed'],arm=r['arm'],step=r['step'],active_seconds=r['active_seconds'],mse=r['metrics']['mse'],mae=r['metrics']['mae']) for r in tr])
    csvwrite(OUT/'attempts.csv',[dict(phase='CPU_toy_structure',attempts=s.get('cpu_toy_optimizer_updates',0),optimizer_updates=s.get('cpu_toy_optimizer_updates',0),forecasting_fits=0),dict(phase='structure_disposable',attempts=s['check_update_attempts'],optimizer_updates=s['check_updates'],forecasting_fits=0)]+[dict(phase='training',fit=f['fit'],status=f['status'],updates=f['updates'],error=f.get('error')) for f in fits])
    csvwrite(OUT/'resources.csv',[{k:f[k] for k in ['fit','dataset','seed','arm','updates','wall_seconds','active_seconds','median_step_seconds','peak_allocated','peak_reserved','budget_limited'] if k in f} for f in fits]);save(OUT/'decisions.json',dict(source_decisions=decisions,auto_followup=False))
    save(OUT/'independent_verification.json',dict(status='VERIFIED_RECORDED_OUTCOME',prediction_metric_replays=len(verified),metric_max_abs_error=maxerr if verified else None,checks=verified,historical_files_unchanged=len(c['historical_hashes']),new_gpu_forward=0,new_optimizer_updates=0,selected_gpu_replays=sum(f['status']=='COMPLETE' for f in fits),forecasting_evaluation='MEASURED' if ev else 'NOT_RUN'))
    lines=['# 채널 공유 파일럿 — 기존 원천을 재사용한 개발 비교','',f"**상태 {s['status']}. CPU toy optimizer {s.get('cpu_toy_optimizer_updates',0)} updates; 실모델 GPU 구조 점검 {s['check_updates']}/{s['check_update_attempts']} updates/attempts. 본학습 {s['fits_completed']}/{s['fit_attempts']} fits 완료/시도, {s['training_updates']} updates.**",'',
        '## 실제 실행과 미실행','',
        'CPU FP64 구조 환원·순열·초기화·활성 경로, 실제 MOMENT-small hidden512/patch8/head, 공식 SPECIFIC 출력/손실 비교를 수행했다. GPU 준비가 막힌 경우 이 CPU 검사 통과를 GPU 학습 완료로 쓰지 않는다. 시도0은 예측 성능0이나 성능 FAIL이 아니다.','',
        '## 무엇을 공유했는가','',
        '공개 Time-PEFT의 LoRA/frequency/head를 유지하면서 채널 up을 독립/공유/폭 증가/정적 그룹/정적 기저합으로 바꿨다. 입력 채널 간 새 attention은 없다.','',
        '| arm | 채널 블록 | 전체 trainable | 총 모델 |','| --- | ---: | ---: | ---: |']
    for r in inventory:
        if r['seed']==40000:lines.append(f"| {r['arm']} | {r['groups']['channel_adapter']:,} | {r['trainable']:,} | {r['total']:,} |")
    lines+=['','SHARED_WIDE/BASIS4의 채널 블록 차이는1개, GROUP4/BASIS4는256개다. LORA_HEAD/SHARED/SPECIFIC은 동일 예산 대조군이 아니다. count 절약은 학습 속도나 예측 이득과 별개다.','',
        '## 원점수와 같은 예산 비교','']
    if not ev:lines+=['신규 V/E 예측 비교가 없어 원점수, 최강 예측 대조군, 계수 활용/경량화 신호는 모두 미판정이다. 빈 지표 표는 미실행을 뜻한다.']
    else:
        lines+=['| 원천 | seed | arm/role | MSE | MAE |','| --- | --- | --- | ---: | ---: |']
        for r in rows:lines.append(f"| {r['dataset']} | {r['seed']} | {r['arm']}/{r['role']} | {r['mse']:.9g} | {r['mae']:.9g} |")
        lines+=['','평균은 원천별 두seed의 원점수를 먼저 평균한다. MSE는 이미 train-standardized 공간이며 std로 다시 나누지 않았다. [개선율/CI](comparisons.csv), [실제 자원](resources.csv).']
        for d in decisions:lines.append(f"- {d['dataset']}: 최저 MSE {d['best_arm']}; 제한된 경량화 {d['limited_compression_signal']}, 계수 활용 {d['coefficient_signal']}; LORA_HEAD가 더 작고 정확함 {d['lora_head_smaller_and_more_accurate']}.")
    lines+=['','## 한계와 종료','',
        'C-LoRA 등 채널별 공유·분해는 이미 알려진 원리다. 이번은 Time-PEFT 구조의 통제 변형이며 신규성 확정이나 공개 논문 전체 재현이 아니다. 공식 C-LoRA와 같은 백본 비교, 새 원천 독립 확인, 더 넓은 최적화 비교가 남아 있다. 단일64채널/길이512/단일LR/2seed 개발 비교로 범용성·수렴·새채널 일반화를 주장하지 않는다. 선택1024와 말기 V 감소는 BUDGET_LIMITED로 기록한다.','',
        f"원래 결과 {len(c['historical_hashes'])}개는 그대로 보존됐다. [검증 기록](independent_verification.json), [고정 계약](contract.json), [선행/환경 차이](../../sources/RELATED_WORK.md). 추가 후보나 자동 후속 학습은 없다.",'']
    if s.get('error'):lines+=['종료 근거: `'+s['error']+'`.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines));print('C VERIFIED',s['status'],'prediction replays',len(verified),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true');p.parse_args();finalize()
if __name__=='__main__':main()
