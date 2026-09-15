"""One bounded CPU analysis of exposed LH residuals; no forecasting model fits."""
from pathlib import Path
import hashlib,json,math,subprocess,time,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/channel_residual_evidence_20260916'
OLD=ROOT/'results/channel_sharing_screen_v1_20260915'
METHODS=['UNCHANGED','EXPANDING_BIAS','EXPANDING_DAILY','LAST_BIAS']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(name,rows):
    with open(OUT/name,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def load(r):
    p=ROOT/r['prediction_path'];assert sha(p)==r['prediction_sha256']
    with np.load(p) as z:a={k:z[k].copy() for k in z.files}
    assert np.isfinite(a['prediction']).all() and np.isfinite(a['target']).all()
    return a

def correct(p,y,origins):
    n,c,h=p.shape;assert h==96 and np.all(np.diff(origins)>=h)
    total=np.zeros(c);num=0;phase_total=np.zeros((c,24));phase_num=np.zeros(24);latest=np.zeros(c);cursor=0
    outputs={m:[] for m in METHODS}; receipts=[]
    for i,o in enumerate(origins):
        while cursor<i and origins[cursor]+h<=o:
            residual=p[cursor]-y[cursor];total+=residual.sum(-1);num+=h;latest=residual.mean(-1)
            phases=(origins[cursor]+np.arange(h))%24
            for phase in range(24):
                ix=phases==phase;phase_total[:,phase]+=residual[:,ix].sum(-1);phase_num[phase]+=int(ix.sum())
            cursor+=1
        bias=total/num if num else np.zeros(c)
        daily=np.divide(phase_total,phase_num[None],out=np.zeros_like(phase_total),where=phase_num[None]>0)
        outputs['UNCHANGED'].append(p[i].copy());outputs['EXPANDING_BIAS'].append(p[i]-bias[:,None])
        outputs['EXPANDING_DAILY'].append(p[i]-daily[:,(o+np.arange(h))%24]);outputs['LAST_BIAS'].append(p[i]-latest[:,None])
        receipts.append(dict(origin=int(o),completed_forecasts=cursor,latest_label_time_exclusive=int(origins[cursor-1]+h) if cursor else None))
    return {m:np.stack(v) for m,v in outputs.items()},receipts

def scores(p,y,std):
    e=p-y; mse=float((e**2).mean((0,2)).mean());mae=float(abs(e).mean((0,2)).mean());raw=float((abs(e).mean((0,2))*std).mean())
    scalar=math.fsum(math.fsum((float(a)-float(b))**2 for a,b in zip(p[:,c].flat,y[:,c].flat))/(p.shape[0]*p.shape[2]) for c in range(p.shape[1]))/p.shape[1]
    assert math.isclose(mse,scalar,abs_tol=1e-12,rel_tol=1e-12)
    return dict(mse=mse,mae=mae,raw_mae=raw,scalar_mse_abs_difference=abs(mse-scalar))

def main():
    assert not (OUT/'seal.json').exists(),'No repeated diagnostic under same run ID'
    before={str(p.relative_to(ROOT)):sha(p) for folder in ['results','research'] for p in (ROOT/folder).rglob('*') if p.is_file() and OUT not in p.parents}
    fits=[f for f in read(OLD/'fits.json') if f['arm']=='LH'];evaluation=read(OLD/'evaluation.json');assert len(fits)==4
    source={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'PROTOCOL.md',OLD/'fits.json',OLD/'evaluation.json']}
    for f in fits:
        source[f['best']['prediction_path']]=f['best']['prediction_sha256']
        e=next(e for e in evaluation if e['dataset']==f['dataset'] and e['seed']==f['seed'] and e['arm']=='LH' and e['role']=='selected')
        source[e['prediction_path']]=e['prediction_sha256']
    save('seal.json',dict(at=time.time(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hashes=source,methods=METHODS,neural_fits=0,optimizer_updates=0,scope='Post-hoc exposed-data diagnostic'))
    save('historical_hashes.json',before)
    # Synthetic absolute-phase counterexample crossing the V/E offset: same periodic error must be removed exactly.
    origins=np.array([0,111,210]);pred=np.zeros((3,2,96));y=np.stack([np.tile(np.sin(2*np.pi*((o+np.arange(96))%24)/24),(2,1)) for o in origins])
    pp,rr=correct(pred,y,origins);assert np.max(abs(pp['EXPANDING_DAILY'][1:]-y[1:]))<1e-14
    summaries=[];decomp=[];channels=[];causal=[];receipts_all=[]
    for f in fits:
        d,seed=f['dataset'],f['seed'];v=load(f['best']);er=next(e for e in evaluation if e['dataset']==d and e['seed']==seed and e['arm']=='LH' and e['role']=='selected');e=load(er)
        assert np.array_equal(v['std'],e['std'])
        p=np.concatenate([v['prediction'],e['prediction']]).astype(np.float64);y=np.concatenate([v['target'],e['target']]).astype(np.float64);origins=np.concatenate([v['origins'],e['origins']]);nv=len(v['origins'])
        pp,rr=correct(p,y,origins)
        for j in [0,nv,nv+(len(e['origins'])//2)]:
            poison=y.copy();poison[j:]=np.random.default_rng(61694).normal(size=poison[j:].shape)*1e6
            poisoned,_=correct(p,poison,origins)
            for m in METHODS:assert np.array_equal(pp[m][:j+1],poisoned[m][:j+1])
            causal.append(dict(dataset=d,seed=seed,poison_start_index=j,predictions_unchanged_through_index=j,methods=METHODS))
        receipts_all.append(dict(dataset=d,seed=seed,origins=rr,statistics_state_scalars=32+1+32*24+24+32,forecast_blocks_seen=len(origins)-1))
        for split,ix in [('V',slice(0,nv)),('E',slice(nv,None))]:
            yy=y[ix];err=p[ix]-yy;b=err.mean(-1,keepdims=True);daily=err.reshape(-1,32,4,24).mean(2)-b
            periodic=np.tile(daily,(1,1,4));rest=err-b-periodic
            total=float(np.mean(err**2));dc=float(np.mean(b**2));pe=float(np.mean(periodic**2));re=float(np.mean(rest**2))
            assert math.isclose(total,dc+pe+re,abs_tol=1e-12,rel_tol=1e-12)
            decomp.append(dict(dataset=d,seed=seed,split=split,mse=total,level_mse=dc,daily_zero_mean_mse=pe,remainder_mse=re,level_share_percent=100*dc/total,daily_share_percent=100*pe/total,remainder_share_percent=100*re/total,orthogonality_error=abs(total-dc-pe-re),interpretation='Future-label descriptive decomposition, not deployable oracle recovery'))
            base=scores(pp['UNCHANGED'][ix],yy,v['std'])
            for m in METHODS:
                sm=scores(pp[m][ix],yy,v['std']);summaries.append(dict(dataset=d,seed=seed,split=split,method=m,origins=len(yy),**sm,gain_percent=100*(base['mse']-sm['mse'])/base['mse']))
                for c in range(32):channels.append(dict(dataset=d,seed=seed,split=split,method=m,channel=c,mse=float(np.mean((pp[m][ix,c]-yy[:,c])**2))))
            expected=f['best']['metrics']['mse'] if split=='V' else er['metrics']['mse'];assert math.isclose(base['mse'],expected,abs_tol=1e-12,rel_tol=1e-12)
    table('scores.csv',summaries);table('decomposition.csv',decomp);table('channel_scores.csv',channels)
    save('causality.json',dict(poison_checks=causal,phase_alignment_counterexample_passed=True,state_receipts=receipts_all))
    for p,h in source.items():assert sha(ROOT/p)==h,p
    for p,h in before.items():assert sha(ROOT/p)==h,p
    save('verification.json',dict(source_hashes_unchanged=True,historical_files_unchanged=len(before),scalar_mse_checks=len(summaries),maximum_scalar_mse_difference=max(r['scalar_mse_abs_difference'] for r in summaries),decomposition_checks=len(decomp),maximum_decomposition_error=max(r['orthogonality_error'] for r in decomp),poisoned_stream_checks=len(causal)*len(METHODS),phase_alignment_counterexample=True,neural_fits=0,optimizer_updates=0,statistical_optimizers=0,unique_streams=4))
    lines=['# 단순 LoRA 잔차의 구조와 보정 가능성','', '**사후 개발 진단이며 신규 PEFT 방법 또는 독립 PASS가 아니다.** 신경망 학습0회, GPU 작업0회. 기존 LH의 두 원천·두 seed 예측을 해시 검산해 사용했다.','', '## 실제로 쓸 수 있는 과거 잔차 보정','', 'V에서 이미 선택된 checkpoint를 고정했다. 시점o에 origin+96<=o인 이전 forecast의 잔차만 보정 통계에 넣었다. 현재 정답을 먼저 본 예측은 없다. 다만 V 자체는 전체 V로 선택된 고정 checkpoint의 사후 backtest이므로 전체 pipeline의 순수 online 평가가 아니다. E 또한 과거 연구에 노출돼 독립 확증은 아니다.','', '| 원천 | seed | 구간 | 방법 | MSE | MAE | raw MAE | LH 대비 개선 |','|---|---:|---|---|---:|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['dataset']} | {r['seed']} | {r['split']} | {r['method']} | {r['mse']:.6f} | {r['mae']:.6f} | {r['raw_mae']:.6f} | {r['gain_percent']:+.3f}% |")
    lines+=['','## 남은 오차의 직교 분해','', '각96시간 잔차를 전체 평균, 평균0인24시간 반복, 나머지로 분해했다. 정답을 알고 난 뒤의 구조다. 아래 비율만큼 미래 오차를 제거할 수 있다는 뜻이 아니다.','', '| 원천 | seed | 구간 | 평균 성분 | 24h 반복 성분 | 나머지 |','|---|---:|---|---:|---:|---:|']
    for r in decomp:lines.append(f"| {r['dataset']} | {r['seed']} | {r['split']} | {r['level_share_percent']:.2f}% | {r['daily_share_percent']:.2f}% | {r['remainder_share_percent']:.2f}% |")
    lines+=['','## 검증과 해석 범위','',f"32개 집계 MSE를 별도 scalar 계산으로 검산했다. 미래값 오염48개 비교, phase offset 반례,8개 직교 분해 검사 및 기존 {len(before)}개 결과·연구 파일 보존을 확인했다. [검증](verification.json), [인과성 상태 원장](causality.json), [원점수](scores.csv), [채널별 원점수](channel_scores.csv), [고정 프로토콜](PROTOCOL.md).",'', '통계 상태는 각 스트림857 scalars이며 optimizer 학습은 없다. 이는 neural PEFT의 학습 파라미터 절감 성과가 아니다. 최신 잔차의 채널 평균, 누적 평균, 누적24시간 평균은 알려진 단순 통계 대조다. 좋게 나오는 방법·구간만 골라 새 PASS를 만들지 않았다. 단순 대조가 좋아지면 이후 후보가 넘어야 할 기준선이 추가되는 것이며, 새 방법의 성과가 확보되는 것은 아니다.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(evidence_complete=True,new_neural_fits=0,e_scores=[r for r in summaries if r['split']=='E'],decomposition=decomp),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
