"""Read-only numerical audit plus Korean report for the sealed diagnostic."""
import csv,json,math,sys,time
from pathlib import Path
import numpy as np
from model import ROOT
from common import read,save,sha,csvwrite
from rank2_data import independent,batch,load
import importlib.util
_spec = importlib.util.spec_from_file_location('identity_diagnostic_runner', Path(__file__).with_name('run.py'))
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)
OUT, OLD, DATA, check_hashes = _runner.OUT, _runner.OLD, _runner.DATA, _runner.check_hashes


def main():
    status=read(OUT/'status.json'); seal=read(OUT/'seal.json')
    check_hashes(seal['source_hashes']);check_hashes(read(OUT/'historical_hashes.json'))
    fits=read(OUT/'fits.json'); rows=[]; differences=[]; checked=0;mae_checked=0
    results=read(OUT/'evaluation.json') if (OUT/'evaluation.json').exists() else []
    trajectory=read(OUT/'trajectory.json') if (OUT/'trajectory.json').exists() else []
    all_predictions=trajectory+results+[f['replay'] for f in fits if 'replay' in f]+read(OUT/'reused_controls.json')
    for r in all_predictions:
        assert sha(ROOT/r['prediction_path'])==r['prediction_sha256']
        with np.load(ROOT/r['prediction_path']) as z: p=z['prediction'];y=z['target']
        scalar=independent(p,y);differences.append(abs(scalar-r['metrics']['mse']));checked+=1
        assert math.isclose(scalar,r['metrics']['mse'],rel_tol=1e-12,abs_tol=1e-12)
        per_channel=[]
        for c in range(y.shape[1]):
            pairs=[(float(a),float(b)) for a,b in zip(p[:,c].flat,y[:,c].flat) if math.isfinite(float(b))]
            if pairs:per_channel.append(math.fsum(abs(a-b) for a,b in pairs)/len(pairs))
        mae=math.fsum(per_channel)/len(per_channel);assert math.isclose(mae,r['metrics']['mae'],rel_tol=1e-12,abs_tol=1e-12);mae_checked+=1
        if 'checkpoint_path' in r:assert sha(ROOT/r['checkpoint_path'])==r['checkpoint_sha256']
    future=[]
    for d,dc in seal['data'].items():
        v,std=load(DATA,d,'development');poison=v.copy();poison[dc['t1']:]=np.random.default_rng(61693).normal(size=poison[dc['t1']:].shape)*1e4
        # All scheduled train origins, not a chosen favourable subset.
        a,b=batch(v,dc['origins']['train'],'cpu');x,y=batch(poison,dc['origins']['train'],'cpu')
        assert np.array_equal(a.numpy(),x.numpy(),equal_nan=True) and np.array_equal(b.numpy(),y.numpy(),equal_nan=True)
        future.append(dict(dataset=d,train_origins=len(dc['origins']['train']),future_value_poison_training_batch_unchanged=True))
    assert len(fits)<=4 and status['training_updates']==sum(f['updates'] for f in fits)<=5080 and status['smoke_updates']<=4
    oldfits=read(OLD/'fits.json');controls=read(OUT/'reused_controls.json')
    for r in results:
        d,seed=r['dataset'],r['seed'];f=next(f for f in fits if f['dataset']==d and f['seed']==seed)
        for baseline in ['LH','SHARED_BUDGET','SEASONAL_NAIVE']:
            b=next(b for b in controls if b['dataset']==d and b['arm']==baseline and (b['seed']==seed or baseline=='SEASONAL_NAIVE') and b['role']!='INIT')
            bf=next((z for z in oldfits if z['dataset']==d and z['seed']==seed and z['arm']==baseline),None)
            rm=float(np.mean(r['metrics']['channel_raw_mae']));bm=float(np.mean(b['metrics']['channel_raw_mae']))
            rows.append(dict(dataset=d,seed=seed,baseline=baseline,candidate_mse=r['metrics']['mse'],baseline_mse=b['metrics']['mse'],gain_percent=100*(b['metrics']['mse']-r['metrics']['mse'])/b['metrics']['mse'],candidate_mae=r['metrics']['mae'],baseline_mae=b['metrics']['mae'],candidate_raw_mae=rm,baseline_raw_mae=bm,candidate_epoch=r['epoch'],baseline_epoch=bf['best']['epoch'] if bf else '',candidate_trainable=f['trainable'],baseline_trainable={'LH':761952,'SHARED_BUDGET':1319711,'SEASONAL_NAIVE':0}[baseline],candidate_peak_mib=f['peak_allocated']/2**20,baseline_peak_mib=bf['peak_allocated']/2**20 if bf else '',candidate_active_seconds=f['active_seconds'],baseline_active_seconds=bf['active_seconds'] if bf else ''))
    csvwrite(OUT/'comparisons.csv',rows)
    decisions=[]
    for d in seal['data']:
        sr=[r for r in rows if r['dataset']==d and r['baseline']=='SHARED_BUDGET'];lr=[r for r in rows if r['dataset']==d and r['baseline']=='LH']
        complete=len(sr)==len(lr)==2
        decisions.append(dict(dataset=d,execution_complete=complete,preservation_development_signal=complete and all(r['gain_percent']>0 for r in sr),added_value_over_LH_development_signal=complete and all(r['gain_percent']>0 for r in lr)))
    verification=dict(at=time.time(),prediction_mse_recalculations=checked,prediction_mae_recalculations=mae_checked,max_mse_abs_error=max(differences,default=0),future_poison=future,historical_files_preserved=len(read(OUT/'historical_hashes.json')),source_seal_unchanged=True,fit_count=len(fits),replayed_selected_checkpoints=sum('replay' in f for f in fits),report_source_sha256=sha(Path(__file__)))
    save(OUT/'verification.json',verification)
    save(OUT/'decision.json',dict(execution=status['status'],predictive_evidence='DEVELOPMENT_DIAGNOSTIC_ONLY',novelty='DIRECT_EQUIVALENCE_REZERO',new_method_topic='NOT_ESTABLISHED',by_source=decisions,additional_training='NONE'))
    def fmt(x):return f'{x:.6f}'
    lines=['# 표현 보존 채널 PEFT 진단 — 한국어 결과', '',
        '**이번 실행은 알려진 ReZero 대조의 원인 진단이다. 새 PEFT 방법론 주제나 독립 SCREEN_PASS를 확보했다는 결과가 아니다.**','',
        f"실행 상태 {status['status']}. 신규 본학습 {status['fits_completed']}/{len(seal['cells'])} fits, optimizer updates {status['training_updates']}/5080. smoke {status['smoke_updates']}/4 updates. 기존 LH·SHARED_BUDGET 8 fits를 다시 학습하지 않고 해시 검산 후 재사용했다.",'',
        '## 무엇을 확인했나','',
        '사전학습 표현 h를 새 채널 모듈 출력 u로 교체하던 경로에 h+a·u를 적용했다. a는0에서 학습하는 스칼라이고 나머지 설정은 기존 R2와 같다. 초기 CPU 두 seed 및 실제 BF16 smoke에서 LH와 출력이 정확히 같았다. 이후 gate·주파수·채널·head·LoRA가 업데이트되고 동결 파라미터/버퍼가 보존됐다.','',
        '이는 잔차 경로와0 초기화를 함께 바꾼 대조다. 두 요인의 개별 인과효과나 과거 모든 실패의 원인을 증명하지 않는다. 공개 Time-PEFT 전체 재현도 아니다.','',
        '## 원점수와 같은 seed 비교','',
        '지표는 기존과 같은 정규화된 채널-macro MSE다. 개선율은100×(대조−진단군)/대조. 양수일수록 진단군이 좋다. 기존 E는 이미 여러 번 노출됐으며 이번에도 개발 자료(DISCOVERY_REUSED_E)다.','',
        '| 원천 | seed | 선택 epoch | 진단군 MSE | 대조 | 대조 MSE | 개선율 |','|---|---:|---:|---:|---|---:|---:|']
    for r in rows:lines.append(f"| {r['dataset']} | {r['seed']} | {r['candidate_epoch']} | {fmt(r['candidate_mse'])} | {r['baseline']} | {fmt(r['baseline_mse'])} | {r['gain_percent']:+.3f}% |")
    lines += ['', 'normalized MAE와 원 단위 channel-macro MAE, 모든 비용은 [comparisons.csv](comparisons.csv)에 함께 보관했다. 두 데이터의 단위가 다르므로 raw MAE를 원천 간 단일 성능으로 합산하지 않는다.','', '## 같은 원천 두 seed의 평균','', '| 원천 | 진단군 MSE | LH MSE | SHARED MSE | 계절 반복 MSE |','|---|---:|---:|---:|---:|']
    for d in seal['data']:
        rr=[r for r in rows if r['dataset']==d]
        if not rr:continue
        get=lambda b:float(np.mean([r['baseline_mse'] for r in rr if r['baseline']==b]))
        candidate=float(np.mean([r['candidate_mse'] for r in rr if r['baseline']=='LH']))
        lines.append(f"| {d} | {fmt(candidate)} | {fmt(get('LH'))} | {fmt(get('SHARED_BUDGET'))} | {fmt(get('SEASONAL_NAIVE'))} |")
    lines += ['', '## 추가 가치와 연구 판단','']
    for d in decisions:
        lines.append(f"- {d['dataset']}: SHARED 대비 두 seed 모두 양수={d['preservation_development_signal']}; LH 대비 두 seed 모두 양수={d['added_value_over_LH_development_signal']}.")
    lines += ['', '이 기록은 봉인 프로토콜의 서술 규칙에 따른 개발 신호다. 기존 실험의1%/CI/seed 기준을 소급 변경하지 않는다. 같은 데이터에서 다른 이름·gate 초기값·학습률로 재시도하지 않았다.','',
        '## 자원과 실제 실행','',
        '학습 파라미터는 진단군1,319,712개, SHARED1,319,711개, LH761,952개다. SHARED보다1개 많고 LH보다 약73.20% 많으므로 파라미터 절감은 없다. 아래 시간은 실제 완료 trajectory의 optimizer-step 누적 시간이며 선택 checkpoint까지의 배포 비용과 다르다. 기존 실험과 현재 측정의 실행 시점이 달라 인과적인 속도 우위로 해석하지 않는다.','',
        '| fit | epochs | updates | 선택 epoch | peak MiB | active seconds |','|---|---:|---:|---:|---:|---:|']
    for f in fits:lines.append(f"| {f['fit']} | {f['epochs']} | {f['updates']} | {f.get('best',{}).get('epoch','N/A')} | {f.get('peak_allocated',0)/2**20:.3f} | {f.get('active_seconds',0):.2f} |")
    wall=read(OUT/'finite_diagnostic_wall.json') if (OUT/'finite_diagnostic_wall.json').exists() else {}
    lines += ['',f"실행 wall {wall.get('seconds',0):.1f}초, 관측 최소 GPU 여유 {wall.get('minimum_free_mib','N/A')}MiB. GPU 원장에는 승인된 RustDesk와 자체 작업을 구분해 남겼다.",'',
        '## 검증·미실행·한계','',
        f"저장 예측 {checked}개 MSE 및 {mae_checked}개 MAE를 독립 float64 scalar로 재계산했다. 최대 MSE 절대차{max(differences,default=0):.3g}. 선택 체크포인트 {verification['replayed_selected_checkpoints']}개를 새 모델에서 재생해 예측 차이0을 확인했다. 두 원천의 모든 학습 origin에서 V 이후 값을 독으로 바꿔도 학습 batch가 변하지 않았다. 기존 결과·연구 {verification['historical_files_preserved']}개 파일의 해시는 보존됐다.",'',
        f"미완료 본학습은 {4-status['fits_completed']} fits. 계획에 없는 다른 후보·추가 seed·재튜닝은 실행하지 않았다. 독립 미사용 데이터 평가와 새 방법론 비교는 이 진단 범위에 없으며 미실행이다.",'',
        'ReZero는 알려진 방법이며 이번 적용 자체의 신규성은 DIRECT_EQUIVALENCE다. 채널별 저랭크, hypernetwork, 변하는 채널 수의 TSFM 적응에도 가까운 선행이 있다. [선행·연구 검토](RESEARCH_REVIEW.md), [봉인 프로토콜](PROTOCOL.md), [검증](verification.json), [판단](decision.json)을 함께 읽어야 한다.','',
        '현재 새 방법론 주제는 미확보다. 논문 주제를 정하려면 실제 남은 오류 구조와 강한 단순 대조로 설명되지 않는 이득, 가까운 선행과의 구체적 차이가 더 필요하다. GPU 실행이 정상이라는 사실을 예측·신규성 PASS로 바꾸지 않는다. 원시 자료·가중치·예측 배열은 로컬 cache에 남고 GitHub에는 해시와 검증 결과만 보관한다.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(verification,ensure_ascii=False),flush=True)
    print(json.dumps(decisions,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
