"""CPU-only independent gradient/CSV audit and scoped D report."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from tsfm_peft_screen.reproducibility import ROOT, sha, write_json

OUT=ROOT/'results/temporal_transfer_diagnostic_v1_D'
PARENT=ROOT/'results/temporal_transfer_diagnostic_v1'


def read(p): return json.loads(Path(p).read_text())
def rows(p): return list(csv.DictReader(open(p)))
def close(a,b):
    assert np.isclose(float(a),float(b),rtol=1e-10,atol=1e-12),(a,b)


def main():
    receipt=read(OUT/'execution_receipt.json')
    assert receipt['status']=='COMPLETED'
    assert receipt['states_completed']==12 and receipt['forward']==180 and receipt['backward']==60
    assert receipt['optimizer_updates']==receipt['new_fits']==receipt['E_array_reads']==receipt['permanent_weight_changes']==0
    geom=rows(OUT/'gradient_geometry.csv');eff=rows(OUT/'local_perturbation_effects.csv');base=rows(OUT/'precision_baselines.csv')
    restores=read(OUT/'restoration_checks.json')
    assert len(geom)==180 and len(eff)==96 and len(base)==36 and len(restores)==48
    assert all(all(r[k] for k in ('trainable_restored','frozen_unchanged','checkpoint_unchanged','rng_restored')) for r in restores)
    states={}; checks=0
    for p in sorted(OUT.glob('*_state.json')):
        s=read(p); sid=p.name[:-len('_state.json')]
        assert sha(ROOT/s['gradient_file'])==s['gradient_sha256']
        with np.load(ROOT/s['gradient_file'],allow_pickle=False) as f: g={n:f[n].copy() for n in f.files}
        assert all(v.shape==(1179648,) and np.isfinite(v).all() for v in g.values())
        for r in [r for r in geom if r['state']==sid]:
            a,b=g[r['a']],g[r['b']];na=np.linalg.norm(a);nb=np.linalg.norm(b);dot=np.dot(a,b)
            close(na,r['norm_a']);close(nb,r['norm_b']);close(dot,r['dot']); checks+=3
            if na and nb: close(dot/na/nb,r['cosine']);checks+=1
            else: assert r['cosine']=='UNDEFINED_ZERO_GRADIENT'
        for r in [r for r in eff if r['state']==sid]:
            direction=-g[r['direction']]/np.linalg.norm(g[r['direction']])
            pred=float(r['epsilon'])*np.dot(g['eval_'+r['region']],direction)
            close(pred,r['first_order_delta']); close(float(r['perturbed_loss'])-float(r['base_loss']),r['actual_delta'])
            close(float(r['factor'])*max(1,float(r['phi_norm'])),r['epsilon']);checks+=3
        states[sid]=s
    groups=defaultdict(list)
    for r in eff:groups[(r['state'],r['direction'],r['region'])].append(r)
    pair=[]; decisions={}
    for (sid,direction,region),rr in sorted(groups.items()):
        rr=sorted(rr,key=lambda r:float(r['factor']));assert [float(r['factor']) for r in rr]==[1e-4,1e-3]
        good=[]
        for r in rr:
            a,p,f=float(r['actual_delta']),float(r['first_order_delta']),float(r['numerical_floor'])
            reliable=abs(a)>f and abs(p)>f and a*p>0 and abs(a-p)<=.5*abs(p)+f
            assert reliable==(r['local_reliable']=='True');good.append(reliable);checks+=1
        stable=all(good) and float(rr[0]['actual_delta'])*float(rr[1]['actual_delta'])>0
        decision=('IMPROVES' if float(rr[0]['actual_delta'])<0 else 'WORSENS') if stable else 'NONLOCAL_OR_NUMERICAL'
        decisions[(sid,direction,region)]=decision
        pair.append(dict(state=sid,direction=direction,region=region,decision=decision,
            small_delta=float(rr[0]['actual_delta']),large_delta=float(rr[1]['actual_delta']),
            small_prediction=float(rr[0]['first_order_delta']),large_prediction=float(rr[1]['first_order_delta'])))
    evidence=[]
    for sid in sorted(states):
        objectives=[r for r in ['S','D'] if decisions[sid,'native_T',r]=='WORSENS' and decisions[sid,'eval_T',r]=='IMPROVES']
        temporal=[d for d in ['native_T','eval_T'] if {decisions[sid,d,'S'],decisions[sid,d,'D']}=={'IMPROVES','WORSENS'}]
        evidence.append(dict(state=sid,source=states[sid]['state']['source'],objective_conflict_regions=objectives,
            temporal_conflict_directions=temporal,decisions={f'{d}_{r}':decisions[sid,d,r] for d in ['native_T','eval_T'] for r in ['S','D']}))
    objective=sum(bool(e['objective_conflict_regions']) for e in evidence)
    temporal=sum(bool(e['temporal_conflict_directions']) for e in evidence)
    status=[]
    if objective:status.append('LOCAL_OBJECTIVE_CONFLICT')
    if temporal:status.append('LOCAL_TEMPORAL_CONFLICT')
    if not status:status=['INSUFFICIENT_EVIDENCE']
    old=read(OUT/'source_and_history_hashes.json');assert all(sha(ROOT/p)==h for p,h in old.items())
    summary=dict(status=status,objective_conflict_states=objective,temporal_conflict_states=temporal,
        pair_decisions=dict(Counter(p['decision'] for p in pair)),states=evidence,pairs=pair,
        precision=dict(max_abs_bf16_loss_difference=max(abs(float(r['bf16_minus_fp32'])) for r in base),
                       max_cached_bf16_prediction_difference=max(float(r['cached_bf16_prediction_max_abs']) for r in base if r['cached_bf16_prediction_max_abs']),
                       fp32_sort_ties=sum(int(r['fp32_sort_ties']) for r in base),fp32_target_ties=sum(int(r['fp32_target_ties']) for r in base)))
    write_json(OUT/'analysis_summary.json',summary)
    write_json(OUT/'artifact_verification.json',dict(status='PASS',independent_numeric_checks=checks,
        gradient_files=12,gradient_pairs=180,perturbation_measurements=96,restorations=48,
        historical_files_unchanged=len(old),actual_metric_implementations='During GPU run each of 168 losses checked against metrics.score and independent scalar replay; CPU synthetic finite-difference tests passed. Reporting rechecks stored gradients and CSV algebra without GPU.',
        limits='Numerical consistency PASS only; not a method-performance or paper PASS.'))
    lines=['# 고정 체크포인트 기울기·미소 교란 진단 D','',
        f"진단 완료: 12개 상태, forward {receipt['forward']}회, backward {receipt['backward']}회, 미소 교란 48개. 새 학습·optimizer update·E 배열 접근·영구 가중치 변경은 모두 0이다.",
        f"실행 시간 {receipt['elapsed_seconds']/60:.2f}분(모델/데이터 사전 해시 검사 및 구현·보고서 작성 시간 제외). GPU 최대 할당 {receipt['resources']['gpu_peak_bytes']/2**30:.2f} GiB.",'',
        f"판정: {', '.join(status)}. 국소 목적 충돌 {objective}/12 상태, 국소 시간 구간 충돌 {temporal}/12 상태.",
        '이는 고정 step450 상태와 각 구간 마지막 2개 origin에 한정된 결과다. 실제 Adam 업데이트나 장기 학습 실패의 원인을 입증하지 않으며, 모든 상태·시간 구간으로 일반화할 수 없다.','',
        '| 상태 | native→S | native→D | eval→S | eval→D |','|---|---|---|---|---|']
    for e in evidence:
        v=e['decisions'];lines.append('| '+e['state']+' | '+' | '.join(v[k] for k in ['native_T_S','native_T_D','eval_T_S','eval_T_D'])+' |')
    lines+=['','두 epsilon 모두 부호가 일치하고 1차 예측과 근사가 맞는 경우만 IMPROVES/WORSENS로 읽었다. NONLOCAL_OR_NUMERICAL에는 효과가 FP32 수치 바닥보다 작거나, 두 크기의 부호가 다르거나, 근사가 나쁜 경우가 포함된다. epsilon이나 임계값을 결과에 맞춰 변경하지 않았다.',
        '수치 바닥은 8×FP32 epsilon×max(1,|baseline loss|), 허용 근사 오차는 |예측|의 50%+수치 바닥이다. 이것은 진단의 수치 신뢰성 필터이며 성능 PASS 기준이 아니다. 원시 효과와 1차 예측은 local_perturbation_effects.csv에 보존했다.','',
        f"BF16/FP32 평가 손실 차이의 최대 절댓값: {summary['precision']['max_abs_bf16_loss_difference']:.8g}. 원래 저장 BF16 V 예측과 재계산의 최대 절댓값 차이: {summary['precision']['max_cached_bf16_prediction_difference']:.8g}.",
        f"FP32 baseline sort tie {summary['precision']['fp32_sort_ties']}개, pinball target tie {summary['precision']['fp32_target_ties']}개. 결측값 마스크와 채널별 유효 관측 수, train-only scale을 유지했다.",
        'native task는 기존 native_loss/21이며 anchor는 기존 0.1×정렬된 F0 출력 L1/채널 scale이다. native arm에서도 anchor 기울기를 기하 비교용으로 계산했다. 모든 gradient는 동일한 LoRA 1,179,648개 좌표다.','',
        '48개 교란마다 trainable/frozen/checkpoint/RNG 복원 또는 보존 검사를 통과했다. 교란 48개는 모두 폐기했다. 새 optimizer는 생성하지 않았다. 원본 A–C의 D 보류 기록은 당시 상태를 나타내므로 그대로 보존했다.',
        'GPU에는 학습 프로세스가 없었고 RustDesk 표시 서비스만 있었다. 정확한 실행 경로와 512MiB 이하 메모리에만 예외를 허용했으며, 다른 계산 프로세스·메모리 부족·기존 RAM/2시간 제한을 검사했다.','',
        '## A–D 통합 해석과 다음 한 가지 제안','',
        'A는 seed별 gain 보고식 오류를 수정했지만 기존 주된 실패 판정은 바뀌지 않았다. B는 시간 구간에 따라 선택 가능한 체크포인트의 유효성이 달라지는 사례와 F0 선택이 뒤 구간의 개선 후보를 놓치는 사례를 분리했다. C는 S에서 고른 중간 출력 보정 강도가 4/24 셀에서 뒤 구간의 두 끝점보다 좋았지만 한 seed에 집중되었다.',
        'D는 위 표의 제한된 상태에서만 목적/시간 방향을 판독한다. 서로 다른 실험의 상태·origin·선택 규칙이므로 D만으로 B/C의 현상을 모두 설명했다고 주장할 수 없다.','',
        '다음 우선순위는 기존 제안인 새 미래 구간에서 RECENT4와 SPREAD4 선택 규칙을 같은 검증 비용으로 비교하는 0-fit 실험 하나다. D에서 국소 충돌이 관찰되어도 바로 gradient surgery나 새로운 PEFT 방법을 채택하지 않는다. 이 후속 실험은 제안만 유지하며 이번 실행에서 시작하지 않았다.',
        '방법론 논문 후보를 평가하기 전에 선택 절차가 실제 미래 구간으로 전달되는지 분리해서 확인하는 단계다. 논문 PASS나 성능 개선을 보장하는 결과가 아니다.','',
        '[A–C 원본 보고서](../temporal_transfer_diagnostic_v1/REPORT.md) · [후속 실험 설계](../temporal_transfer_diagnostic_v1/next_experiment_proposal.md) · [실행 영수증](execution_receipt.json) · [수치 검산](artifact_verification.json)','']
    report='\n'.join(lines)
    (OUT/'gradient_probe_report.md').write_text(report)
    (OUT/'REPORT.md').write_text(report)
    write_json(OUT/'reporting_receipt.json',dict(status='COMPLETED',diagnostic_status=status,
        reporting_gpu_forward=0,reporting_gpu_backward=0,git_push=False,automatic_followup=False,
        report_sha256=sha(OUT/'REPORT.md'),runner_sha256=sha(ROOT/'scripts/run_temporal_transfer_gradient_v1.py'),
        reporting_script_sha256=sha(Path(__file__))))
    print(json.dumps(dict(status=status,objective_states=objective,temporal_states=temporal,checks=checks,precision=summary['precision']),indent=2))


if __name__=='__main__':main()
