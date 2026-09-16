"""Finish the human-readable report using verified, already sealed results only."""
import csv
from completion_audit import audit
from runtime import *

def read_csv(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))

def finalize():
    assert state_json()['status']=='COMPLETE'
    audit()
    macro=read_csv('macro_scores.csv');effects=read_csv('effects.csv');cost=read_csv('selection_cost.csv');main=read_csv('main_resources.csv');trades=read_csv('resource_tradeoffs.csv')
    sensitivity=[]
    for r in macro:
        if r['panel']!='E_MIXED':continue
        fixed=next(x for x in macro if x['dataset']==r['dataset'] and x['arm']==r['arm'] and x['policy']==r['policy'] and x['panel']=='E_FIXED')
        a,b=float(r['mse']),float(fixed['mse'])
        sensitivity.append(dict(dataset=r['dataset'],arm=r['arm'],policy=r['policy'],mixed_mse=a,fixed_mse=b,mixed_minus_fixed=a-b,mixed_relative_change_pct=100*(a/b-1),interpretation='Different target sets; descriptive panel sensitivity, not pure phase causal effect'))
    csvwrite(OUT/'panel_sensitivity.csv',sensitivity)
    wall=read(OUT/'all_wall.json');fits=read(OUT/'fits.json');s=state_json();newrows=[r for r in main if r['mode']=='NEW']
    totals=dict(new_fits=s['new_fits_completed'],reused_fits=s['reused_fits_completed'],new_updates=s['training_updates'],reused_updates=s['reused_updates'],smoke_updates=s['smoke_updates'],resource_updates=s['resource_updates'],this_run_optimizer_updates=s['training_updates']+s['smoke_updates']+s['resource_updates'],gpu_controller_seconds=wall['seconds'],wait_seconds=wall['wait_seconds'],new_active_training_seconds=sum(float(r['active_seconds']) for r in newrows),all_current_validation_seconds=sum(f['validation_seconds'] for f in fits),new_resume_save_seconds=sum(f['resume_save_seconds'] for f in fits if f['mode']=='NEW'),new_fit_wall_seconds=sum(f['wall_seconds'] for f in fits if f['mode']=='NEW'),comparison_exposure='DISCOVERY_REUSED_PERIODS')
    save(OUT/'execution_totals.json',totals)
    source=OUT/'REPORT.md';archive=OUT/'REPORT_before_completion_audit.md'
    assert not archive.exists(),'Finalization already applied; no duplicate append'
    archive.write_bytes(source.read_bytes())
    text=source.read_text()
    appendix=['','## 선택 비용·자원 절충·평가 패널 민감도 추가 검산','',
        f"이번 GPU controller wall은 {wall['seconds']/60:.2f}분이며 이 중 GPU 안전 대기는 {wall['wait_seconds']:.2f}초다. 신규 본학습의 active optimizer 구간 합은 {totals['new_active_training_seconds']/60:.2f}분, 신규 fit 전체 wall 합은 {totals['new_fit_wall_seconds']/60:.2f}분이다. 추가 검증·모델 로드·저장·안전 감시 비용 때문에 active 시간과 전체 시간을 구분했다. 이 실행에서 실제 수행한 optimizer 업데이트는 본학습+smoke+자원 합 {totals['this_run_optimizer_updates']:,}회다. 재사용 {totals['reused_updates']:,}회는 과거 작업으로 별도 집계한다.", '',
        '선택된 prefix와 두 LR 전체 학습의 비용은 [selection_cost.csv](selection_cost.csv)에 모든 정책·seed별로 있다. 20 epoch에서 V 최소를 갱신한 경로는 [budget_status.csv](budget_status.csv)의 BUDGET_LIMITED로 표시했으며 자동 연장하지 않았다. 전체 신규 28개 완전 재개 상태를 [resume_audit.json](resume_audit.json)에서 검산했다.', '',
        '| 원천 | 자원 옵션 vs LH-current | 같은 seed의 정확도 이득 % | 학습 peak 절감 % | step 시간 절감 % |','|---|---|---:|---:|---:|']
    for r in trades:appendix.append(f"| {r['dataset']} | {r['option']} | {float(r['accuracy_gain_pct']):+.4f} | {float(r['training_memory_saving_pct']):+.2f} | {float(r['step_time_saving_pct']):+.2f} |")
    appendix += ['', '이 자원 표의 정확도는 동일한 선택 seed41000이고, 주 성능 판정은 앞의 3seed 평균과 별개다. 자원 benchmark는 선택 가중치에서 같은 arm의 저장된 빈 Adam/RNG를 복구하고 동일한 초기 LR로 warmup2+timed9를 수행했다. 과거 optimizer 연속 학습을 복원한 것이 아니다. GUI 부하는 제거하지 않았으므로 시간은 이 데스크톱 환경의 실측이며 전용 GPU 조건의 속도를 보증하지 않는다. 비승인 compute가 없었다는 기록과 GUI 부하 통제는 다르다.', '',
        '| 원천 | P_MAIN 방법 | E_FIXED MSE | E_MIXED MSE | MIXED 상대 변화 % |','|---|---|---:|---:|---:|']
    for r in sensitivity:
        if r['policy']=='P_MAIN':appendix.append(f"| {r['dataset']} | {r['arm']} | {r['fixed_mse']:.9f} | {r['mixed_mse']:.9f} | {r['mixed_relative_change_pct']:+.4f} |")
    appendix += ['', '양의 패널 변화는 E_MIXED 점수가 더 높다는 뜻이다. 두 패널의 target 묶음은 달라서 이 수치 전체를 순수한 위상 변화의 효과로 해석하지 않는다. 모든 정책 및 baseline은 [panel_sensitivity.csv](panel_sensitivity.csv)에 남겼다.', '',
        '추가 검산은 새 fit·forward·성능 선택 없이 저장된 산출물만 읽었다. [completion_audit.json](completion_audit.json), [실제 실행 집계](execution_totals.json), [모듈별 smoke 변화](smoke_group_audit.json).']
    appendix += ['', 'native checkpoint를 켠 LH는 두 원천에서 학습 peak 401.59MiB로 SIDE-fast 424.74MiB와 PRIOR-current 425.87MiB보다 작았다. 단, step 시간은 약 72ms로 SIDE-fast의 약 36~37ms보다 길었다. 따라서 side 경로의 메모리 이득은 checkpoint를 끈 LH 대비에 한정되며, PRIOR 고유 이점으로 볼 수 없다. 이 옵션은 선택 가중치에서 자원만 측정했고 두 원천 각각 동일 배치의 2 updates 출력·gradient·가중치가 bitwise 일치했다. native checkpoint로 20 epochs 전체 경로를 새로 학습한 것은 아니다.']
    source.write_text(text+'\n'.join(appendix)+'\n')
    decision_path=OUT/'FINAL_DECISION.md'
    roles='\n## 구현의 역할\n\n- LH: 교정된 TRAIN의 정확도 기준 구현으로 보존.\n- SIDE: 동일 예산의 저메모리 대조 구현으로 보존.\n'
    roles += '- PRIOR: 이번 구현과 recipe만 고정 보존. 새 head·gate·온도·후속 학습은 자동 실행하지 않음.\n' if s['decision']=='A' else '- PRIOR: 추가 연구 후보 확장과 후속 학습 중단. 기존 코드·가중치·결과는 재현용 기록으로 보존.\n'
    roles += '\nLH의 native checkpoint 옵션도 자원 대조 구현으로 보존한다. 이번 동일 배치 2-update 검사는 두 원천에서 bitwise 통과했고 peak는 SIDE-fast보다 낮았다. 전체 20-epoch checkpoint 학습을 추가 실행한 것은 아니며, 시간 비용은 더 컸다.\n'
    decision_path.write_text(decision_path.read_text()+roles)
    needed=['PROTOCOL.md','SCOPE.md','phase_panels.csv','exposure_ledger.md','schedules.json','fit_manifest.csv','fit_attempts.csv','training_curves.csv','V_FIXED_scores.csv','V_MIXED_scores.csv','V_FIXED_predictions.json','V_MIXED_predictions.json','selection_seal.json','E_FIXED_scores.csv','E_MIXED_scores.csv','effects.csv','uncertainty.csv','prior_dependency.csv','resources.csv','independent_verification.json','REPORT.md','FINAL_DECISION.md','validation_curves.png','seed_gains.png','accuracy_resources.png','phase_distributions.png']
    assert all((OUT/name).is_file() for name in needed)
    save(OUT/'required_artifacts.json',{name:sha(OUT/name) for name in needed})
    print('REPORT_FINALIZED',totals,flush=True)

if __name__=='__main__':finalize()
