"""Explain preregistered selected/fixed120 contrasts after independent verification."""
import sys
from pathlib import Path
import finalize_building_peft_topic_decision_20260916 as f

def main():
    status=f.read(f.OUT/'final_status.json');assert status['execution']=='COMPLETE'
    d=f.read(f.OUT/'scientific_decision.json');macro={r['arm']:r for r in d['macro']}
    import csv
    buildings=list(csv.DictReader((f.OUT/'building_summary.csv').open()))
    groups={r['method']:r for r in f.read(f.OUT/'resource_summary.json')['groups'] if r['role']=='LOCKED'}
    contrasts=[('STD','F0','타깃 LoRA 학습 / 선택 recipe'),('SIMPLE','STD','균일 보존 / 선택 recipe'),('CANDIDATE','SIMPLE','잔차 방향 배분 / 선택 recipe'),('STD_FIXED120','F0','타깃 LoRA 학습 / 고정120'),('SIMPLE_FIXED120','STD_FIXED120','균일 보존 / 고정120'),('CANDIDATE_FIXED120','SIMPLE_FIXED120','잔차 방향 배분 / 고정120')]
    rows=[]
    for arm,base,component in contrasts:
        by={r['building']:float(r['primary']) for r in buildings if r['arm']==arm};bb={r['building']:float(r['primary']) for r in buildings if r['arm']==base};ids=sorted(by);assert ids==sorted(bb)
        eff=f.paired_effect([bb[b] for b in ids],[by[b] for b in ids])
        row=dict(component=component,arm=arm,baseline=base,arm_primary=macro[arm]['primary'],baseline_primary=macro[base]['primary'],gain_percent=eff['gain_percent'],ci95_low_percent=eff['ci95_percent'][0] if eff['ci95_percent'] else None,ci95_high_percent=eff['ci95_percent'][1] if eff['ci95_percent'] else None,raw_RMSE_gain_percent=f.gain(macro[base]['raw_RMSE'],macro[arm]['raw_RMSE']),interpretation='Fixed120 is a preregistered audit contrast, not a newly selected deployment policy' if arm.endswith('120') else 'Frozen DISCOVERY recipe, applied to both LOCKED seeds')
        rows.append(row)
    f.csvwrite(f.OUT/'component_contributions.csv',rows)
    resources=[]
    for method in ['SIMPLE','CANDIDATE']:
        base=groups['STD'];r=groups[method]
        resources.append(dict(method=method,baseline='STD',same_trainable=147456,parameter_reduction_percent=0.,full_trajectory_median_peak_allocated_reduction_percent=f.gain(base['median_peak_allocated_MiB'],r['median_peak_allocated_MiB']),fixed120_gradient_time_reduction_percent=f.gain(macro['STD_FIXED120']['selected_gradient_seconds_mean'],macro[method+'_FIXED120']['selected_gradient_seconds_mean']),full_trajectory_median_peak_allocated_MiB=r['median_peak_allocated_MiB'],baseline_full_trajectory_median_peak_allocated_MiB=base['median_peak_allocated_MiB'],fixed120_gradient_seconds=macro[method+'_FIXED120']['selected_gradient_seconds_mean'],baseline_fixed120_gradient_seconds=macro['STD_FIXED120']['selected_gradient_seconds_mean'],selected_adaptation_updates=macro[method]['selected_mean_updates']))
    f.save(f.OUT/'resource_benefits.json',resources)
    text='\n\n## 구성요소별 추가 가치와 자원 이득\n\n아래 비교는 사전에 남기기로 한 선택 recipe와 고정120을 분리한다. 평가에서 더 좋은 고정120을 찾아 기존 ZERO 선택을 바꾸는 배포 정책이 아니다. 양수는 비교 대상보다 좋은 방향이다.\n\n| 추가 구성요소 | 기준 primary | 변경 primary | 이득 % | 건물 paired95% CI % | raw RMSE 이득 % |\n|---|---:|---:|---:|---|---:|\n'
    for r in rows:
        ci=None if r['ci95_low_percent'] is None else [r['ci95_low_percent'],r['ci95_high_percent']]
        text+=f"| {r['component']} | {r['baseline_primary']:.9f} | {r['arm_primary']:.9f} | {f.fnum(r['gain_percent'])} | {f.interval(ci)} | {f.fnum(r['raw_RMSE_gain_percent'])} |\n"
    text+='\n| 방법 vs STD | 파라미터 감소 % | 전체 trajectory median peak allocated 감소 % | fixed120 gradient 시간 감소 % |\n|---|---:|---:|---:|\n'
    for r in resources:text+=f"| {r['method']} | 0 | {f.fnum(r['full_trajectory_median_peak_allocated_reduction_percent'])} | {f.fnum(r['fixed120_gradient_time_reduction_percent'])} |\n"
    text+='\n[확인] 세 방법의 선택은 모두 ZERO이므로 선택된 배포의 adaptation updates는0이고, 학습 시간 비율0/0은 N/A다. 표의 메모리는 전체 trajectory에서 측정한 peak이며 ZERO 배포의 추가 메모리 측정치가 아니다. fixed120 시간은 gradient 구간만의 값이다. Fit wall은 점점 커지는 원장의 저장 비용과 고정 실행 순서에도 영향을 받으므로 방법 자체의 속도 차이라고 단정하지 않는다. 메모리·시간이 같다는 수학적 주장은 하지 않는다.\n'
    refs=f.read(f.OUT/'independent_references.json');cal=f.read(f.OUT/'independent_calibration.json')
    text+=f"\n[확인] 공통 참조 {refs['episodes']} episodes의 계산을 독립 복원했고 affine ill-conditioning fallback은 {refs['affine_fallbacks']}회였다. 이력 {len(cal)}개의 native F0와 과거 잔차로 lambda·레벨/형상 가중치를 독립 재계산했다. [후처리 개발 기록](POSTPROCESSING_REVIEW.md)은 학습 오류와 별도로 보존한다.\n"
    report=f.OUT/'REPORT.md';original=report.read_text().split('\n\n## 구성요소별 추가 가치와 자원 이득')[0];report.write_text(original+text)
    f.save(f.OUT/'component_report_provenance.json',dict(source_sha256=f.sha(__file__),scientific_decision_sha256=f.sha(f.OUT/'scientific_decision.json'),extra_fits=0,changes_to_selection=0,changes_to_scientific_decision=0))
    print('Component contribution and measured resource benefit tables appended; no additional training')
if __name__=='__main__':main()
