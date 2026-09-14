"""Consolidate bounded reopen results; no new experiments or auto research."""
import csv,json,subprocess
from pathlib import Path
from tsfm_peft_screen.reproducibility import ROOT,sha,write_json
R=ROOT/'research/reopen_review_20260914'
def read(p):return json.loads(Path(p).read_text())

def main():
    assessment=read(R/'assessment.json')
    a=read(R/'anchor_evidence.json');p=read(R/'patch_evidence.json');q=read(R/'query_evidence.json')
    fr=read(ROOT/'results/reopen_fr_diagnostic_20260914/summary.json')['states'];ce=read(ROOT/'results/reopen_censor_diagnostic_20260914/summary.json')['states']
    # No numeric performance gate: report tail fractions; materiality is reviewed explicitly below.
    matrix=[]
    def add(method,historical,why,reason,execution,result,status,next_action):matrix.append(dict(method=method,historical_verdict=historical,why_historical_stop=why,new_reason_to_reopen=reason,new_execution=execution,new_result=result,current_status=status,recommended_next_action=next_action))
    add('Uniform Prediction Anchoring','WINDOW_BUDGET_HYPOTHESIS_NOT_SUPPORTED','Sparse-window interaction not supported','Preserve positive and mixed source effects','No repeat fit; replay30 saved E caches',a['verdict'],'CONDITIONAL','Retain uniform anchoring as baseline; no universal regularizer claim')
    hist=read(ROOT/'results/candidate_03/status.json')
    add('PatchPhase v2',hist['verdict'],'Original sparse phase support and fixed pilot criterion','Full rank sin/cos train support','12 fits /8640 updates',p['verdict'],
        {'PATCH_V2_SUPPORTED':'SUPPORTED_FOR_FURTHER_STUDY','PATCH_V2_MIXED':'CONDITIONAL','PATCH_V2_NOT_SUPPORTED':'NOT_SUPPORTED_CURRENT_FORM'}[p['verdict']],
        'No v3; replicate only if mechanism earns a separate plan' if p['verdict']!='PATCH_V2_NOT_SUPPORTED' else 'Close current final-hidden gating branch')
    add('Forecast-Query PEFT','STOP_CURRENT_QUERY','Quality behind Standard; old fixed resource gates','Partial-checkpoint Pareto comparison','18 configs planned; no forecasting fits',q['verdict'],
        'NOT_SUPPORTED_CURRENT_FORM' if q['verdict']=='QUERY_DOMINATED' else 'SYSTEMS_TRADEOFF_ONLY','Close current efficiency branch' if q['verdict']=='QUERY_DOMINATED' else 'Keep only demonstrated memory-time-quality tradeoff; fresh systems replication needed')
    frver='FR_REOPEN_POSSIBLE' if any(x['verdict']=='FR_REOPEN_POSSIBLE' for x in fr) else 'FR_NO_ENTRY'
    add('FR-LoRA','FAIL','All original arms selected step0','Check positive adapted states with nondegenerate revision','No fits; stored-state diagnostic',frver,'DIAGNOSTIC_ONLY','Entry evidence only; no automatic FR fit')
    add('Censor-Preserve LoRA',read(ROOT/'results/candidate_07/status.json')['verdict'],'Behind censored-loss baseline','Check saturated zero-gradient tail','No fits;2 saved states on360 original train batches each',assessment['censor_verdict'],'DIAGNOSTIC_ONLY','If material, tail-loss correction requires separate design; no Censor v2 fit')
    excluded={
        'Freshness v1/v2':('FAIL','No extra value over simple feature baseline after v2','results/candidate_01_v2/RESULT.md'),
        'DualClock extra extension':('WEAK/NOT_SUPPORTED','Fair 360-to1440 extension already covered Standard/Summary/DualClock','results/reassessment_diagnostics_20260914/REPORT.md'),
        'Maturity-PEFT':('FAIL','Recovered implementation remains unfavorable in quality and cost','results/candidate_05/RESULT.md'),
        'Block-shape':('STOP','Acceptance does not beat pooled/simple controls','research/full_reassessment_20260914/REPORT.md'),
        'Calibration-weighted anchor':('STOP','No extra value over uniform/shuffled anchoring','results/calibration_anchor_20260914/REPORT.md'),
        'Drift-conditioned LoRA':('STOP','No supporting additional value; excluded by this instruction','research/full_reassessment_20260914/REPORT.md'),
        'Adaptive freeze/probe/overlap':('STOP','Strong simple controls win after actual cost','research/full_reassessment_20260914/REPORT.md'),
        'Activation compression global/local':('STOP','Unfavorable relative to nearby simple memory/gradient controls','results/local_backward_feasibility/RESULT.md'),
        'Conditional Path':('NOVELTY_COLLISION','Contribution not differentiated; not a quality failure','results/candidate_06/RESULT.md'),
        'Context distillation':('NO_TEACHER_HEADROOM','Student training entry unmet; not a student failure','research/full_reassessment_20260914/REPORT.md')}
    for method,(ver,reason,path) in excluded.items():add(method,ver,reason,'Not reopened per user scope','0 new fits','Historical evidence retained: '+path,'NOT_REOPENED','Do not rerun under current plan')
    with open(R/'METHOD_REOPEN_MATRIX.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(matrix[0]),lineterminator='\n');w.writeheader();w.writerows(matrix)
    receipts={name:read(ROOT/'results'/folder/'receipt.json') for name,folder in [('patch','patchphase_v2_support_complete'),('query','reopen_query_resource_20260914'),('fr','reopen_fr_diagnostic_20260914'),('censor','reopen_censor_diagnostic_20260914')]}
    assert all(r['status']=='COMPLETED' for r in receipts.values())
    assert receipts['patch']['counts']['updates']==8640 and receipts['fr']['optimizer_updates']==receipts['censor']['optimizer_updates']==0
    evidence=dict(assessment=assessment,anchor=a,patch=p,query_verdict=q['verdict'],FR=fr,censor=ce,receipts=receipts,
        main_before_final_archive=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),tests_passed=98,
        explicit_scope='Original verdicts preserved. No excluded fits, no v3/v4, no PR/branch/force push, no auto follow-up.',
        documentation_note='Prior archived CSV CRLF and original proposal final blank line were preserved byte-for-byte to retain historical hashes; new code and final working diff checked.')
    write_json(R/'evidence.json',evidence)
    lines=['# Bounded PEFT reopen review','', '허용된 재검토를 완료했다. 과거 FAIL/STOP은 해당 고정 실험의 판정으로 보존하며 방법 전체의 반증으로 바꾸지 않았다.','',
        f"- Anchor: {a['verdict']}; macro {a['macro']:+.6f}%, sparse/dense interaction {a['interaction']:+.6f}pp. 기존48 fits 재실행0.",
        f"- PatchPhase v2: {p['verdict']}; 12 fits/8640 updates 완료, smoke6 updates 별도.",
        f"- Query: {q['verdict']}; 신규 forecasting fits0, 부분 checkpointing의 함수·gradient·Adam 동등성 후 기존 quality와 결합.",
        f"- FR: {frver}; 신규 fits0. 실제 미래 타깃이 있는 사후 개발 쌍에서 진입 조건만 검사.",
        '- Censor: 신규 fits0. 선택된 두 상태에서 원래360 train batches를 재생한 tail-gradient 비율 보고.','',
        '구체적인 후보 투자 우선순위와 tail materiality의 수치 해석은 FINAL_ASSESSMENT.md에 기록한다. 연구 가치·방법론 성공·진입 조건을 구분한다. 추가 연구는 자동 생성하거나 실행하지 않는다.','',
        '[Anchor](ANCHOR_REVIEW.md) · [PatchPhase](PATCHPHASE_V2_RESULT.md) · [Query](QUERY_RESOURCE_FRONTIER.md) · [FR](FR_ENTRY_DIAGNOSTIC.md) · [Censor](CENSOR_TAIL_DIAGNOSTIC.md) · [방법별 표](METHOD_REOPEN_MATRIX.csv)','',
        '실행 명령(이미 완료한 run은 재실행 금지):','```bash',
        'scripts/with_cuda.sh .venv/bin/python scripts/run_patchphase_support_v2.py smoke',
        'scripts/with_cuda.sh .venv/bin/python scripts/run_patchphase_support_v2.py run',
        'scripts/with_cuda.sh .venv/bin/python scripts/run_reopen_query_resources.py',
        'scripts/with_cuda.sh .venv/bin/python scripts/run_reopen_fr_censor.py fr',
        'scripts/with_cuda.sh .venv/bin/python scripts/run_reopen_fr_censor.py censor',
        'scripts/with_cuda.sh env CUDA_VISIBLE_DEVICES=\'\' .venv/bin/python -m pytest -q tests automation/research_followup/test_observe.py automation/overnight_followup/test_watch.py','```','']
    (R/'STATUS.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
