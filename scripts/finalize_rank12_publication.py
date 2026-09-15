"""Postprocess completed reports, audit publication and index; no model calls."""
import csv,hashlib,json,math,re,statistics,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/peft_rank12_20260915';Q=ROOT/'results/query_budget_repair_v2_20260915';C=ROOT/'results/channel_sharing_screen_v1_20260915'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def main():
    q=read(Q/'status.json');c=read(C/'status.json');assert (OUT/'completion.json').exists(),'Wait for parent finalization'
    assert c['status']=='COMPLETE','No complete-result postprocessing for unexecuted track'
    history=read(OUT/'historical_hashes.json')
    for p,h in history.items():assert sha(ROOT/p)==h,p
    for folder in [Q,C]:
        for p,h in read(folder/'contract.json')['source_hashes'].items():assert sha(ROOT/p)==h,p
    verification=read(C/'independent_verification.json');assert verification['E_scope']=='MEASURED' and verification['max_metric_abs_error']<=1e-10
    decisions=read(C/'decisions.json')['source_decisions'];fits=read(C/'fits.json');metric=rows(C/'metrics.csv');comp=rows(C/'comparisons.csv');params={r['arm']:int(r['total_trainable']) for r in rows(C/'PARAMETER_BUDGET.csv')}
    checks=read(Q/'numeric_checks.json');numeric=[]
    for r in checks:
        numeric.append(dict(dataset=r['dataset'],arm=r['arm'],pair=str(r['pair']),kind=r['kind'],precision=r['precision'],passed=r['passed'],legacy_fp32_1e5_pass=r['legacy_fp32_1e5_pass'],raw_scaled_max_abs=r['metrics']['raw_scaled']['max_absolute'],raw_gradient_relative_l2=r['metrics']['raw_gradient']['relative_l2'],update_rms_error=r['metrics']['update']['rms_error'],update_rms_reference=r['metrics']['update']['rms_reference'],update_relative_l2=r['metrics']['update']['relative_l2'],exp_avg_rms_error=r['metrics']['exp_avg']['rms_error'],exp_avg_sq_rms_error=r['metrics']['exp_avg_sq']['rms_error'],sign_flips=r['pinball_sign_flips']))
    with (Q/'numeric_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(numeric[0]),lineterminator='\n');w.writeheader();w.writerows(numeric)
    lines=['# R1/R2 최종 실행 검토','',f"**R1 {q['status']} / R2 {c['status']}.** R1 실제 수치 {q['numeric_updates']} updates·자원 {q['A_updates']} updates·본학습 {q['B_fit_attempts']} fits. R2 smoke {c['smoke_updates']} updates·본학습 {c['fits_completed']}/{c['fit_attempts']} fits·{c['training_updates']:,} updates와 평가를 완료했다.",'', '## 핵심 판정','', '| 원천 | 실행 유효 | 예산 이득 신호 | 압축 신호 | REF가 LH보다 정확함 | 학습6군 중 최저 MSE | 신규성 |','| --- | --- | --- | --- | --- | --- | --- |']
    for d in decisions:lines.append(f"| {d['dataset']} | {d['EXECUTION_VALID']} | {d['BUDGET_SIGNAL']} | {d['COMPRESSION_SIGNAL']} | {d['REFERENCE_EFFECT_REPRODUCED_OR_NOT']} | {d['best_arm']} | {d['NOVELTY_STATUS']} |")
    lines+=['','## 두 seed 평균 원점수','', '| arm | 전체 trainable | Electricity MSE | Traffic MSE |','| --- | ---: | ---: | ---: |']
    for a,n in params.items():lines.append(f"| {a} | {n:,} | {decisions[0]['means'][a]:.9f} | {decisions[1]['means'][a]:.9f} |")
    lines+=['','Train 표준화 공간의 equal-channel MSE이며 낮을수록 좋다. 아래 개선율은 원천별 두 seed 원점수를 먼저 평균했다.','', '| 원천 | 기준 | 기준 MSE | BASIS MSE | BASIS 개선율(%) |','| --- | --- | ---: | ---: | ---: |']
    for r in comp:
        if r['seed']=='mean':lines.append(f"| {r['dataset']} | {r['baseline']} | {float(r['baseline_mse']):.9f} | {float(r['basis_mse']):.9f} | {float(r['gain_percent']):+.4f} |")
    lines+=['','## 구성요소와 자원의 추가 가치','']
    reduction=100*(1-params['BASIS_BUDGET']/params['INDIV_REF']);lines.append(f'BASIS의 전체 학습 파라미터는 INDIV_REF보다 {reduction:.3f}% 적다. SHARED_BUDGET과 같은 전체 trainable 수를 갖는다. 압축 신호와 학습된 계수의 효과는 별개다.');lines.append('')
    for d in decisions:
        means=d['means'];basis=means['BASIS_BUDGET'];b=[f for f in fits if f['dataset']==d['dataset'] and f['arm']=='BASIS_BUDGET'];ref=[f for f in fits if f['dataset']==d['dataset'] and f['arm']=='INDIV_REF'];lh=[f for f in fits if f['dataset']==d['dataset'] and f['arm']=='LH']
        lines.append(f"- {d['dataset']}: BASIS가 SHARED보다 {'낮은' if basis<means['SHARED_BUDGET'] else '높은'} 평균 MSE, FACTOR보다 {'낮은' if basis<means['FACTOR_BUDGET'] else '높은'} 평균 MSE다. LH 대비 MSE 변화 {100*(basis/means['LH']-1):+.3f}%. BASIS 평균 active {statistics.mean(f['active_seconds'] for f in b):.3f}초 / REF {statistics.mean(f['active_seconds'] for f in ref):.3f}초 / LH {statistics.mean(f['active_seconds'] for f in lh):.3f}초. 서로 epoch 수가 달라 총 시간 감소를 동일 작업량 속도 향상으로 해석하지 않는다.")
    lines+=['','Traffic에서는 seasonal-naive MSE0.981448809가 학습6군 중 최선인 LH1.235361901보다도 낮다. Electricity에서는 LH0.469379966이 seasonal-naive0.525609095보다 낮다. 두 원천 모두 BASIS가 무학습 기준보다 더 정확하다는 근거는 없다.','', '자원 표의 allocated/reserved peak는 기록된 training step 최대값이다. 모델 생성·V·체크포인트 재로딩을 포함한 전 구간 tensor peak를 측정했다고 하지 않는다. 전 구간 GPU NVML/free 감시는 별도 원본 로그에 있다. 추가 inference timing/static merge benchmark는 실행하지 않았다.','', '| 선택 종류 | fit 수 |','| --- | ---: |',f"| INIT 선택 | {sum(f['best']['epoch']==0 for f in fits)} |",f"| patience 조기 종료 | {sum(f['early_stopped'] for f in fits)} |",f"| BUDGET_LIMITED | {sum(f['budget_limited'] for f in fits)} |",'', 'INIT가 선택된 BASIS에서는 계수 개입이 불변이어도 학습된 특화의 인과 효과를 시험한 결과로 해석하지 않는다. 초기 residual basis가0인 구조와 선택된 epoch를 함께 봐야 한다.','', '## R1 수치 중단과 미실행','', '| 검사 | 통과/전체 |','| --- | --- |']
    for kind,precision in [('checkpoint','fp32'),('microbatch','fp32'),('microbatch','bf16')]:
        rr=[r for r in checks if r['kind']==kind and r['precision']==precision];lines.append(f"| {kind}/{precision} | {sum(r['passed'] for r in rr)}/{len(rr)} |")
    lines+=['','[48개 수치 요약](../../results/query_budget_repair_v2_20260915/numeric_summary.csv). FP64 의미 검사와 실제 origin 격리는 통과했다. 새 RMS 정책은 기존 실패 이후 고정된 개정이다. BF16 실패를 없애려고 FP32 본학습으로 바꾸지 않았다. 의미 오류가 입증되지 않아 추측으로 학습 코드를 수정하거나 남은 수치 예산12회를 재튜닝에 쓰지 않았다. 자원 검사480updates와 조건부12fits는 진입 조건 미충족으로 NOT_RUN이며 예측 성능 실패가 아니다.','', '## 준비 오류·검산·남은 범위','', '초기 R1 CPU 검사를 R2 환경에서 수집한 명령 오류는 환경 분리로 바로잡았다. R2 환경 목록 기록은 기존 uv 환경에 pip가 없어 한 차례 실패했다. 설치 변경 없이 importlib.metadata로 수정하고 완료된 CPU/모델/데이터 검사를 보존한 채 준비 끝부분만 마무리했다. 본학습 재시도나 cell 대체는 없다. 학습 전 끝 공백 정리는 계약 이전 버전과 함께 기록했다.','',f"R2 예측 기록 {verification['prediction_replays']}개를 독립 scalar float64로 재계산했고 최대 MSE 절대차 {verification['max_metric_abs_error']:.3g}였다. 기존 결과·연구 {len(history)}개와 실행 소스 해시는 불변이다. 원점수·분모·seed별 gain·CI·실제 자원은 [R2 상세 보고서](../../results/channel_sharing_screen_v1_20260915/REPORT.md)에 있다.",'', 'R2는 알려진 affine factorization의32채널·길이96·고정 recipe 비교다. 이전64채널 실험에서 바뀐 여러 조건의 개별 효과나 미노출 원천의 범용성·공식 Time-PEFT 전체 재현은 미검증이다. 공식 C-LoRA/MoLA의 같은 백본 직접 비교도 미실행이다.','', 'NEXT_ACTION R1: BF16 수치 동등성 미확정을 남기고 현재 자원 제약 후보 투자를 보류한다. NEXT_ACTION R2: 이번 최강 단순 대조군을 기준으로 결과를 검토해 후속 투자 여부를 결정한다. 트랙별 추가 학습·후속 후보는 자동 실행하지 않는다.','']
    for name in ['parameters_vs_error','basis_gains','validation_curves']:
        if (C/(name+'.png')).exists():lines += [f'![{name}](../../results/channel_sharing_screen_v1_20260915/{name}.png)','']
    lines += ['## 검토 자료와 재현 범위', '', '[기준 commit 차이와 중복 검사](AUDIT.md), [최종 보존·GPU·초기값 감사](publication_audit.json), [실제 epoch별 학습 곡선](../../results/channel_sharing_screen_v1_20260915/training_curves.csv). 원시자료·모델 가중치·예측 tensor cache는 로컬에 보존하고 GitHub에는 코드·원점수·해시·검산 기록을 공개한다. 저장소만으로 저장 예측을 수치 재검산할 수 있다고 주장하지 않는다.', '']
    detail=C/'REPORT.md'
    detail.write_text(detail.read_text().replace('최저 MSE=LH,', '학습6군 중 최저 MSE=LH,'))
    # Keep original auto-generated combined report and add a readable outcome-led report.
    if not (OUT/'REPORT.automated.md').exists():(OUT/'REPORT.automated.md').write_bytes((OUT/'REPORT.md').read_bytes())
    (OUT/'REPORT.md').write_text('\n'.join(lines).rstrip()+'\n')
    live=OUT/'LIVE_PROGRESS.md';old=live.read_text()
    if '**이후 실행 종료:**' not in old:live.write_text('# R1/R2 보존된 진행 기록\n\n**이후 실행 종료:** [최종 한국어 보고서](REPORT.md). 아래는 학습 중 저장한 스냅샷이다.\n\n'+old)
    p=ROOT/'README.md';text=p.read_text();anchor='## Latest completed execution — Q/C authorized resumption';new='## Latest completed execution — R1/R2 budget study\n\nThe [Korean R1/R2 report](research/peft_rank12_20260915/REPORT.md) records R1 numerical diagnosis (84 updates, FP32 passed, BF16 microbatch checks unresolved; no profile or forecasting fits) and the independent R2 32-channel matched-budget study (24/24 fits with sealed evaluation and scalar metric replay). Signals, INIT selection, resource costs and known-parameterization limits are reported separately. No automatic follow-up training is running.\n\n## Previous completed execution — Q/C authorized resumption'
    if anchor in text:text=text.replace(anchor,new,1);p.write_text(text)
    p=ROOT/'docs/RESULTS_INDEX.md';text=p.read_text();text=text.replace('최신 Q/C 재개 실행은 종료됐다. Q 수치 진단180 updates, C24 fits와 최종 평가·검산을 완료했으며 자동 후속 학습은 없다.', '최신 R1/R2 실행은 종료됐다. R1 수치84 updates, R2 24 fits·9,602 updates와 평가·검산을 완료했다. 이전 Q/C 결과와 판정은 그대로 보존한다.');text=re.sub(r'\n진행 중: \[R1/R2 새 고정 계약\].*?\n','\n',text);entry='| R1/R2 완료: R1 수치84updates·본학습0, R2 24/24 fits·고정 epoch 파일럿 | [한국어 REPORT](../research/peft_rank12_20260915/REPORT.md), [R1](../results/query_budget_repair_v2_20260915/REPORT.md), [R2](../results/channel_sharing_screen_v1_20260915/REPORT.md) | [완료 장부](../research/peft_rank12_20260915/completion.json), [R2 독립 검산](../results/channel_sharing_screen_v1_20260915/independent_verification.json) |\n'
    if entry not in text:text=text.replace('| --- | --- | --- |\n','| --- | --- | --- |\n'+entry,1)
    p.write_text(text)
    print(json.dumps(dict(Q=q['status'],R2=c['status'],fits=c['fits_completed'],updates=c['training_updates'],replays=verification['prediction_replays'],historical_files=len(history)),indent=2))
if __name__=='__main__':main()
