"""Combine bounded Q/C outcomes without turning unexecuted work into performance claims."""
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/priority12_20260915';Q=ROOT/'results/query_budget_numeric_v2_20260915';C=ROOT/'results/channel_basis_pilot_20260915'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvrows(p):
    if not p.exists():return []
    with p.open() as f:return list(csv.DictReader(f))
def main():
    q,c=read(Q/'status.json'),read(C/'status.json');assert not any(v in q['status'] or v in c['status'] for v in ['WAITING','TRAINING','RUNNING','PREPARED'])
    history=read(OUT/'historical_hashes.json')
    for p,h in history.items():assert sha(ROOT/p)==h,p
    for directory,key in [(Q,'sources'),(C,'source_hashes')]:
        for p,h in read(directory/'contract.json')[key].items():assert sha(ROOT/p)==h,p
    if q['numeric_updates']==q['A_updates']==q['B_updates']==0:
        old=Q/'REPORT_automated_at_stop.md'
        if not old.exists():old.write_bytes((Q/'REPORT.md').read_bytes())
        wall=read(Q/'numeric_wall.json');replay=read(Q/'historical_parity_replay.json')
        (Q/'REPORT.md').write_text('\n'.join([
            '# Query v2 — simulated tensor budget / 기존 개발 구간 재사용','',
            f"**{q['status']}. 새 GPU 수치진단0 updates, 자원측정0 updates, 본학습0/12 fits.**",'',
            '실행기는 실제 진입했으나 외부 GPU compute가 계속되어 시작 안전 조건을 얻지 못했다.','',
            f"누적 대기 {wall['gpu_wait_seconds']:.2f}초로 고정600초 한도를 적용했다. 최소 free VRAM {wall['minimum_free_mib']}MiB, 외부 compute 관측 {wall['external_compute_samples']}회. 다른 프로세스를 종료하거나 동시 학습하지 않았다.",'',
            '## 수치 중단의 해소 여부','',
            '이번 지시문은 과거 실패를 본 뒤 새로 정한 v2 수치 정책이다. raw 출력은 Train std로 보정하고 단일 FP32 gradient/update 상대 허용치1e-4, 짧은 경로1e-3 등을 한 번 고정했다. 이전 run의1e-5 판정을 바꾸지 않았다. 새 인증 배치와 5-step 경로는 GPU 대기로 실행하지 못했으므로 NUMERICS_BOUNDED로 인증되지 않았다.','',
            f"기존 저장 비교 {len(replay)}개를 원래 정책으로 재계산해 원 판정을 확인했다. CPU FP64의 partition/mask/gradient 검사 및 새 수치 정책 단위 검사는 통과했다. 이는 새 실제 모델 수치 인증을 대체하지 않는다.",'',
            '## 자원과 예측','',
            '1/2/4/8GiB 예산별 신규 반복 측정이 없어 B*, 최속 옵션, Query 자원 이점을 정하지 않았다. 신규 V/E 점수도 없어 Standard/Side/F0 중 최강 대조군이나 예측 이득을 비교할 수 없다. 값0으로 성능을 표시하지 않는다. 기존 결과를 새 실행의 점수로 가져오지 않았다.','',
            'Q 전용 종료는 C의 성능 관문이 아니다. C는 별도600초 대기 예산으로 독립 진입했다. [통합 보고서](../priority12_20260915/REPORT.md), [원래 재검산](historical_parity_replay.json), [대기 기록](numeric_wall.json).','',
            '현재 Q 실행은 종료했으며 새 후보나 자동 재시도는 없다. 구현된 GPU 수치/자원/학습 경로는 이번 실행에서 검증되지 않은 부분으로 남는다.','']))
        with (Q/'resource_budget_table.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['budget_gib','status','reason'],lineterminator='\n');w.writeheader();w.writerows([dict(budget_gib=b,status='NOT_MEASURED',reason='GPU_STARTUP_BLOCKED') for b in [1,2,4,8]])
    with (Q/'attempts.csv').open('w',newline='') as f:
        fields=['phase','planned_cap','attempts','optimizer_updates','forecasting_fits'];w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        w.writerows([dict(phase='numeric',planned_cap=192,attempts=q['numeric_update_attempts'],optimizer_updates=q['numeric_updates'],forecasting_fits=0),dict(phase='resource',planned_cap=512,attempts=q['A_update_attempts'],optimizer_updates=q['A_updates'],forecasting_fits=0),dict(phase='forecasting',planned_cap=12,attempts=q['B_fit_attempts'],optimizer_updates=q['B_updates'],forecasting_fits=q['B_fits_completed'])])
    inv=read(C/'parameter_inventory.json');counts={r['arm']:r for r in inv if r['seed']==40000};specific=counts['SPECIFIC']['trainable'];basis=counts['BASIS4']['trainable']
    qw=read(Q/'numeric_wall.json');cw=read(C/'preflight_wall.json');qm=csvrows(Q/'metrics.csv');cm=csvrows(C/'metrics.csv')
    lines=['# PEFT 우선순위 1·2 실행 결과 — 기존 원천을 재사용한 개발 비교','',
        f"**Q: {q['status']}. C: {c['status']}.** 이번에 완료한 구현·CPU 검사와 실제 GPU 학습을 아래에서 분리한다. 수치 인증 미실행과 예측 성능 FAIL을 혼동하지 않는다.",'',
        '## 1. 실제 실행 횟수와 미실행','',
        '| 작업 | 폐기용 실모델 GPU updates | 본학습 fits 완료/시도 (상한) | 본학습 updates |','| --- | ---: | ---: | ---: |',
        f"| Q 수치진단/자원 | {q['numeric_updates']} / {q['A_updates']} | {q['B_fits_completed']}/{q['B_fit_attempts']} (12) | {q['B_updates']} |",
        f"| C 구조점검 | {c['check_updates']} | {c['fits_completed']}/{c['fit_attempts']} (24) | {c['training_updates']} |",'',
        f"C에는 별도로 tiny FP64 구조 검사의 실제 optimizer {c.get('cpu_toy_optimizer_updates',0)}회가 있다. 초기 검사와 prepare 회귀 검사에서 각각5개 작은 block×3회였다. 실모델 CPU optimizer는0이며 이 toy 업데이트를 예측 fits로 세지 않았다. 보수적으로 구조점검72회 예산에서 차감했다. [장부](../channel_basis_pilot_20260915/cpu_update_ledger.json).",'',
        f"Q 시작 대기 {qw['gpu_wait_seconds']:.2f}초, C 시작 대기 {cw['wait_seconds']:.2f}초. 원래1290개 파일은 불변이다. GPU 진입 조건은 외부 compute 없음30초와 free4GiB 이상이다. 다른 작업의 프로세스를 종료하지 않았다.",'',
        '## 2. Q 수치 중단이 해소됐는가','',
        f"새 단일/연속 인증 결과는 [Q 수치검증](../query_budget_numeric_v2_20260915/numeric_verification.json)에 있다. 실제 새 GPU 진단 update는 {q['numeric_updates']}회다. 기존42개 artifact 비교는 원래 v1 기준으로 재계산했다. v1의4개 microbatch 기준 초과와 INCONCLUSIVE_NUMERICS는 보존한다.",'',
        'v2 기준은 사용자가 이전 중단을 보고 명시적으로 개정한 정책이다. raw 단위 절대값 관문을 Train std 보정으로 바꾸고 gradient/update 및5-step 경로 한계를 새로 고정했다. 결과에 맞춰 반복 조정하지 않았다. 실제 인증이 미실행이면 정책 구현만으로 중단 원인을 규명하거나 NUMERICS_BOUNDED라 할 수 없다.','',
        '## 3. 강한 Standard 대조 뒤 Query의 자원 이점','',
        f"자원 비교 실제 update는 {q['A_updates']}회다. [Q 보고서](../query_budget_numeric_v2_20260915/REPORT.md)와 [예산 표](../query_budget_numeric_v2_20260915/resource_budget_table.csv)를 참조한다. 측정이 없으면 B*, 반복 속도 이점, 최속 Standard/Side를 정할 수 없다. 예산은 실제 소형 GPU 실행이 아닌1/2/4/8GiB tensor allocated 시뮬레이션이다.",'',
        '## 4. C의 구조와 정확한 파라미터 수','',
        'Time-PEFT의 MOMENT-small·LoRA q/k/v·frequency/down/head를 기준으로 채널 up의 공유 형태만 비교한다. 실제 D512, patch8,64patches, horizon96을 확인했다. 두seed에서 공식 SPECIFIC 출력/목적식과 wrapper가 CPU FP32에서 차이0으로 일치했다. SPECIFIC~BASIS4의 초기 출력 차이도0이었다. 이 관찰은 GPU backward 검증과 다르다.','',
        '| arm | 채널 블록 | 전체 trainable | 총 모델 |','| --- | ---: | ---: | ---: |']
    for a,r in counts.items():lines.append(f"| {a} | {r['groups']['channel_adapter']:,} | {r['trainable']:,} | {r['total']:,} |")
    lines+=['',f"BASIS4의 전체 trainable은 SPECIFIC의 {100*basis/specific:.3f}%로 {100*(1-basis/specific):.3f}% 감소한다. 채널 블록만의 절약을 전체 절약이라고 바꾸지 않았다. 파라미터 조건만으로 예측1% 이내 유지 조건을 만족했다고 볼 수 없다.",'',
        'Electricity 원본26304×321, Traffic17544×862에서 Train-only 조건의 첫64개 열을 골랐다. V/E origins는 각각54/54 및36/36이며 원본4채널을 복제하지 않았다. [자료 manifest](channel_data_manifest.json).','',
        '## 5. 같은 예산 공유/고정 그룹보다 BASIS4가 좋은가','']
    if cm:
        lines+=['[원점수](../channel_basis_pilot_20260915/metrics.csv)와 [개선율/시간 블록 CI](../channel_basis_pilot_20260915/comparisons.csv), [사전 목표 판정](../channel_basis_pilot_20260915/decisions.json)을 확인한다.']
    else:lines+=['신규 MSE/MAE 원점수가 없다. BASIS4 대 SHARED_WIDE/GROUP4의0.5% 평균 이득 및 두seed 같은 방향 조건은 미판정이다. CPU 구조 정상 작동을 계수 활용의 예측 이득으로 승격하지 않는다.']
    lines+=['','## 6. SPECIFIC과 LORA_HEAD 대비 실제 가치','']
    if not cm:lines+=['SPECIFIC 대비1% 이내 예측 유지, LORA_HEAD 대비 예측 우위 모두 확인하지 못했다. 가장 강한 예측 대조군을 선정할 근거가 없다. LORA_HEAD는 전체3,317,856 trainable로 BASIS4보다 작다는 구조적 사실만 확인됐다.']
    lines+=['','| 작업 | 신규 원점수 | 자원 이득 | 예측 이득 |','| --- | --- | --- | --- |',
        f"| Q | {'측정됨: Q metrics.csv' if qm else '미실행'} | {'자원 표 참조' if q['A_updates'] else '미판정'} | {'Q 보고서 참조' if qm else '미판정'} |",
        f"| C | {'측정됨: C metrics.csv' if cm else '미실행'} | 파라미터 수 확인; GPU 속도/peak는 실측 여부 별도 | {'C 보고서 참조' if cm else '미판정'} |",'',
        '## 7. 계속할 근거와 남은 한계','',
        '이번 구현은 제한 실험을 실행할 준비와 CPU 의미 검사를 제공한다. GPU 시작이 막힌 경우에는 성능을 근거로 연구 방향을 채택하거나 폐기할 수 없다. CPU 검사는 Q 전체112개, C 전용6개가 통과했다. C의 끝 불완전 시간 블록을 포함하는 bootstrap도 추가 모델 실행 없이 검산했다.','',
        '[Time-PEFT 공개 구조](https://github.com/kaist-dmlab/TimePEFT/blob/ea4e7e1887bb35587bab7ea93e2af3685ac55852/run.py)의 통제 변형이며 원문 전체 재현이 아니다. [C-LoRA](https://arxiv.org/html/2407.17246v1) 등 채널 공유·저랭크 조합은 이미 알려진 원리다. 공식 C-LoRA 같은 백본 비교, 가까운 혼합 adapter와의 차이, 새 원천 독립 확증 및 최적화 조건 확장이 남아 있다. 이번 작업만으로 신규 방법이나 논문 PASS를 선언하지 않는다. [선행/환경 차이](../../sources/RELATED_WORK.md).','',
        '공개 requirements와 momentfm 메타데이터의 Transformers 버전 충돌은 격리된 C 환경에서 공개 코드의4.44.2를 적용해 해결했다. Q 환경·드라이버는 변경하지 않았다. C의 best-V 저장/복원은 upstream의 마지막 state 반환과 다르며 모든 arm에 동일 적용한다.','',
        '**이번 두 실행은 종료한다.** 같은 run을 덮어쓰거나 이름만 바꾼 재시도, 추가 seed/LR/후보/자동 후속 학습은 실행하지 않는다. 코드·계약·검증·실패 근거는 GitHub에 보관하고, 원자료·가중치·큰 cache는 로컬에 둔다. GPU 학습 경로가 미실행이면 그 경로의 실동작까지 검증됐다고 하지 않는다.','',
        '[Q 상세](../query_budget_numeric_v2_20260915/REPORT.md) · [C 상세](../channel_basis_pilot_20260915/REPORT.md) · [C 검증](../channel_basis_pilot_20260915/independent_verification.json)','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    audit=dict(status='VERIFIED_RECORDED_OUTCOMES',Q_status=q['status'],C_status=c['status'],historical_files_unchanged=len(history),Q_new_gpu_updates=q['numeric_updates']+q['A_updates']+q['B_updates'],C_new_gpu_updates=c['check_updates']+c['training_updates'],C_CPU_toy_updates=c.get('cpu_toy_optimizer_updates',0),Q_forecasting_fits=q['B_fit_attempts'],C_forecasting_fits=c['fit_attempts'],forecast_scores_available=dict(Q=bool(qm),C=bool(cm)),sealed_execution_sources_unchanged=True,old_fail_stop_unchanged=True,report_source_sha256=sha(Path(__file__)))
    (OUT/'completion_verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    index=ROOT/'docs/RESULTS_INDEX.md';text=index.read_text()
    row='| Q v2 + 채널 공유 독립 파일럿: '+q['status']+' / '+c['status']+'; 신규 본학습 '+str(q['B_fit_attempts']+c['fit_attempts'])+' fits | [통합 한국어 보고서](../results/priority12_20260915/REPORT.md), [Q](../results/query_budget_numeric_v2_20260915/REPORT.md), [C](../results/channel_basis_pilot_20260915/REPORT.md) | [완료 검증](../results/priority12_20260915/completion_verification.json), [C 실제 CPU 모델 비교](../results/channel_basis_pilot_20260915/initial_cpu_parity.json) |'
    if '../results/priority12_20260915/REPORT.md' not in text:index.write_text(text.replace('| --- | --- | --- |','| --- | --- | --- |\n'+row,1))
    readme=ROOT/'README.md';text=readme.read_text()
    if 'results/priority12_20260915/REPORT.md' not in text:
        at=text.index('## Latest completed work')
        text=text[:at]+'## Latest bounded execution — 2026-09-15\n\nThe [independent Query v2 and channel sharing tracks](results/priority12_20260915/REPORT.md) ended at **'+q['status']+' / '+c['status']+'**. New forecasting fits: '+str(q['B_fit_attempts'])+'/12 Query and '+str(c['fit_attempts'])+'/24 channel. Actual GPU updates: '+str(audit['Q_new_gpu_updates'])+' / '+str(audit['C_new_gpu_updates'])+'. CPU model/structure checks and parameter counts are reported separately from unmeasured forecasting outcomes. Historical results are preserved; no automatic retry or follow-up training is running.\n\n'+text[at:].replace('## Latest completed work','## Previous completed work',1)
        readme.write_text(text)
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
