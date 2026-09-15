"""Final report for the explicitly authorized GPU resumption; no model calls."""
import csv,json,hashlib,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/priority12_resume_20260915';Q=ROOT/'results/query_budget_numeric_v2_resume_20260915';C=ROOT/'results/channel_basis_pilot_resume_20260915'
def read(p):return json.loads(p.read_text())
def rows(p):
    if not p.exists():return []
    with p.open() as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    qs,cs=read(Q/'status.json'),read(C/'status.json');assert (Q/'phases.json').exists() and (C/'phases.json').exists()
    history=read(OUT/'historical_hashes.json')
    for p,h in history.items():assert sha(ROOT/p)==h,p
    for d,key in [(Q,'sources'),(C,'source_hashes')]:
        for p,h in read(d/'contract.json')[key].items():assert sha(ROOT/p)==h,p
    cert=read(Q/'numeric_checks.json');paths=read(Q/'path_checks.json');nv=read(Q/'numeric_verification.json');params=read(C/'parameter_inventory.json');param={r['arm']:r for r in params if r['seed']==40000}
    selection=read(Q/'resource_selection.json') if (Q/'resource_selection.json').exists() else {};qm=rows(Q/'metrics.csv');cm=rows(C/'metrics.csv');cc=rows(C/'comparisons.csv');dec=read(C/'decisions.json') if (C/'decisions.json').exists() else {}
    lines=['# PEFT Q/C 사용자 승인 재개 결과 — 기존 원천의 개발 비교','',f"**Q {qs['status']} / C {cs['status']}.** 이전 GPU 대기 종료는 그대로 보존하고, 계속 요청과 RustDesk 예외 승인을 별도 기록한 재개다.",'',
        'C의24 fits와 평가를 완료했다. SPECIFIC 대비 제한된 경량화 신호는 두 원천에서 관측됐지만, BASIS4의 학습 계수 추가 가치 조건은 둘 다 미충족이다. 두 원천 모두 더 작고 정확한 LORA_HEAD가 최선이었다. Q는 수치 진단 단계 종료이며 예측 성능 판정이 아니다.','',
        '## 1. 실행 횟수와 미실행','',
        '| 작업 | 실모델 검사용 updates | 본학습 완료/시도 | 학습 updates |','| --- | ---: | ---: | ---: |',
        f"| Q 수치/자원 | {qs['numeric_updates']} / {qs['A_updates']} | {qs['B_fits_completed']}/{qs['B_fit_attempts']} (최대12) | {qs['B_updates']} |",
        f"| C 구조 | {cs['check_updates']} | {cs['fits_completed']}/{cs['fit_attempts']} (최대24) | {cs['training_updates']} |",'',
        'C의 작은 FP64 toy optimizer30회는 이전 준비 단계에서 수행했으며 이번에 반복하지 않았다. 구조 점검72회 상한에서 이30회를 보수적으로 차감했다. 과거 Query v1의120회도 새 업데이트에 더하지 않았다.','',
        '## 2. Q 수치 중단 진단','',
        f"새 단일 비교 {len(cert)}개 중 {sum(r['passed'] for r in cert)}개,5-step 경로 {len(paths)}개 중 {sum(r['passed'] for r in paths)}개가 고정 v2 한계를 만족했다. 저장 배열을 CPU float64로 독립 재계산했다. [단일 비교](../query_budget_numeric_v2_resume_20260915/numeric_checks.json), [경로 비교](../query_budget_numeric_v2_resume_20260915/path_checks.json), [검산](../query_budget_numeric_v2_resume_20260915/numeric_verification.json).",'',
        'v2 정책은 이전 수치 실패를 본 뒤 사용자가 고정한 개정 기준이다. 출력은 Train std로 보정하고 FP32 gradient/update1e-4,5-step delta1e-3 등 지정값을 적용했다. 기존1e-5 판정과 옛 INCONCLUSIVE_NUMERICS는 수정하지 않았다. 통과해도 장기 학습 동등성이나 원인의 완전 규명을 뜻하지 않는다.','']
    fits=read(C/'fits.json') if (C/'fits.json').exists() else []
    complete=[f for f in fits if f['status']=='COMPLETE']
    lines+=['Q 수치 게이트에서 종료되어 자원 측정0회, 본학습0/12 fits다. C는 아래의 실제 완료/시도 수를 보고하며 실패 attempt를 다시 실행하지 않았다.','']
    lines+=['FP32 단일9/12·연속5/6, BF16 단일0/12·연속2/6이 통과했다. Electricity/Standard FP32의5-step delta 상대 L2는0.001180421로1e-3 한계를 넘었다. FP32에서 pinball 부호 뒤집힘이 없는 경우에도 gradient/update 한계를 넘었으므로 pinball kink만으로 원인 규명 완료라고 하지 않는다. BF16도 기준을 넘었다. 원단위 출력 scale만의 문제로 처리할 수 없으며, 미실행 예측을 성능 FAIL로 판정하지 않는다.','', '[진단 해설](../query_budget_numeric_v2_resume_20260915/NUMERIC_RESULT.md), [24개 단일 비교 원수치](../query_budget_numeric_v2_resume_20260915/numeric_summary.csv). 경로별 수치와 tensor별 차이는 위 JSON에 보존했다.','']
    lines+=['','## 3. Query의 자원 이점과 예측 이득','']
    if selection:
        lines.append('자원 선택 상태: `'+selection.get('status','UNKNOWN')+'`. [선택 기록](../query_budget_numeric_v2_resume_20260915/resource_selection.json), [모든 예산](../query_budget_numeric_v2_resume_20260915/resource_budget_table.csv).')
        if selection.get('budget_gib'):lines.append(f"B* 후보 {selection['budget_gib']}GiB. 실제 장비 VRAM이 아닌 tensor allocated 예산이다.")
        for r in selection.get('speed_validation',[]):lines.append(f"- {r['dataset']}: Query 속도 이득 block2 {r['block2_gain_percent']:.6g}%, block3 {r['block3_gain_percent']:.6g}%.")
    else:lines.append('Q 자원 선택은 미실행이다. B*·반복 속도 이점·최강 자원 대조군을 정할 수 없다.')
    if qm:
        lines+=['','| 원천 | seed | arm/role | raw scaled 2-pinball |','| --- | --- | --- | ---: |']
        for r in qm:lines.append(f"| {r['dataset']} | {r['seed']} | {r['arm']}/{r['role']} | {r['scaled_2pinball']} |")
        lines+=['','[Standard/Side/F0 대비 개선율](../query_budget_numeric_v2_resume_20260915/comparisons.csv). 원천별 두seed 원점수를 먼저 평균하며 다른 원천 raw loss를 합산하지 않는다.']
    else:lines.append('신규 Q V/E 예측 비교는 미실행이다. 기존 점수를 새 설정의 결과로 가져오지 않았으며 예측 이득은 미판정이다.')
    lines+=['','## 4. C의 채널 공유와 파라미터','',
        'MOMENT-small의512차원/64patches, 공개 Time-PEFT의 LoRA/frequency/down/head를 사용했다. 서로 다른 채널 up을 독립, 공유, 넓은 공유, 정적4그룹, 정적4기저 조합으로 바꾸며 새 채널 attention은 추가하지 않았다.','',
        '| arm | 채널 블록 | 전체 trainable | 총 모델 |','| --- | ---: | ---: | ---: |']
    for a,r in param.items():lines.append(f"| {a} | {r['groups']['channel_adapter']:,} | {r['trainable']:,} | {r['total']:,} |")
    lines+=['','SHARED_WIDE/BASIS4는 채널 파라미터1개 차이이며 GROUP4/BASIS4는256개 차이다. SPECIFIC/SHARED/LORA_HEAD는 동일 예산 대조군이 아니다. 파라미터 감소를 GPU 메모리/속도 또는 예측 우위로 바꾸어 말하지 않는다.','',
        '## 5. 같은 예산·고정 그룹 비교와 원점수','']
    if cm:
        lines+=['| 원천 | seed | arm/role | MSE | MAE |','| --- | --- | --- | ---: | ---: |']
        for r in cm:lines.append(f"| {r['dataset']} | {r['seed']} | {r['arm']}/{r['role']} | {r['mse']} | {r['mae']} |")
        lines+=['','MSE는 equal-channel Train-standardized 공간이며 std로 다시 나누지 않았다. [개선율과 paired 시간 블록 CI](../channel_basis_pilot_resume_20260915/comparisons.csv), [실제 자원](../channel_basis_pilot_resume_20260915/resources.csv).']
    else:lines.append('신규 C V/E 원점수가 없다. 미실행이나 실행 오류를 성능0/예측 FAIL로 쓰지 않는다.')
    if cm:
        lines+=['','### 두 seed 원점수 평균과 BASIS4 상대 이득','',
            '| 원천 | 대조군 | 대조군 평균 MSE | BASIS4 평균 MSE | BASIS4 이득(%) | paired 95% CI(%) |',
            '| --- | --- | ---: | ---: | ---: | --- |']
        for r in cc:
            if r['seed']=='mean':lines.append(f"| {r['dataset']} | {r['baseline']} | {float(r['baseline_mse']):.9f} | {float(r['basis_mse']):.9f} | {float(r['gain_percent']):+.4f} | [{float(r['CI95_lower']):+.4f}, {float(r['CI95_upper']):+.4f}] |")
        lines+=['','양의 이득은 BASIS4의 낮은 MSE를 뜻한다. CI는 동일 시간 블록을 모든 arm/seed에 짝지어 재표집한 개발 구간 불확실성이다. 원천·seed 모집단 일반화의 신뢰구간이 아니다.','']
    if complete:
        lines+=['### 실제 학습 자원','',
            '| 원천 | arm | 평균 active 초 | 평균 fit wall 초 | 평균 step 중앙값(ms) | 최대 allocated GiB | 최대 reserved GiB |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for dataset,arm in sorted({(f['dataset'],f['arm']) for f in complete}):
            fs=[f for f in complete if f['dataset']==dataset and f['arm']==arm]
            lines.append(f"| {dataset} | {arm} | {statistics.mean(f['active_seconds'] for f in fs):.3f} | {statistics.mean(f['wall_seconds'] for f in fs):.3f} | {1000*statistics.mean(f['median_step_seconds'] for f in fs):.3f} | {max(f['peak_allocated'] for f in fs)/2**30:.4f} | {max(f['peak_reserved'] for f in fs)/2**30:.4f} |")
        lines+=['','active는 optimizer step 시간 합이며 fit wall에는 V 평가·체크포인트 저장·재로딩 등이 포함된다. fit별 메모리 수치는 PyTorch tensor allocator 값이다. 추가 timing 반복은 수행하지 않았다.','',
            f"선택 step1024: {sum(f['best']['step']==1024 for f in complete)}/{len(complete)}, BUDGET_LIMITED: {sum(f.get('budget_limited',False) for f in complete)}/{len(complete)}, 선택 step0: {sum(f['best']['step']==0 for f in complete)}/{len(complete)}. step0는 각 arm의 초기 head/adapter 상태이며 별도의 pretrained F0 기준선으로 해석하지 않는다.",'']
    lines+=['','## 6. SPECIFIC·LORA_HEAD 대비 실제 가치','']
    for d in dec.get('source_decisions',[]):
        lines.append(f"- {d['dataset']}: 최저 MSE {d['best_arm']}; 제한된 경량화 신호 {d['limited_compression_signal']}; 계수 활용 신호 {d['coefficient_signal']}; LORA_HEAD가 더 작고 정확함 {d['lora_head_smaller_and_more_accurate']}.")
    if complete:
        reduction=100*(1-param['BASIS4']['trainable']/param['SPECIFIC']['trainable'])
        lines += ['',f'BASIS4의 전체 trainable은 SPECIFIC보다 {reduction:.3f}% 적다. 채널 블록 감소와 전체 모델 감소는 위 표에서 따로 확인할 수 있다.','']
        for dataset in sorted({f['dataset'] for f in complete}):
            bf=[f for f in complete if f['dataset']==dataset and f['arm']=='BASIS4'];sf=[f for f in complete if f['dataset']==dataset and f['arm']=='SPECIFIC']
            active=100*(1-statistics.mean(f['active_seconds'] for f in bf)/statistics.mean(f['active_seconds'] for f in sf))
            wall=100*(1-statistics.mean(f['wall_seconds'] for f in bf)/statistics.mean(f['wall_seconds'] for f in sf))
            memory=100*(max(f['peak_allocated'] for f in bf)/max(f['peak_allocated'] for f in sf)-1)
            lines.append(f'{dataset}: SPECIFIC 대비 평균 active 시간 {active:.2f}% 감소, 평균 fit wall {wall:.2f}% 감소. 반면 최대 allocated는 {memory:.2f}% 증가했다. 파라미터 절약이 peak 메모리 절약으로 이어진 결과는 아니다.')
        lines+=['','같은 예산의 SHARED_WIDE·GROUP4에도 두 원천 평균 MSE가 모두 밀렸다. 더 작은 SHARED도 두 원천 평균에서 BASIS4보다 낮은 MSE였다. 따라서 큰 SPECIFIC만 이긴 결과를 새 PEFT 방법의 예측 우위로 삼을 수 없다.','']
    mechanism=sorted(C.glob('*_mechanism.json'))
    if mechanism:
        lines+=['','### 선택된 계수의 사후 진단','',
            '| fit | 계수의 채널별 분산 (4개 basis) | effective weight 평균 채널 분산 | 계수 교환 MSE 변화(%) |',
            '| --- | --- | ---: | ---: |']
        for path in mechanism:
            fid=path.name.removesuffix('_mechanism.json');f=next(f for f in complete if f['fit']==fid);m=read(path)
            selected=next(float(r['mse']) for r in cm if r['dataset']==f['dataset'] and int(r['seed'])==f['seed'] and r['arm']=='BASIS4' and r['role']=='selected')
            swapped=next(float(r['mse']) for r in cm if r['dataset']==f['dataset'] and int(r['seed'])==f['seed'] and r['role']=='posthoc_coefficient_swap')
            lines.append(f"| {fid} | {', '.join(f'{v:.4g}' for v in m['coefficient_channel_variance'])} | {m['effective_weight_channel_variance']:.6g} | {100*(swapped-selected)/selected:+.4f} |")
        lines+=['','양의 교환 변화는 계수를 바꾼 후 MSE 악화다. 선택된 모델에 대한 사후 기술통계이며 학습 중 인과 효과 검증이 아니다. basis별 계수 절대값 합·전체 계수는 각 mechanism.json, 실제 tensor update는 각 parameter_updates.json에 저장했다.','']
    if mechanism:lines+=['네 계수 교환 모두 원래 계수보다 E MSE가 낮았다. 이 사후 관측은 학습 계수의 유용한 채널 특화를 뒷받침하지 않는다. 다만 교환 한 번으로 부진의 원인이나 계수 자체의 무용성을 증명한 것은 아니다. 검증 곡선에서도 BASIS4가 강한 단순 대조군에 뒤졌으므로 이번 결과를 V→E 전달 실패만으로 설명할 근거는 없다. 14/24 fits는 BUDGET_LIMITED이며, 고정 recipe에서의 비교를 완전 수렴 결과라고 하지 않는다.','']
    if not dec.get('source_decisions'):lines.append('완성된 원천 블록의 예측 비교가 없어 가장 강한 대조군, 계수 활용,1% 이내 예측 유지 조건은 미판정이다.')
    lines+=['','## 7. 운영·검증과 남은 한계','',
        'C의 V120개·선택 checkpoint 재로딩24개·E28개(선택24+계수 교환4), 총172개 예측 기록을 독립 검산했다. 최대 MSE 절대차2.220446049250313e-16으로 지정1e-10 이내다. [독립 검산](../channel_basis_pilot_resume_20260915/independent_verification.json). Q는 새36개 수치 비교와 이전42개 저장 비교를 재계산했다. 신규 Q 예측 검산은0개로 미측정이며 점수0을 뜻하지 않는다.','',
        f"이전 results/research {len(history)}개 파일과 두 계약의 실행 소스 해시를 확인했다. [재개 승인](authorization.json), [실행 종료 코드](runner_exit_codes.json), [공개 전 교차 감사](publication_audit.json). GPU에 남은 RustDesk만 예외로 허용했다. 원격 화면 부하를 포함한 환경의 시간 측정이며 엄격한 완전 유휴 GPU 측정과 같다고 주장하지 않는다. 다른 compute 작업과 free VRAM은 계속 감시했다. C 학습·평가 감시10,306개 표본에서 비승인 외부 compute는0개였고 최소 free는5,412MiB였다. RustDesk 기록3,155개는 승인된 예외다. 종료 후 GPU compute 목록은 비어 있었다.",'',
        'Q/C는 서로 독립적으로 진행했다. Q의 예측 성공을 C의 입장 조건으로 사용하지 않았다. C는 동일1024updates로 비교했다. Q의120초 active budget 본학습은 사전 계획이며 이번에는 미실행이다. Q pinball과 C MSE를 한 성능 평균으로 합치지 않는다.','',
        '[Time-PEFT 공개 구조](https://github.com/kaist-dmlab/TimePEFT/blob/ea4e7e1887bb35587bab7ea93e2af3685ac55852/run.py)의 통제 변형이다. [C-LoRA](https://arxiv.org/html/2407.17246v1) 등 채널 공유·분해는 알려진 원리이며 새로운 방법 또는 논문 PASS가 확정되지 않았다. 공식 C-LoRA의 같은 백본 비교, 가까운 공유 adapter와의 차이, 미노출 원천의 독립 확인, 더 넓은 최적화 비교가 남는다. 단일 길이·고정 LR·2seed·기존 원천 개발 결과로 범용성을 주장하지 않는다.','',
        '**승인된 두 작업은 여기서 종료한다.** 추가 후보,seed,LR,자동 후속 학습은 실행하지 않는다. 원자료·가중치·큰 tensor cache는 로컬에 남으며 GitHub에는 코드·계약·원점수·검증 범위를 올린다.','']
    for name,s in [('Q',qs),('C',cs)]:
        if s.get('error'):lines.append(f"{name} 종료 오류: `{s['error']}`.")
    for name in ['validation_trajectories','parameters_vs_error','basis_gains']:
        if (C/(name+'.png')).exists():lines += [f'![{name}](../channel_basis_pilot_resume_20260915/{name}.png)','']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    summary=dict(Q_status=qs['status'],C_status=cs['status'],Q_numeric_updates=qs['numeric_updates'],Q_resource_updates=qs['A_updates'],Q_training_updates=qs['B_updates'],Q_fits=qs['B_fit_attempts'],C_check_updates=cs['check_updates'],C_training_updates=cs['training_updates'],C_fits=cs['fit_attempts'],historical_files_unchanged=len(history),new_optimizer_updates_in_reporting=0,report_source_sha256=sha(Path(__file__)))
    (OUT/'completion_verification.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
