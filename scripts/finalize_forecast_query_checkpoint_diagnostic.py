"""CPU-only independent replay of every stored CP parity pair and resource summary."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
from tsfm_peft_screen.reproducibility import ROOT, sha, write_json

OUT = ROOT/'results/forecast_query_checkpoint_diagnostic'
CACHE = ROOT/'.cache/forecast_query_checkpoint_diagnostic'


def main():
    status = json.loads((OUT/'status.json').read_text())
    assert status['status'] == 'COMPLETE', status
    contract = json.loads((OUT/'contract.json').read_text())
    rows = json.loads((OUT/'measurements.json').read_text())
    eqs = json.loads((OUT/'equivalence.json').read_text())
    cfg = contract['config']
    assert len(rows) == 32 and len(eqs) == 16
    assert status['counts'] == dict(warmup_optimizer_updates=6, measured_optimizer_updates=24,
                                   fp32_equivalence_optimizer_updates=8, instrumented_backward_only=4,
                                   fits=0, evaluation_accesses=0)
    assert sha(CACHE/'shared_state.pt') == contract['shared_state_sha256']
    old = contract['historical_result_hashes']
    assert all(sha(ROOT/p) == h for p,h in old.items())
    replay = []
    for eq in eqs:
        pair = []
        for cp in (False, True):
            record = next(r for r in rows if all(r[k] == eq[k] for k in ('arm','dataset','precision','repeat')) and r['checkpoint'] == cp)
            assert record['initial_state_sha256'] == contract['shared_state_content_sha256']
            path = CACHE/(record['tag']+'.pt')
            assert sha(path) == record['cache_sha256']
            pair.append(torch.load(path, map_location='cpu', weights_only=True))
        errors = {}
        for key in ('z','raw','loss','gradient','update','adam'):
            a,b = [v[key].numpy().astype(np.float64).reshape(-1) for v in pair]
            assert np.isfinite(a).all() and np.isfinite(b).all()
            d = a-b
            errors[key] = dict(max_absolute=float(np.max(np.abs(d))),
                               relative_l2=float(np.sqrt(np.sum(d*d))/max(np.sqrt(np.sum(a*a)),1e-30)))
            for kind, value in errors[key].items():
                assert abs(value-eq['metrics'][key][kind]) <= 1e-10
                assert value <= cfg['tolerances'][eq['precision']][kind]
        assert pair[0]['rng_after_sha256'] == pair[1]['rng_after_sha256']
        assert abs(pair[0]['gradient_norm']-pair[1]['gradient_norm']) <= cfg['tolerances'][eq['precision']]['max_absolute']
        replay.append(dict(arm=eq['arm'], dataset=eq['dataset'], precision=eq['precision'],
                           repeat=eq['repeat'], errors=errors))
    groups = []
    for dataset in cfg['datasets']:
        for arm in cfg['arms']:
            for cp in cfg['checkpoint']:
                rr = [r for r in rows if r['precision']=='bf16' and r['dataset']==dataset and r['arm']==arm and r['checkpoint']==cp]
                assert len(rr) == 3
                groups.append(dict(dataset=dataset, arm=arm, checkpoint=cp,
                                   peak_allocated_mib=max(r['peak_allocated_bytes'] for r in rr)/2**20,
                                   peak_reserved_mib=max(r['peak_reserved_bytes'] for r in rr)/2**20,
                                   median_seconds=float(np.median([r['seconds'] for r in rr])),
                                   min_seconds=min(r['seconds'] for r in rr), max_seconds=max(r['seconds'] for r in rr)))
    decisions = []
    for dataset in cfg['datasets']:
        g = {(r['arm'],r['checkpoint']):r for r in groups if r['dataset']==dataset}
        q,s = g[('query',True)],g[('standard',True)]
        mr = q['peak_allocated_mib']/s['peak_allocated_mib']
        tr = q['median_seconds']/s['median_seconds']
        decisions.append(dict(dataset=dataset, query_on_vs_standard_on_memory_ratio=mr,
                              query_on_vs_standard_on_time_ratio=tr, old_20_percent_memory_target_met=mr<=.8,
                              memory_reduced_and_time_overhead_le_5_percent=mr<1 and tr<=1.05,
                              query_checkpoint_memory_ratio=q['peak_allocated_mib']/g[('query',False)]['peak_allocated_mib'],
                              query_checkpoint_time_ratio=q['median_seconds']/g[('query',False)]['median_seconds']))
    if all(d['memory_reduced_and_time_overhead_le_5_percent'] for d in decisions):
        verdict = 'RESOURCE_SIGNAL_ONLY'
    elif all(d['query_on_vs_standard_on_memory_ratio'] >= 1 and d['query_on_vs_standard_on_time_ratio'] >= 1 for d in decisions):
        verdict = 'STOP_CURRENT_QUERY_STORAGE'
    else:
        verdict = 'TRADEOFF_ONLY'
    verification = dict(passed=True, parity_pairs_replayed=len(replay), raw_cache_files=32,
                        historical_files_unchanged=len(old), all_initial_states_identical=True,
                        max_parity_absolute_error=max(v['max_absolute'] for r in replay for v in r['errors'].values()),
                        max_parity_relative_error=max(v['relative_l2'] for r in replay for v in r['errors'].values()),
                        fits=0, evaluation_accesses=0)
    summary = dict(verdict=verdict, resources=groups, decisions=decisions,
                   original_forecast_query_verdict='FAIL', counts=status['counts'],
                   note='Fixed train-state resource diagnosis only. No new quality gate, side comparison or paper PASS.')
    if '--verify-only' in sys.argv:
        assert json.loads((OUT/'summary.json').read_text()) == summary
        assert json.loads((OUT/'verification.json').read_text()) == verification
        print('CHECKPOINT DIAGNOSTIC REPLAY PASS', verification)
        return
    write_json(OUT/'summary.json', summary)
    write_json(OUT/'verification.json', verification)
    write_json(OUT/'independent_parity_replay.json', replay)
    with (OUT/'resources.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(groups[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(groups)
    text = [
        '# Forecast-query checkpoint 통제 진단 결과', '',
        f'판정: **{verdict}**. 기존 forecast-query의 **FAIL은 그대로 유지**한다.',
        '',
        '## 실행 범위', '',
        'BF16 자원 측정 24회, FP32 동등성 업데이트 8회, 워밍업 6회: 총 38 optimizer updates.',
        '별도 저장값 계측 backward 4회는 optimizer update 없이 수행했다.',
        '새 학습 fit, V 선택, E 접근은 모두 0이다. 모든 비교는 동일 LoRA/Adam/RNG 상태에서 시작했다.',
        f"실행 commit: {contract['execution_commit']}. 두 데이터셋 각각 train batch 하나를 3회 반복했다.",
        '',
        '## 실제 자원 비교', '',
        '| 데이터 | 방식 | checkpoint | peak allocated MiB | peak reserved MiB | step 중앙값 s |',
        '|---|---|---|---:|---:|---:|']
    for r in groups:
        text.append(f"| {r['dataset']} | {r['arm']} | {'on' if r['checkpoint'] else 'off'} | {r['peak_allocated_mib']:.2f} | {r['peak_reserved_mib']:.2f} | {r['median_seconds']:.4f} |")
    text += ['', 'peak는 3회 중 최대값, 시간은 3회 중앙값이다. frozen pass, cache 생성, backward, clipping, Adam step을 포함한다.',
             '복원/CPU export/모니터 호출은 timing 밖에 있다. 계측 hook이 없는 24회만 자원 표에 사용했다.',
             '데스크톱 graphics가 공유하는 GPU이므로 작은 시간 차이와 3회 반복을 통계적 우위로 해석하지 않는다.',
             '', '## 사전 기준에 따른 분리 판정', '']
    for d in decisions:
        text += [f"- {d['dataset']}: query-on / standard-on peak={d['query_on_vs_standard_on_memory_ratio']:.6f}, time={d['query_on_vs_standard_on_time_ratio']:.6f}. "
                 f"기존 20% memory target 충족={d['old_20_percent_memory_target_met']}; "
                 f"memory 감소 및 시간 overhead≤5%={d['memory_reduced_and_time_overhead_le_5_percent']}. "
                 f"query 자체 checkpoint on/off memory={d['query_checkpoint_memory_ratio']:.6f}, time={d['query_checkpoint_time_ratio']:.6f}."]
    text += ['', '## 수치 및 보존 검증', '',
             f"독립 CPU 재계산: {len(replay)} on/off 쌍, raw cache 32개. 출력, loss, 전체 clipped gradient, "
             '실제 parameter delta, Adam moments/step 및 RNG 사후 상태를 비교했다.',
             f"최대 absolute={verification['max_parity_absolute_error']:.3g}, 최대 relative L2={verification['max_parity_relative_error']:.3g}.",
             f"기존 결과 {len(old)}개 파일 SHA256 불변. frozen backbone 파라미터도 실행 전후 불변이다.",
             '역사적 forward와 새 구현의 precision별 출력 비교는 historical_forward_parity.json에 기록했다.',
             '', '## 저장값과 단계별 peak', '',
             'storage_profiles.json은 별도 계측 4회를 보존한다. storage union과 실제 allocator peak를 구분한다.']
    profiles=json.loads((OUT/'storage_profiles.json').read_text())
    for p in profiles:
        cache=p['cache']
        phases=sorted(p['phases'], key=lambda r:r['peak_allocated_bytes'],reverse=True)
        text.append(f"- {p['arm']} checkpoint={p['checkpoint']}: outer saved CUDA nonparameter storage union "
                    f"{p['saved_unique_cuda_nonparameter_bytes']/2**20:.2f}MiB; "
                    f"계측 중 가장 높은 phase={phases[0]['phase']} ({phases[0]['peak_allocated_bytes']/2**20:.2f}MiB)."
                    +(f" 과거 K/V unique={cache['kv_unique_bytes']/2**20:.2f}MiB, 전체 cache unique={cache['all_unique_bytes']/2**20:.2f}MiB." if cache else ''))
    text += ['', '이 계측은 nested checkpoint hook 내부를 모두 볼 수 없고 연산별 전체 임시 메모리를 분해한 profiler가 아니다.',
             '따라서 saved-storage union을 peak와 동일시하거나, 전체 peak 차이를 K/V만으로 설명하지 않는다.',
             '', '## 해석과 다음 범위', '',
             'checkpointing은 기존 기술이며 이번 진단은 새 PEFT 방법의 성공 증거가 아니다.',
             'RESOURCE_SIGNAL_ONLY이면 동등한 저장 전략에서 자원 이점이 남는다는 개발 근거만 얻은 것이다.',
             'TRADEOFF_ONLY이면 메모리/시간 교환을 확인한 것이며, 목표를 바꿔 과거 FAIL을 PASS로 바꾸지 않는다.',
             'STOP_CURRENT_QUERY_STORAGE이면 현재 구조·저장 방식의 추가 확장은 중단한다.',
             '어느 경우에도 side/head 대비 우위, 새 데이터 예측 정확도, time-to-quality 또는 논문 신규성을 입증하지 않았다.',
             '후속 학습이 필요하다면 강한 side 대조와 별도 미노출 평가 계획을 먼저 고정해야 한다.',
             '', '## 재현', '',
             'CPU: PYTHONPATH=src .venv/bin/python scripts/finalize_forecast_query_checkpoint_diagnostic.py --verify-only',
             'GPU: scripts/with_cuda.sh .venv/bin/python scripts/run_forecast_query_checkpoint_diagnostic.py',
             'GPU 실행기는 기존 결과 디렉터리가 존재하면 덮어쓰기를 거부한다. 원시 상태/gradient cache는 .cache 아래 ignored 파일이다.', '']
    (OUT/'RESULT.md').write_text('\n'.join(text))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
