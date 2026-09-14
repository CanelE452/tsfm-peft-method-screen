"""Independent CPU replays and source-paired report, no post-D selection."""
import csv
import json
from pathlib import Path
import numpy as np
import run_temporal_transfer_diagnostic_v1 as A
from tsfm_peft_screen.metrics import score,independent
from tsfm_peft_screen.reproducibility import ROOT,sha,digest,write_json
OUT=ROOT/'results/future_selection_transfer_v1'


def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(open(p)))
def gain(a,b):return 100*(b-a)/b


def main():
    rec=read(OUT/'execution_receipt.json');assert rec['status']=='COMPLETED'
    assert rec['forward']<=1536 and rec['backward']==rec['new_fits']==rec['optimizer_updates']==rec['old_E_array_reads']==rec['D_reads_before_seal']==0
    plan=read(OUT/'plan.json');seal=read(OUT/'selection_seal.json')
    assert sha(OUT/'plan.json')==seal['plan_sha256'] and sha(OUT/'selection_scores.csv')==seal['selection_scores_sha256']
    assert digest({k:v for k,v in seal.items() if k!='sha256'})==seal['sha256']
    index=read(OUT/'prediction_index.json');cache={};checks=0;maxerr=0.;perorigin=[]
    allowed={(r['source']+'_F0' if c['step']==0 else c['id']) for r in seal['selections'] for c in r['selected'].values()}
    for row in index:
        assert sha(ROOT/row['prediction_file'])==row['prediction_sha256']
        if row['split']=='D':assert row['canonical'] in allowed
        with np.load(ROOT/row['prediction_file'],allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
        loss=score(d['prediction'],d['target'],d['scale'])['scaled_2pinball'];scalar=independent(d['prediction'],d['target'],d['scale'])
        err=max(abs(loss-row['loss']),abs(loss-scalar));assert err<=1e-10;maxerr=max(maxerr,err);checks+=2
        num,count=A.components(d['prediction'],d['target'],d['scale'])
        for i,o in enumerate(d['origins']):
            for c in range(4):perorigin.append(dict(canonical=row['canonical'],source=row['source'],split=row['split'],origin=int(o),channel=c,numerator=float(num[i,c]),count=int(count[i,c])))
        cache[row['canonical'],row['split']]=d
    reported={(r['cell'],r['rule'],r['candidate']):float(r['loss']) for r in rows(OUT/'selection_scores.csv')}
    for cell in plan['cells']:
        chosen=next(s for s in seal['selections'] if s['cell']==cell['id'])
        for rule,idx in plan['rules'].items():
            losses={}
            for c in cell['candidates']:
                key=cell['source']+'_F0' if c['step']==0 else c['id'];d=cache[key,'S']
                value=score(d['prediction'][idx],d['target'][idx],d['scale'])['scaled_2pinball']
                assert abs(value-reported[cell['id'],rule,c['id']])<=1e-10;checks+=1;losses[c['id']]=value
            selected=min(cell['candidates'],key=lambda c:(losses[c['id']],c['step'],c['lr'],c['id']))
            assert selected['id']==chosen['selected'][rule]['id'];checks+=1
    evaluation=rows(OUT/'evaluation.csv');assert len(evaluation)==60
    pairs=[];source_rows=[];rule_rows=[]
    for cell in plan['cells']:
        rr={r['rule']:r for r in evaluation if r['cell']==cell['id']}
        assert set(rr)=={'F0','FIXED','RECENT4','SPREAD4','ALL16'}
        for rule,r in rr.items():
            d=cache[r['canonical'],'D'];loss=score(d['prediction'],d['target'],d['scale'])['scaled_2pinball']
            assert abs(float(r['loss'])-loss)<=1e-10;checks+=1
        f0=float(rr['F0']['loss']);recent=float(rr['RECENT4']['loss']);spread=float(rr['SPREAD4']['loss'])
        pairs.append(dict(cell=cell['id'],source=cell['source'],seed=cell['seed'],arm=cell['arm'],f0_loss=f0,
            recent_loss=recent,spread_loss=spread,recent_minus_spread=recent-spread,
            recent_gain_f0=gain(recent,f0),spread_gain_f0=gain(spread,f0),spread_minus_recent_gain_pp=100*(recent-spread)/f0,
            same_checkpoint=rr['RECENT4']['canonical']==rr['SPREAD4']['canonical']))
        for rule,r in rr.items():rule_rows.append(dict(source=cell['source'],cell=cell['id'],rule=rule,loss=float(r['loss']),gain_f0=gain(float(r['loss']),f0)))
    for source in plan['ranges']:
        pp=[r for r in pairs if r['source']==source]
        source_rows.append(dict(source=source,recent_loss=float(np.mean([r['recent_loss'] for r in pp])),
            spread_loss=float(np.mean([r['spread_loss'] for r in pp])),recent_minus_spread=float(np.mean([r['recent_minus_spread'] for r in pp])),
            recent_gain_f0=float(np.mean([r['recent_gain_f0'] for r in pp])),spread_gain_f0=float(np.mean([r['spread_gain_f0'] for r in pp])),
            spread_minus_recent_gain_pp=float(np.mean([r['spread_minus_recent_gain_pp'] for r in pp])),same_checkpoint_cells=sum(r['same_checkpoint'] for r in pp)))
    macro=float(np.mean([r['spread_minus_recent_gain_pp'] for r in source_rows]))
    write_json(OUT/'analysis_summary.json',dict(source_rows=source_rows,paired_cells=pairs,source_balanced_spread_minus_recent_gain_pp=macro,
        rule_source_means=[dict(source=s,rule=r,mean_loss=float(np.mean([x['loss'] for x in rule_rows if x['source']==s and x['rule']==r])),
            mean_gain_f0=float(np.mean([x['gain_f0'] for x in rule_rows if x['source']==s and x['rule']==r]))) for s in plan['ranges'] for r in ['F0','FIXED','RECENT4','SPREAD4','ALL16']],
        interpretation_scope='Three sources, two optimization seeds and two arms per source; no independent n=12 claim, no significance/paper PASS threshold, no D oracle or post-D tuning.'))
    for name,rr in [('paired_cells.csv',pairs),('source_comparison.csv',source_rows),('per_origin_components.csv',perorigin)]:
        with open(OUT/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    history=read(OUT/'source_and_history_hashes.json');assert all(sha(ROOT/p)==h for p,h in history.items())
    write_json(OUT/'artifact_verification.json',dict(status='PASS',numeric_checks=checks,prediction_caches=len(index),max_metric_error=maxerr,
        selection_replay='36 selections exactly match independent direct metric replay',historical_files_unchanged=len(history),gpu_forward=0,gpu_backward=0))
    lines=['# 새 미래 구간 선택 전달 비교','',
        f"완료: 새 학습 0, backward 0, forward {rec['forward']}회(S {rec['S_forward']}, D {rec['D_forward']}). 실행 시간 {rec['elapsed_seconds']/60:.2f}분. 검증에서 12개 셀의 모든 선택을 봉인한 뒤 미래 구간을 평가했다.",
        f"주 비교의 source-balanced 차이는 {macro:+.4f} pp (%F0 gain 기준, 양수는 SPREAD4 우세)다. RECENT4/SPREAD4는 모두 4개 검증 origin을 사용했다. ALL16은 정보량이 다른 참조다.",'',
        '| 원천 | RECENT4 %F0 gain | SPREAD4 %F0 gain | SPREAD−RECENT pp | 같은 선택 /4 |','|---|---:|---:|---:|---:|']
    for r in source_rows:lines.append(f"| {r['source']} | {r['recent_gain_f0']:+.4f}% | {r['spread_gain_f0']:+.4f}% | {r['spread_minus_recent_gain_pp']:+.4f} | {r['same_checkpoint_cells']} |")
    lines+=['','양수 %F0 gain은 frozen 대비 손실 개선, 음수는 악화다. 원천별 평균은 두 seed와 두 arm을 포함하되 이들을 독립 데이터셋으로 세지 않았다. 각 셀·참조 결과는 paired_cells.csv 및 analysis_summary.json에 있다.',
        '모든 미래 D 후보를 추론하는 oracle은 만들지 않았다. 평가한 것은 각 규칙에서 선택된 모델, 사전 고정 step150/LR3e-5, F0의 합집합뿐이다. 따라서 D에서 다른 미평가 후보가 더 좋았는지는 알 수 없다.',
        '노출 감사는 이 체크아웃의 origin 메타데이터와 실행/데이터 기록 범위다. Electricity는 과거 실험과 겹치는 날짜의 다른 채널(4..7)을 사용한다. 외부 실험이나 foundation model 사전학습에 대한 미노출을 보장하지 않는다. 원시 데이터 형식/결측 검사는 과거에 이루어졌으므로 한 번도 읽지 않은 원시 데이터라는 뜻도 아니다.',
        '새 미래 구간의 결과를 본 뒤 검증 규칙·학습률·후보 집합을 조정하지 않았다. 이번 결과는 표준 선택 설계의 검증이며 새로운 PEFT 방법의 성능 입증이 아니다.','',
        '[실행 계획](plan.json) · [선택 봉인](selection_seal.json) · [노출 감사](exposure_audit.json) · [검산](artifact_verification.json) · [이전 A–D](../temporal_transfer_diagnostic_v1_D/REPORT.md)','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(macro_pp=macro,sources=source_rows,numeric_checks=checks),indent=2))


if __name__=='__main__':main()
