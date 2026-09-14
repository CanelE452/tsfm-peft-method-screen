"""Append-only correction and train/V review of the completed window study."""
import json
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'results/anchor_window_study_20260914'
OUT = ROOT/'research/window_study_followup_20260914'


def gain(method, reference):
    return 100*(1-method/reference)


def main():
    assert not OUT.exists(), 'Preserve an existing review'
    files = [p for p in PARENT.rglob('*') if p.is_file()]
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    read = lambda p: json.loads(p.read_text())
    summary = read(PARENT/'summary.json')
    evaluation = read(PARENT/'evaluation.json')
    selections = read(PARENT/'evaluation_seal.json')['selections']
    rows=[]
    for cell in summary['cells']:
        ds=cell['dataset']
        for seed in (34000,34001):
            picked={a:next(r for r in evaluation if (r['dataset'],r['seed'],r['arm'])==(ds,seed,a))
                    for a in ('native','native_anchor')}
            plain,anchor=[picked[a]['metrics']['scaled_2pinball'] for a in ('native','native_anchor')]
            selected={a:next(r for r in selections if (r['dataset'],r['seed'],r['arm'])==(ds,seed,a))
                      for a in picked}
            rows.append(dict(dataset=ds,seed=seed,plain_loss=plain,anchor_loss=anchor,
                corrected_gain_percent=gain(anchor,plain),old_loss_ratio_percent=100*anchor/plain,
                plain_step=selected['native']['step'],anchor_step=selected['native_anchor']['step']))
        rr=rows[-2:]
        macro=gain(sum(r['anchor_loss'] for r in rr),sum(r['plain_loss'] for r in rr))
        assert abs(macro-cell['anchor_gain_percent'])<1e-10
    OUT.mkdir(parents=True)
    payload=dict(corrections=rows,original_primary_effects_unchanged=True,
        original_decision=summary['decision'],parent_hashes=hashes,
        new_training_fits=0,new_evaluation_accesses=0,
        scope='Existing aggregate evaluation metrics and train/V selections only; no new E forecasts.')
    (OUT/'correction.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    lines=['# Window study 후속 검토 및 정정','',
      '48 fits / 43,200 updates와 평가·검증·GitHub 저장은 정상 완료됐다. 실행기는 archive 뒤 종료하며 다음 연구를 호출하지 않았다. 이는 학습 실패가 아니라 후속 제어 누락이다.','',
      '기존 보고서 seed_gains_percent는 loss ratio × 100을 개선율로 잘못 표기했다. 올바른 식은 100 × (1 − anchor / plain)이다. 동일 성능은 0%이며 100% 개선이 아니다. 기존 원본을 보존하고 아래에 정정한다. 데이터별 평균 주효과와 원래 가설 판정은 달라지지 않는다.','',
      '| Dataset | Seed | 정정 개선율 % | Native 선택 step | Anchor 선택 step |',
      '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['seed']} | {r['corrected_gain_percent']:+.4f} | {r['plain_step']} | {r['anchor_step']} |")
    lines += ['',
      'Beijing dense의 +4.17% 개선은 보존 규제의 조건부 효과를 검토할 근거다. 그러나 sparse에서 더 유리하다는 가설은 지지되지 않았다. ETTm2는 모두 step 0을 선택했으므로 학습된 두 방법의 동률이라는 해석도 피해야 한다. Electricity의 악화와 함께 모두 보고한다.',
      '다음 과학 작업은 train/V에서 학습 악화 시점과 과제/보존 기울기의 관계를 확인해 개입 대상을 좁히는 것이다. 여기서는 새 학습·새 E 추론을 수행하지 않았다.', '',
      '후속 자동 재개 활성화는 승인 대기다. 완료 감지 코드만으로 모델이 계속 추론한다고 주장하지 않는다.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    print(json.dumps({'parent_files_unchanged':len(hashes),'corrected_seed_rows':len(rows),'original_primary_effects_unchanged':True}))


if __name__=='__main__':main()
