"""Independent cache audit and Korean report for the fixed 0-fit diagnostic."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/lora_projection_diagnostic_20260916'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    status=read(OUT/'status.json');assert status['status']=='COMPLETE'
    assert status['fits']==status['optimizer_updates']==0 and status['predictions_completed']==16
    seal=read(OUT/'seal.json');history=read(OUT/'historical_hashes.json')
    for mapping in [history,seal['source_hashes'],seal['references'],seal['model_files']]:
        for p,h in mapping.items():assert sha(ROOT/p)==h,p
    for d in seal['data'].values():
        for p,h in d['staged'].items():assert sha(ROOT/p)==h,p
    records=read(OUT/'predictions.json');assert len(records)==16
    checks=read(OUT/'checks.json');assert len(checks)==16
    assert all(r['unchanged_during_forward'] and r['only_declared_B_changed'] for r in checks)
    errors=[];scores=[];arrays={};replays=0
    for r in records:
        assert sha(ROOT/r['prediction_path'])==r['prediction_sha256']
        with np.load(ROOT/r['prediction_path']) as z:a={k:z[k].copy() for k in z.files}
        arrays[r['dataset'],r['seed'],r['arm']]=a
        cm=[];ca=[]
        for c in range(32):
            pairs=[(float(p),float(y)) for p,y in zip(a['prediction'][:,c].flat,a['target'][:,c].flat) if math.isfinite(float(y))]
            cm.append(math.fsum((p-y)**2 for p,y in pairs)/len(pairs));ca.append(math.fsum(abs(p-y) for p,y in pairs)/len(pairs))
        mse=math.fsum(cm)/32;mae=math.fsum(ca)/32;raw=math.fsum(v*float(s) for v,s in zip(ca,a['std']))/32
        for key,val in [('mse',mse),('mae',mae)]:assert math.isclose(val,r['metrics'][key],rel_tol=1e-12,abs_tol=1e-12)
        assert math.isclose(raw,float(np.mean(r['metrics']['channel_raw_mae'])),rel_tol=1e-12,abs_tol=1e-12)
        errors.append(abs(mse-r['metrics']['mse']))
        assert len(r['zeroed_B_names'])=={'FULL':0,'QK_ONLY':8,'V_ONLY':16,'NONE':24}[r['arm']]
        cp=next(c['checkpoint'] for c in seal['cells'] if (c['dataset'],c['seed'])==(r['dataset'],r['seed']))
        assert (r['checkpoint_path'],r['checkpoint_sha256'])==(cp['checkpoint_path'],cp['checkpoint_sha256'])
        with np.load(ROOT/cp['prediction_path']) as z:
            for key in ['target','std','origins']:assert np.array_equal(a[key],z[key],equal_nan=True)
            if r['arm']=='FULL':assert np.array_equal(a['prediction'],z['prediction']);replays+=1
        scores.append(dict(dataset=r['dataset'],seed=r['seed'],arm=r['arm'],selected_epoch=r['selected_epoch'],mse=mse,mae=mae,raw_mae=raw,forward_wall_seconds=r['forward_wall_seconds'],forward_peak_allocated_mib=r['forward_peak_allocated_mib']))
    assert replays==4
    effects=[];macro=[]
    for d in ['electricity','traffic']:
        for seed in [41000,41001]:
            m={r['arm']:r['mse'] for r in scores if (r['dataset'],r['seed'])==(d,seed)}
            total=m['NONE']-m['FULL']
            qk=.5*((m['NONE']-m['QK_ONLY'])+(m['V_ONLY']-m['FULL']))
            v=.5*((m['NONE']-m['V_ONLY'])+(m['QK_ONLY']-m['FULL']))
            assert math.isclose(total,qk+v,rel_tol=1e-12,abs_tol=1e-12)
            effects.append(dict(dataset=d,seed=seed,full=m['FULL'],qk_only=m['QK_ONLY'],v_only=m['V_ONLY'],none=m['NONE'],qk_removal_cost=m['V_ONLY']-m['FULL'],v_removal_cost=m['QK_ONLY']-m['FULL'],total_reduction=total,qk_symmetric_allocation=qk,v_symmetric_allocation=v,v_allocation_percent=100*v/total if total else None,nonadditivity=m['NONE']-m['QK_ONLY']-m['V_ONLY']+m['FULL']))
        for arm in seal['arms']:
            rows=[r for r in scores if (r['dataset'],r['arm'])==(d,arm)]
            macro.append(dict(dataset=d,arm=arm,**{k:sum(r[k] for r in rows)/2 for k in ['mse','mae','raw_mae']}))
    csvwrite(OUT/'scores.csv',scores);csvwrite(OUT/'macro_scores.csv',macro);csvwrite(OUT/'effects.csv',effects)
    events=[json.loads(x) for x in (OUT/'gpu_projection.jsonl').read_text().splitlines()]
    external=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in e['apps']) for e in events)
    assert external==0
    verify=dict(predictions=16,full_checkpoint_replays_exact=4,mse_mae_raw_mae_scalar_records=16,max_mse_abs_error=max(errors),same_target_origin_std=True,unchanged_head_and_frozen_tensors=True,only_declared_B_ablation=True,source_checkpoint_model_data_hashes_valid=True,historical_files_preserved=len(history),unapproved_compute_samples=external,new_fits=0,optimizer_updates=0,E_predictions=0,report_source_sha256=sha(Path(__file__)))
    save(OUT/'verification.json',verify)
    save(OUT/'decision.json',dict(execution='COMPLETE',evidence='JOINT_CHECKPOINT_DEPENDENCE_ON_QK_AND_V',novelty='KNOWN_ABLATION_DIAGNOSTIC',new_method_topic='NOT_CONFIRMED',independent_screen_pass='NOT_EVALUATED',time_relationship_only_hypothesis='NOT_ESTABLISHED',retraining_ablation='NOT_RUN'))
    wall=read(OUT/'projection_wall.json')
    lines=['# LoRA Q/K·V 진단: 시간 연결 수정만으로 충분하다는 근거는 없음','','**신규 학습0회로16개 검증 예측과 검산을 완료했다. Q/K와 V 어느 한쪽을 제거해도 네 checkpoint 모두 악화했고, V 제거 비용이 더 컸다.**','','## 원점수','','같은 학습된 head를 고정한 상태다. FULL은 기존 V 선택 예측과exact하다. NONE은 LoRA와 공동 학습한 head를 남긴 상태이므로, 별도로 학습한 head-only의 성능으로 읽으면 안 된다. 모든 점수는 재사용 V이며 E 점수가 아니다.','','| 원천 | FULL | Q/K만 유지 | V만 유지 | 모두 제거 |','|---|---:|---:|---:|---:|']
    for d in ['electricity','traffic']:
        m={r['arm']:r['mse'] for r in macro if r['dataset']==d}
        lines.append(f"| {d} | {m['FULL']:.6f} | {m['QK_ONLY']:.6f} | {m['V_ONLY']:.6f} | {m['NONE']:.6f} |")
    lines+=['','| 원천 | seed | Q/K 제거 MSE 증가 | V 제거 MSE 증가 | 전체 LoRA 이득 중 V 대칭배분 |','|---|---:|---:|---:|---:|']
    for r in effects:lines.append(f"| {r['dataset']} | {r['seed']} | {r['qk_removal_cost']:.6f} | {r['v_removal_cost']:.6f} | {r['v_allocation_percent']:.2f}% |")
    lines+=['','대칭 배분은 두 구성요소를 넣는 순서의 기여를 평균한 설명용 계산이다. 절대 MSE 감소량의 합을 보존하지만, 재학습한 방법의 독립 가치나 보편적 인과 비중을 측정한 것은 아니다. [seed별 원점수](scores.csv), [분해](effects.csv), [사전 프로토콜](PROTOCOL.md).','','## 다음 주제에 주는 근거와 반례','','[확인] 같은 용량의 마지막 adapter가 LoRA를 따라가지 못한 뒤, encoder 내부 변화의 역할을 분리했다. 이번 저장 모델에서는 V 경로도 크게 작용하고 Q/K 경로도 제거할 수 없었다. 앞서 제안한 시간 패치 관계만의 경량 적응을 정답으로 선택할 근거는 확보되지 않았다.','','[한계] V는 뒤 layer의 입력을 바꾸므로 그 뒤 attention 관계에도 영향을 준다. q/k를 바꾸면 전달 내용도 달라진다. 이 실험은 각 projection의 국소적 변경을 제거했으며, 시간과 내용을 완전히 분리한 실험이 아니다. 또한 head와 encoder가 공동 적응했으므로 삭제 손실을 그대로 학습 기여로 해석할 수 없다. QK_ONLY/V_ONLY를 처음부터 학습하거나 예산을 맞춘 비교는 미실행이다.','','[제안] 다음 설계에서 설명해야 할 것은 **시간 연결과 전달 내용의 공동 적응을 어떻게 유지하는가**다. 단순한 시간 평균·입력 유사도 weighting·adapter 폭 확대만으로 충분하다는 주장은 현재 증거가 지지하지 않는다. 그러나 이 요구조건 자체가 새 알고리즘은 아니다. 구체적 수식과 가까운 단순 대조의 차이를 정당화하기 전에는 후보 확보나 PASS라고 기록하지 않는다.','','## 선행과 신규성 경계','','[TRACE, arXiv v1(2025), §4.2](https://arxiv.org/html/2503.16991v1)는 다른 LoRA 모듈을 mask할 때 남은 모듈 중요도가 달라지는 문제와 gate를 이용한 반복 masking·선택을 다룬다. 따라서 이번 상호작용 관찰이나 중요도에 따라 projection을 선택한다는 발상만으로 신규성을 주장할 수 없다. 이 논문의 모든 주장·실험을 재현한 것은 아니다.','','[Beyond LoRA, arXiv v1(2024), §2](https://arxiv.org/html/2409.11302v1)는 Chronos에 BitFit·LayerNorm tuning·VeRA·FourierFT를 비교한다. 기존 PEFT를 다른 이름으로 적용하는 것과 새 방법론 개발을 구분한다. 현재 MOMENT 진단의 결과를 해당 ICU 과제나 Chronos-2 건물 실험에 일반화하지 않는다.','','## 실행·검산·미실행 범위','',f"실제fits0, optimizer updates0, V 예측16/16, E 예측0. Controller {wall['seconds']:.1f}초, 최소 GPU 여유{wall['minimum_free_mib']}MiB, 비승인compute0표본. 메모리는 순전파 진단 비용이며 학습 메모리 절감의 새 측정이 아니다.",'',f"16개 예측의 MSE·MAE·rawMAE를 독립float64 scalar로 검산했다. MSE 최대차{max(errors):.3g}, FULL4개 예측은기존checkpoint출력과exact다. head·frozen tensor/buffer 불변, 지정한B만0, 원상복구·source/checkpoint/model/data hash 및 기존{len(history)}개 결과파일 보존을 확인했다. [검산](verification.json).",'', '새 후보 구현·학습, 재학습에 의한 구성요소 비교, 미노출 평가, 다른backbone 일반화는 아직 미실행이다. EXECUTION=COMPLETE, 새방법론주제=NOT_CONFIRMED, 독립SCREEN_PASS=NOT_EVALUATED. 예측배열·원자료·가중치는 로컬, 코드·원점수·manifest·보고서는GitHub에 보존한다.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(verification=verify,macro=macro,effects=effects),ensure_ascii=False))

if __name__=='__main__':main()
