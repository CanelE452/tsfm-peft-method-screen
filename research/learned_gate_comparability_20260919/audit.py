"""Read-only historical selection and feature-access audit. No training/inference."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sources={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):
 p=ROOT/p;sources[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())
def main():
 new=read('results/learned_gate_comparison_20260919/SEAL.json');rows=[];packets=[]
 for source in ['electricity','ettm1']:
  for arm,oldarm,folder in [('MAG_ONLY','MAG_ONLY','c3_weakness_controls_20260918'),('PLAIN','C2','additive_b0_adapter_v1_20260917'),('C3','C3','additive_b0_adapter_v1_20260917')]:
   choice=read(f'results/{folder}/LR_SELECTION.json')[source][oldarm];receipts=[]
   for lr in [1e-4,3e-4]:
    r=read(f'results/{folder}/fits/{source}_{oldarm}_s81550_lr{lr:g}/receipt.json');assert r['status']=='COMPLETE' and r['updates']==1024 and r['microbatch']==32
    assert [c['step'] for c in r['checkpoints']]==[0,256,512,768,1024]
    for c in r['checkpoints']:assert sha(ROOT/c['checkpoint'])==c['sha256'];sources[c['checkpoint']]=c['sha256']
    receipts.append(r)
   selected=min(receipts,key=lambda r:(r['selected']['objective'],r['lr'],r['selected']['step']))
   assert choice['fit']==selected['fit'] and choice['lr']==selected['lr']
   repeats=[]
   for seed in [81551,81552]:
    r=read(f"results/{folder}/fits/{source}_{oldarm}_s{seed}_lr{choice['lr']:g}/receipt.json")
    assert r['status']=='COMPLETE' and r['updates']==1024 and r['microbatch']==32 and r['frozen_unchanged']
    repeats.append(dict(seed=seed,lr=r['lr'],selected_step=r['selected']['step'],checkpoint_sha256=r['selected']['sha256']))
   rows.append(dict(source=source,arm=arm,LR_candidates=[r['lr'] for r in receipts],selection_seed=81550,selection_updates=2048,selected_lr=choice['lr'],checkpoints=[0,256,512,768,1024],repeat_seed_details=repeats))
  for file in ['train_x','train_y','train_sigma','V_SELECT_x','V_SELECT_y','V_SELECT_sigma']:
   p=ROOT/f'.cache/additive_persistence_validation_v1_20260917/conditions/{source}/{file}.npy';h=sha(p)
   assert new['hashes'][str(p.relative_to(ROOT))]==h
   for folder in ['additive_b0_adapter_v1_20260917','c3_weakness_controls_20260918','c3_training_factorial_20260918']:
    q=ROOT/f'.cache/{folder}/conditions/{source}/{file}.npy';assert sha(q)==h
   packets.append(dict(source=source,file=file,sha256=h,exact_same_across_historical_and_current=True))
 modelrows=read('results/temporal_response_peft_20260919/MODEL_SELECTION.json')
 for r in rows:
  for repeat in r['repeat_seed_details']:
   m=next(v for v in modelrows if (v['source'],v['arm'],v['seed'],v['stage'])==(r['source'],r['arm'],repeat['seed'],'selected'))
   assert m['sha256']==repeat['checkpoint_sha256'] and m['lr']==repeat['lr']
 for name in ['experiments/c3_weakness_controls_20260918/model.py','experiments/outlier_signal_peft_v1_20260917/model.py','experiments/learned_gate_comparison_20260919/model.py','experiments/learned_gate_comparison_20260919/train.py','experiments/additive_persistence_validation_v1_20260917/model.py']:
  sources[name]=sha(ROOT/name)
 sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
 result=dict(status='VERIFIED_HISTORICAL_SELECTION_AND_INFORMATION_ACCESS',old_selection_groups=rows,packet_checks=packets,new_fit_count=0,new_optimizer_updates=0,new_forecast_inferences=0,new_evaluation_labels_read=False,methodology_goal_achieved=False,current_gate_training_completion_not_claimed=True,source_hashes=sources,feature_difference='MAG global robust observed-window statistics versus learned gate on original pre-attention patch embedding; same permitted observations does not imply identical gate features',causal_limit='Cannot isolate trainability alone from representation, functional family, initialization and parameter-count differences')
 (OUT/'AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 lines=['| source | 기존 arm | 선택 LR | seed81551 step | seed81552 step |','| --- | --- | --- | --- | --- |']
 for r in rows:lines.append(f"| {r['source']} | {r['arm']} | {r['selected_lr']:g} | {r['repeat_seed_details'][0]['selected_step']} | {r['repeat_seed_details'][1]['selected_step']} |")
 report='''# 학습형 gate 직접 비교의 공정성과 해석 범위

이번 감사는 실행 중인16경로를 변경하지 않는 읽기 전용 검사다. 새fit/optimizer update/forecast inference/E채점0. 기존 선택receipt·checkpoint SHA·TRAIN/V 배열을 확인했다. 방법론 목표 완료나 현재 실험 완료를 뜻하지 않는다.

## 선택 기회와 재사용

기존 MAG/PLAIN/C3의 두원천 모두 selection seed81550에서 LR1e-4/3e-4 각각1024updates, checkpoint0/256/512/768/1024를 실행했다. objective→LR→step 동률 규칙으로 선택 결과를 독립 재계산했다. 반복81551/81552는 선택LR을 사용했고 각1024updates/micro32/frozen 보존을 확인했다. 현재 참조 MODEL_SELECTION의 selected SHA가 이 원래 선택과 같음을 확인했다. factorial의 다른 초기값·순서 조합에서 E가 좋은 모델을 골라 재사용한 것이 아니다.

'''+ '\n'.join(lines)+'''

두원천×TRAIN/V의6파일=12개를 과거4개 경로와 현재SEAL에서 hash 일치로 확인했다. 기존3번째seed는 지우지 않지만 이번 직접 비교의 두 공통seed와 구분한다. 새 두군의 실제 완료·최종 선택은 실행 후 별도 AUDIT가 확인해야 한다. 이 감사로 미래 완료를 선행 확정하지 않는다.

## 같은 관측 권한과 서로 다른 gate 표현

MAG는 전체512입력의 median/MAD 및 기존 TRAIN sigma로 scale floor를 만든다. patch16개 중 robust절댓값3 초과 비율을 세어1에서 뺀다. 학습형 gate는 Chronos의 원래 instance normalization과 patch embedding을 지난 h_j에 Linear512→1/sigmoid를 적용한다. 이 h는 encoder self-attention 이전이다. 두군 모두 합법적인 관측과기존TRAIN만 사용하지만, MAG의전역robust통계가학습gate에별도feature로주어지는것은아니다.

따라서 이 비교는 **이 고정 robust-statistic gate와 이 학습형 embedding gate의 전체 설계 비교**다. 어떤 차이가 나도 ‘학습하지 않는다는 속성 하나’나 ‘MAG의 진폭값 하나’의 인과효과로 분리할 수 없다. 표현·함수군·초기 gate·파라미터 수의 차이가 함께 있다. 데이터 권한 공정성과 gate feature 동일성을 혼동하지 않는다.

초기 residual up=0이므로 모든 예측은B0와 같지만, gate 자체는 MAG의입력의존값 대학습군0.5다. 초기 residual의 gradient도 gate에 따라 달라질 수 있다. 동일 초기예측은 동일 optimization trajectory를 뜻하지 않는다. 이 차이를 없애기 위한 재초기화·추가훈련을 실시하지 않는다.

MAG 추가parameter8712, 학습군9225(+513,5.89%). 학습군 둘의 차이는 사전고정entropy벌점 유무이며, 같은표현/초기값/학습기회로 비교한다. 이를 공식GateRA 전체 재현이라고하지않는다. 기존 B0 사전학습LoRA294912와본체는공유·동결이며8712만으로모든B0훈련비가대체된다고계산하지않는다.

## 논문에서 지킬 표현

결과가 지지하는 경우에만 ‘고정된 B0 위에서 관측의 robust 통계를 이용해 추가 patch residual을 제한하는 설계가, 이 통제된 embedding-gate 대조와 비교해 특정 조건에서 유리한 정확도·파라미터 절충을 보였다’고 쓸 수 있다. 수치와조건은전체채점완료뒤확정한다.

‘고정 gate가학습gate보다일반적으로우수하다’, ‘학습형gate가불필요함을증명했다’, ‘실제센서오류와regime change를식별했다’, ‘Time-PEFT/GateRA전체를이겼다’는현재비교가검증하지않는주장이다. 신규성은성능대조와별개다. 기존 gate원리와현설계의차이를정확히설명하고정식선행전체비교·독립source의한계를남긴다.

사전등록된4개주비교·기준·LR·seed·형태·표본을이감사로변경하지않았고후속학습을추가하지않았다.
'''
 (OUT/'COMPARABILITY_KO.md').write_text(report)
 print(json.dumps({'historical_selection_groups':len(rows),'identical_packet_groups':len(packets),'new_training':0,'forecast_inference':0}))
if __name__=='__main__':main()
