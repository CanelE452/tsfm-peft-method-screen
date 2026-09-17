"""Reporting completion: separate historical training costs and fixed512 effects.
No changed primary decision, no new selection, no new model call or training.
"""
import sys,csv,json,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.history_compression_v1_20260917.common import *
assert read(OUT/'PUBLICATION_AUDIT.json')['passed'];assert read(OUT/'objective_initialization_audit.json')['passed'];validate_seal()
d=Data();y=np.load(OC/'E_DISCOVERY_labels.npz')['y'];reuse=read(OUT/'REUSE_RECEIPT.json');new=read(OUT/'fits.json');rows=[]
for a in ['SHORT','LONG']+ARMS:
 arm={'SHORT':'B0','LONG':'B1'}.get(a,a);ff=[f for f in reuse['fits']+new if f['arm']==arm]
 rows.append(dict(arm=a,peak_training_MiB=float(np.mean([f['peak_allocated_bytes'] for f in ff])/2**20),optimizer_seconds_per512=float(np.mean([f['optimizer_seconds'] for f in ff])),fits=len(ff),measurement='historical_same_recipe' if a in ['SHORT','LONG'] else 'current_run'))
csvwrite(OUT/'training_resource_summary.csv',rows)
# Fixed512 was already a sealed secondary evaluation. These contrasts are explicitly post-run diagnostics.
man=read(OUT/'predictions_manifest.json')+[dict(r,arm={'B0':'SHORT','B1':'LONG'}[r['arm']]) for r in reuse['predictions']];pred={}
for r in man:
 if r['policy']=='fixed512':
  p=np.load(ROOT/r['path'])['pred'];pred[r['arm'],r['seed']]=p[:,0] if p.ndim==4 else p
w=block_counts(d.x['E_DISCOVERY']['origins'],168);den=w.sum(1)
def scores(a):
 point=[];boot=[]
 for seed in [73101,73102]:
  q=pred[a,seed];point.append(nrms(q,y,d.sigma));mse=(((q-y)/d.sigma[None,:,None])**2).mean(-1);boot.append(np.sqrt((w@mse)/den[:,None]).mean(1))
 return np.mean(point),np.mean(boot,0)
cs=[]
for a,b in [('LEARN_KD','POOL_KD'),('LEARN_KD','LEARN'),('POOL_KD','POOL'),('LEARN','POOL')]:
 av,ab=scores(a);bv,bb=scores(b);boot=100*(bb-ab)/bb;lo,hi=np.quantile(boot,[.025,.975]);cs.append(dict(proposed=a,baseline=b,policy='fixed512',gain_percent=float(100*(bv-av)/bv),CI_low=float(lo),CI_high=float(hi),post_run_diagnostic=True,primary_decision_changed=False))
csvwrite(OUT/'fixed512_component_diagnostics.csv',cs)
text='## 학습 메모리와 선택 효과의 추가 해석\n\n'
text+='추론 자원과 별도로 학습 중 peak를 비교하면 다음과 같다. 각 군의 동일한 네 경로를 평균했다. SHORT/LONG은 기존 동일 recipe의 측정값 재사용이고 새 군과 같은 시각에 재측정한 학습 benchmark가 아니다. 따라서 아래 속도 차이를 격리된 시스템 우위로 읽으면 안 된다.\n\n| 군 | 학습 peak MiB | 512 updates optimizer 초 | 측정 |\n|---|---:|---:|---|\n'
for r in rows:text+=f"| {r['arm']} | {r['peak_training_MiB']:.2f} | {r['optimizer_seconds_per512']:.2f} | {r['measurement']} |\n"
l=next(r for r in rows if r['arm']=='LONG');p=next(r for r in rows if r['arm']=='LEARN_KD');gain=100*(l['peak_training_MiB']-p['peak_training_MiB'])/l['peak_training_MiB']
text+=f'\nLEARN_KD의 학습 peak는 이 기록상 LONG보다 **{gain:.2f}% 낮다**. 따라서 자원 이점이 전혀 없다는 결론도 맞지 않는다. 다만 주 결과의 정확도 손실 2.89%, 거의 줄지 않은 추론 비용, 별도 시점의 학습 속도 측정이라는 제한을 함께 남긴다. 사전 판정에 사용한 추론 자원·정확도 조건을 이 결과를 보고 학습 메모리 조건으로 바꾸지 않았다.\n\n'
text+='증류 없는 POOL/LEARN은 두 seed 모두 INIT, 증류군은 256 업데이트가 선택됐다. 따라서 주 비교의 약 3.6% 이득에는 검증 선택의 차이가 함께 반영된다. 처음부터 보존한 보조 정책인 고정 512 결과에서도 다음 직접 비교를 계산했다. **이 표는 실행 후 해석용이며 주 판정을 대체하지 않는다.** 모든 군에 같은 시간 block을 적용했고 추가 학습이나 모델 선택은 없다.\n\n| 고정512 직접 비교 | gain % | 95% 시간 block 구간 |\n|---|---:|---|\n'
for r in cs:text+=f"| {r['proposed']} / {r['baseline']} | {r['gain_percent']:+.4f} | [{r['CI_low']:+.4f}, {r['CI_high']:+.4f}] |\n"
text+='\n학습 압축의 선택 가중치가 균등 평균에 가깝다는 관찰, 두 핵심 구성요소의 주 비교, 고정512 진단을 모두 보존한다. 작은 변화를 더 크게 만들면 성공할 것이라고 추론하지 않으며 LR·초기화·학습량을 변경하지 않았다. [고정512 진단](fixed512_component_diagnostics.csv), [목적함수·초기화의 정확한 검산](objective_initialization_audit.json).\n\n'
p=OUT/'REPORT.md';s=p.read_text();assert '## 학습 메모리와 선택 효과의 추가 해석' not in s
s=s.replace('| LEARN / POOL | +0.0000 | [+0.0000, +0.0000] | NEGATIVE_UNCERTAIN |','| LEARN / POOL | +0.0000 | [+0.0000, +0.0000] | IDENTICAL_SELECTED_OUTPUTS |')
s=s.replace('## 신규성과 일반화의 한계',text+'## 신규성과 일반화의 한계');p.write_text(s)
audit=read(OUT/'PUBLICATION_AUDIT.json');audit.update(report_sha256=sha(p),objective_and_initialization_verified=True,training_resource_comparison_labelled_historical=True,fixed512_diagnostics_labelled_post_run=True,at=time.time());save(OUT/'PUBLICATION_AUDIT.json',audit)
print('Training resource and selection-effect interpretation added',gain,cs)
