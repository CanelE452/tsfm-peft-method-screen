"""Independent timing ledger audit and manuscript supplement; no model calls."""
import ast,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];NAME='mag_inference_cost_20260919';OUT=ROOT/'results'/NAME;EXP=ROOT/'experiments'/NAME
ARMS=['B0','PLAIN','MAG_ONLY','TOKEN_GATE','TOKEN_GATE_ENTROPY','PETSA_XY_OFFLINE']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
 assert read(OUT/'status.json')['execution'] in ['COMPLETE_MEASUREMENT','COMPLETE_VERIFIED']
 seal=read(OUT/'SEAL.json')
 for p,h in seal['hashes'].items():assert sha(ROOT/p)==h,p
 df=pd.read_json(OUT/'TIMINGS.jsonl',lines=True);blocks=pd.read_json(OUT/'BLOCKS.jsonl',lines=True)
 counts=read(OUT/'COUNTS.json');assert counts==dict(timed=9216,warmup=864,parity=48,total=10128,optimizer_updates=0,backward_calls=0,E_forwards=0)
 keys=['source','seed','arm','round','batch'];assert len(df)==9216 and len(blocks)==288 and not blocks.contaminated.any()
 assert not df.duplicated(keys+['repeat']).any() and not blocks.duplicated(keys).any()
 assert set(df.arm)==set(ARMS) and set(df.source)=={'electricity','ettm1'} and set(df.seed)=={81551,81552}
 assert np.isfinite(df[['wall_ms','cuda_stream_ms']]).all().all() and (df[['wall_ms','cuda_stream_ms']]>0).all().all()
 assert (df.peak_allocated>=df.allocated_baseline).all()
 assert (df.peak_allocated-df.allocated_baseline==df.transient_allocated).all()
 assert (df.groupby(keys).size()==32).all() and (df.groupby(keys).repeat.nunique()==32).all()
 for r in df[['arm','round','position']].drop_duplicates().itertuples():assert (ARMS.index(r.arm)-r.round)%6==r.position
 assert df.groupby(['source','seed']).input_hash.nunique().eq(1).all()
 expected=dict(B0=0,PLAIN=8712,MAG_ONLY=8712,TOKEN_GATE=9225,TOKEN_GATE_ENTROPY=9225,PETSA_XY_OFFLINE=19010)
 for a,n in expected.items():assert df[df.arm==a].adaptation_parameters.eq(n).all()
 checks=read(OUT/'MODEL_CHECKS.json');assert len(checks)==24 and len({(r['source'],r['seed'],r['arm']) for r in checks})==24
 assert all(r['state_unchanged'] and r['output_exact'] and r['input_unchanged'] for r in checks)
 gpu=[json.loads(s) for s in (OUT/'gpu_inference.jsonl').read_text().splitlines()]
 unapproved=sum(any(not a['own'] and not (a.get('allowed_desktop') and a['name']=='/usr/share/rustdesk/rustdesk') for a in r['apps']) for r in gpu);assert unapproved==0
 code=(EXP/'run.py').read_text();tree=ast.parse(code)
 calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
 assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ['step','backward'] for n in calls)
 assert 'torch.inference_mode()' in code and 'train_y.npy' not in code and 'E_DISCOVERY' not in code
 metrics=[]
 for key,z in df.groupby(['source','seed','arm','batch']):
  metrics.append(dict(zip(['source','seed','arm','batch'],key))|dict(n=len(z),wall_median_ms=float(np.median(z.wall_ms)),wall_p10_ms=float(np.quantile(z.wall_ms,.1)),wall_p90_ms=float(np.quantile(z.wall_ms,.9)),cuda_median_ms=float(np.median(z.cuda_stream_ms)),peak_allocated_MiB=float(z.peak_allocated.max()/2**20),transient_peak_MiB=float(z.transient_allocated.max()/2**20),adaptation_parameters=int(z.adaptation_parameters.iloc[0])))
 pd.DataFrame(metrics).to_csv(OUT/'BY_CASE.csv',index=False)
 rounds=df.groupby(keys,as_index=False)[['wall_ms','cuda_stream_ms']].median();rounds.to_csv(OUT/'ROUND_MEDIANS.csv',index=False)
 ratio=[]
 for (source,seed,round_id,batch),z in rounds.groupby(['source','seed','round','batch']):
  z=z.set_index('arm')
  for baseline in ARMS:
   if baseline=='MAG_ONLY':continue
   ratio.append(dict(source=source,seed=seed,round=round_id,batch=batch,baseline=baseline,wall_ratio=float(z.loc['MAG_ONLY','wall_ms']/z.loc[baseline,'wall_ms']),cuda_ratio=float(z.loc['MAG_ONLY','cuda_stream_ms']/z.loc[baseline,'cuda_stream_ms'])))
 rr=pd.DataFrame(ratio);assert len(rr)==240;rr.to_csv(OUT/'PAIRED_ROUND_RATIOS.csv',index=False)
 summary=[]
 for (arm,batch),z in df.groupby(['arm','batch']):
  summary.append(dict(arm=arm,batch=batch,wall_median_ms=float(np.median(z.wall_ms)),wall_p10_ms=float(np.quantile(z.wall_ms,.1)),wall_p90_ms=float(np.quantile(z.wall_ms,.9)),cuda_median_ms=float(np.median(z.cuda_stream_ms)),peak_allocated_MiB=float(z.peak_allocated.max()/2**20),transient_peak_MiB=float(z.transient_allocated.max()/2**20),additional_parameters=expected[arm]))
 summary=pd.DataFrame(summary);summary.to_csv(OUT/'SUMMARY.csv',index=False);df.to_csv(OUT/'TIMINGS.csv.gz',index=False,compression='gzip')
 ratio_summary=rr.groupby(['batch','baseline']).agg(wall_ratio_median=('wall_ratio','median'),wall_ratio_min=('wall_ratio','min'),wall_ratio_max=('wall_ratio','max'),cuda_ratio_median=('cuda_ratio','median')).reset_index();ratio_summary.to_csv(OUT/'RATIO_SUMMARY.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 labels=['B0','Plain','MAG','Gate','Gate+H','PETSA cell']
 for ax,batch in zip(axes,[1,32]):
  z=summary[summary.batch==batch].set_index('arm').loc[ARMS]
  ax.bar(range(6),z.wall_median_ms,color=['#aab5c0','#aab5c0','#287b6d','#aab5c0','#aab5c0','#aab5c0'])
  ax.errorbar(range(6),z.wall_median_ms,yerr=[z.wall_median_ms-z.wall_p10_ms,z.wall_p90_ms-z.wall_median_ms],fmt='none',color='0.2',capsize=3)
  ax.set_xticks(range(6),labels,rotation=30,ha='right');ax.set_title(f'Batch {batch}');ax.set_ylabel('Synchronized request wall time (ms)')
 fig.suptitle('Same-device inference; saved selected models; bars median, whiskers p10–p90')
 fig.tight_layout();fig.savefig(OUT/'inference_cost.png',dpi=180);fig.savefig(OUT/'inference_cost.pdf');plt.close(fig)
 rows=[[r.arm,r.batch,f'{r.wall_median_ms:.3f}',f'[{r.wall_p10_ms:.3f}, {r.wall_p90_ms:.3f}]',f'{r.peak_allocated_MiB:.3f}',r.additional_parameters] for r in summary.itertuples()]
 rrows=[[r.batch,r.baseline,f'{r.wall_ratio_median:.3f}',f'[{r.wall_ratio_min:.3f}, {r.wall_ratio_max:.3f}]'] for r in ratio_summary.itertuples()]
 report='''# 동일 조건 추론 자원 검사

기존24개 선택 모델을 그대로 사용한 측정을 완료했다. 신규 학습0, optimizer/backward0, E예측·정답채점0. 기존 성능·선택·기준을 변경하지 않았다. 이 결과는 과거 optimizer timer 경계 차이를 소급 해결하지 않으며 **현재 구현의 추론 비용**만 보완한다.

## 고정 범위와 검산

Electricity/ETTm1 × 기존 seed81551/81552 × 6군. 각 원천 TRAIN epoch0 index0,33,…,1023의32개 관측 입력과 sigma를 사용했다. REFERENCE/POINT/BURST/SHIFT가8개씩이며 변형 진폭은4뿐이다. 새 변화 강도 전반의 비용을 확인한 것으로 해석하지 않는다. 모델은 state 이름·clean 입력·future label을 받지 않았다. 입력은 CPU→GPU 전송 후 동일하게 재사용했다.

FP32/TF32off/eval/inference_mode/CPUthreads4, batch1/32,6개 순서 회전, block별 warmup3+timed32회. 288blocks, timed9216+warmup864+전후출력확인48=10128forward를 실행했다. batch1은32입력을각각1회, batch32는같은32입력을32회 반복했다. 24개모델의 state hash와 전후출력exact·입력불변,133개코드/입력/checkpoint hash,모든장부의유일성·크기·timing양수·메모리일관성을검사했다. 외부GPU compute는기존허용RustDesk외없었다.

## 직접 측정

'''+table(['방법','batch','wall median ms','p10–p90 ms','max allocated MiB','추가학습 params'],rows)+'''

wall은forward호출전부터CUDA완료까지이며모델로딩/전송/hash/guard/I/O는제외한다. 코드의CPUassert·normalization·gate·분위수계산은포함한다. CUDA stream event도별도공개하지만kernel-only라고부르지않는다. 메모리는한모델만GPU에둔상태의PyTorch allocated peak다. nvidia-smi의전체사용량·CUDAcontext·RustDesk메모리는포함하지않는다. 네source-seed사례에동일개수로관측한요청의pooled분포이며독립과제반복이나신뢰구간이아니다.

![동일 조건 추론 비용](inference_cost.png)

## 같은 round에서의 MAG/대조 시간비

'''+table(['batch','대조','MAG/대조 median','24개round비의min–max'],rrows)+'''

1보다크면MAG가느리다.같은source/seed/batch/round의요청median을짝지었다. 범위는24개고정round의관측범위이며모집단신뢰구간이아니다.추가파라미터가적다는사실을자동으로속도이득으로바꾸지않는다.선택step0도기존adapter를실행해측정했고모듈생략을적용하지않았다.

## 논문에 반영할 범위

학습속도·총적응비용우위는여전히별도문제다.B0의기존LoRA294912와그학습비용을제외한추가파라미터만표시했다.이동일장치측정으로추론시간/메모리를정확도와나란히보고할수있지만평균화된accuracy-cost점수로새우승자를선택하지않는다.PETSA는offline공개cell이식이며Time-PEFT·정식온라인PETSA전체성능우위를증명하지않는다.신규성·독립source/seed일반화도이측정으로해결되지않는다.

모든원시timing은TIMINGS.csv.gz/JSONL,48개source-seed-arm-batch값은BY_CASE.csv,240개짝비는PAIRED_ROUND_RATIOS.csv다.기존실험및원고v2는보존한다.신규후속학습/구조0.아티팩트검산후scoped commit/push한다.
'''
 observation='\n## 이번 측정의 핵심 해석\n\n동일round의MAG/B0시간비중앙값은batch1 1.059, batch32 1.064다. MAG/PLAIN은각각1.027/1.028, MAG/PETSA cell은1.007/1.015다. 적은추가파라미터가추론속도우위로이어지지는않았다. 개별round의비가1아래/위를오가므로작은차이를일반적인속도서열로확정하지않는다. 이값은현재PyTorch구현의dispatch·assert·GPU완료대기를포함하며연산별원인profile을수행한것은아니다. GPUclock을고정하지않았고허용된RustDesk가실행중이므로일반적인hardware벤치마크기록으로확대하지않는다.\n'
 for baseline,expected_ratio in [('B0',1.064),('PLAIN',1.028),('PETSA_XY_OFFLINE',1.015)]:
  value=ratio_summary[(ratio_summary.batch==32)&(ratio_summary.baseline==baseline)].wall_ratio_median.item();assert round(value,3)==expected_ratio
 report+=observation
 (OUT/'REPORT.md').write_text(report)
 (OUT/'FINAL_DECISION.md').write_text('# 추론 자원 판단\n\n동일조건추론측정과검산완료. 신규학습/optimizer/backward0회, E추론0회. MAG의추론속도우위는확보되지않았다. 기존예측이득과추가파라미터수를삭제하지않고실측추론overhead를함께보고한다. 학습시간·총적응비용·신규성·독립source일반화는이측정의완료범위가아니다.\n\n사용자의“현재학습은하지말아줘”지시를유지하며추가fit이나학습성대조를시작하지않는다.\n')
 appendix=ROOT/'papers/persistence_adaptation/inference_cost_20260919/ADDENDUM_KO.md'
 appendix.parent.mkdir(parents=True,exist_ok=True)
 appendix.write_text('# 기존 원고 v2의 추론 자원 보충\n\n원고 PDF의 과거 학습 timer 한계는 유지된다. 이후 시행한 같은 장치·같은 입력의 추론 측정으로 추론 비용에 한정된 직접 비교를 추가한다. 기존 성능이나 주 비교 기준을 바꾸지 않았다.\n\n'+table(['방법','batch','wall median ms','p10–p90 ms','max allocated MiB','추가학습 params'],rows)+'\n\n[전체 측정 보고서](../../../results/'+NAME+'/REPORT.md) · [원시 검산](../../../results/'+NAME+'/AUDIT.json) · [그림 PDF](../../../results/'+NAME+'/inference_cost.pdf).\n\n표의 요청 반복은 새로운seed·과제표본이 아니며분포구간은신뢰구간이아니다.입력은기존TRAIN의진폭4조건에한정되고raw/future정답채점은없다.학습속도/전체적응비용/신규성/논문PASS를이표로주장하지않는다.\n')
 with appendix.open('a') as f:f.write(observation)
 write(OUT/'AUDIT.json',dict(status='VERIFIED_INFERENCE_ONLY',timed_forwards=9216,warmup_forwards=864,parity_forwards=48,total_forwards=10128,blocks=288,model_states=24,case_rows=48,paired_ratios=240,optimizer_updates=0,backward_calls=0,E_forwards=0,labels_loaded=False,unapproved_gpu_samples=unapproved,sealed_hashes=len(seal['hashes']),code_sha256=sha(EXP/'audit_report.py'),input_timing_sha256=sha(OUT/'TIMINGS.jsonl'),paper_goal_complete=False,artifact_hashes={str(p.relative_to(ROOT)):sha(p) for p in list(OUT.iterdir())+[appendix] if p.is_file() and p.name not in ['AUDIT.json','status.json']}))
 write(OUT/'status.json',dict(execution='COMPLETE_VERIFIED',optimizer_updates=0,total_forwards=10128))
 print(summary.to_string(index=False));print(ratio_summary.to_string(index=False));print('INDEPENDENT_AUDIT_COMPLETE')
if __name__=='__main__':main()
