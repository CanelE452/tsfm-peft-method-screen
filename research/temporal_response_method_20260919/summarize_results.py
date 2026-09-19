"""Post-score manuscript evidence. Never loads labels or creates predictions.

Requires the full sealed prediction marker and verified scorer outputs. Adds no
optimizer updates, model inference, selection, hyperparameter search or criteria.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/temporal_response_peft_20260919'
DEST=Path(__file__).resolve().parent/'result_evidence'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(df):
    head='| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'
    return head+'\n'.join('| '+' | '.join(f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def main():
    verified=json.loads((OUT/'VERIFICATION.json').read_text());assert verified['status']=='VERIFIED'
    marker=json.loads((OUT/'ALL_PREDICTIONS_SAVED.json').read_text());assert marker['manifest_sha256']==digest(OUT/'PREDICTIONS.json')
    effects=pd.read_csv(OUT/'EFFECTS.csv');raw=pd.read_csv(OUT/'RAW_SCORES.csv');seeds=pd.read_csv(OUT/'SEED_EFFECTS.csv');costs=pd.read_csv(OUT/'RESOURCES.csv')
    ledger=pd.read_json(OUT/'UPDATE_LEDGER.jsonl',lines=True);assert len(ledger)==16384 and not ledger.duplicated(['fit','step']).any()
    assert set(ledger.groupby('fit').size())=={1024} and len(ledger.fit.unique())==16
    for metric in ['supervised','regularizer','loss','gradient_norm']:assert np.isfinite(ledger[metric]).all()
    np.testing.assert_allclose(ledger.loss,ledger.supervised+ledger.regularizer,rtol=2e-7,atol=2e-7)
    DEST.mkdir(parents=True,exist_ok=True)
    phases=ledger.assign(phase=np.where(ledger.step<=256,'first256',np.where(ledger.step>768,'last256','middle512')))
    loss=phases.groupby(['source','arm','seed','phase'],as_index=False)[['supervised','regularizer','gradient_norm']].mean();loss.to_csv(DEST/'TRAIN_LOSS_COMPONENTS.csv',index=False)
    change=[]
    import torch
    for p in (OUT/'fits').glob('*/receipt.json'):
        r=json.loads(p.read_text());items={c['step']:c for c in r['checkpoints']}
        states=[]
        for step in [0,1024]:
            q=ROOT/items[step]['checkpoint'];assert digest(q)==items[step]['sha256'];states.append(torch.load(q,map_location='cpu',weights_only=True))
        for name in states[0]:
            a,b=states[0][name].double(),states[1][name].double();delta=(b-a).norm().item()
            change.append(dict(source=r['source'],arm=r['arm'],seed=r['seed'],parameter=name,initial_norm=a.norm().item(),final_norm=b.norm().item(),change_norm=delta))
    pd.DataFrame(change).to_csv(DEST/'PARAMETER_MOVEMENT.csv',index=False)
    baseline=['PLAIN','ANCHOR','SHUFFLE','IDEAL','B0','C3','MAG_ONLY'];panels=['electricity','electricity_transfer','ettm1'];conditions=['REFERENCE','FAULT','SHIFT8']
    fig,axes=plt.subplots(3,3,figsize=(14,10),sharey=True)
    for i,panel in enumerate(panels):
        for j,cond in enumerate(conditions):
            ax=axes[i,j];q=effects[(effects.panel==panel)&(effects.stage=='selected')&(effects.condition==cond)].set_index('baseline').loc[baseline]
            lo=q.ci_low.copy();hi=q.ci_high.copy()
            if panel=='electricity_transfer' and cond=='SHIFT8':
                lo.loc[baseline[:4]]=q.loc[baseline[:4],'bonferroni4_low'];hi.loc[baseline[:4]]=q.loc[baseline[:4],'bonferroni4_high']
            for k,b in enumerate(baseline):
                x=q.loc[b,'gain_pct'];color='#287b60' if x>=0 else '#ad4848'
                ax.plot([lo[b],hi[b]],[k,k],color=color,linewidth=2);ax.scatter([x],[k],color=color,s=28,zorder=3)
            ax.axvline(0,color='#333',linewidth=.8);ax.set_yticks(range(7),baseline);ax.set_ylim(6.7,-.7);ax.grid(axis='x',alpha=.2)
            ax.set_title(f'{panel} / {cond}',fontsize=10);ax.set_xlabel('TRP gain over comparator (%)')
    fig.suptitle('Prespecified TRP comparisons; selected checkpoint; all gains and harms retained',fontsize=13)
    fig.text(.5,.005,'Two previously used seeds, reused development periods. Primary four transfer SHIFT8 intervals: Bonferroni4; others: pointwise 95%.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.025,1,.96));fig.savefig(DEST/'effects.pdf');fig.savefig(DEST/'effects.png',dpi=160);plt.close(fig)
    core=costs[~costs.reused].groupby(['source','arm'],as_index=False)[['train_seconds','validation_seconds','peak_allocated_GiB']].mean();core.to_csv(DEST/'COST_SUMMARY.csv',index=False)
    primary=effects[effects.primary_family];primary_seeds=seeds[(seeds.panel=='electricity_transfer')&(seeds.condition=='SHIFT8')&(seeds.stage=='selected')&seeds.baseline.isin(baseline[:4])]
    lines=['# TRP 방법론 근거 추가 검토','',f"봉인된 학습·평가 완료 여부는 원 REPORT.md를 따른다. 고정 투자 신호={verified['four_comparisons_meet_predeclared_signal']}. 이 자료의 새 학습·모델 추론·예측 채점은0회다.",'','## 주 비교와 seed','',table(primary[['baseline','TRP_nmae','baseline_nmae','gain_pct','bonferroni4_low','bonferroni4_high']]),'',table(primary_seeds[['baseline','seed','TRP_nmae','baseline_nmae','gain_pct']]),'','## 강한 기존 방법과의 관계','']
    context=effects[(effects.stage=='selected')&(effects.condition.isin(['REFERENCE','FAULT','SHIFT8']))&effects.baseline.isin(['B0','C3','MAG_ONLY'])]
    lines+=[table(context[['panel','condition','baseline','gain_pct','ci_low','ci_high']]),'','C3/MAG는 기존 학습률 정책을 사용한 맥락 비교다. 이를 이번 동일 loss 대조군처럼 쓰지 않으며, 후보가 단순 MAG보다 불리하면 네 새 대조군의 일부 이득만으로 실용적 우위를 주장하지 않는다.','', '## 추가 보정 억제 여부','', 'TRAIN_LOSS_COMPONENTS.csv는 실제 training ledger의 초반/중반/후반 supervised·regularizer·gradient norm이다. PARAMETER_MOVEMENT.csv는 checkpoint0→1024의 행렬 이동량이며 함수 보정량이나 성능 인과 기여율과 다르다. 이 수치만으로 특정 gradient 경로가 최종 성능의 원인이라고 판단하지 않는다.','', '## 비용·한계','',table(core),'','아직 독립 새 source·추가 seed·정식 PEFT 선행 비교는 없다. 최적 λ/LR을 탐색하지 않은 고정 개발 파일럿이므로 실패를 모든 가능한 반응 보존 방법의 반증으로 확대하지 않는다. 반대로 탐색하지 않았다는 이유만으로 현 후보를 성공으로 간주하지도 않는다.']
    (DEST/'EVIDENCE_REVIEW_KO.md').write_text('\n'.join(lines)+'\n')
    source_paths=['VERIFICATION.json','ALL_PREDICTIONS_SAVED.json','PREDICTIONS.json','EFFECTS.csv','RAW_SCORES.csv','SEED_EFFECTS.csv','RESOURCES.csv','UPDATE_LEDGER.jsonl']
    manifest={'input_sha256':{str((OUT/n).relative_to(ROOT)):digest(OUT/n) for n in source_paths},'new_fits':0,'new_inference':0,'new_E_scoring':0,'existing_ledger_rows_verified':len(ledger),'parameter_movement_rows':len(change),'primary_comparisons':len(primary),'code_sha256':digest(Path(__file__))}
    (DEST/'AUDIT.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
