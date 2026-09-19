"""Post-selection descriptive response audit from already saved predictions.

No model input, optimizer, checkpoint selection or model inference is changed.
Uses known synthetic evaluation offsets only for analysis, never for prediction.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/temporal_response_peft_20260919'
DEST=Path(__file__).resolve().parent/'result_evidence'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    marker=json.loads((OUT/'ALL_PREDICTIONS_SAVED.json').read_text());assert marker['manifest_sha256']==sha(OUT/'PREDICTIONS.json')
    assert json.loads((OUT/'VERIFICATION.json').read_text())['status']=='VERIFIED'
    from experiments.persistence_evidence_extension_20260918 import common as ext
    manifest=json.loads((OUT/'PREDICTIONS.json').read_text());rows=[];used={}
    def load(r):
        p=ROOT/r['path'];assert sha(p)==r['sha256'];used[r['path']]=r['sha256']
        return np.load(p,mmap_mode='r').reshape(10,2,128,-1,9,64)
    for key,r in manifest.items():
        if r['kind']!='standard':continue
        m=[q for q in manifest.values() if q['panel']==r['panel'] and q['kind']=='standard' and q['stage']==r['stage'] and q['seed']==r['seed'] and q['arm']=='B0'];assert len(m)==1
        p=load(r);base=load(m[0]);sigma=np.load(ext.data_path(r['panel'])/'E_DISCOVERY_inputs.npz')['sigma'][None,None,:,None]
        offset=np.load(ext.panel_path(r['panel'],'standard')/'E_DISCOVERY_offset.npy').reshape(10,2,128,-1,1)
        ref=ext.STATES.index('REFERENCE');reference=p[ref,...,4,:].astype(float);bref=base[ref,...,4,:].astype(float)
        for condition in ['REFERENCE','SHIFT4','SHIFT8','SHIFT_POINT']:
            i=ext.STATES.index(condition);pp=p[i,...,4,:].astype(float);bp=base[i,...,4,:].astype(float);response=pp-reference;teacher_response=bp-bref
            row=dict(panel=r['panel'],stage=r['stage'],arm=r['arm'],seed=r['seed'],condition=condition,checkpoint_step=r['step'],absolute_correction_nmae=float((np.abs(pp-bp)/sigma).mean()),response_distortion_from_B0=float((np.abs(response-teacher_response)/sigma).mean()))
            if condition in ['SHIFT4','SHIFT8']:
                a=offset[i];assert (a!=0).all()
                row.update(ideal_response_error=float((np.abs(response-a)/sigma).mean()),B0_ideal_response_error=float((np.abs(teacher_response-a)/sigma).mean()),mean_response_to_true_shift_ratio=float((response/a).mean()))
            # SHIFT_POINT has both an error and a shift: do not mislabel the fault response as pure shift equivariance error.
            rows.append(row)
    f=pd.DataFrame(rows);assert len(f)==384;DEST.mkdir(exist_ok=True,parents=True);f.to_csv(DEST/'RESPONSE_DIAGNOSTIC.csv',index=False)
    g=f.groupby(['panel','stage','arm','condition'],as_index=False)[['absolute_correction_nmae','response_distortion_from_B0','ideal_response_error','B0_ideal_response_error','mean_response_to_true_shift_ratio']].mean();g.to_csv(DEST/'RESPONSE_DIAGNOSTIC_MEAN.csv',index=False)
    b=f[f.arm=='B0'];assert (b.absolute_correction_nmae==0).all() and (b.response_distortion_from_B0==0).all()
    note='''# 저장 예측의 반응 진단\n\n선택·채점 완료 뒤 저장 예측만 사용했다. 추가 학습·모델 추론0회다. 과거·미래를 일관되게 이동시킨 합성 SHIFT4/SHIFT8에서만 이상적 반응 오차를 계산한다. 알려진 합성 offset은 진단 정답에만 사용했으며 모델·gate·정규화 입력이 아니다. SHIFT_POINT는 오류까지 섞여 있으므로 이상적 shift 오차로 해석하지 않는다.\n\n- absolute_correction_nmae: 같은 B0 예측에서 실제로 얼마나 달라졌는가.\n- response_distortion_from_B0: REFERENCE→SHIFT 예측 변화가 B0의 예측 변화와 얼마나 다른가.\n- ideal_response_error: 예측 변화가 실제 합성 수준 이동과 얼마나 다른가.\n- mean_response_to_true_shift_ratio: 예측 변화/합성 이동의 평균. 1이면 평균 크기 일치이지만 오차0이나 좋은 전체 예측을 보장하지 않는다.\n\n작은 반응 왜곡은 B0의 반응을 유지했다는 뜻이다. B0의 반응 자체가 부정확하면 이것만으로 예측 개선을 뜻하지 않는다. 이 진단은 알려진 합성 변형에 조건부이며 실제 센서 사건이나 모든 변화에서의 인과 설명으로 확대하지 않는다.\n'''
    (DEST/'RESPONSE_DIAGNOSTIC_KO.md').write_text(note)
    audit=dict(code_sha256=sha(Path(__file__)),predictions_sha256=used,rows=len(f),new_training=0,new_inference=0,checkpoint_selection_changed=False,metric_role='descriptive mechanism diagnostic, not primary success criterion',synthetic_offset_used_only_after_prediction=True)
    (DEST/'RESPONSE_DIAGNOSTIC_AUDIT.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
