import csv,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];RUN='building_peft_topic_decision_20260916';OUT=ROOT/'results'/RUN;CACHE=ROOT/'.cache'/RUN;EXP=ROOT/'experiments'/RUN
T=ROOT/'results/building_transfer_subspace_v1_20260915';C=ROOT/'results/building_coldstart_coverage_v1_20260915'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def csvread(p):return list(csv.DictReader(Path(p).open()))
def csvwrite(p,rows):
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
    if (OUT/'audit.json').exists():raise RuntimeError('Audit already exists; preserve it')
    previous={str(p.relative_to(ROOT)):sha(p) for d in ['results','research'] for p in (ROOT/d).rglob('*') if p.is_file() and OUT not in p.parents}
    save(OUT/'historical_hashes.json',previous)
    baseline='6ceec43d11747505bc1d5ddaa70de435c65dd370';head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();diff=subprocess.check_output(['git','diff',baseline,'--stat'],cwd=ROOT,text=True)
    held=read(C/'building_split.json')['heldout'];old_eps=read(C/'episodes.json');transfer=read(T/'episodes.json')
    assert not set(held)&set(read(T/'split.json')['source']+read(T/'split.json')['tune']+read(T/'split.json')['dev'])
    manifests=read(C/'prediction_manifest.json');assert not any(r['building_id'] in held for r in manifests)
    oldfits=read(C/'fit_attempts.json');assert not any(f['building_id'] in held for f in oldfits)
    trscores=csvread(T/'scores.csv');assert not any(r['building'] in held for r in trscores)
    # Search runtime outputs for any later use of the locked IDs; metadata-only names are audited separately.
    suspect=[];allow_meta={'eligibility.csv','episodes.json','episode_manifest.csv','data_manifest.json','prepare_seal.json','building_split.json','split.json','building_groups.json','historical_hashes.json','publication_audit.json'}
    for p in (ROOT/'results').rglob('*'):
        if not p.is_file() or OUT in p.parents or p.suffix not in ['.json','.jsonl','.csv'] or p.name in allow_meta:continue
        if any(b in p.read_text(errors='replace') for b in held):suspect.append(str(p.relative_to(ROOT)))
    cache_hits=[]
    for p in (ROOT/'.cache').rglob('*'):
        if p.is_file() and p.suffix in ['.npz','.pt'] and any(b in p.name for b in held):cache_hits.append(str(p.relative_to(ROOT)))
    assert not suspect and not cache_hits,('LOCKED_DATA_EXPOSED',suspect,cache_hits)
    discovery=[];locked=[]
    for e in transfer:
        if e['role']=='tune':discovery.append(dict(id=e['id'],building=e['building'],role='DISCOVERY',days=e['days'],origin=e['origin'],history=e['history'],target=e['target'],provenance='previous transfer tune4; already exposed'))
    for i,bid in enumerate(sorted(held)):
        day='Wednesday' if i%2==0 else 'Saturday'
        ee=[e for e in old_eps if e['building_id']==bid and e['forecast_type']==day];assert len(ee)==2
        for e in ee:locked.append(dict(id=e['id'],building=bid,physical_group=e['physical_group'],role='LOCKED',days=e['history_days'],origin=e['origin'],history={'path':e['history_file'],'sha256':e['history_sha256']},target={'path':e['target_file'],'sha256':e['target_sha256']},provenance='original coverage heldout6; canonical ID index parity chooses weekday'))
    assert len(discovery)==8 and len(locked)==12
    for e in discovery+locked:
        assert sha(ROOT/e['history']['path'])==e['history']['sha256']
        # Target bytes may be hashed for integrity, never decoded here for LOCKED.
        assert sha(ROOT/e['target']['path'])==e['target']['sha256']
    save(OUT/'episodes.json',discovery+locked)
    evidence=[];hashes={};maxidentity=0.
    def summarize(source,episode,building,days,arm,q,raw,y,h,path):
        nonlocal maxidentity
        for kind,pred in [('sorted',q),('raw',raw)]:
            err=np.asarray(pred[10],dtype=np.float64)-y;b=float(err.mean());mse=float(np.mean(err**2));shape=float(np.mean((err-b)**2));delta=abs(mse-b*b-shape);maxidentity=max(maxidentity,delta)
            evidence.append(dict(source=source,status='DISCOVERY_ONLY',episode=episode,building=building,H=days,arm=arm,output=kind,raw_RMSE=float(np.sqrt(mse)),scaled_RMSE=float(np.sqrt(mse)/max(float(h.std()),1e-6)),bias=b,bias_squared=b*b,demeaned_MSE=shape,demeaned_RMSE=float(np.sqrt(shape)),MSE_identity_error=delta,cache=path))
    for row in trscores:
        if row['arm'] not in ['F0','LOCAL','LOCAL_FIXED120','AFFINE'] and not row['arm'].startswith('local_'):continue
        p=ROOT/row['replay_file'];assert sha(p)==row['replay_sha256'];hashes[row['replay_file']]=row['replay_sha256'];z=np.load(p);q=z['q'];raw=q.copy()
        # Locate original raw forecast at the matching checkpoint; AFFINE is transformed consistently.
        rec=read(T/'fits'/f"{row['fit']}.json");d=rec['checkpoints'][row['step']];assert sha(ROOT/d['prediction_file'])==d['prediction_sha256'];pred=np.load(ROOT/d['prediction_file']);raw=pred['raw'][0]
        if row['arm']=='AFFINE':raw=float(row['a'])*raw+float(row['b'])
        summarize('transfer',row['episode'],row['building'],int(row['days']),row['arm'],q,raw,z['y'],z['h'],row['replay_file'])
    for row in manifests:
        if row['method'] not in ['F0','STANDARD','AFFINE']:continue
        p=ROOT/row['path'];assert sha(p)==row['sha256'];hashes[row['path']]=row['sha256'];z=np.load(p)
        summarize('coverage',row['episode'],row['building_id'],row['history_days'],row['method'],z['q'],z['raw'],z['target'],z['history'],row['path'])
    csvwrite(OUT/'evidence_episode_decomposition.csv',evidence)
    curves=[]
    for p in (T/'fits').glob('local_*.json'):
        f=read(p);c0=f['checkpoints']['0'];assert sha(ROOT/c0['prediction_file'])==c0['prediction_sha256'];z0=np.load(ROOT/c0['prediction_file'])
        for k,d in f['checkpoints'].items():
            assert sha(ROOT/d['prediction_file'])==d['prediction_sha256'];assert sha(ROOT/d['checkpoint_file'])==d['checkpoint_sha256'];z=np.load(ROOT/d['prediction_file'])
            hashes[d['prediction_file']]=d['prediction_sha256'];hashes[d['checkpoint_file']]=d['checkpoint_sha256']
            for kind in ['q','raw']:
                change=z[kind][0,10]-z0[kind][0,10];curves.append(dict(fit=f['id'],rank=f['rank'],seed=61600,step=int(k),output=kind,mean_change=float(change.mean()),demeaned_change_RMS=float(np.sqrt(np.mean((change-change.mean())**2))),total_change_RMS=float(np.sqrt(np.mean(change**2))),last_train_loss=f['losses'][int(k)-1]['loss'] if int(k)>0 else None))
    csvwrite(OUT/'evidence_training_changes.csv',curves);save(OUT/'evidence_cache_hashes.json',hashes)
    lines=['# 기존 증거 재검토','','[확인] 기존 두 건물 실험의 local 예측·체크포인트를 hash 확인 후 읽었다. 모든 과거 성능 표본은 DISCOVERY다. 새 GPU 학습은 이 재검토에0 fits다. 아래 raw는 정렬 여부와 별개인 원래 부하 단위 RMSE이며 출력 정렬 여부를 별도 표기한다.','','| 실험/비교 | 표본과 직접 비교 범위 | raw/scaled 오차 | 편향/평균 제거 분해 | 학습량에 따른 변화 |','| --- | --- | --- | --- | --- |', '| transfer F0/LOCAL/fixed120/AFFINE | 같은 episode끼리만 paired; rank1·seed61600 선택 epoch1와 fixed120 | episode CSV 전체 공개 | 각 출력의 b²와 centered MSE 공개 | 같은 trajectory checkpoint raw/q 둘 다 공개 |','| transfer tune rank1/rank8 | 동일 건물·origin·H에서 rank별; 이번 신규seed61680과 같다고 하지 않음 | rank/step별 공개 | 사후 정답 분해 | 0/n/4n/16n/120 checkpoint |','| coverage F0/STANDARD/AFFINE | 기존 rank8·선택recipe; transfer rank1과 직접 대조 금지 | 원점수와 분해 공개 | 같은 시점 안의 비교만 | 최종120 및 별도 recipe 곡선은 기존 장부 참조 |','', '[확인] 상세: [episode 분해](evidence_episode_decomposition.csv), [학습량별 raw/정렬 변화](evidence_training_changes.csv), [직접 선택-vs120 기존 표](../building_transfer_subspace_v1_20260915/budget_control_comparison.csv).','',f'[확인] MSE=b²+centered MSE 독립 분해의 최대 절대차 {maxidentity:.3g}. RMSE의 선형 합을 사용하지 않는다. 평균 잔차는 미래 정답으로 계산한 진단이며 배포 입력이나 oracle 선택 성능이 아니다.','','| 경쟁 설명 | 지지 관찰 | 반례 | 없는 근거 |','| --- | --- | --- | --- |','| 적응량 선택이 건물별로 달라짐 | 기존 tune와 dev의 fixed120 순위 역전 | fixed120도 일부 건물에서 좋음 | target 미래 없이 신뢰할 선택 규칙 |','| 레벨과 형상 보정의 혼동 | affine 보정으로 일부 내부 적응 손해 회복 | affine 자체가 raw/일부 건물에서 악화 | 두 잔차 항을 분리한 변경의 직접 비교 |','| 단순 local LoRA/보정만으로 충분 | 기존 fixed120이 최저 dev macro | 짧은 H3에서는 악화 | 새 변경이 가장 가까운 단순 대조를 넘는 잠금 평가 |','', '[설계] 새 STD16 fits는 동일 origin이지만 seed61680·LR2개·ZERO 선택 포함으로 기존 실행과 다르다. 동일실험을 이름만 바꿔 반복하지 않는다. 이 새 seed의 결과를 받은 후 한 후보의 필요성과 반례를 명세한다.','']
    (OUT/'EVIDENCE.md').write_text('\n'.join(lines))
    record=dict(at=time.time(),baseline=baseline,current=head,diff=diff,branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),initial_dirty=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),locked_exposure_status='NO_EVALUATION_FOUND',locked=held,suspect_runtime_records=suspect,suspect_cache_files=cache_hits,locked_values_decoded=False,discovery_buildings=sorted({e['building'] for e in discovery}),cache_files_verified=len(hashes),old_files=len(previous),no_previous_training_repeated=True)
    save(OUT/'audit.json',record)
    (OUT/'AUDIT.md').write_text(f"# 버전·노출·중복 감사\n\n[확인] HEAD `{head}`; 기준 commit과 tracked diff 없음: {not bool(diff)}. 원래 결과 {len(previous)}개 hash를 보존했다. 현재 신규 run의 파일만 추가 중이다.\n\n[확인] 기존 heldout6은 이전 fit/예측 원장과 이후 source/tune/dev에서 사용되지 않았고, runtime 결과 및 cache의 예측/weight 파일명에도 노출 증거가 없다. Manifest·eligibility·해시에서의 등장과 성능 노출을 구분했다. 이 감사에서 LOCKED target 배열을 decode하지 않았다.\n\n[확인] 현재 동일 실험 worker 없음, GPU RTX3080 free9031MiB, compute는 기존 승인 RustDesk272MiB만 있었다.\n\n[설계] 이 파일의 유한한120-fit 계약을 먼저 이행한다. 사용자 상위 목표의 이후 후보 탐색은 이번 판정·heldout을 되돌려 PASS로 바꾸는 권한이 아니다. 이번 run 종료 후 별도 근거와 새 평가 보호를 갖춘 작업으로만 검토할 수 있다.\n\n[확인] 상세 분할/감사: [audit.json](audit.json), [episodes](episodes.json).\n")
    print(json.dumps(record,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
