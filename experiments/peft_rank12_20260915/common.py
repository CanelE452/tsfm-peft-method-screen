"""Explicit run paths and shared guards. Historical execution modules are never rebound."""
import sys,json,hashlib,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from priority12.common import save,read,sha,digest,csvwrite,parameters,cpu_state,tensor_hash,frozen_hash,restore,preserve_rng,cleanup,ResourceError
from priority12.common import Watch as OriginalWatch
EXP=ROOT/'experiments/peft_rank12_20260915';RESEARCH=ROOT/'research/peft_rank12_20260915'
R1=ROOT/'results/query_budget_repair_v2_20260915';R2=ROOT/'results/channel_sharing_screen_v1_20260915'
C1=ROOT/'.cache/query_budget_repair_v2_20260915';C2=ROOT/'.cache/channel_sharing_screen_v1_20260915'
class Watch(OriginalWatch):
    def sample(self):
        r=super().sample()
        for a in r['apps']:a['allowed_desktop']=a['name']=='/usr/share/rustdesk/rustdesk'
        r['busy']=any(not a['own'] and not a['allowed_desktop'] for a in r['apps']) or r['free_mib']<1024
        return r

def source_hashes():
    paths=list(EXP.glob('*.py'))+list(EXP.glob('*.json'))+[RESEARCH/'PROTOCOL.md']+list((ROOT/'src').rglob('*.py'))+list((ROOT/'sources/timepeft_ea4e7e1').rglob('*.py'))
    paths += [ROOT/'scripts'/x for x in ['priority12/common.py','priority12/channel_model.py','run_forecast_query_checkpoint_diagnostic.py']]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
def verify_contract(out):
    c=read(out/'contract.json')
    for p,h in c['source_hashes'].items():assert sha(ROOT/p)==h,('SOURCE_CHANGED',p)
    for d in c['data'].values():
        for p,h in d['staged'].items():assert sha(ROOT/p)==h,('STAGED_DATA_CHANGED',p)
    return c

def audit_history():
    for p,h in read(RESEARCH/'historical_hashes.json').items():assert sha(ROOT/p)==h,('HISTORICAL_CHANGE',p)

def phase0():
    assert not (RESEARCH/'audit.json').exists(),'No repeated audit or run overwrite'
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    files=[p for folder in ['results','research'] for p in (ROOT/folder).rglob('*') if p.is_file() and RESEARCH not in p.parents]
    save(RESEARCH/'historical_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)})
    changed=subprocess.check_output(['git','diff','--name-status','9051d03..HEAD'],cwd=ROOT,text=True)
    (RESEARCH/'BASE_COMMIT_DIFF.txt').write_text(changed)
    save(RESEARCH/'audit.json',dict(head=head,base_commit='9051d03cebe23ae35d9689e7af576c63c4d5edd5',branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),dirty=dirty,historical_files=len(files),protocol_sha256=sha(RESEARCH/'PROTOCOL.md'),previous_Q=read(ROOT/'results/query_budget_numeric_v2_resume_20260915/status.json'),previous_C=read(ROOT/'results/channel_basis_pilot_resume_20260915/status.json'),rustdesk_exception='User explicitly approved RustDesk only in prior turn; persists',duplicate_study=False,reason='R1 new pairs and RMS contract; R2 L96 C32 new budget/factor arms, initialization, split .7/.8, BF16 and epoch recipe differ from completed L512 C64 FP32 study.',disk=subprocess.check_output(['df','-h',str(ROOT)],text=True)))
    (RESEARCH/'AUDIT.md').write_text(f'''# 기준 commit 이후 변경과 중복 검사\n\n시작 HEAD `{head}`, 기준 `9051d03`. [변경 파일](BASE_COMMIT_DIFF.txt). 이전 결과·연구 {len(files)}개를 hash로 봉인했다.\n\n기준 이후 Q 수치 진단180 updates/본학습0과 C24 fits/24,576 updates가 완료됐다. 기존 C는64채널·길이512·1024updates·FP32와 SHARED_WIDE/GROUP4/BASIS4 비교다. 이번 R2는32채널·길이96·동일 예산4종(FACTOR 포함)·다른 초기화·70/80분할·BF16·20epoch 상한으로 동일 실험이 아니다. R1도 검사 쌍과 RMS 기준이 다르다. 이미 본 부진 뒤의 사용자 지정 변경이므로 독립확증이나 사전 무노출 설계라고 하지 않는다.\n\n과거 실행기를 재호출하지 않고 독립 경로·실행 소스를 사용한다. R1 종료와 R2 진입은 분리하며 공통 환경 위험만 전파한다. 원시자료와 모델은 해시 확인 후 재사용하고, 기존 점수를 새 실험 점수로 복사하지 않는다.\n''')
if __name__=='__main__':phase0()
