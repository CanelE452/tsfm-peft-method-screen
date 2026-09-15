"""Explicit user-authorized resumption of GPU-unstarted Q/C tracks; preserve originals."""
import argparse,copy,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES={'q':'query_budget_numeric_v2_resume_20260915','c':'channel_basis_pilot_resume_20260915','all':'priority12_resume_20260915'}
OLD={'q':'query_budget_numeric_v2_20260915','c':'channel_basis_pilot_20260915','all':'priority12_20260915'}
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def path(k):return ROOT/'results'/NAMES[k]
def prepare():
    assert all(not path(k).exists() for k in NAMES),'No repeated resumption'
    history={str(p.relative_to(ROOT)):sha(p) for folder in ['results','research'] for p in (ROOT/folder).rglob('*') if p.is_file()}
    path('all').mkdir()
    authorization=dict(user_instruction='계속해줘',resource_exception='RustDesk만 예외 허용',allowed_executable='/usr/share/rustdesk/rustdesk',allowed_process_policy='Only exact RustDesk path; every other compute process remains blocking',original_commit='a5cdc6817dc0f8bb9a0b8da8d3cdedb5ead3f8a0',requested_at=time.time(),reason='Both original tracks had zero real-model GPU updates and zero fits; resume unstarted work without changing training/numerical settings',new_fit_allowance=False,new_wait_budget_per_track_seconds=600,wait_history_retained=True)
    save(path('all')/'authorization.json',authorization);save(path('all')/'historical_hashes.json',history)
    for k in ['q','c']:
        old=ROOT/'results'/OLD[k];s=read(old/'status.json');assert s['status']=='BLOCKED_GPU_BUSY'
        keys=['numeric_updates','A_updates','B_updates','B_fit_attempts'] if k=='q' else ['check_updates','fit_attempts','training_updates']
        assert all(s[v]==0 for v in keys)
        out=path(k);out.mkdir();cache=ROOT/'.cache'/NAMES[k];assert not cache.exists();cache.mkdir()
        contract=read(old/'contract.json');sourcekey='sources' if k=='q' else 'source_hashes'
        for p,h in contract[sourcekey].items():assert sha(ROOT/p)==h
        for d in contract['data'].values():
            staged={}
            for p,h in d['staged'].items():
                assert sha(ROOT/p)==h;target=cache/Path(p).name;os.link(ROOT/p,target);staged[str(target.relative_to(ROOT))]=h
            d['staged']=staged
        for n in ['PROTOCOL.md','cpu_tests.json','schedules.json','fit_manifest.json']+(['profile_order.json'] if k=='q' else ['parameter_counts.csv','parameter_inventory.json','initial_cpu_parity.json','master_initial_hashes.json','constructor_counts.csv','leakage_checks.json','cpu_update_ledger.json']):shutil.copyfile(old/n,out/n)
        contract['historical_hashes']=history;contract[sourcekey][str(Path(__file__).relative_to(ROOT))]=sha(__file__)
        contract['resumption']=dict(authorization=authorization,prior_run=OLD[k],prior_contract_sha256=sha(old/'contract.json'),prior_status_sha256=sha(old/'status.json'),cpu_tests_reused_without_rerun=True,real_gpu_budget_carried_forward=True)
        contract['config']['run_id']=NAMES[k]
        save(out/'contract.json',contract)
        s={key:value for key,value in s.items() if key not in ['error','traceback']};s['status']='PREPARED';save(out/'status.json',s)
        (out/'RESUMPTION.md').write_text('# 사용자 승인 재개\n\n이전 GPU 대기 종료 기록은 불변으로 남긴다. 사용자 계속 요청 및 RustDesk만 예외 허용에 따라 실모델 GPU 업데이트0/본학습0인 작업을 재개한다. 수치 정책, 데이터, 모델, LR, 표본 순서, fit 상한은 그대로다. CPU 검사 증거를 재사용하며 toy update30회를 다시 실행하지 않는다. 새 GPU 대기 한도는 track별600초다. RustDesk의 사용량을 로그에 보존하며 다른 compute는 차단한다. 원격 화면 부하가 있는 timing이라는 한계를 보고한다.\n')
    print('PREPARED RESUMPTION; original files',len(history),'preserved',flush=True)

def allow_desktop(r):
    for a in r['apps']:a['allowed_desktop']=a['name']=='/usr/share/rustdesk/rustdesk'
    r['busy']=any(not a['own'] and not a['allowed_desktop'] for a in r['apps']) or r['free_mib']<1024
    return r

def bind_q():
    import run_query_numeric_v2 as q
    import finalize_query_numeric_v2 as f
    q.OUT=f.OUT=path('q');q.CACHE=f.CACHE=ROOT/'.cache'/NAMES['q'];q.RUN=f.RUN=NAMES['q']
    original=q.Watch.read_gpu
    q.Watch.read_gpu=lambda self:allow_desktop(original(self))
    q.configure_backend();return q,f

def bind_c():
    import run_channel_basis_pilot as c
    import finalize_channel_basis_pilot as f
    c.OUT=f.OUT=path('c');c.CACHE=ROOT/'.cache'/NAMES['c'];c.COMBINED=path('all')
    original_read=c.read
    c.read=lambda p:original_read(path('q')/'status.json' if Path(p)==ROOT/'results'/OLD['q']/'status.json' else p)
    original=c.Watch.sample;c.Watch.sample=lambda self:allow_desktop(original(self))
    c.configure();return c,f

def execute(k):
    auth=read(path('all')/'authorization.json');assert auth['resource_exception']=='RustDesk만 예외 허용'
    assert read(path(k)/'contract.json')['resumption']['authorization']==auth
    module,finalizer=bind_q() if k=='q' else bind_c()
    save(path(k)/'execution_revision.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),driver_sha256=sha(__file__),authorization_sha256=sha(path('all')/'authorization.json')))
    phases=[]
    if k=='q':
        module.numeric();phases.append(dict(phase='numeric',status=read(path(k)/'status.json')['status']))
        if read(path(k)/'status.json')['status']=='NUMERICS_BOUNDED':module.profile();phases.append(dict(phase='profile',status=read(path(k)/'status.json')['status']))
        if read(path(k)/'status.json')['status']=='RESOURCE_SIGNAL':module.train_evaluate();phases.append(dict(phase='train_evaluate',status=read(path(k)/'status.json')['status']))
    else:
        module.preflight();phases.append(dict(phase='preflight',status=read(path(k)/'status.json')['status']))
        if read(path(k)/'status.json')['status']=='PREFLIGHT_PASSED':module.train_evaluate();phases.append(dict(phase='train_evaluate',status=read(path(k)/'status.json')['status']))
    finalizer.finalize();phases.append(dict(phase='finalize',exit_code=0));save(path(k)/'phases.json',phases)
    print(k.upper(),'TRACK FINISHED',read(path(k)/'status.json'),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','q','c','verify-q','verify-c']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage in ['q','c']:execute(a.stage)
    else:
        m,f=bind_q() if a.stage=='verify-q' else bind_c();f.finalize()
if __name__=='__main__':main()
