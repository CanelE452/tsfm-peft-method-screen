"""Fixed subprocess supervisor: one attempt per stage, bounded resources, raw evidence."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from . import NAME,BASE_COMMIT
from .util import require,now,write_json,read_json,file_hash,redact,git,verify_manifest
from .data_audit import audit,SOURCE
from .ledger import summarize


def available_gib():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):return int(line.split()[1])*1024/2**30
    except OSError:pass
    return None


def rss_gib(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])*1024/2**30
    except OSError:pass
    return 0.


def run_child(cmd,log: Path,cfg,env,global_deadline):
    free=available_gib()
    if free is not None and free<cfg['min_available_ram_gib']:
        write_json(log.with_suffix('.process.json'),{'status':'NOT_RUN_RESOURCE','available_gib':free,'command':cmd})
        return {'returncode':125,'status':'NOT_RUN_RESOURCE','command':cmd}
    start=time.monotonic(); peak=0.; reason=None
    timeout=min(cfg['per_process_timeout_seconds'],max(0,global_deadline-start))
    if timeout<=0:return {'returncode':124,'status':'NOT_RUN_TOTAL_TIMEOUT','command':cmd}
    with log.open('w',encoding='utf-8') as f:
        process=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True,text=True)
        while process.poll() is None:
            peak=max(peak,rss_gib(process.pid))
            if peak>cfg['max_process_rss_gib']:
                reason='RESOURCE_RSS_STOP'
            elif time.monotonic()-start>timeout:
                reason='TIMEOUT_STOP'
            if reason:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                break
            time.sleep(.2)
    result={'returncode':process.returncode,'status':reason or ('EXIT_OK' if process.returncode==0 else 'EXIT_NONZERO'),
            'elapsed_seconds':time.monotonic()-start,'peak_child_rss_gib':peak,'command':cmd}
    # Never expose a possible token embedded by a library error message.
    log.write_text(redact(log.read_text(encoding='utf-8',errors='replace')),encoding='utf-8')
    write_json(log.with_suffix('.process.json'),result)
    return result


def main():
    p=argparse.ArgumentParser(description='Fixed v5 readiness execution. No source edits, installation, or research fitting.')
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--backend',choices=('chronos','tiny'),default='chronos')
    p.add_argument('--local-test-root',type=Path,help='Only for the author local test; never an automatic backend fallback.')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    verify_manifest(root)
    cfg=read_json(root/'RUN_CONFIG.json')
    repo=a.repo.resolve()
    if a.local_test_root:
        require(a.backend=='tiny','Local test root can only use the labelled test double')
        public=a.local_test_root.resolve()/a.run_id/'public'
        private=a.local_test_root.resolve()/a.run_id/'private'
    else:
        require(a.backend=='chronos','Production execution cannot use a test double')
        public=repo/'results'/NAME/a.run_id
        private=repo/'runs'/NAME/a.run_id
    require(not public.exists() and not private.exists(),'Run already exists; automatic retry/overwrite prohibited')
    public.mkdir(parents=True);private.mkdir(parents=True)
    (private/'.gitignore').write_text('*\n',encoding='utf-8')
    env=os.environ.copy()
    env.update({'PYTHONPATH':str(root),'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':str(cfg['threads']),'MKL_NUM_THREADS':str(cfg['threads']),
                'CUDA_VISIBLE_DEVICES':''})
    start=time.monotonic();deadline=start+cfg['total_timeout_seconds']
    stages={}
    manifest={'started':now(),'package_sha256':file_hash(root/'PACKAGE_MANIFEST.json'),
              'backend_requested':a.backend,'config':cfg,'baseline_commit':BASE_COMMIT,'research_fits':0,
              'new_simulation_calls':0,'model_downloads_allowed':False,'retries':0,'stages':stages,
              'counting_scope':'Event counts are native/test-double lifecycle worker operations including failed attempts. Standalone formula/PEFT-parity and unittest operations are separately reported and never perform optimizer steps.'}
    if not a.local_test_root:
        manifest['head_at_start']=git(repo,'rev-parse','HEAD')
    write_json(public/'RUN_MANIFEST.json',manifest)
    # The fixed unit-test module uses no optimizer steps and no external resources.
    cmd=[sys.executable,'-m','unittest','transient_v5.selftest','-v']
    stages['unit_tests']=run_child(cmd,public/'unit_tests.log',cfg,env,deadline)
    write_json(private/'backend_requested.json',{'kind':a.backend})
    args=[sys.executable,'-m','transient_v5.worker','--mode','preflight','--backend-json',str(private/'backend_requested.json'),
          '--public',str(public/'preflight'),'--private',str(private/'preflight'),'--run-config',str(root/'RUN_CONFIG.json')]
    (public/'preflight').mkdir()
    stages['preflight']=run_child(args,public/'preflight.log',cfg,env,deadline)
    ready=stages['preflight']['returncode']==0 and stages['unit_tests']['returncode']==0
    resolved=public/'preflight'/'backend_resolved.json'
    arm_status={}
    for arm in cfg['arms']:
        if not ready:
            arm_status[arm]={'status':'NOT_RUN_PREFLIGHT_FAILED'}
            continue
        pub=public/arm;prv=private/arm
        cmd=[sys.executable,'-m','transient_v5.worker','--mode','arm','--backend-json',str(resolved),
             '--arm',arm,'--public',str(pub),'--private',str(prv),'--run-config',str(root/'RUN_CONFIG.json')]
        stages[arm]=run_child(cmd,public/(arm+'.log'),cfg,env,deadline)
        arm_status[arm]={'training_stage':stages[arm]}
        # Replay the actual saved checkpoint even if updates were classified inconclusive.
        if (prv/'checkpoint'/'metadata.json').exists():
            cmd=[sys.executable,'-m','transient_v5.worker','--mode','restore','--backend-json',str(resolved),
                 '--arm',arm,'--public',str(pub),'--private',str(prv),'--run-config',str(root/'RUN_CONFIG.json')]
            stages[arm+'_restore']=run_child(cmd,public/(arm+'_restore.log'),cfg,env,deadline)
            arm_status[arm]['restore_stage']=stages[arm+'_restore']
        else:
            arm_status[arm]['restore_stage']={'status':'NOT_RUN_NO_TRAINED_CHECKPOINT','returncode':126}
        if (pub/'arm_result.json').exists():arm_status[arm]['result']=read_json(pub/'arm_result.json')
        manifest['stages']=stages;write_json(public/'RUN_MANIFEST.json',manifest)
    # Existing data audit is independent from model installation/access.
    if a.local_test_root:
        data_result={'status':'NOT_RUN_LOCAL_TEST_DOUBLE','reason':'Actual BOPTEST CSVs are not present in the author runtime.'}
    else:
        try:
            for rel in ('fast_probe.csv','transient_response.csv','TAU_FAST.json'):
                full=repo/SOURCE/rel
                prior=subprocess.run(['git','-C',str(repo),'show',f'{BASE_COMMIT}:{SOURCE}/{rel}'],capture_output=True,check=True)
                import hashlib
                require(file_hash(full)==hashlib.sha256(prior.stdout).hexdigest(),f'Previously reviewed CSV changed: {rel}')
            data_result=audit(repo,public/'data_audit')
        except Exception as e:
            data_result={'status':'DATA_AUDIT_BLOCKED','error':redact(str(e)),'traceback':redact(traceback.format_exc())}
            write_json(public/'data_audit_FAILURE.json',data_result)
    counts={}
    for path in public.rglob('*events.jsonl'):
        for op,items in summarize(path).items():
            dest=counts.setdefault(op,{'START':0,'COMPLETED':0,'FAILED':0})
            for state,n in items.items():dest[state]+=n
    all_ok=ready and all(v.get('training_stage',{}).get('returncode')==0 and v.get('restore_stage',{}).get('returncode')==0 for v in arm_status.values())
    updates=counts.get('optimizer_step',{}).get('COMPLETED',0)
    require(updates<=cfg['max_optimizer_updates'],'Unexpected update budget overrun')
    verify_manifest(root)
    manifest.update({'finished':now(),'elapsed_seconds':time.monotonic()-start,'stages':stages,
                     'actual_counts_including_failed_attempts':counts,'optimizer_updates_completed':updates,
                     'source_manifest_preserved':True})
    write_json(public/'RUN_MANIFEST.json',manifest)
    status=('NATIVE_FIXTURE_LIFECYCLE_VERIFIED' if a.backend=='chronos' else 'TEST_DOUBLE_ONLY_VERIFIED') if all_ok else 'READINESS_NOT_CONFIRMED'
    result={'status':status,'backend':a.backend,'arm_status':arm_status,'data_status':data_result['status'],
            'optimizer_updates':updates,'science':{'difficulty_exists':'UNTESTED','PEFT_improvement':'UNTESTED',
                                                  'new_method_novelty':'UNRESOLVED','physical_target_and_horizon':'NOT_SELECTED'},
            'human_result_analysis':'Return the pushed commit to ChatGPT. CLI must not rewrite the code, choose a target, or tune.'}
    write_json(public/'SUMMARY.json',result)
    lines=['# 고정 실행 결과 — 기계 집계, 연구 결과 해석 아님','',f'- 상태: `{status}`',
           f'- 백엔드: `{a.backend}`',f'- 실제 완료 optimizer update: {updates}',
           f'- 기존 데이터 검산: `{data_result["status"]}`','',
           '이 결과는 제한된 fixture의 학습·저장·추론 복원 검사다. 설비 예측 성능이나 신규성의 증거가 아니다.',
           '실패·미확인 결과도 그대로 보고하며, CLI에서 코드/학습률/초기화를 바꾸지 않는다.','',
           '| 구성 | 실행 | 새 프로세스 복원 |','|---|---|---|']
    for arm,v in arm_status.items():
        lines.append(f'| {arm} | {v.get("training_stage",{}).get("status",v.get("status"))} | {v.get("restore_stage",{}).get("status","NOT_RUN")} |')
    (public/'REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # File index is an allowlist for later review/push, not an assertion of scientific success.
    file_index={str(p.relative_to(public)):file_hash(p) for p in public.rglob('*') if p.is_file()}
    write_json(public/'FILES_SHA256.json',file_index)
    print(json.dumps({'result_path':str(public),'status':status,'optimizer_updates':updates},ensure_ascii=False))
    return 0 if all_ok else 2

if __name__=='__main__':
    raise SystemExit(main())
