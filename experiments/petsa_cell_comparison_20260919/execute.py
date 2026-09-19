"""Bounded execution and publication; no automatic retry or successor training."""
import os,subprocess,traceback
from .common import *

def publish():
    index=ROOT/'docs/RESULTS_INDEX.md';text=index.read_text();header='## PETSA 공개 보정 부품 비교 (2026-09-19)'
    if header not in text:
        state=read(OUT/'status.json')['execution']
        text+='\n'+header+'\n\n상태: '+state+'. 기존 MAG 고정, 최대8경로/8192+4updates, 기존 개발 E 재사용. [보고서](../results/'+NAME+'/REPORT.md) · [결정](../results/'+NAME+'/FINAL_DECISION.md) · [고정 계약](../experiments/'+NAME+'/PROTOCOL.md). 독립source/논문PASS로 주장하지 않음.\n'
        index.write_text(text)
    paths=[str(EXP.relative_to(ROOT)),str(OUT.relative_to(ROOT)),'docs/RESULTS_INDEX.md']
    subprocess.run(['git','diff','--check','--',*paths],cwd=ROOT,check=True)
    dirty=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    assert not dirty,'Unrelated staged changes require review; no automatic publication'
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
    state=read(OUT/'status.json')['execution']
    subprocess.run(['git','commit','-m','Record bounded PETSA-cell comparison: '+state,'--',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','HEAD:main'],cwd=ROOT,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True).split()[0]
    assert head==remote
    save(CACHE/'PUBLICATION.json',dict(commit=head,verified_remote=remote,at=time.time()))
    print('PUSH_VERIFIED',head,flush=True)

def main():
    require_authorization()
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        from .runner import run
        run()
    except BaseException as e:
        save(OUT/'EXECUTION_ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),at=time.time()))
        state=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
        state.update(execution='BLOCKED' if isinstance(e,ResourceError) else 'IMPLEMENTATION_OR_AUDIT_ERROR',error=str(e),automatic_retry=False);save(OUT/'status.json',state)
        if not (OUT/'REPORT.md').exists():
            (OUT/'REPORT.md').write_text('# 실행 중단 기록\n\n성능 실패와 구분한다. 자동 재시작·추가 학습 없음.\n\n```json\n'+json.dumps(state,ensure_ascii=False,indent=2)+'\n```\n\n완료receipt와UPDATE_LEDGER를 보존했다. 미실행 횟수는 최대8에서 completed_fits를 뺀 값이다. 오류 원문은 EXECUTION_ERROR.json. 중단 원인 수정과 journal 일치 검토 뒤에만 같은 예산의 재개가 가능하다.\n')
            (OUT/'FINAL_DECISION.md').write_text('# 결정 보류\n\n실행 또는 검산 오류로 방법의 성능·논문 성공을 판정하지 않는다. 새 후보·자동 재학습 없음.\n')
    try:publish()
    except BaseException:
        save(CACHE/'PUBLICATION_ERROR.json',dict(traceback=traceback.format_exc(),at=time.time()));raise

if __name__=='__main__':main()
