"""Bounded experiment only. No automatic retry or successor training."""
import subprocess,traceback,os
from .common import *
def main():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    import psutil
    save(CACHE/'PROCESS.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),command='mag_standalone_ablation_v1_20260919.execute'))
    try:
        from .runner import run
        run()
    except BaseException as e:
        save(OUT/'EXECUTION_ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),at=time.time()));status(execution='RESOURCE_BLOCKED' if isinstance(e,ResourceError) else 'IMPLEMENTATION_OR_AUDIT_ERROR',error=str(e))
        s=read(OUT/'status.json')
        if not (OUT/'REPORT.md').exists():
            (OUT/'REPORT.md').write_text('# 실행 중단 기록\n\n성능 실패와 구분한다. 다섯 성능 질문은 평가·검산 완료 전에는 답할 수 없다. 자동 재시작 없음. 완료receipt/journal보존.\n\n```json\n'+json.dumps(s,ensure_ascii=False,indent=2)+'\n```\n\n재개는같은설정·남은예산·journal일치검토뒤가능하며모호한update는자동재실행하지않는다.\n')
            (OUT/'FINAL_DECISION.md').write_text('# 판단 보류\n\n실행 또는 검산 차단을 성능 실패로 분류하지 않는다. 새 후보·자동후속학습없음.\n')
        raise
if __name__=='__main__':main()
