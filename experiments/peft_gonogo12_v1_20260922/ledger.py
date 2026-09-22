import time
from common import RESULTS, CONFIG, read, save, event


class Ledger:
    def __init__(self):
        self.path = RESULTS / 'OPTIMIZER_LEDGER.json'
        if self.path.exists():
            self.state = read(self.path)
        else:
            self.state = dict(fits={},counts={group:dict(main_updates=0,smoke_updates=0,main_fits=0) for group in ['h1','h2']})
            save(self.path,self.state)

    def start(self,key,group,phase,steps):
        if key in self.state['fits']:
            raise RuntimeError(f'Existing fit attempt {key}: do not replay optimizer work')
        if phase=='main':
            assert self.state['counts'][group]['main_fits'] < CONFIG[group]['fit_cap']
            self.state['counts'][group]['main_fits'] += 1
        assert phase in ['main','smoke']
        self.state['fits'][key]=dict(group=group,phase=phase,planned_steps=steps,reserved=0,completed=0,status='RUNNING')
        save(self.path,self.state)
        event('fit_start',key=key,group=group,phase=phase,steps=steps)

    def reserve(self,key):
        fit=self.state['fits'][key];group=fit['group'];phase=fit['phase']
        assert fit['reserved']==fit['completed'], 'Uncommitted update: inspect failure; no replay'
        counter=phase+'_updates'
        cap=CONFIG[group]['main_updates_cap' if phase=='main' else 'smoke_updates_cap']
        assert self.state['counts'][group][counter] < cap
        assert fit['reserved'] < fit['planned_steps']
        self.state['counts'][group][counter] += 1
        fit['reserved'] += 1
        save(self.path,self.state)

    def completed(self,key):
        fit=self.state['fits'][key]
        assert fit['reserved']==fit['completed']+1
        fit['completed']+=1
        save(self.path,self.state)

    def finish(self,key):
        fit=self.state['fits'][key]
        assert fit['completed']==fit['reserved']==fit['planned_steps']
        fit['status']='COMPLETE'
        save(self.path,self.state)
        event('fit_complete',key=key,completed=fit['completed'],counts=self.state['counts'])


class PhaseTimer:
    def __init__(self,group,phase):
        self.group=group; self.phase=phase; self.start=time.perf_counter()
        self.path=RESULTS/'GPU_PHASE_TIME.json'
        self.before=read(self.path) if self.path.exists() else {'h1':[],'h2':[]}
        self.previous=sum(row['seconds'] for row in self.before[group])

    def check(self):
        assert self.previous+time.perf_counter()-self.start < 45*60, 'RESOURCE_BLOCK: candidate phase wall cap'

    def close(self):
        self.before[self.group].append(dict(phase=self.phase,seconds=time.perf_counter()-self.start))
        save(self.path,self.before)
