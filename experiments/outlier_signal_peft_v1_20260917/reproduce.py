"""Bounded public-model reduced replication: 49 series/model, no optimizer."""
from .common import *
from .reference_core import rng

def reproduce(watch):
    if (OUT/'reproduction.json').exists():return
    from chronos import ChronosBoltPipeline,ChronosPipeline
    values=np.cos(np.arange(-512,64)*2*np.pi/160+.3)
    xx=[values[:512].astype(np.float32)];meta=[dict(count=0,amplitude=0,seed=-1)]
    for count in [1,16]:
        for amp in [1,10,100]:
            for seed in range(8):
                r=rng(82200,count,amp,seed);x=values[:512].copy();ix=r.choice(512,count,replace=False)
                x[ix]+=amp*r.normal(size=count);xx.append(x.astype(np.float32));meta.append(dict(count=count,amplitude=amp,seed=seed))
    results={};rows=[]
    for name,cls in [('amazon/chronos-bolt-small',ChronosBoltPipeline),('amazon/chronos-t5-small',ChronosPipeline)]:
        watch.boundary();rec=read(OUT/'download_receipts.json')[name];short=name.split('/')[-1]
        journal=OUT/(short+'_reproduction.jsonl')
        assert not journal.exists(),'INCOMPLETE_REFERENCE_NO_SILENT_REPEAT'
        try:pipe=cls.from_pretrained(str(ROOT/rec['snapshot']),device_map='cuda',torch_dtype=torch.float32)
        except Exception as e:
            if 't5' in name:results[name]=dict(status='BLOCKED_LOADER',error=repr(e),series=0);continue
            raise
        model=pipe.model;model.eval();predictions=[];count_calls=[0]
        def hook(*args):count_calls[0]+=1
        # T5 wrapper calls an inner HF model; count actual decoder/model forward calls.
        inner=getattr(model,'model',model)
        handle=inner.register_forward_hook(hook)
        for i,x in enumerate(xx):
            watch.boundary();torch.manual_seed(82300+i);torch.cuda.manual_seed_all(82300+i)
            with torch.no_grad():
                if 'bolt' in name:p=pipe.predict(torch.tensor(x),prediction_length=64)[0,4].cpu().numpy()
                else:p=pipe.predict(torch.tensor(x),prediction_length=64,num_samples=20)[0].quantile(.5,dim=0).cpu().numpy()
            assert np.isfinite(p).all();predictions.append(p)
            row=dict(model=short,**meta[i],mae=float(np.abs(p-values[-64:]).mean()))
            rows.append(row);append(journal,row)
        handle.remove();np.save(CACHE/f'{short}_cosine_predictions.npy',predictions)
        results[name]=dict(status='COMPLETE',series=49,internal_forward_calls=count_calls[0],samples_per_series=20 if 't5' in name else 1)
        del pipe,model,inner;cleanup()
    csvwrite(OUT/'reproduction_scores.csv',rows)
    save(OUT/'reproduction.json',dict(models=results,optimizer_updates=0,maximum_series=98,
         source='cos(t+.3), period160/context512/horizon64',kind='reduced replication',
         amplitude_rule='seeded Gaussian additive amplitudes; not exact original notebook sweeps',
         patch_size1='SKIPPED_UNAVAILABLE_REFERENCE'))
    print('REPRODUCTION_COMPLETE',results,flush=True)
