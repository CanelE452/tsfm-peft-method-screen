"""Post-selection CPU diagnostics/figures. No optimizer or model selection."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/outlier_followup_mpl')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *
from .model import ForecastModel,coordinates
from .reference_core import STATES
from .prepare import OLD

@torch.no_grad()
def render():
    setup();assert read(OUT/'verification.json')['status']=='VERIFIED'
    rows=[];parameter=[];draws={};examples={}
    for sel in read(OUT/'MODEL_SELECTION.json'):
        source,arm,seed=sel['source'],sel['arm'],sel['seed']
        if arm not in ['B3','B4','B5']:continue
        model=ForecastModel(torch.nn.Identity(),arm,seed)
        state=torch.load(ROOT/sel['checkpoint'],weights_only=True,map_location='cpu')
        if arm in ['B4','B5']:model.gate.copy_(state['gate'])
        for role in ['V_SELECT','E_DISCOVERY']:
            f=CACHE/'conditions'/source;x=np.load(f/f'{role}_x.npy',mmap_mode='r');s=np.load(f/f'{role}_sigma.npy',mmap_mode='r')
            gates=[];excess=[]
            for lo in range(0,len(x),256):
                xx=torch.tensor(x[lo:lo+256].copy());ss=torch.tensor(s[lo:lo+256].copy());eff,g,p,e=model.transform(xx,ss)
                gates.append(g.numpy());excess.append(e.numpy())
            g=np.concatenate(gates);e=np.concatenate(excess);chunk=len(g)//10
            for j,condition in enumerate(STATES):
                gg=g[j*chunk:(j+1)*chunk];ee=e[j*chunk:(j+1)*chunk]
                for subset,vals in [('all_slots',gg.ravel()),('clipped_slots',gg[ee>0])]:
                    if len(vals)==0:continue
                    quant=np.quantile(vals,[0,.1,.25,.5,.75,.9,1]);hist,bins=np.histogram(vals,np.linspace(0,1,21))
                    rows.append(dict(source=source,arm=arm,seed=seed,role=role,condition=condition,subset=subset,count=len(vals),mean=float(vals.mean()),std=float(vals.std()),**{name:float(v) for name,v in zip(['min','p10','p25','median','p75','p90','max'],quant)},histogram=json.dumps(hist.tolist())))
                if role=='E_DISCOVERY' and condition in ['POINT8','BURST8','SHIFT4','SHIFT8']:
                    draws[(source,arm,seed,condition)]=np.histogram(gg[ee>0],np.linspace(0,1,21))[0]
            if role=='V_SELECT' and seed==81551:
                for condition in ['POINT8','SHIFT8']:
                    i=STATES.index(condition)*chunk;xx=torch.tensor(x[i:i+1].copy());ss=torch.tensor(s[i:i+1].copy());eff=model.transform(xx,ss)[0]
                    examples[(source,arm,condition)]=(xx.numpy()[0],eff.numpy()[0],coordinates(xx,ss)[0].numpy()[0])
    for p in (OUT/'fits').glob('*/receipt.json'):
        rec=read(p)
        if rec['arm'] not in ['B4','B5']:continue
        selected=torch.load(ROOT/rec['selected']['checkpoint'],weights_only=True,map_location='cpu')['gate'].tolist()
        end=next(c for c in rec['checkpoints'] if c['step']==1024);final=torch.load(ROOT/end['checkpoint'],weights_only=True,map_location='cpu')['gate'].tolist()
        parameter.append(dict(source=rec['source'],arm=rec['arm'],seed=rec['seed'],lr=rec['lr'],selected_step=rec['selected']['step'],initial=[-4.]+[0.]*(len(final)-1),selected=selected,final=final))
    pd.DataFrame(rows).to_csv(OUT/'gate_behavior.csv',index=False);save(OUT/'gate_parameters.json',parameter)
    # Zero-training V dependence diagnostics compared to stored normal V predictions.
    ab=[];di=read(OUT/'B5_validation_diagnostics.json')
    for sel in read(OUT/'MODEL_SELECTION.json'):
        if sel['arm']!='B5':continue
        source,seed=sel['source'],sel['seed'];key=f'{source}_B5_{seed}'
        p=np.load(CACHE/'predictions'/f'{key}_V.npy',mmap_mode='r');y=np.load(CACHE/'conditions'/source/'V_SELECT_y.npy')
        sig=np.tile(read(OUT/'DATA_MANIFEST.json')[source]['sigma_train_population'],len(y)//4)
        normal=(np.abs(p[:,4].astype(float)-y)/sig[:,None]).mean(-1).reshape(10,2,64,4).mean((1,2,3))
        for mode,values in di[key].items():
            for condition,a,b in zip(STATES,normal,values):ab.append(dict(source=source,seed=seed,condition=condition,mode=mode,normal_nmae=float(a),ablated_nmae=b,error_increase_pct=float(100*(b/a-1)),optimizer_updates=0))
    pd.DataFrame(ab).to_csv(OUT/'B5_validation_ablation_effects.csv',index=False)
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    def savefig(name):
        plt.tight_layout();plt.savefig(figdir/(name+'.png'),dpi=160);plt.savefig(figdir/(name+'.pdf'));plt.close()
    exposure=pd.read_csv(OUT/'clip_exposure_summary.csv');fig,axes=plt.subplots(2,2,figsize=(11,7))
    for j,source in enumerate(SOURCES):
        f=exposure[(exposure.source==source)&(((exposure.role=='TRAIN')&(exposure.state=='SHIFT'))|((exposure.role=='E_DISCOVERY')&exposure.state.isin(['SHIFT4','SHIFT8'])))].copy()
        f=f.sort_values(['role','state'],ascending=[False,True]);labels=[f'{r.role}\n{r.state}' for r in f.itertuples()]
        for k,metric in enumerate(['shift_clipped_fraction','normalized_mass']):
            axes[j,k].bar(labels,f[metric]);axes[j,k].set_title(source+' / '+metric)
    savefig('training_evaluation_clip_exposure')
    scores=pd.read_csv(OUT/'scores_by_condition.csv');scores['panel']=scores.condition.map(lambda x:'FAULT' if x.startswith(('POINT','BURST')) else x)
    summary=scores.groupby(['source','arm','panel']).nmae.mean()
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for i,source in enumerate(SOURCES):
        for j,panel in enumerate(['FAULT','REFERENCE','SHIFT4','SHIFT8']):
            axes[i,j].bar(ARMS,[summary[source,a,panel] for a in ARMS]);axes[i,j].set_title(source+' '+panel);axes[i,j].set_ylabel('nMAE (lower better)')
    savefig('fault_reference_shift_tradeoff')
    fig,axes=plt.subplots(2,4,figsize=(15,7));centers=(np.arange(20)+.5)/20
    for i,source in enumerate(SOURCES):
        for j,condition in enumerate(['POINT8','BURST8','SHIFT4','SHIFT8']):
            for arm in ['B3','B4','B5']:
                hist=sum(draws[(source,arm,seed,condition)] for seed in [81551,81552]);axes[i,j].plot(centers,hist/max(hist.sum(),1),label=arm)
            axes[i,j].set_title(source+' '+condition);axes[i,j].set_xlabel('restore fraction on clipped slots');axes[i,j].legend()
    savefig('restore_fraction_distribution')
    fig,axes=plt.subplots(2,2,figsize=(13,7))
    for i,source in enumerate(SOURCES):
        for j,condition in enumerate(['POINT8','SHIFT8']):
            raw,_,clip=examples[(source,'B3',condition)];axes[i,j].plot(raw,label='observed',alpha=.5);axes[i,j].plot(clip,label='clip6',alpha=.7)
            for arm in ['B3','B4','B5']:axes[i,j].plot(examples[(source,arm,condition)][1],label=arm,alpha=.7)
            axes[i,j].set_title(f'{source} / {condition} / first sealed V origin, ch0, g0');axes[i,j].legend()
    savefig('predeclared_transform_examples')
    # Fit ledger and direct resource costs, no normalization into a composite score.
    inf=read(OUT/'inference_resources.json');resources=[];ledger=[]
    for p in sorted((OUT/'fits').glob('*/receipt.json')):
        rec=read(p);key=f"{rec['source']}_{rec['arm']}_{rec['seed']}";ii=inf.get(key,{})
        ledger.append({k:rec[k] for k in ['fit','source','arm','seed','lr','status','updates','optimizer_seconds','validation_seconds','checkpoint_io_seconds','peak_allocated','peak_reserved']})
        resources.append(dict(source=rec['source'],arm=rec['arm'],seed=rec['seed'],lr=rec['lr'],selected_step=rec['selected']['step'],optimizer_seconds=rec['optimizer_seconds'],validation_seconds=rec['validation_seconds'],checkpoint_io_seconds=rec['checkpoint_io_seconds'],train_peak_allocated_mib=rec['peak_allocated']/2**20,train_peak_reserved_mib=rec['peak_reserved']/2**20,E_inference_seconds=ii.get('inference_seconds'),E_peak_allocated_mib=ii.get('peak_allocated',0)/2**20))
    pd.DataFrame(ledger).to_csv(OUT/'FIT_LEDGER.csv',index=False);pd.DataFrame(resources).to_csv(OUT/'resources.csv',index=False)
    guards=[]
    for p in OUT.glob('gpu_*.jsonl'):
        guards.extend(json.loads(l) for l in open(p))
    used=sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file() and not p.is_symlink())
    summary=dict(main_optimizer_seconds=sum(r['optimizer_seconds'] for r in resources),validation_seconds=sum(r['validation_seconds'] for r in resources),checkpoint_io_seconds=sum(r['checkpoint_io_seconds'] for r in resources),E_inference_seconds=sum(r['inference_seconds'] for r in inf.values()),min_free_gpu_mib=min(r['free_mib'] for r in guards),unapproved_external_compute_samples=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in r['apps']) for r in guards),cache_bytes_excluding_symlink_files=used)
    save(OUT/'execution_resource_summary.json',summary)
    # Post-run immutable dependencies/data/checkpoints are re-audited, plus all selection decisions.
    from .prepare import audit
    audit();check_seal()
    for p,h in read(OUT/'AUGMENTATION_MANIFEST.json')['hashes'].items():assert sha(ROOT/p)==h
    from .train import fit_id
    choices=read(OUT/'LR_SELECTION.json')
    for source in SOURCES:
        for arm in ARMS:
            rs=[read(OUT/'fits'/fit_id(source,arm,81550,lr)/'receipt.json') for lr in [1e-4,3e-4]]
            chosen=min(rs,key=lambda r:(r['selected']['objective'],r['lr'],r['selected']['step']))
            assert choices[source][arm]['fit']==chosen['fit']
    save(OUT/'publication_audit.json',dict(status='VERIFIED',immutable_v1_hashes=True,augmented_and_reused_arrays_verified=True,LR_selection_recomputed=True,gate_diagnostics_optimizer_updates=0,postprocessing_did_not_modify_selection=True,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in EXP.glob('*.py')}))
    print('POST_ANALYSIS_VERIFIED',summary,flush=True)

if __name__=='__main__':render()
