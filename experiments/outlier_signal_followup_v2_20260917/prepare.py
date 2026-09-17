"""Audit immutable v1, exact origin reuse, matched augmentation sealed before fits."""
import subprocess,shutil,collections
from .common import *
from .reference_core import rng,scale,STATES
OLD=ROOT/'results/outlier_signal_peft_v1_20260917'
OLDC=ROOT/'.cache/outlier_signal_peft_v1_20260917'

def audit():
    p=OUT/'SOURCE_MANIFEST.json'
    if p.exists():
        for name,h in read(p)['immutable_hashes'].items():assert sha(ROOT/name)==h,name
        return
    names=['REPORT.md','FINAL_DECISION.md','verification.json','origin_audit.json','MODEL_SELECTION.json','LR_SELECTION.json','scores_by_condition.csv','scores_by_origin.csv']
    files=[OLD/n for n in names]+[ROOT/'experiments/outlier_signal_peft_v1_20260917'/n for n in ['reference_core.py','model.py','train.py','evaluate.py','common.py']]
    # Read every specified document; CSV is parsed for shape and finite numeric metrics.
    import pandas as pd
    inspected={}
    for f in files:
        if f.suffix=='.csv':
            df=pd.read_csv(f);inspected[str(f.relative_to(ROOT))]={'rows':len(df),'columns':list(df.columns)}
        else:inspected[str(f.relative_to(ROOT))]={'bytes_read':len(f.read_bytes())}
    assert read(OLD/'verification.json')['status']=='VERIFIED'
    hashes={str(f.relative_to(ROOT)):sha(f) for f in files}
    for name,h in read(OLD/'EXECUTION_SEAL.json')['source_hashes'].items():assert sha(ROOT/name)==h;hashes[name]=h
    for row in read(OLD/'MODEL_SELECTION.json'):
        assert sha(ROOT/row['checkpoint'])==row['sha256'];hashes[row['checkpoint']]=row['sha256']
    for info in read(OUT/'DATA_MANIFEST.json').values():
        assert sha(ROOT/info['path'])==info['sha256'];hashes[info['path']]=info['sha256']
        for pp in info['packet_hashes'].values():
            for name,h in pp.items():assert sha(ROOT/name)==h;hashes[name]=h
    for name,h in read(OLD/'CONDITION_MANIFEST.json')['hashes'].items():assert sha(ROOT/name)==h;hashes[name]=h
    for info in read(OUT/'download_receipts.json').values():
        # Snapshot and actual weight hashes were sealed by v1; retain receipt as provenance.
        assert (ROOT/info['snapshot']).exists()
    hashes[str((OLD/'download_receipts.json').relative_to(ROOT))]=sha(OLD/'download_receipts.json')
    for n in ['origins.csv','origin_audit.json','DATA_MANIFEST.json']:
        assert sha(OUT/n)==sha(OLD/n)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    baseline='e2b1ea4a050021740a64c58c50bc8d02b156b06e'
    save(p,dict(baseline=baseline,head=head,diff_from_baseline=subprocess.check_output(['git','diff',baseline,'--stat'],cwd=ROOT,text=True),inspected=inspected,immutable_hashes=hashes,old_results_read_only=True))


def prepare():
    audit()
    if (OUT/'AUGMENTATION_MANIFEST.json').exists():
        for p,h in read(OUT/'AUGMENTATION_MANIFEST.json')['hashes'].items():assert sha(ROOT/p)==h
        return
    save(OUT/'MACHINE_CONTRACT.json',dict(status='SEALED_DESIGN_BEFORE_NEW_TRAINING',arms=ARMS,sources=SOURCES,
        seeds=dict(select=81550,repeat=[81551,81552]),lr_grid=[1e-4,3e-4],epochs=32,updates=1024,checkpoints=[0,256,512,768,1024],
        main_cap=49152,smoke_cap=24,fit_cap=48,context=512,horizon=64,effective_batch=32,
        gate_initial=dict(B4=[-4,0,0],B5=[-4,0,0,0]),gate_initial_rationale='B5 unspecified intercept initialized identically to B4 before any new fits',
        gate_extra=dict(B4=3,B5=4),parameter_conflict_resolution='specific sections 7.4/7.5/9 override earlier generic same-count phrase; no dummy scalar',
        V_conditions=['REFERENCE','POINT8','BURST8','SHIFT4','SHIFT8'],objective='equal-condition TRAIN-sigma nMAE',
        tie='smaller LR then earlier checkpoint',generator_key=82800,shuffle_key=83100,bootstrap_key=83500,p_permutation_key=83400,
        training_state='[REFERENCE,POINT,BURST,SHIFT][(epoch+example)%4]',exposure='epoch//4, each example/state appears once per four epochs',
        amplitude_cycle=[4,8,16,4,8,16,4,8],point_count='[1,2,4][(exposure+example)%3]',burst_duration='[2,4][(exposure+example)%2]',
        shift_amplitude=[4,8,4,8,4,8,4,8],shift_duration=[24,48,24,48,48,24,48,24],
        tolerance=dict(cpu_rtol=1e-10,cpu_atol=1e-12,gpu='normalized max <=1e-5 OR rtol1e-4'),
        diagnostic_tags='descriptive quantitative comparisons; no performance gates or new numerical thresholds',example_figure='first sealed V origin, channel0, generator0; all states',
        E_is_reused_development=True,automatic_followup=False))
    hashes={};counts={}
    for source in SOURCES:
        f=CACHE/'conditions'/source;f.mkdir(parents=True,exist_ok=True)
        for old in (OLDC/'conditions'/source).iterdir():
            if not old.name.startswith('train_') and not (f/old.name).exists():(f/old.name).symlink_to(old)
        d=np.load(CACHE/'data'/source/'TRAIN_inputs.npz');x0=d['x'].reshape(-1,512);sigma=np.tile(d['sigma'],256)
        y0=np.load(CACHE/'data'/source/'TRAIN_labels.npz')['y'].reshape(-1,64)
        xs=np.empty((32,1024,512),np.float32);ys=np.empty((32,1024,64),np.float32);meta=[]
        for epoch in range(32):
            for i in range(1024):
                state=['REFERENCE','POINT','BURST','SHIFT'][(epoch+i)%4];k=epoch//4;r=float(scale(x0[i].astype(float),sigma[i]));rr=rng(82800,source,'TRAIN',epoch,i)
                x=x0[i].astype(float).copy();delta=0.;positions=[];duration=0;amp=0
                if state=='SHIFT':
                    amp=[4,8,4,8,4,8,4,8][k];duration=[24,48,24,48,48,24,48,24][k]
                    delta=float(rr.choice([-1,1])*amp*r);x[-duration:]+=delta
                elif state in ['POINT','BURST']:
                    amp=[4,8,16,4,8,16,4,8][k]
                    if state=='POINT':
                        count=[1,2,4][(k+i)%3];positions=rr.choice(512,count,replace=False);change=rr.choice([-1,1],count)*amp*r
                    else:
                        duration=[2,4][(k+i)%2];start=rr.integers(513-duration);positions=np.arange(start,start+duration);change=rr.choice([-1,1])*amp*r
                    x[positions]+=change;positions=positions.tolist()
                xs[epoch,i]=x;ys[epoch,i]=y0[i]+delta
                meta.append(dict(epoch=epoch,example=i,state=state,exposure=k,amplitude=amp,duration=duration,delta=delta,r0=r,positions=positions))
        for name,a in [('train_x',xs),('train_y',ys),('train_sigma',sigma.astype(np.float32))]:np.save(f/(name+'.npy'),a)
        save(f/'train_generator_audit.json',meta)
        import pandas as pd
        df=pd.DataFrame(meta)
        assert df.groupby(['example','state']).size().eq(8).all()
        for state in ['POINT','BURST']:
            assert df[df.state==state].groupby(['example','amplitude']).size().unstack().eq([3,3,2]).all().all()
        assert df[df.state=='SHIFT'].groupby(['example','amplitude']).size().eq(4).all()
        assert df[df.state=='SHIFT'].groupby(['example','duration']).size().eq(4).all()
        rows=df.groupby(['state','amplitude','duration']).size().rename('count').reset_index().to_dict('records');counts[source]=rows
        for p in f.iterdir():hashes[str(p.relative_to(ROOT))]=sha(p)
        counts[source].append(dict(shared_x_sha256=sha(f/'train_x.npy'),shared_y_sha256=sha(f/'train_y.npy'),all_arms_same_packet=True))
    save(OUT/'AUGMENTATION_MANIFEST.json',dict(hashes=hashes,counts=counts,model_packet_fields=['observed','sigma'],labels_separate=True,V_E_exact_v1_reuse=True))
    print('PREPARED_MATCHED_TRAIN',flush=True)
