"""Independent arithmetic replay of each prepared TRAIN example; no optimizer."""
import numpy as np
from .common import *

def check():
    receipts={}
    for source in SOURCES:
        f=CACHE/'conditions'/source;d=np.load(CACHE/'data'/source/'TRAIN_inputs.npz');x0=d['x'].reshape(-1,512).astype(float);sig=np.tile(d['sigma'],256)
        y0=np.load(CACHE/'data'/source/'TRAIN_labels.npz')['y'].reshape(-1,64)
        x=np.load(f/'train_x.npy',mmap_mode='r');y=np.load(f/'train_y.npy',mmap_mode='r');meta=read(f/'train_generator_audit.json')
        peak=0.;clip_by_amp={4:[],8:[]}
        for m in meta:
            ep,i=m['epoch'],m['example'];original=x0[i];observed=x[ep,i];actual=observed.astype(float)-original
            expected_y=(y0[i]+m['delta']).astype(np.float32);assert np.array_equal(y[ep,i],expected_y)
            affected=np.zeros(512,bool)
            if m['state']=='SHIFT':
                affected[-m['duration']:]=True;expected=original.copy();expected[affected]+=m['delta'];assert np.array_equal(observed,expected.astype(np.float32))
                med=np.median(observed.astype(float));r=max(1.4826*np.median(abs(observed.astype(float)-med)),.1*sig[i]);clipped=abs(observed.astype(float)-med)>6*r
                clip_by_amp[m['amplitude']].append(float(clipped[affected].mean()))
            elif m['state'] in ['POINT','BURST']:
                affected[m['positions']]=True;err=np.max(abs(abs(actual[affected])-m['amplitude']*m['r0']))/sig[i];peak=max(peak,float(err));assert err<=1e-5
                if m['state']=='BURST':assert np.all(np.sign(actual[affected])==np.sign(actual[affected][0]))
            assert np.array_equal(observed[~affected],original[~affected].astype(np.float32))
        receipts[source]=dict(examples_checked=len(meta),labels_exact=True,unaffected_past_exact=True,shift_past_exact=True,fault_amplitude_max_normalized_rounding_error=peak,matched_train_shift_clipped_fraction={str(k):float(np.mean(v)) for k,v in clip_by_amp.items()},shared_train_x_sha256=sha(f/'train_x.npy'),shared_train_y_sha256=sha(f/'train_y.npy'))
    save(OUT/'independent_generator_replay.json',dict(status='VERIFIED',optimizer_updates=0,timing='additional independent audit during main execution; original severity counts passed before launch',sources=receipts))
    print(receipts)

if __name__=='__main__':check()
