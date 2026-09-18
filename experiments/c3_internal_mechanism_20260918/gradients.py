import itertools
import pandas as pd
from .common import *

def initial(watch):
 vectors={};checks=[];geometry=[];signalrows=[]
 for source in SOURCES:
  a=arrays(source);models={b:build('C3',(b,81551,81551),source) for b in LEVELS};captures={b:Capture(m) for b,m in models.items()};before={b:tensor_hash(m.state_dict()) for b,m in models.items()};adapters={i:ResidualAdapter(i).cuda().eval() for i in LEVELS};features={i:[] for i in LEVELS}
  for start in range(0,1024,32):
   watch.boundary();x,y,s=batch(a,'C3',0,np.arange(start,start+32));pred={};v={};loss={}
   for b,m in models.items():
    pred[b]=forward(m,x,s);loss[b]=loss_2pinball(pred[b],y,s,m.base.quantiles);v[b]=grad(loss[b],pred[b],retain_graph=True)[0].detach()
   h=captures[81551].data['h'];assert torch.equal(h,captures[81552].data['h']),'INPUT_EMBEDDING_DIFFERS'
   assert all(torch.equal(captures[b].data['u'],h) for b in LEVELS)
   signalrows.append(dict(source=source,batch=start//32,v_disagreement_fraction=float((v[81551]!=v[81552]).float().mean()),v_cosine=cos(v[81551].cpu(),v[81552].cpu()),h_max_difference=0.))
   rr={}
   for b,m in models.items():
    for error_b in LEVELS:rr[b,error_b]=grad(pred[b],captures[b].data['u'],grad_outputs=v[error_b],retain_graph=True)[0].detach()
    params=parameters(m);gs=grad(loss[b],tuple(params.values()),retain_graph=True);native=dict(zip(params,gs));assert torch.count_nonzero(native['adapter.down.weight'])==0 and torch.count_nonzero(native['adapter.down.bias'])==0
    expected=torch.cat([native['adapter.up.weight'].flatten(),native['adapter.up.bias']]).detach().cpu().numpy()
    with torch.no_grad():actual,_=formula(h,captures[b].data['g'],rr[b,b],m.adapter)
    actual=actual.cpu().numpy();r=rel(actual,expected);d=float(np.max(np.abs(actual-expected)));assert r<=1e-4 and d<=2e-6,('CHAIN_RULE',r,d)
    checks.append(dict(source=source,b=b,batch=start//32,relative_l2=r,max_abs=d,down_gradient_zero=True))
   with torch.no_grad():
    for i in LEVELS:
     z=torch.nn.functional.gelu(adapters[i].down(h));features[i].append(z.cpu().numpy().reshape(-1,8))
     for arm in ARMS:
      g=gateof(x,s,arm)
      for b,error_b in itertools.product(LEVELS,repeat=2):
       value,_=formula(h,g,rr[b,error_b],adapters[i]);key=f'{source}__{arm}__i{i}__j{b}__v{error_b}';vectors[key]=vectors.get(key,np.zeros(4608,dtype=float))+value.cpu().numpy().astype(float)/32
   # Alternate init/gate autograd confirms the closed form on the first common batch.
   if start==0:
    for b,i,arm in itertools.product(LEVELS,LEVELS,ARMS):
     if i==81551 and arm=='C3':continue
     m=build(arm,(b,i,81551),source);capture=Capture(m);p=forward(m,x,s);assert torch.equal(p,pred[b]);params=parameters(m);gs=grad(loss_2pinball(p,y,s,m.base.quantiles),tuple(params.values()));gs=dict(zip(params,gs))
     assert torch.count_nonzero(gs['adapter.down.weight'])==0
     with torch.no_grad():f,_=formula(capture.data['h'],capture.data['g'],rr[b,b],m.adapter)
     expected=torch.cat([gs['adapter.up.weight'].flatten(),gs['adapter.up.bias']]).detach().cpu().numpy();r=rel(f.cpu().numpy(),expected);d=float(np.max(np.abs(f.cpu().numpy()-expected)));assert r<=1e-4 and d<=2e-6
     checks.append(dict(source=source,b=b,i=i,arm=arm,batch=0,relative_l2=r,max_abs=d,alternate_autograd=True,initial_prediction_exact=True));capture.close();del m,p,gs,capture
   del pred,v,loss,rr;save_counts()
  z1=np.concatenate(features[81551]);z2=np.concatenate(features[81552]);z1=z1-z1.mean(0);z2=z2-z2.mean(0);cka=float(np.sum((z1.T@z2)**2)/np.sqrt(np.sum((z1.T@z1)**2)*np.sum((z2.T@z2)**2)))
  geometry.append(dict(source=source,contexts=1024,patches=32768,initial_feature_linear_CKA=cka,B0_input_embedding_identical=True))
  for b,m in models.items():assert tensor_hash(m.state_dict())==before[b];captures[b].close()
  del models,captures,adapters,features,h;cleanup();print('INITIAL_GRADIENTS',source,COUNTS,flush=True)
 np.savez_compressed(CACHE/'initial_gradients.npz',**vectors);pd.DataFrame(checks).to_csv(OUT/'CHAIN_RULE_CHECKS.csv',index=False);pd.DataFrame(geometry).to_csv(OUT/'FEATURE_GEOMETRY.csv',index=False);pd.DataFrame(signalrows).to_csv(OUT/'OUTPUT_SIGNAL_CHECKS.csv',index=False)
 rows=[];gate_rows=[]
 for source,arm,i in itertools.product(SOURCES,ARMS,LEVELS):
  get=lambda b,v:vectors[f'{source}__{arm}__i{i}__j{b}__v{v}'];a=get(81551,81551);b=get(81551,81552);c=get(81552,81551);d=get(81552,81552);J=.5*((c-a)+(d-b));V=.5*((b-a)+(d-c));delta=d-a;np.testing.assert_allclose(J+V,delta,rtol=1e-10,atol=1e-12)
  rows.append(dict(source=source,arm=arm,init=i,gradient_low_norm=np.linalg.norm(a),gradient_high_norm=np.linalg.norm(d),native_cosine=cos(a,d),total_difference_norm=np.linalg.norm(delta),Jacobian_component_norm=np.linalg.norm(J),output_signal_component_norm=np.linalg.norm(V),components_cosine=cos(J,V),Jacobian_change_fixed_signal_cosine=cos(a,c),signal_change_fixed_J_cosine=cos(a,b)))
 for source,i,b in itertools.product(SOURCES,LEVELS,LEVELS):
  a=vectors[f'{source}__C3__i{i}__j{b}__v{b}'];d=vectors[f'{source}__MAG_ONLY__i{i}__j{b}__v{b}'];gate_rows.append(dict(source=source,init=i,B0=b,C3_MAG_gradient_cosine=cos(a,d),C3_norm=np.linalg.norm(a),MAG_norm=np.linalg.norm(d),difference_norm=np.linalg.norm(a-d)))
 pd.DataFrame(rows).to_csv(OUT/'INITIAL_GRADIENT_DECOMPOSITION.csv',index=False);pd.DataFrame(gate_rows).to_csv(OUT/'INITIAL_GATE_GRADIENTS.csv',index=False)

def order_first_batch(watch):
 vec={};rows=[]
 for source,b,i,arm in itertools.product(SOURCES,LEVELS,LEVELS,ARMS):
  a=arrays(source);m=build(arm,(b,i,81551),source);before=tensor_hash(m.state_dict());indices={}
  for o in LEVELS:
   watch.boundary();idx=rng(84100,source,o,0).permutation(1024)[:32];indices[o]=idx;x,y,s=batch(a,arm,0,idx);p=forward(m,x,s);loss=loss_2pinball(p,y,s,m.base.quantiles);g=flat(grad(loss,tuple(parameters(m).values()))).detach().cpu().numpy();vec[f'{source}_{arm}_{b}_{i}_{o}']=g
  g1=vec[f'{source}_{arm}_{b}_{i}_81551'];g2=vec[f'{source}_{arm}_{b}_{i}_81552'];rows.append(dict(source=source,B0=b,init=i,arm=arm,first_batch_overlap=len(set(indices[81551])&set(indices[81552])),gradient_cosine=cos(g1,g2),gradient_relative_difference=rel(g2,g1)))
  assert tensor_hash(m.state_dict())==before;del m;cleanup()
 np.savez_compressed(CACHE/'first_batch_gradients.npz',**vec);pd.DataFrame(rows).to_csv(OUT/'ORDER_FIRST_BATCH.csv',index=False);save_counts();print('FIRST_BATCH_DONE',COUNTS,flush=True)

def trajectory(watch):
 rows=[];vectors={};probe=read(OUT/'PROBE.json')
 for row in read(PRIOR/'GRID.json'):
  m=load(row,step=0);a=arrays(row['source']);rec=read(PRIOR/'fits'/row['fit']/'receipt.json');initial_state=cpu_state(m);frozen=frozen_hash(m);capture=Capture(m)
  for step in STEPS:
   c=next(x for x in rec['checkpoints'] if x['step']==step);restore(m,torch.load(ROOT/c['checkpoint'],weights_only=True,map_location='cpu'));before=tensor_hash(m.state_dict());losses=[];gs=[];res=[];saturation=[];features=[]
   for selection in probe:
    watch.boundary();x,y,s=batch(a,row['arm'],selection['epoch'],np.array(selection['indices']));p=forward(m,x,s);loss=loss_2pinball(p,y,s,m.base.quantiles);params=parameters(m);g=grad(loss,tuple(params.values()));gs.append(flat(g).detach().cpu().numpy().astype(float));losses.append(float(loss))
    with torch.no_grad():
     h=capture.data['h'];u=capture.data['u'];res.append(float((u-h).norm()/h.norm()));z=torch.nn.functional.gelu(m.adapter.down(h));t=torch.tanh(m.adapter.up(z));saturation.append(float((t.abs()>.95).float().mean()));features.append(z.cpu().numpy().reshape(-1,8))
   g=np.mean(gs,axis=0);key=f"{row['fit']}__{step}";vectors[key]=g
   state=cpu_state(m);dn=torch.cat([(state[k]-initial_state[k]).flatten() for k in ['adapter.down.weight','adapter.down.bias']]);un=torch.cat([state[k].flatten() for k in ['adapter.up.weight','adapter.up.bias']]);dn0=torch.cat([initial_state[k].flatten() for k in ['adapter.down.weight','adapter.down.bias']]);down_n=4104
   rows.append(dict(fit=row['fit'],source=row['source'],arm=row['arm'],b=row['seed'][0],i=row['seed'][1],o=row['seed'][2],step=step,probe_loss=float(np.mean(losses)),gradient_norm=float(np.linalg.norm(g)),down_gradient_norm=float(np.linalg.norm(g[:down_n])),up_gradient_norm=float(np.linalg.norm(g[down_n:])),down_relative_drift=float(dn.norm()/dn0.norm()),up_parameter_norm=float(un.norm()),residual_relative_norm=float(np.mean(res)),tanh_saturation_fraction=float(np.mean(saturation))))
   assert tensor_hash(m.state_dict())==before and frozen_hash(m)==frozen
   if step==0:assert np.linalg.norm(g[:down_n])==0
  capture.close();del m,capture,a;cleanup();save_counts();print('TRAJECTORY',row['fit'],COUNTS['autograd_calls'],flush=True)
 np.savez_compressed(CACHE/'trajectory_gradients.npz',**vectors);pd.DataFrame(rows).to_csv(OUT/'TRAJECTORY.csv',index=False)
 pairs=[]
 for source,b,i,o,step in itertools.product(SOURCES,LEVELS,LEVELS,LEVELS,STEPS):
  rr=[r for r in read(PRIOR/'GRID.json') if r['source']==source and r['seed']==[b,i,o]];by={r['arm']:r for r in rr};u=vectors[by['C3']['fit']+f'__{step}'];v=vectors[by['MAG_ONLY']['fit']+f'__{step}'];pairs.append(dict(source=source,b=b,i=i,o=o,step=step,C3_MAG_gradient_cosine=cos(u,v),C3_MAG_gradient_difference_norm=np.linalg.norm(u-v)))
 pd.DataFrame(pairs).to_csv(OUT/'TRAJECTORY_GRADIENT_PAIRS.csv',index=False)
