"""Official component information-flow probes; no forecaster and no optimizer."""
from pathlib import Path
import ast,hashlib,json,math
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
CACHE=ROOT/'.cache/method_baseline_compatibility_20260919'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def classes(p,names):
 tree=ast.parse(p.read_text());selected=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name in names];assert {x.name for x in selected}==set(names)
 scope={'torch':torch,'nn':nn,'math':math}
 exec(compile(ast.Module(body=selected,type_ignores=[]),str(p),'exec'),scope)
 return [scope[x] for x in names]
def count(m):return sum(p.numel() for p in m.parameters())
def main():
 torch.set_num_threads(4);torch.manual_seed(91980)
 tp=ROOT/'.cache/temporal_response_peft_20260919/prior/TimePEFT_run.py'
 receipt=json.loads((ROOT/'results/temporal_response_peft_20260919/PRIOR_CODE_AUDIT.json').read_text());assert sha(tp)==receipt['sha256']
 Frequency,Channel=classes(tp,['FrequencyAdapter','ChannelAdapter'])
 f=Frequency(3,512).eval();c=Channel(512,4,2).eval();x=torch.randn(2,4,32,512,requires_grad=True)
 y=c(x,f(x));changed=x.detach().clone();changed[:,1]+=torch.linspace(-10,10,32)[None,:,None]
 changed_y=c(changed,f(changed));difference=float((y[:,0]-changed_y[:,0]).abs().max().detach());assert difference==0
 grad=torch.autograd.grad(y[:,0].square().sum(),x)[0];cross_gradient=float(grad[:,1:].abs().max());assert cross_gradient==0
 order=torch.tensor([2,0,3,1]);yp=c(x[:,order],f(x[:,order]))[:,torch.argsort(order)]
 permutation_difference=float((y-yp).abs().max().detach());assert permutation_difference>0
 transfer_error=None
 try:c(x[:,:1],f(x[:,:1]))
 except ValueError as e:transfer_error=str(e)
 assert transfer_error is not None
 params=dict(frequency=count(f),channel_four=count(c),total_four=count(f)+count(c),total_one=count(f)+count(Channel(512,1,2)))
 assert params==dict(frequency=262656,channel_four=789760,total_four=1052416,total_one=657664)
 pr=json.loads((CACHE/'receipts.json').read_text());pp=next(v for v in pr if v['file']=='tta/petsa.py');path=ROOT/pp['local'];assert sha(path)==pp['sha256'];GCM,=classes(path,['GCM'])
 cells=[]
 for length in [512,64]:
  m=GCM(length,n_var=1,low_rank=16).eval();xx=torch.randn(3,length,1);yy=m(xx);assert torch.equal(xx,yy)
  initial=count(m);expected=2*length*16+length+1;assert initial==expected
  # Fixed synthetic weight perturbation illustrates routing only; it is not a trained candidate.
  with torch.no_grad():m.lora_B.fill_(.05)
  zz=m(xx);delta=float((zz-xx).abs().max().detach());assert delta>0
  cells.append(dict(length=length,parameters=initial,initial_identity=True,nonzero_calibration_can_change_values=True,synthetic_parameter_perturbation_not_training=True))
 rows=dict(status='CPU_COMPONENT_INFORMATION_FLOW_VERIFIED',timepeft_commit=receipt['commit'],timepeft=dict(shape=[2,4,32,512],top_k=3,hidden_ratio=2,other_channel_intervention_max_difference=difference,other_channel_jacobian_max=cross_gradient,permutation_inverse_max_difference=permutation_difference,channel_count_change_exception=transfer_error,component_parameters=params,full_MOMENT_tested=False,head_and_LoRA_parameters_excluded=True),petsa=dict(commit=pp['commit'],cells=cells,total_shared_single_channel_XY_parameters=sum(x['parameters'] for x in cells),full_online_runner_tested=False,Chronos_connection_tested=False),optimizer_updates=0,real_dataset_forecast_inferences=0,model_weights_loaded=False,new_candidate=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [tp,path,Path(__file__)]})
 (HERE/'CPU_PROBES.json').write_text(json.dumps(rows,indent=2)+'\n');(HERE/'PETSA_CODE_RECEIPTS.json').write_text(json.dumps(pr,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
