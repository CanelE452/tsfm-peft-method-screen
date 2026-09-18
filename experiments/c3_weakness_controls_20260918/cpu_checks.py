import inspect
from .common import *
from .model import ControlModel

def run():
    setup();checks=[]
    # Gate math without a foundation model; same actual method called on a minimal module.
    m=object.__new__(ControlModel);torch.nn.Module.__init__(m);m.arm='MAG_ONLY'
    x=torch.zeros(2,512);x[0,100:108]=8;x[1,100:108]=-8;s=torch.ones(2);g=m.gate(x,s)
    expected=torch.ones(2,32);expected[:,6]=.5;assert torch.equal(g,expected);checks.append('magnitude_gate_scalar_positions_and_sign_symmetry')
    m.arm='POS_ONLY';m.position_logits=torch.nn.Parameter(torch.zeros(32));assert torch.equal(m.gate(x,s),torch.full((2,32),.5));assert torch.equal(m.gate(x,s),m.gate(x.flip(-1)+10,s));checks.append('position_gate_no_input_dependence')
    assert list(inspect.signature(ControlModel.forward).parameters)==['self','observed','sigma','residual_mode','persistence_mode'];checks.append('no_future_or_generator_metadata_interface')
    from experiments.additive_persistence_validation_v1_20260917.model import ResidualAdapter
    a,b=ResidualAdapter(81550),ResidualAdapter(81550);assert tensor_hash(a.state_dict())==tensor_hash(b.state_dict());assert sum(p.numel() for p in a.parameters())+32==8744;assert 64*72+64+1==4673;checks.append('same_adapter_initialization_and_parameter_arithmetic')
    # Native affine normalization makes mean(normalized block)=(mean(block)-loc)/scale.
    from chronos.chronos_bolt import InstanceNorm
    norm=InstanceNorm();xx=torch.arange(1024,dtype=torch.float32).reshape(2,512);z,ls=norm(xx);loc,scale=ls;torch.testing.assert_close(z.reshape(2,8,64).mean(-1),(xx.reshape(2,8,64).mean(-1)-loc)/scale);checks.append('observed_only_normalized_eight_features')
    OUT.mkdir(exist_ok=True,parents=True);save(OUT/'CPU_CHECKS.json',dict(status='VERIFIED',checks=checks,optimizer_updates=0));print('CPU CHECKS',len(checks))
if __name__=='__main__':run()
