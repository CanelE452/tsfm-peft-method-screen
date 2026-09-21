from common import *
import gc
from types import MethodType
import numpy as np
import torch
from torch import nn
import bitsandbytes as bnb
from chronos import ChronosBoltPipeline
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

torch.set_num_threads(4)
GROUPS = ['input_projection','encoder_attention','encoder_ffn','decoder_self_attention',
          'decoder_cross_attention','decoder_ffn','output_projection']


def group(name):
    if name.startswith('input_patch_embedding.'): return GROUPS[0]
    if name.startswith('output_patch_embedding.'): return GROUPS[6]
    if name.startswith('encoder.'):
        if '.SelfAttention.' in name: return GROUPS[1]
        if '.DenseReluDense.' in name: return GROUPS[2]
    if name.startswith('decoder.'):
        if '.SelfAttention.' in name: return GROUPS[3]
        if '.EncDecAttention.' in name: return GROUPS[4]
        if '.DenseReluDense.' in name: return GROUPS[5]
    raise ValueError('Unmapped actual linear: ' + name)


def clear():
    gc.collect()
    torch.cuda.empty_cache()


def load_base(size='small', quant=False, io16=False):
    pin = read(RESULTS/'UPSTREAM.json')['models'][size]
    kwargs = {'device_map':'cuda','dtype':torch.bfloat16,'local_files_only':True}
    if quant:
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=['input_patch_embedding','output_patch_embedding'] if io16 else None)
    pipeline = ChronosBoltPipeline.from_pretrained(pin['path'], **kwargs)
    base = pipeline.model.eval()
    assert base.chronos_config.prediction_length == 64
    assert np.allclose(base.chronos_config.quantiles, np.arange(1,10)/10)
    base.requires_grad_(False)
    return pipeline, base


def attach(base, seed, scope='qv', ranks=None, ffa=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    names = [n for n,m in base.named_modules() if isinstance(m, nn.Linear)
             and (scope == 'all' or n.rsplit('.',1)[-1] in ['q','v'])]
    assert names
    rank = 4 if scope == 'all' else 8
    rank_pattern = {n:ranks[group(n)] for n in names} if ranks else {}
    base.get_input_embeddings = MethodType(lambda self:self.shared, base)
    net = get_peft_model(base, LoraConfig(r=rank, lora_alpha=rank, target_modules=names,
                         rank_pattern=rank_pattern, alpha_pattern=rank_pattern,
                         lora_dropout=0, bias='none'))
    if ffa:
        for n,p in net.named_parameters():
            if '.lora_A.' in n: p.requires_grad_(False)
    net.eval()
    return net


def predict(net, x):
    return net(context=x).quantile_preds.transpose(1,2).float()


def train_state(net):
    return {n:p.detach().cpu().clone() for n,p in net.named_parameters() if p.requires_grad}


def restore(net, state):
    params = dict(net.named_parameters())
    with torch.no_grad():
        for n,v in state.items(): params[n].copy_(v)


def tensor_hash(items):
    h=hashlib.sha256()
    for n,t in sorted(items):
        t=t.detach().contiguous().cpu()
        h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def frozen_hash(net):
    train_names={n for n,p in net.named_parameters() if p.requires_grad}
    return tensor_hash((n,t) for n,t in net.state_dict().items() if n not in train_names)


def audit(net):
    return {'trainable':{n:{'shape':list(p.shape),'count':p.numel(),'dtype':str(p.dtype)}
                        for n,p in net.named_parameters() if p.requires_grad},
            'count':sum(p.numel() for p in net.parameters() if p.requires_grad),
            'frozen_hash':frozen_hash(net),
            'quantized':{n:{'packed_dtype':str(m.weight.dtype),'packed_shape':list(m.weight.shape),
                           'logical_shape':[m.out_features,m.in_features],
                           'nested':m.weight.quant_state.nested,'type':m.weight.quant_state.quant_type}
                         for n,m in net.named_modules() if isinstance(m,bnb.nn.Linear4bit)},
            'high_precision':{n:str(p.dtype) for n,p in net.named_parameters() if not p.requires_grad and p.dtype!=torch.uint8}}


def batch(values, tuples, with_y=True):
    x=np.stack([values[o-512:o,s] for s,o in tuples])
    xt=torch.from_numpy(x.copy()).to('cuda')
    if not with_y:return xt
    y=np.stack([values[o:o+64,s] for s,o in tuples])
    return xt,torch.from_numpy(y.copy()).to('cuda')


def loss(q,y,sigma):
    q=q.sort(dim=-1).values
    delta=y[:,:,None]-q
    tau=torch.arange(1,10,device=q.device,dtype=torch.float32)/10
    return (2*torch.maximum(tau*delta,(tau-1)*delta)/sigma[:,None,None]).mean()


class PrivateHead(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm=arm
        self.coeff=nn.Parameter(torch.zeros({'F_AFFINE':2,'F_HEAD':128,'F_PERIODIC':7}.get(arm,0),device='cuda'))

    def forward(self,q,origins,sigma):
        c=self.coeff
        if c.numel()==0:return q
        if self.arm=='F_HEAD':return c[:64].exp()[None,:,None]*q+sigma[:,None,None]*c[64:][None,:,None]
        if self.arm=='F_AFFINE':return c[0].exp()*q+sigma[:,None,None]*c[1]
        # Origin is the first target slot. h=0..63 therefore o+h indexes that target.
        h=torch.arange(64,device=q.device,dtype=torch.float32)
        t=origins[:,None]+h[None,:]
        b=c[1]+c[2]*(h/64)+c[3]*torch.sin(2*torch.pi*t/24)+c[4]*torch.cos(2*torch.pi*t/24)
        b=b+c[5]*torch.sin(2*torch.pi*t/168)+c[6]*torch.cos(2*torch.pi*t/168)
        return c[0].exp()*q+sigma[:,None,None]*b[:,:,None]


if __name__=='__main__':
    for quant in [False,True]:
        p,m=load_base(quant=quant)
        x=torch.arange(512,device='cuda',dtype=torch.float32).sin()[None]
        with torch.no_grad():
            a=p.predict(x,prediction_length=64).transpose(1,2).float().cpu()
            b=predict(m,x).cpu()
        assert torch.equal(a,b)
        record=audit(m)
        record['native_parity_max_abs']=float((a-b).abs().max())
        record['linears']={n:{'shape':[v.out_features,v.in_features],'group':group(n)} for n,v in m.named_modules() if isinstance(v,nn.Linear)}
        write(RESULTS/('MODEL_NF4.json' if quant else 'MODEL_BF16.json'),record)
        print(quant,len(record['linears']),len(record['quantized']),flush=True)
        del p,m
        clear()
