from contextlib import contextmanager
import math
import torch
from torch import nn
from chronos.chronos2.layers import Chronos2RotaryEmbedding
from ..backbone import load_base
from ..lora import attach,disabled
from ..memory.compression import checkpoint_blocks

class Head(nn.Module):
    def __init__(self):
        super().__init__();self.a=nn.Linear(768,768,bias=False);self.b=nn.Linear(768,768,bias=False);nn.init.zeros_(self.b.weight)
    def forward(self,h):return h+self.b(nn.functional.silu(self.a(nn.functional.layer_norm(h,(768,)))))

class SideLayer(nn.Module):
    def __init__(self):
        super().__init__();self.lateral=nn.Linear(768,64,bias=False)
        self.q=nn.Linear(64,64,bias=False);self.k=nn.Linear(64,64,bias=False);self.v=nn.Linear(64,64,bias=False);self.o=nn.Linear(64,64,bias=False)
        self.a=nn.Linear(64,224,bias=False);self.b=nn.Linear(224,64,bias=False)
    def attend(self,h,mask):
        shape=lambda v:v.reshape(*v.shape[:2],4,16).transpose(1,2)
        q,k,v=[shape(layer(h)) for layer in [self.q,self.k,self.v]]
        a=nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        return self.o(a.transpose(1,2).reshape(*h.shape[:2],64))
    def forward(self,h,frozen,time_mask,group_mask):
        h=(h+self.lateral(frozen))/math.sqrt(2)
        h=h+self.attend(nn.functional.layer_norm(h,(64,)),time_mask)
        trans=h.transpose(0,1);trans=trans+self.attend(nn.functional.layer_norm(trans,(64,)),group_mask);h=trans.transpose(0,1)
        return h+self.b(nn.functional.silu(self.a(nn.functional.layer_norm(h,(64,)))))

class Side(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([SideLayer() for _ in range(12)]);self.out=nn.Linear(64,768,bias=False);nn.init.zeros_(self.out.weight)
    def forward(self,features,time_mask,group_mask,base):
        h=torch.zeros(*features[0].shape[:2],64,device=base.device,dtype=base.dtype)
        for layer,f in zip(self.layers,features):h=layer(h,f,time_mask,group_mask)
        return base+self.out(nn.functional.layer_norm(h[:,-3:],(64,)))

class ForecastModel(nn.Module):
    def __init__(self,arm,seed=30000):
        super().__init__();torch.manual_seed(seed);self.arm=arm;self.base=load_base()
        if arm in ['standard','query']:attach(self.base,seed=seed)
        else:
            torch.manual_seed(seed+2000);self.adapter=(Head() if arm=='head' else Side()).to('cuda')
        count=sum(p.numel() for p in self.parameters() if p.requires_grad)
        assert count==1179648,(arm,count)
    def encode_frozen(self,x,g):
        m=self.base;cache=dict(k=[],v=[],features=[]);handles=[]
        def pre(module,args,kwargs):
            if 'initial' not in cache:
                cache['initial']=args[0][:,-4:].detach().clone()
                cache['positions']=kwargs['position_ids'];cache['time_mask']=kwargs['attention_mask'];cache['group_mask']=kwargs['group_time_mask']
        if self.arm in ['query','side']:
            handles.append(m.encoder.block[0].register_forward_pre_hook(pre,with_kwargs=True))
        if self.arm=='query':
            for block in m.encoder.block:
                for kind in ['k','v']:
                    def save(module,args,out,key=kind):cache[key].append(out[:,:-4].detach().clone())
                    handles.append(getattr(block.layer[0].self_attention,kind).register_forward_hook(save))
        elif self.arm=='side':
            for block in m.encoder.block:
                def save(module,args,out):cache['features'].append(out[0].detach())
                handles.append(block.register_forward_hook(save))
        try:
            with torch.no_grad():
                if self.arm=='query':
                    with disabled(m):enc,(loc,scale),_,n=m.encode(context=x,group_ids=g,num_output_patches=3)
                else:enc,(loc,scale),_,n=m.encode(context=x,group_ids=g,num_output_patches=3)
        finally:
            for h in handles:h.remove()
        assert n==(x.shape[-1]+15)//16
        cache['base']=enc.last_hidden_state[:,-3:].detach();return cache,loc,scale
    def query_hidden(self,c):
        h=c['initial'];m=self.base
        assert len(c['k'])==len(c['v'])==12
        for i,block in enumerate(m.encoder.block):
            layer=block.layer[0];a=layer.self_attention;norm=layer.layer_norm(h)
            shape=lambda v:v.reshape(v.shape[0],v.shape[1],a.n_heads,a.kv_proj_dim).transpose(1,2)
            q=shape(a.q(norm));k=shape(torch.cat((c['k'][i],a.k(norm)),dim=1));v=shape(torch.cat((c['v'][i],a.v(norm)),dim=1))
            cos,sin=a.rope_embed(v,c['positions']);cos=cos.unsqueeze(1);sin=sin.unsqueeze(1)
            q=q*cos[:,:,-4:]+Chronos2RotaryEmbedding.rotate_half(q)*sin[:,:,-4:]
            k=k*cos+Chronos2RotaryEmbedding.rotate_half(k)*sin
            out=nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=c['time_mask'],scale=1.)
            h=h+a.o(out.transpose(1,2).reshape(h.shape[0],4,768))
            h=block.layer[1](h,attention_mask=c['group_mask'][-4:])[0]
            h=block.layer[2](h)
        return m.encoder.final_layer_norm(h)[:,-3:]
    def forward(self,x,g):
        if self.arm=='standard':
            with checkpoint_blocks(self.base):enc,(loc,scale),_,n=self.base.encode(context=x,group_ids=g,num_output_patches=3)
            assert n==(x.shape[-1]+15)//16;h=enc.last_hidden_state[:,-3:]
        else:
            c,loc,scale=self.encode_frozen(x,g)
            if self.arm=='query':h=self.query_hidden(c)
            elif self.arm=='head':h=self.adapter(c['base'])
            else:h=self.adapter(c['features'],c['time_mask'],c['group_mask'],c['base'])
        z=self.base.output_patch_embedding(h).reshape(len(x),3,21,16).permute(0,2,1,3).reshape(len(x),21,48).float()
        p=z.sinh()*scale[:,None,:]+loc[:,None,:]
        return z,p,loc,scale
    def f0(self,x,g):
        with torch.no_grad():
            if self.arm in ['standard','query']:
                with disabled(self.base):enc,(l,s),_,_=self.base.encode(context=x,group_ids=g,num_output_patches=3)
            else:enc,(l,s),_,_=self.base.encode(context=x,group_ids=g,num_output_patches=3)
            h=enc.last_hidden_state[:,-3:];z=self.base.output_patch_embedding(h).reshape(len(x),3,21,16).permute(0,2,1,3).reshape(len(x),21,48).float()
            return z,z.sinh()*s[:,None,:]+l[:,None,:],l,s
