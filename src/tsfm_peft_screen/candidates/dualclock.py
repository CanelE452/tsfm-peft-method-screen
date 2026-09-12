import torch
from torch import nn

def event_features(x):
    # At most 336 events, all inside the given window; no external history.
    n,t=x.shape;seq=x.new_zeros(n,t,4);lengths=[];summary=[]
    for i,row in enumerate(x):
        pos=torch.where(torch.isfinite(row)&(row>0))[0];k=len(pos);lengths.append(max(k,1))
        if k:
            gaps=torch.diff(pos,prepend=pos.new_tensor([-1])).float();mag=row[pos]
            seq[i,:k]=torch.stack([pos.float()/336,gaps/336,torch.log1p(mag),((335-pos).float()/336)],-1)
            summary.append(torch.stack([(335-pos[-1]).float()/336,gaps.mean()/336,torch.log1p(mag[-1]),row.new_tensor(k/336)]))
        else:summary.append(row.new_tensor([1,1,0,0]))
    return seq,torch.tensor(lengths,device='cpu'),torch.stack(summary)

class EventAdapter(nn.Module):
    def __init__(self,width,full):
        super().__init__();self.full=full
        self.encoder=nn.GRU(4,32,batch_first=True) if full else nn.Linear(4,32)
        self.down=nn.Linear(width+32,8,bias=False);self.up=nn.Linear(8,width,bias=False);nn.init.zeros_(self.up.weight)
    def forward(self,h,state):
        seq,lengths,summary=state
        if self.full:
            packed=nn.utils.rnn.pack_padded_sequence(seq,lengths,batch_first=True,enforce_sorted=False)
            _,e=self.encoder(packed);e=e[-1]
        else:e=torch.tanh(self.encoder(summary))
        e=e[:,None,:].expand(-1,h.shape[1],-1)
        return h+self.up(torch.nn.functional.silu(self.down(torch.cat([h,e],-1))))
