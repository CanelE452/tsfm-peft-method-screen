"""One owner for LoRA, response parameters, modulation, optimizer and checkpoint.

Standard LoRA: W(x)+(alpha/r)*B(A(x)); A is Kaiming-uniform, B is zero.
Only time-attention LoRA receives modulation. Group-attention LoRA remains plain.
The response bank is SHARED across tasks; response VALUES remain task-specific.
No class/site-packages patch, numpy conversion in the learned input path, or manual
parameter change is used by the training entry point.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from typing import Any
import torch
from torch import nn
from .util import require, ContractError, state_hash

ARMS = ('L0', 'FI', 'FM', 'LI', 'LM')

@dataclass(frozen=True)
class ModelSpec:
    arm: str
    rank: int = 8
    alpha: float = 16.0
    tau_init: tuple[float, ...] = (4.0, 16.0)
    dt: float = 1.0
    tau_min: float = 0.25
    tau_max: float = 256.0
    seed: int = 924510
    scope: str = 'time_only_modulation_all_attention_plain_lora'
    state_sharing: str = 'shared_parameters_task_specific_values'
    loss_contract: str = 'TARGET_ROW_MEAN_CORRECTION_v1'

    def validate(self):
        require(self.arm in ARMS, 'Unknown arm')
        require(self.rank > 0 and self.alpha > 0 and self.dt > 0, 'Invalid LoRA/time config')
        require(all(self.tau_min < t < self.tau_max for t in self.tau_init), 'Invalid initial tau')
        require(self.scope == 'time_only_modulation_all_attention_plain_lora', 'Unsupported scope')
        require(self.state_sharing == 'shared_parameters_task_specific_values', 'Unsupported ownership')
        require(self.loss_contract == 'TARGET_ROW_MEAN_CORRECTION_v1', 'Unsupported loss')

@dataclass
class Task:
    task_id: int
    y_context: torch.Tensor
    command: torch.Tensor           # exactly context + forecast horizon; future portion is a known PLAN
    y_future: torch.Tensor
    y_context_mask: torch.Tensor | None = None
    command_mask: torch.Tensor | None = None
    y_future_mask: torch.Tensor | None = None

    def with_tensors(self, device):
        return Task(self.task_id, self.y_context.to(device), self.command.to(device), self.y_future.to(device),
                    None if self.y_context_mask is None else self.y_context_mask.to(device),
                    None if self.command_mask is None else self.command_mask.to(device),
                    None if self.y_future_mask is None else self.y_future_mask.to(device))

class ResponseState(nn.Module):
    """Signed command-change bank, not a claim of separately identified physical modes.

Missing command segments: no hidden value is used; the first observation after a
missing segment starts a new zero state. Such patches are explicitly masked.
Learned INPUT gap support is deliberately rejected in this readiness release;
this avoids claiming observed-only native scaling for zero-filled state rows.
"""
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.dt, self.lo, self.hi = spec.dt, spec.tau_min, spec.tau_max
        initial = torch.log(torch.tensor(spec.tau_init, dtype=torch.float64))
        if spec.arm.startswith('L') and spec.arm != 'L0':
            self.log_tau = nn.Parameter(initial)
        else:
            self.register_buffer('log_tau', initial)

    def forward(self, command: torch.Tensor, mask: torch.Tensor):
        require(command.ndim == 1 and mask.shape == command.shape, 'Command/mask shape mismatch')
        mask = mask.bool()
        safe = torch.where(mask, command, torch.zeros_like(command))
        require(bool(torch.isfinite(safe).all()), 'Non-finite observed command')
        tau = self.log_tau.clamp(math.log(self.lo), math.log(self.hi)).exp()
        decay = torch.exp(-self.dt / tau)
        prev = torch.zeros_like(tau)
        cols = []
        for t in range(command.numel()):
            if not bool(mask[t]):
                prev = torch.zeros_like(prev)
            elif t == 0 or not bool(mask[t-1]):
                prev = torch.zeros_like(prev)
            else:
                delta = (safe[t] - safe[t-1]).to(tau.dtype)
                prev = decay * prev + delta
            cols.append(prev)
        return torch.stack(cols, dim=-1).to(command.dtype)


def aggregate_state(raw, observed, length, horizon, p_in, p_out, has_reg):
    """Last observed state inside each real patch; no H-extrapolation or implicit REG."""
    nctx, nout = math.ceil(length/p_in), math.ceil(horizon/p_out)
    left = nctx*p_in - length
    states, valid, mapping = [], [], []
    def add(start, end, kind):
        ids = torch.nonzero(observed[start:end].bool(), as_tuple=False).flatten()
        last = int(ids[-1]) + start if ids.numel() else None
        states.append(raw[:, last] if last is not None else raw.new_zeros(raw.shape[0]))
        valid.append(last is not None)
        mapping.append({'kind':kind, 'raw_start':start, 'raw_stop_exclusive':end, 'last_valid':last})
    for k in range(nctx):
        add(max(0,k*p_in-left), min(length,(k+1)*p_in-left), 'context')
    if has_reg:
        states.append(raw.new_zeros(raw.shape[0])); valid.append(False)
        mapping.append({'kind':'REG','raw_start':None,'raw_stop_exclusive':None,'last_valid':None})
    for k in range(nout):
        add(length+k*p_out, min(length+horizon,length+(k+1)*p_out), 'future')
    return torch.stack(states,0), torch.tensor(valid,device=raw.device,dtype=torch.bool), mapping

class OwnedLoRALinear(nn.Module):
    """Explicit standard LoRA implementation, checked against PEFT in the remote run."""
    def __init__(self, base: nn.Linear, rank: int, alpha: float, modulated: bool):
        super().__init__()
        require(isinstance(base, nn.Linear), 'Expected nn.Linear projection')
        self.base = base
        self.base.requires_grad_(False)
        self.A = nn.Linear(base.in_features,rank,bias=False,device=base.weight.device,dtype=base.weight.dtype)
        self.B = nn.Linear(rank,base.out_features,bias=False,device=base.weight.device,dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.A.weight,a=math.sqrt(5)); nn.init.zeros_(self.B.weight)
        self.scale = alpha/rank
        self.modulated = modulated
        self.condition = None
        self.disabled = False
        self.calls = 0
        self.last_shape = None
        self.in_features, self.out_features = base.in_features, base.out_features

    def forward(self, x):
        if self.disabled:
            self.calls += 1
            self.last_shape = list(x.shape[:-1]) + [self.A.out_features]
            return self.base(x)
        a = self.A(x)
        if self.modulated:
            require(self.condition is not None, 'Modulated projection called outside owned forward')
            require(a.ndim == 3 and a.shape == self.condition.shape,
                    f'Modulation shape mismatch: {tuple(a.shape)} != {tuple(self.condition.shape)}')
            a = a * (1+self.condition)
        else:
            require(self.condition is None, 'Plain/group projection received modulation')
        self.calls += 1
        self.last_shape = list(a.shape)
        return self.base(x) + self.scale * self.B(a)

class TransientModel(nn.Module):
    def __init__(self, base, spec: ModelSpec, time_type: type, group_type: type):
        super().__init__()
        spec.validate()
        self.base = base
        self.spec = spec
        base.requires_grad_(False)
        torch.manual_seed(spec.seed)
        cfg = base.chronos_config
        self.p_in, self.p_out = int(cfg.input_patch_size), int(cfg.output_patch_size)
        require(int(cfg.input_patch_stride) == self.p_in, 'Overlapping native patches unsupported')
        self.max_context = int(cfg.context_length)
        self.max_out = int(cfg.max_output_patches)
        self.has_reg = bool(cfg.use_reg_token)
        self.scope = {'time':[], 'group':[]}
        parents = [(n,m,'time' if isinstance(m,time_type) else 'group') for n,m in base.named_modules()
                   if isinstance(m,(time_type,group_type))]
        require(parents and len([x for x in parents if x[2]=='time']) == int(base.config.num_layers),
                'Time attention module count mismatch')
        require(len([x for x in parents if x[2]=='group']) == int(base.config.num_layers),
                'Group attention module count mismatch')
        self._projections = {}   # non-owning lookup: native base remains the sole owner of these modules
        for name, parent, kind in parents:
            for projection in ('q','k','v','o'):
                layer = getattr(parent.self_attention,projection)
                full = name+'.self_attention.'+projection
                modulated = kind == 'time' and spec.arm in ('FM','LM')
                new = OwnedLoRALinear(layer,spec.rank,spec.alpha,modulated)
                setattr(parent.self_attention,projection,new)
                self._projections[full] = new
                self.scope[kind].append(full)
        self.response = None if spec.arm == 'L0' else ResponseState(spec)
        self.modulator = None
        if spec.arm in ('FM','LM'):
            self.modulator = nn.Linear(len(spec.tau_init),spec.rank,bias=False,
                                       device=next(base.parameters()).device,dtype=torch.float32)
            nn.init.zeros_(self.modulator.weight)
        if self.response is not None:
            self.response.to(next(base.parameters()).device)
        self._busy = False
        self.last_debug = {}
        self.eval()  # dropout OFF in the readiness trainer; gradients are still enabled explicitly.

    def native(self, tasks, row_order=None):
        """Same input view, before/without LoRA. Used ONLY as an initial reference."""
        built = self.build_inputs(tasks,row_order)
        # Disabled mode is used only with B exactly zero in initial parity tests.
        return built

    def build_inputs(self, tasks: list[Task], row_order=None):
        require(len(tasks)>0 and len({x.task_id for x in tasks})==len(tasks), 'Duplicate or empty tasks')
        length, horizon = tasks[0].y_context.numel(), tasks[0].y_future.numel()
        require(length<=self.max_context and horizon>0 and math.ceil(horizon/self.p_out)<=self.max_out,
                'Only bounded direct forecasting supported')
        records, state_tokens, state_valid, mappings = {}, {}, {}, {}
        for task in tasks:
            y,u,ft = task.y_context, task.command, task.y_future
            require(y.ndim==u.ndim==ft.ndim==1 and len(y)==length and len(ft)==horizon and len(u)==length+horizon,
                    'All fixture tasks require common context/horizon')
            ym = torch.isfinite(y) if task.y_context_mask is None else task.y_context_mask.bool()
            um = torch.isfinite(u) if task.command_mask is None else task.command_mask.bool()
            fm = torch.isfinite(ft) if task.y_future_mask is None else task.y_future_mask.bool()
            require(ym.shape==y.shape and um.shape==u.shape and fm.shape==ft.shape, 'Mask shape mismatch')
            require(bool(ym.any()) and bool(um[:length].any()) and bool(fm.any()), 'Empty target/command support')
            require(bool(torch.isfinite(y[ym]).all()) and bool(torch.isfinite(u[um]).all()) and bool(torch.isfinite(ft[fm]).all()),
                    'Nonfinite unmasked input')
            # Original raw observations use NaN for native observed-only normalization.
            yc = torch.where(ym,y,torch.full_like(y,float('nan')))
            uc = torch.where(um[:length],u[:length],torch.full_like(y,float('nan')))
            uf = torch.where(um[length:],u[length:],torch.zeros_like(ft))
            tf = torch.where(fm,ft,torch.zeros_like(ft))
            zero, ones = torch.zeros_like(ft), torch.ones_like(ft,dtype=torch.bool)
            records[(task.task_id,'target')] = (yc,ym,zero,~ones,tf,fm)
            records[(task.task_id,'command')] = (uc,um[:length],uf,um[length:],zero,~ones)
            if self.response is not None:
                raw = self.response(u,um)
                agg, valid, mapping = aggregate_state(raw,um,length,horizon,self.p_in,self.p_out,self.has_reg)
                state_tokens[task.task_id], state_valid[task.task_id], mappings[task.task_id] = agg,valid,mapping
                if self.spec.arm in ('FI','LI'):
                    require(bool(um.all()), 'INPUT response features with command gaps are not supported in v5; no silent zero-fill')
                    for j in range(raw.shape[0]):
                        records[(task.task_id,f'state{j}')] = (raw[j,:length],torch.ones_like(ym),raw[j,length:],ones,zero,~ones)
        order = list(records) if row_order is None else [tuple(x) for x in row_order]
        require(len(order)==len(records) and set(order)==set(records), 'Row order does not cover exact rows')
        cols = [torch.stack([records[k][i] for k in order],0) for i in range(6)]
        kwargs = dict(context=cols[0],context_mask=cols[1].float(),future_covariates=cols[2],
                      future_covariates_mask=cols[3].float(),future_target=cols[4],future_target_mask=cols[5].float(),
                      group_ids=torch.tensor([k[0] for k in order],device=cols[0].device,dtype=torch.long),
                      num_output_patches=math.ceil(horizon/self.p_out))
        c_rows = None
        valid_rows = None
        if self.modulator is not None:
            cr, vr = [], []
            for key in order:
                st, valid = state_tokens[key[0]], state_valid[key[0]].clone()
                # Context tokens with no observation in THIS row are also neutral.
                obs = records[key][1]
                nctx = math.ceil(length/self.p_in); pad=nctx*self.p_in-length
                for k in range(nctx):
                    s,e=max(0,k*self.p_in-pad),min(length,(k+1)*self.p_in-pad)
                    valid[k] &= obs[s:e].any()
                c = self.modulator(st.to(self.modulator.weight.dtype))
                cr.append(c*valid[:,None]); vr.append(valid)
            c_rows,valid_rows=torch.stack(cr),torch.stack(vr)
        return {'kwargs':kwargs,'rows':order,'c':c_rows,'valid':valid_rows,
                'state_tokens':state_tokens,'mappings':mappings,'horizon':horizon}

    def forward(self, tasks: list[Task], row_order=None):
        require(not self._busy, 'Reentrant/concurrent forward unsupported')
        built=self.build_inputs(tasks,row_order)
        self._busy=True
        try:
            for layer in self._projections.values():
                layer.calls=0
                if layer.modulated: layer.condition=built['c']
            out=self.base(**built['kwargs'])
            for name,layer in self._projections.items():
                require(layer.calls==1,f'Projection coverage {name}: calls={layer.calls}')
            active=(built['kwargs']['future_target_mask'].bool() & ~built['kwargs']['future_covariates_mask'].bool()).any(-1)
            require(int(active.sum())==len(tasks),'Not exactly one active target row per task')
            corrected=out.loss*(len(built['rows'])/int(active.sum()))
            by_task={tid:out.quantile_preds[i,:,:built['horizon']] for i,(tid,role) in enumerate(built['rows']) if role=='target'}
            self.last_debug={'rows':built['rows'],'mappings':built['mappings'],
                             'states':{k:v.detach().cpu().clone() for k,v in built['state_tokens'].items()},
                             'c':None if built['c'] is None else built['c'].detach().cpu().clone(),
                             'valid':None if built['valid'] is None else built['valid'].detach().cpu().clone(),
                             'shapes':{n:l.last_shape for n,l in self._projections.items()},
                             'active_target_rows':int(active.sum()),'total_rows':len(built['rows'])}
            return {'loss':corrected,'native_loss':out.loss,'predictions':by_task,'built':built}
        finally:
            for layer in self._projections.values(): layer.condition=None
            self._busy=False

    def compact_keys(self):
        keys=set()
        for name,layer in self._projections.items():
            keys.update({'base.'+name+'.A.weight','base.'+name+'.B.weight'})
        if self.response is not None:
            keys.update('response.'+n for n in self.response.state_dict())
        if self.modulator is not None:
            keys.update('modulator.'+n for n in self.modulator.state_dict())
        trainable={n for n,p in self.named_parameters() if p.requires_grad}
        require(trainable<=keys,'Trainable parameter missing from compact ownership')
        return keys

    def compact_state(self):
        full=self.state_dict()
        keys=self.compact_keys()
        require(keys<=set(full),'Compact key not found')
        return {k:full[k].detach().cpu().clone() for k in sorted(keys)}

    def load_compact(self,state):
        dest=self.state_dict(); expected=self.compact_keys()
        require(set(state)==expected,f'Checkpoint keys mismatch: missing={sorted(expected-set(state))}; extra={sorted(set(state)-expected)}')
        with torch.no_grad():
            for k in sorted(expected):
                require(state[k].shape==dest[k].shape and state[k].dtype==dest[k].dtype,f'Checkpoint tensor mismatch {k}')
                dest[k].copy_(state[k].to(dest[k].device))
        require(state_hash(self.compact_state())==state_hash(state),'Checkpoint tensor mismatch after copy')

    def frozen_state_hash(self):
        return state_hash({n:p for n,p in self.named_parameters() if not p.requires_grad and n.startswith('base.')})

    def inventory(self):
        out={}
        for n,p in self.named_parameters():
            if p.requires_grad:
                category='state' if n.startswith('response.') else 'modulator' if n.startswith('modulator.') else 'lora'
                out[n]={'category':category,'shape':list(p.shape),'numel':p.numel(),'dtype':str(p.dtype)}
        return out
