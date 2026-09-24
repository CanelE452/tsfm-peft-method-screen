"""Fast invariant/negative tests. Uses a test double, never claims native Chronos coverage."""
from __future__ import annotations
import copy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import torch
from torch import nn
from .backend import load_base,resolve_backend
from .model import ModelSpec,TransientModel,ResponseState,OwnedLoRALinear,aggregate_state
from .fixture import make_tasks
from .numerical import check_loss_reduction
from .util import ContractError,state_hash

class Invariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def model(self,arm='LM'):
        base,tt,gt=load_base(resolve_backend('tiny'))
        return TransientModel(base,ModelSpec(arm),tt,gt)
    def test_01_zero_B_preserves_linear(self):
        base=nn.Linear(3,4);m=OwnedLoRALinear(base,2,4,False);x=torch.randn(2,5,3)
        self.assertTrue(torch.equal(m(x),base(x)))
    def test_02_zero_condition_is_plain_not_F0(self):
        base=nn.Linear(1,1,bias=False)
        with torch.no_grad():base.weight.fill_(2)
        m=OwnedLoRALinear(base,1,1,True)
        with torch.no_grad():m.A.weight.fill_(1);m.B.weight.fill_(1)
        m.condition=torch.zeros(1,1,1)
        self.assertEqual(float(m(torch.ones(1,1,1)).detach()),3.)
    def test_03_shape_mismatch_fails(self):
        m=OwnedLoRALinear(nn.Linear(3,4),2,4,True);m.condition=torch.zeros(5,2,2)
        with self.assertRaises(ContractError):m(torch.ones(2,5,3))
    def test_04_no_missing_condition_silent_skip(self):
        m=OwnedLoRALinear(nn.Linear(3,4),2,4,True)
        with self.assertRaises(ContractError):m(torch.ones(2,5,3))
    def test_05_shared_parameter_identity(self):
        m=self.model();ts=make_tasks(m.p_in,m.p_out);m(ts)
        self.assertEqual([n for n,_ in m.named_parameters() if n.startswith('response.')],['response.log_tau'])
        self.assertEqual(m.response.log_tau.numel(),2)
    def test_06_fixed_state_is_buffer(self):
        m=self.model('FI');self.assertEqual(list(m.response.parameters()),[])
        self.assertIn('response.log_tau',m.compact_state())
    def test_07_input_tau_gradient(self):
        m=self.model('LI');m(make_tasks(m.p_in,m.p_out))['loss'].backward()
        self.assertTrue(torch.isfinite(m.response.log_tau.grad).all())
        self.assertGreater(float(m.response.log_tau.grad.norm()),0)
    def test_08_masked_values_do_not_change_state(self):
        r=ResponseState(ModelSpec('LM'));u=torch.arange(12,dtype=torch.float32);mask=torch.ones(12,dtype=torch.bool);mask[3:6]=False
        v=u.clone();v[3:6]=10000
        self.assertTrue(torch.equal(r(u,mask),r(v,mask)))
    def test_09_reg_and_fullmask_neutral(self):
        raw=torch.arange(14,dtype=torch.float32)[None,:].repeat(2,1);mask=torch.ones(14,dtype=torch.bool);mask[1:5]=False
        state,valid,_=aggregate_state(raw,mask,9,5,4,4,True)
        self.assertFalse(bool(valid[1]));self.assertFalse(bool(valid[3]))
        mod=nn.Linear(2,8,bias=False)
        c=mod(state)*valid[:,None]
        self.assertTrue(torch.equal(c[~valid],torch.zeros_like(c[~valid])))
    def test_10_compact_missing_key_rejected(self):
        m=self.model();s=m.compact_state();s.pop(next(iter(s)))
        with self.assertRaises(ContractError):m.load_compact(s)
    def test_11_compact_extra_key_rejected(self):
        m=self.model();s=m.compact_state();s['unexpected']=torch.zeros(1)
        with self.assertRaises(ContractError):m.load_compact(s)
    def test_12_compact_wrong_shape_rejected(self):
        m=self.model();s=m.compact_state();s['response.log_tau']=torch.zeros(99,dtype=torch.float64)
        with self.assertRaises(ContractError):m.load_compact(s)
    def test_13_compact_exact_ownership(self):
        for arm in ('L0','FI','FM','LI','LM'):
            m=self.model(arm);self.assertTrue(set(m.inventory())<=set(m.compact_state()))
    def test_14_group_not_modulated(self):
        m=self.model();self.assertEqual(len(m.scope['time']),8);self.assertEqual(len(m.scope['group']),8)
        self.assertTrue(all(not m._projections[n].modulated for n in m.scope['group']))
    def test_15_future_target_poison(self):
        m=self.model('LI');ts=make_tasks(m.p_in,m.p_out)
        a=m(ts)['predictions'];b=m([replace(t,y_future=t.y_future+999) for t in ts])['predictions']
        self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
    def test_16_response_recomputed_inside_forward(self):
        m=self.model('LI');ts=make_tasks(m.p_in,m.p_out);a=m(ts)['built']['state_tokens'][19].detach().clone()
        # Test-only mutation tests recomputation, never used in lifecycle training.
        with torch.no_grad():m.response.log_tau.add_(.2)
        b=m(ts)['built']['state_tokens'][19].detach()
        self.assertFalse(torch.equal(a,b))
    def test_17_raw_command_present_all_arms(self):
        for arm in ('L0','FI','FM','LI','LM'):
            m=self.model(arm);ts=make_tasks(m.p_in,m.p_out);b=m.build_inputs(ts)
            for t in ts:
                i=b['rows'].index((t.task_id,'command'))
                self.assertTrue(torch.equal(b['kwargs']['context'][i],t.command[:len(t.y_context)]))
                self.assertTrue(torch.equal(b['kwargs']['future_covariates'][i],t.command[len(t.y_context):]))
    def test_18_input_gap_is_explicitly_rejected(self):
        m=self.model('LI');ts=make_tasks(m.p_in,m.p_out);mask=torch.ones_like(ts[0].command,dtype=torch.bool);mask[2]=False
        ts[0]=replace(ts[0],command_mask=mask)
        with self.assertRaises(ContractError):m(ts)
    def test_19_duplicate_id_rejected(self):
        m=self.model();ts=make_tasks(m.p_in,m.p_out);ts[1]=replace(ts[1],task_id=ts[0].task_id)
        with self.assertRaises(ContractError):m(ts)
    def test_20_stale_condition_cleared(self):
        m=self.model();m(make_tasks(m.p_in,m.p_out))
        self.assertFalse(m._busy);self.assertTrue(all(l.condition is None for l in m._projections.values()))
    def test_21_loss_value_and_gradient_invariance(self):
        self.assertEqual(check_loss_reduction()['status'],'PASS')
    def test_22_checkpoint_load_same_object_keys(self):
        m=self.model();s=m.compact_state();m.load_compact(s);self.assertEqual(state_hash(s),state_hash(m.compact_state()))
    def test_23_id_reorder_lookup(self):
        m=self.model('FI');ts=make_tasks(m.p_in,m.p_out)
        a=m(ts)['predictions'];b=m(list(reversed(ts)))['predictions']
        self.assertTrue(all(torch.allclose(a[k],b[k],atol=1e-5,rtol=1e-5) for k in a))
    def test_24_fixed_response_parameters_not_optimizer_candidates(self):
        for arm in ('FI','FM'):
            m=self.model(arm)
            self.assertFalse(any(n.startswith('response.') for n in m.inventory()))
    def test_25_partial_future_patch_uses_last_real_plan(self):
        raw=torch.arange(14,dtype=torch.float32)[None,:];mask=torch.ones(14,dtype=torch.bool)
        st,valid,mapping=aggregate_state(raw,mask,9,5,4,4,True)
        self.assertEqual(float(st[-1,0]),13.);self.assertEqual(mapping[-1]['raw_stop_exclusive'],14)

    def test_26_data_audit_distinguishes_endpoint_and_residual(self):
        import csv,json
        import numpy as np
        from .data_audit import SOURCE,SIGNALS,audit
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);src=repo/SOURCE;src.mkdir(parents=True)
            time=np.arange(181,dtype=float)
            command=np.zeros(181);command[10:110]=1.
            values={name:300+i+4*np.exp(-(time-10)/5) for i,name in enumerate(SIGNALS)}
            header=['time',*SIGNALS,'oveHeaPumY_u']
            for filename in ['fast_probe.csv','transient_response.csv']:
                with (src/filename).open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(header)
                    for i,t in enumerate(time):w.writerow([t,*[values[name][i] for name in SIGNALS],command[i]])
            previous={'per_signal':{name:{'tau_fit_s':5.,'slow_part_span_20min':float(values[name][109]-values[name][10])} for name in SIGNALS}}
            (src/'TAU_FAST.json').write_text(json.dumps(previous))
            result=audit(repo,repo/'audit')
            for item in result['fixed_tau_diagnostics']:
                self.assertTrue(item['span_matches_endpoint_change'])
                self.assertGreater(abs(item['endpoint_change']),3.)
                self.assertLess(item['residual_rmse'],1e-10)
                self.assertFalse(item['physical_modes_identified'])

if __name__=='__main__':unittest.main(verbosity=2)
