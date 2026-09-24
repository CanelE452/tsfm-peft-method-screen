"""Deterministic tensors for wiring checks only; never interpreted as HVAC evidence."""
from __future__ import annotations
import torch
from .model import Task

def make_tasks(p_in=16,p_out=16):
    length,horizon=2*p_in+1,p_out+1
    t=torch.arange(length+horizon,dtype=torch.float32)
    tasks=[]
    for idx,tid in enumerate((19,3)):
        u=torch.zeros_like(t)
        u[p_in//2:]=1.0+idx*0.6
        u[p_in+1:]=-0.4-idx*0.2
        u[length+max(1,p_out//2):]=0.7+idx*0.4
        # These tensor values check differentiability; they are not a proposed data-generation law.
        y=20+idx*3+1.7*torch.sin(t/(3.0+idx))+0.08*t+0.11*u
        tasks.append(Task(tid,y[:length].clone(),u.clone(),y[length:].clone()))
    return tasks


def serialize_tasks(tasks):
    return [{'task_id':t.task_id,'y_context':t.y_context,'command':t.command,'y_future':t.y_future,
             'y_context_mask':t.y_context_mask,'command_mask':t.command_mask,'y_future_mask':t.y_future_mask} for t in tasks]


def restore_tasks(records):
    return [Task(**r) for r in records]
