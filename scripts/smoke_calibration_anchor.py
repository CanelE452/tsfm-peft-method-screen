"""Train-only GPU smoke: production objectives, zero-init identity, real updates."""
import fcntl
import gc
import gzip
import json
import time
import numpy as np
import pandas as pd
import torch
from run_calibration_anchor_followup import CFG, TOPICS, PilotModel, batch, trainable, parameters, tensor_hash, frozen_hash, objective
from tsfm_peft_screen.overnight.methods import mixture_quantiles, teacher_weights
from tsfm_peft_screen.calibration_anchor.method import calibration_weights
from tsfm_peft_screen.forecast_query.gpu import GPUWatch
from tsfm_peft_screen.reproducibility import ROOT, sha, write_json, seed_all, guard

def main():
    lock = open(ROOT/'.cache/gpu.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    folder = ROOT/'.cache/calibration_anchor_smoke'
    folder.mkdir(exist_ok=True)
    watch = GPUWatch(folder/'gpu.json')
    watch.wait_idle()
    source = json.loads((ROOT/CFG['data_receipt']).read_text())
    values = pd.read_csv(ROOT/source['datasets']['etth1']['file']).iloc[:10000, 1:5].to_numpy(dtype=np.float32)
    # Prefix only; no validation or evaluation targets accessed or scored.
    scale = values[2048:9520].std(0, dtype=np.float64).clip(1e-6)
    x, y, g = batch(values, [4096, 6144])
    channel_scale = torch.tensor(np.tile(scale, 2), device='cuda', dtype=torch.float32)
    records = []
    for topic, arms in TOPICS.items():
        for arm in arms:
            watch.check('smoke_'+arm, True)
            guard()
            seed_all(31000)
            model = PilotModel(arm, 31000)
            ps = trainable(model)
            before = parameters(model)
            base_hash = frozen_hash(model)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                _, p0, _, _ = model(x, g, frozen=True)
                _, pi, _, _ = model(x, g)
                assert torch.equal(p0, pi), (arm, 'initial F0 parity')
                components = []
                for context in CFG['teacher_contexts']:
                    _, pc, _, _ = model(x[:, -context:], g, frozen=True)
                    components.append(pc.cpu().numpy().reshape(2, 4, 21, 48))
            components = np.stack(components)
            teacher_prediction = mixture_quantiles(components) if topic == 'distill' else p0.cpu().numpy().reshape(2,4,21,48)
            teacher = {'train': {'prediction': teacher_prediction, 'weights': teacher_weights(components, scale)}}
            weights,_ = calibration_weights(teacher_prediction,y.cpu().numpy().reshape(2,4,48))
            teacher['train']['calibration_weights'] = weights
            initial_gpu = {n: p.cuda() for n, p in before.items()} if arm == 'l2_anchor' else None
            optimizer = torch.optim.AdamW(ps.values(), lr=1e-4, weight_decay=0.)
            steps = []
            for index in range(2):
                watch.check('smoke_update_'+arm)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z, p, loc, ms = model(x, g)
                loss, regularizer = objective(arm,z,p,y,loc,ms,channel_scale,ps,initial_gpu,teacher,np.array([0,1]))
                total = loss+regularizer
                assert torch.isfinite(total)
                total.backward()
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in ps.values())
                if index == 1 and arm.endswith('gate'):
                    assert sum(float(p.grad.abs().sum()) for n,p in ps.items() if '.condition.' in n) > 0
                torch.nn.utils.clip_grad_norm_(list(ps.values()), 1., error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                steps.append(dict(seconds=time.perf_counter()-tick, loss=float(loss.detach()),
                    regularizer=float(regularizer.detach()), peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            assert tensor_hash(before) != tensor_hash(parameters(model))
            assert base_hash == frozen_hash(model)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                _, p1, _, _ = model(x,g)
            assert not torch.equal(p0,p1)
            records.append(dict(topic=topic,arm=arm,zero_init_bitexact=True,frozen_parameters_unchanged=True,
                trained_parameters_changed=True,trained_prediction_changed=True,
                parameters=sum(p.numel() for p in ps.values()),steps=steps))
            print('SMOKE PASS',topic,arm,flush=True)
            del model, optimizer, ps, before, initial_gpu
            gc.collect()
            torch.cuda.empty_cache()
    paths = ['src/tsfm_peft_screen/calibration_anchor/method.py','scripts/run_calibration_anchor_followup.py','scripts/smoke_calibration_anchor.py','configs/calibration_anchor_20260914.json']
    receipt=dict(status='PASS',updates=14,records=records,source_hashes={p:sha(ROOT/p) for p in paths},
        scope='ETTh1 training prefix only; same production objectives; no fit/V/E selection.',
        gpu_log_sha256=sha(folder/'gpu.json'))
    write_json(ROOT/'research/calibration_anchor_20260914/smoke.json',receipt)

if __name__ == '__main__':
    main()
