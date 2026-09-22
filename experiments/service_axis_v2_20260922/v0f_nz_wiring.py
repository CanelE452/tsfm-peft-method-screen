"""V0f: 클래스 수준 NZ 교체가 추론·학습 양쪽에 적용되는지. 반드시 별도 프로세스에서 실행.
사용: python v0f_nz_wiring.py {orig|nz}   -> JSON 한 줄을 stdout 마지막에 출력"""
import sys, json
import numpy as np, torch

MODE = sys.argv[1]
assert MODE in ("orig", "nz")

import chronos.chronos_bolt as cb
_ORIG_FWD = cb.InstanceNorm.forward

def _nz_forward(self, x, loc_scale=None):
    """논문 patch_instancenorm 을 클래스 수준으로. context 안 양수만으로 loc/scale."""
    od = x.dtype; xf = x.to(torch.float32)
    if loc_scale is None:
        nan = torch.isnan(xf); xc = torch.nan_to_num(xf, nan=0.0)
        nzm = (xc > 0) & (~nan)
        cnt = nzm.float().sum(-1, keepdim=True).clamp(min=1.0)
        loc = (xc * nzm.float()).sum(-1, keepdim=True) / cnt
        sc = (((xc - loc) * nzm.float()).square().sum(-1, keepdim=True) / cnt.clamp(min=1.0)).sqrt()
        allz = (nzm.float().sum(-1) == 0)
        if allz.any():
            ol = torch.nan_to_num(torch.nanmean(xf, dim=-1, keepdim=True), nan=0.0)
            os_ = torch.nan_to_num((xf - ol).square().nanmean(-1, keepdim=True).sqrt(), nan=1.0)
            loc = torch.where(allz.unsqueeze(-1), ol, loc); sc = torch.where(allz.unsqueeze(-1), os_, sc)
        sc = torch.where(sc == 0, torch.tensor(self.eps, device=sc.device, dtype=sc.dtype), sc)
        loc_scale = (loc, sc)
    l, s = loc_scale; zz = (xf - l) / s
    if self.use_arcsinh: zz = torch.arcsinh(zz)
    return zz.to(od), (l, s)

if MODE == "nz":
    cb.InstanceNorm.forward = _nz_forward          # 클래스 수준 교체

from chronos import Chronos2Pipeline
out = {"mode": MODE}

pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda", dtype=torch.bfloat16)
probe = torch.tensor([[0., 0., 2., 4.]], dtype=torch.float32).cuda()

def probe_loc(model):
    inz = model.instance_norm
    with torch.no_grad():
        _, (loc, sc) = inz.forward(probe)
    return float(loc.flatten()[0]), float(sc.flatten()[0])

out["infer_loc"], out["infer_scale"] = probe_loc(pipe.inner_model)

# fit 이 새로 만드는 모델에도 적용되는가
ctx = np.random.default_rng(0).poisson(1.0, size=(64, 400)).astype(np.float32)
inputs = [torch.tensor(r) for r in ctx]
ft = pipe.fit(inputs, prediction_length=28, finetune_mode="lora",
              learning_rate=1e-4, num_steps=2, batch_size=8,
              output_dir="/tmp/v0f_" + MODE, remove_printer_callback=True, seed=0)
out["train_loc"], out["train_scale"] = probe_loc(ft.inner_model)
out["same_class"] = (type(ft.inner_model.instance_norm).__name__ == "InstanceNorm")
print("RESULT " + json.dumps(out))
