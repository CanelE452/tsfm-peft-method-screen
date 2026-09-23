"""§4.3 토큰 trace — 실제 형상을 hook 으로 관찰한다. 가상 fixture 전용, 실데이터 아님.

이 스크립트는 구현 배선 검사이며 성능 실험이 아니다.
사용: python trace_tokens.py            (CPU, forward 1회)
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np, torch

OUT = Path(__file__).resolve().parents[2] / "results/transient_peft_readiness_v3_20260923"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = {"model_load": 0, "forward": 0, "backward": 0, "update": 0}

def main():
    from chronos import Chronos2Pipeline
    import chronos.chronos2.model as cmod

    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu", dtype=torch.float32)
    LEDGER["model_load"] += 1
    m = pipe.inner_model
    cfg = m.chronos_config
    p = cfg.input_patch_size
    rec = {"config": {k: getattr(cfg, k) for k in
           ["context_length","input_patch_size","input_patch_stride","output_patch_size",
            "max_output_patches"] if hasattr(cfg, k)}}
    rec["config"]["quantiles_n"] = len(pipe.quantiles)
    rec["config"]["d_model"] = int(m.config.d_model) if hasattr(m.config,"d_model") else None
    print(f"[config] patch_size={p} {rec['config']}", flush=True)

    # ---- §4.2 fixture: 패치 배수가 아닌 context, padding 유발 horizon ----
    ctx_len = 2*p + 1          # 배수 아님
    H = p + 1                  # padding 유발
    rng = np.random.default_rng(20260923)
    def task(seed_shift, n_cov):
        g = np.random.default_rng(20260923 + seed_shift)
        tgt = (np.sin(np.arange(ctx_len)/5.0) * 3 + 20 + g.normal(0, .1, ctx_len)).astype(np.float32)
        d = {"target": tgt}
        if n_cov:
            past = g.normal(0, 1, (n_cov, ctx_len)).astype(np.float32)
            fut  = g.normal(0, 1, (n_cov, H)).astype(np.float32)
            d["past_covariates"]   = {f"u{j}": past[j] for j in range(n_cov)}
            d["future_covariates"] = {f"u{j}": fut[j]  for j in range(n_cov)}
        return d
    inputs = [task(0, 2), task(1, 2)]      # 독립 task 2개, 각 covariate 2행
    rec["fixture"] = {"n_tasks": 2, "ctx_len": ctx_len, "H": H,
                      "note": f"ctx=2p+1={ctx_len}, H=p+1={H}. 가상 텐서이며 설비 데이터가 아니다.",
                      "n_cov_per_task": 2}

    # ---- hook 으로 실제 형상 관찰 ----
    seen = {}
    hooks = []
    def mk(name):
        def f(mod, inp, out):
            def shp(x):
                if torch.is_tensor(x): return list(x.shape)
                if isinstance(x,(tuple,list)): return [shp(y) for y in x]
                return str(type(x).__name__)
            if name not in seen:
                seen[name] = {"module": type(mod).__name__, "in": shp(inp), "out": shp(out)}
        return f
    named = dict(m.named_modules())
    targets = ["instance_norm", "input_patch_embedding", "output_patch_embedding", "encoder"]
    for t in targets:
        if t in named: hooks.append(named[t].register_forward_hook(mk(t)))
    # 첫 encoder block 의 attention projection
    for n_, mod in named.items():
        if n_.endswith("block.0.layer.0.self_attention.q") or n_.endswith("block.0.layer.0.self_attention.o"):
            hooks.append(mod.register_forward_hook(mk(n_)))
            rec.setdefault("proj_dims", {})[n_] = {"in_features": mod.in_features, "out_features": mod.out_features}

    with torch.no_grad():
        q, _ = pipe.predict_quantiles(inputs, prediction_length=H,
                                      quantile_levels=list(pipe.quantiles), batch_size=2)
    LEDGER["forward"] += 1
    for h in hooks: h.remove()
    rec["traced_shapes"] = seen
    rec["prediction"] = {"n_returned": len(q), "shape_per_item": list(q[0].shape),
                         "axes_meaning": "확인 필요 — 아래 axis_check 참조"}
    # 분위수 축이 어디서 생기는지: 출력 projection out_features 로 검산
    op = named.get("output_patch_embedding")
    if op is not None:
        last = [mm for nn, mm in op.named_modules() if isinstance(mm, torch.nn.Linear)]
        if last:
            rec["output_projection_out_features"] = last[-1].out_features
            rec["axis_check"] = {
                "out_features": last[-1].out_features,
                "n_quantiles * output_patch_size": len(pipe.quantiles) * cfg.output_patch_size,
                "match": last[-1].out_features == len(pipe.quantiles) * cfg.output_patch_size,
                "해석": "일치하면 분위수 축은 출력 projection 에서 생긴다(§2-C)"}
    rec["ledger"] = dict(LEDGER)
    json.dump(rec, open(OUT/"TOKEN_TRACE.json","w"), indent=2, ensure_ascii=False)
    print(json.dumps(rec, indent=2, ensure_ascii=False)[:2400], flush=True)

if __name__ == "__main__":
    main()
