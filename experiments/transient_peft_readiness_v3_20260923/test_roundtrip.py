"""§7 저장/복원 roundtrip — custom tau 가 별도 경로로 보존되는지. 배선 검사 전용.
사용: python test_roundtrip.py          (--help 지원)
"""
import argparse, json, hashlib, subprocess, sys, tempfile
from pathlib import Path
import torch, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_state_gradient import ResponseState
OUT = Path(__file__).resolve().parents[2] / "results/transient_peft_readiness_v3_20260923"

def build(seed=0):
    from chronos import Chronos2Pipeline
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu", dtype=torch.float32)
    m = pipe.inner_model
    for q in m.parameters(): q.requires_grad_(False)
    lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                      target_modules=["self_attention.q","self_attention.k",
                                      "self_attention.v","self_attention.o"])
    return get_peft_model(m, lcfg), pipe

def frozen_hash(peft_model):
    """native 출력층 가중치 해시 — 동결 보존 확인용"""
    base = peft_model.base_model.model
    hs = hashlib.sha256()
    for n_, q in base.output_patch_embedding.named_parameters():
        hs.update(q.detach().cpu().numpy().tobytes())
    return hs.hexdigest()[:16]

def main():
    ap = argparse.ArgumentParser(description="§7 save/restore roundtrip 검사 (배선 전용, 성능 실험 아님)")
    ap.add_argument("--child", action="store_true", help="새 프로세스 복원 측에서 내부적으로 사용")
    ap.add_argument("--dir", type=str, default="")
    a = ap.parse_args()

    if a.child:
        d = Path(a.dir)
        pm, _ = build()
        from peft import set_peft_model_state_dict
        sd = torch.load(d/"adapter.pt", map_location="cpu", weights_only=True)
        keys = set_peft_model_state_dict(pm, sd)
        rs = ResponseState([4.0,16.0], learned=True)
        cs = torch.load(d/"state_gen.pt", map_location="cpu", weights_only=True)
        missing, unexpected = rs.load_state_dict(cs, strict=True)  # strict=True
        res = {"child_log_tau": rs.log_tau.detach().tolist(),
               "child_frozen_hash": frozen_hash(pm),
               "adapter_unexpected_keys": list(getattr(keys,"unexpected_keys",[])) if keys else []}
        print("CHILD " + json.dumps(res)); return

    pm, pipe = build()
    h0 = frozen_hash(pm)
    rs = ResponseState([4.0,16.0], learned=True)
    with torch.no_grad(): rs.log_tau += 0.1234        # 학습됐다고 가정한 변경
    d = Path(tempfile.mkdtemp(prefix="rt_"))
    from peft import get_peft_model_state_dict
    torch.save(get_peft_model_state_dict(pm), d/"adapter.pt")
    torch.save(rs.state_dict(), d/"state_gen.pt")
    parent = {"parent_log_tau": rs.log_tau.detach().tolist(), "parent_frozen_hash": h0,
              "saved_keys": {"adapter": len(get_peft_model_state_dict(pm)), "state_gen": list(rs.state_dict().keys())}}
    r = subprocess.run([sys.executable, __file__, "--child", "--dir", str(d)],
                       capture_output=True, text=True, timeout=900)
    line = [l for l in r.stdout.splitlines() if l.startswith("CHILD ")]
    if not line:
        out = {**parent, "child": None, "error": r.stderr[-600:], "pass": False}
    else:
        child = json.loads(line[0][6:])
        same_tau = np.allclose(parent["parent_log_tau"], child["child_log_tau"], atol=0, rtol=0)
        same_hash = parent["parent_frozen_hash"] == child["child_frozen_hash"]
        out = {**parent, **child,
               "tau_exact_match": bool(same_tau), "frozen_hash_preserved": bool(same_hash),
               "strict_load_ok": True,
               "pass": bool(same_tau and same_hash),
               "note": ("custom tau 는 peft adapter 경로로 저장되지 않으므로 별도 파일이 필요하다. "
                        "이 검사는 그 별도 경로가 새 프로세스에서 정확히 복원되는지만 본다.")}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    json.dump(out, open(OUT/"ROUNDTRIP.json","w"), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
