"""§10 전체 checkpoint roundtrip — 새 프로세스에서 prediction parity 까지.
"checkpoint inference roundtrip" 이며 "training resume parity" 가 아니다."""
import argparse, json, subprocess, sys, tempfile, hashlib
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build, OUT, frozen_hash, fwd, LORA_TARGETS

def base_state_hash(tp):
    """base(=LoRA 제외) 파라미터 전체 해시. v4 §10 'base model identifier/hash 가능한 범위'."""
    h = hashlib.sha256()
    for n, q in sorted(tp.peft_model.base_model.model.named_parameters()):
        if "lora_" in n: continue
        h.update(n.encode()); h.update(q.detach().numpy().tobytes())
    return h.hexdigest()[:16]
from fixture import make_tasks, raw_rows, task_states, TAU_INIT

TOL_ATOL = 1e-6          # 실행 전 고정
N_CTX, N_OUT = 2, 1

def make_all(tp):
    base = tp.peft_model.base_model.model
    p = base.chronos_config.input_patch_size
    tasks, ctx_len, H = make_tasks(p, N_CTX, N_OUT)
    rows = raw_rows(tasks, ctx_len, H)
    m_task, tv, _ = task_states(tasks, p, N_CTX, N_OUT, ctx_len, H, learned=True, gens=[tp.state_gen, tp.state_gen])
    return rows, m_task, tv

def main():
    ap = argparse.ArgumentParser(description="§10 full checkpoint roundtrip (inference parity)")
    ap.add_argument("--child", action="store_true"); ap.add_argument("--dir", default="")
    a = ap.parse_args()

    if a.child:
        d = Path(a.dir); meta = json.load(open(d/"contract.json"))
        tp, _ = build(seed=meta["lora_seed"], learned_state=True, with_modulator=True, rank=meta["rank"])
        from peft import set_peft_model_state_dict
        r = set_peft_model_state_dict(tp.peft_model, torch.load(d/"adapter.pt", map_location="cpu", weights_only=True))
        tp.state_gen.load_state_dict(torch.load(d/"state_gen.pt", map_location="cpu", weights_only=True), strict=True)
        tp.modulator.load_state_dict(torch.load(d/"modulator.pt", map_location="cpu", weights_only=True), strict=True)
        rows, m_task, tv = make_all(tp)
        c_rows = tp.build_c_rows(m_task, tv, rows[5])
        with torch.no_grad(): q = fwd(tp, *rows[:5], rows[5], N_OUT, c_rows).quantile_preds
        out = {"log_tau": [float(x) for x in tp.state_gen.log_tau.detach()],
               "mod_norm": float(tp.modulator.lin.weight.norm()),
               "frozen_hash": frozen_hash(tp),
               "pred_sha": hashlib.sha256(q.detach().numpy().tobytes()).hexdigest()[:16],
               "pred_sum": float(q.sum()), "c_sha": hashlib.sha256(c_rows.detach().numpy().tobytes()).hexdigest()[:16],
               "hooked": len(tp.time_names), "hook_calls": tp.hook_calls,
               "unexpected_keys": list(getattr(r,"unexpected_keys",[])) if r else [],
               "missing_keys": list(getattr(r,"missing_keys",[])) if r else [],
               "base_hash": base_state_hash(tp)}
        torch.save(q.detach(), d/"child_pred.pt")          # max_abs 비교용 (v4 §10)
        tp.remove_hooks(); print("CHILD " + json.dumps(out)); return True

    tp, _ = build(seed=0, learned_state=True, with_modulator=True)
    # 학습된 상태를 흉내 — 저장·복원 대상이 비영이어야 의미가 있다
    torch.manual_seed(7)
    import peft.tuners.lora as L
    with torch.no_grad():
        for n_, mod in tp.peft_model.base_model.model.named_modules():
            if isinstance(mod, L.Linear): mod.lora_B["default"].weight.normal_(0, 0.01)
        tp.modulator.lin.weight.normal_(0, 0.02)
        tp.state_gen.log_tau += 0.1234
    rows, m_task, tv = make_all(tp)
    c_rows = tp.build_c_rows(m_task, tv, rows[5])
    tp.reset_audit()
    with torch.no_grad(): q = fwd(tp, *rows[:5], rows[5], N_OUT, c_rows).quantile_preds
    parent = {"log_tau": [float(x) for x in tp.state_gen.log_tau.detach()],
              "mod_norm": float(tp.modulator.lin.weight.norm()),
              "frozen_hash": frozen_hash(tp),
              "pred_sha": hashlib.sha256(q.detach().numpy().tobytes()).hexdigest()[:16],
              "pred_sum": float(q.sum()), "c_sha": hashlib.sha256(c_rows.detach().numpy().tobytes()).hexdigest()[:16],
              "hooked": len(tp.time_names), "hook_calls": tp.hook_calls,
              "base_hash": base_state_hash(tp)}
    d = Path(tempfile.mkdtemp(prefix="v4rt_"))
    torch.save(q.detach(), d/"parent_pred.pt")
    from peft import get_peft_model_state_dict
    import chronos, peft
    torch.save(get_peft_model_state_dict(tp.peft_model), d/"adapter.pt")
    torch.save(tp.state_gen.state_dict(), d/"state_gen.pt")
    torch.save(tp.modulator.state_dict(), d/"modulator.pt")
    contract = {"lora_seed": 0, "rank": tp.rank, "lora_targets": LORA_TARGETS,
                "tau_init": TAU_INIT, "time_module_names": sorted(tp.time_names)[:4],
                "n_time_modules": len(tp.time_names), "n_group_modules": len(tp.group_names),
                "token_aggregation_rule": "last valid observed time in patch; REG/all-masked/beyond-H neutral",
                "input_view": "raw command rows (MOD arm)", "loss_contract": "TARGET_ROW_MEAN_CORRECTION v4.2026-09-23",
                "versions": {"chronos": chronos.__version__, "peft": peft.__version__, "torch": torch.__version__},
                "base_model": "amazon/chronos-2"}
    json.dump(contract, open(d/"contract.json","w"), indent=2)
    tp.remove_hooks()
    r = subprocess.run([sys.executable, __file__, "--child", "--dir", str(d)], capture_output=True, text=True, timeout=1800)
    line = [l for l in r.stdout.splitlines() if l.startswith("CHILD ")]
    if not line:
        res = {"pass": False, "error": r.stderr[-800:], "parent": parent}
    else:
        c = json.loads(line[0][6:])
        pp = torch.load(d/"parent_pred.pt", weights_only=True)
        cp = torch.load(d/"child_pred.pt", weights_only=True)
        pred_max_abs = float((pp - cp).abs().max())
        chk = {"log_tau_exact": parent["log_tau"] == c["log_tau"],
               "modulator_exact": abs(parent["mod_norm"]-c["mod_norm"]) == 0.0,
               "frozen_head_same": parent["frozen_hash"] == c["frozen_hash"],
               "prediction_sha_same": parent["pred_sha"] == c["pred_sha"],
               "prediction_within_tol": pred_max_abs <= TOL_ATOL,
               "prediction_exact": pred_max_abs == 0.0,
               "base_weights_same": parent["base_hash"] == c["base_hash"],
               "c_rows_same": parent["c_sha"] == c["c_sha"],
               "hook_coverage_same": parent["hooked"] == c["hooked"] and parent["hook_calls"] == c["hook_calls"],
               "no_unexpected_keys": len(c["unexpected_keys"]) == 0}
        res = {"parent": parent, "child": c, "contract": contract, "tol_atol": TOL_ATOL,
               "prediction_max_abs_diff": pred_max_abs,
               "checks": chk, "pass": all(chk.values()),
               "scope": "checkpoint inference roundtrip. training resume parity 는 별도 검사다."}
        for k,v in chk.items(): print(f"  [{'PASS' if v else 'FAIL'}] {k}", flush=True)
    json.dump(res, open(OUT/"FULL_ROUNDTRIP.json","w"), indent=2, ensure_ascii=False)
    return res["pass"]

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
