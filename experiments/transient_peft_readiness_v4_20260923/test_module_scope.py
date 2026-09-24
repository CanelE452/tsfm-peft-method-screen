"""§2 module scope — TimeSelfAttention 에만 modulation, Group 은 plain LoRA. silent skip 0."""
import json, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build, OUT, fwd
from fixture import make_tasks, raw_rows, task_states

def main():
    tp, pipe = build(with_modulator=True)
    base = tp.peft_model.base_model.model
    p = base.chronos_config.input_patch_size
    nl = base.config.num_layers
    tasks, ctx_len, H = make_tasks(p, n_ctx_patch=2, n_out_patch=1)
    ctx, fc, fcm, ft, ftm, gids, meta = raw_rows(tasks, ctx_len, H)
    m_task, tv, _ = task_states(tasks, p, 2, 1, ctx_len, H)
    with torch.no_grad(): tp.modulator.lin.weight.normal_(0, 0.02)   # 비영 c
    c_rows = tp.build_c_rows(m_task, tv, gids)
    tp.reset_audit()
    with torch.no_grad(): fwd(tp, ctx, fc, fcm, ft, ftm, gids, 1, c_rows)
    aud = tp.audit()
    hit = sorted({a["module"] for a in aud})
    res = {"num_layers": nl,
           "expected_time_modules": nl*4,
           "time_lora_modules": len(tp.time_names), "group_lora_modules": len(tp.group_names),
           "total_lora_modules": len(tp.time_names)+len(tp.group_names),
           "hooked_modules": len(tp.time_names),
           "hook_invocations": tp.hook_calls,
           "modules_actually_modulated": len(hit),
           "skipped": len(tp.time_names) - len(hit),
           "group_modulated": sorted(set(hit) & tp.group_names),
           # v4 §2: module names 전체 목록과 각 module 의 실제 A out / c shape 를 남긴다
           "time_module_names": sorted(tp.time_names),
           "group_module_names": sorted(tp.group_names),
           "all_lora_module_names": sorted(set(tp.time_names) | set(tp.group_names)),
           "modulated_module_names": hit,
           "per_module_shapes": [{"module": a["module"], "a_out": a["a_out"], "c": a["c"]} for a in aud],
           "sample_shapes": aud[:2],
           "c_rows_shape": list(c_rows.shape)}
    res["checks"] = {
        "time_count_matches_config": res["time_lora_modules"] == nl*4,
        "all_time_modules_modulated": res["modules_actually_modulated"] == len(tp.time_names),
        "zero_skipped": res["skipped"] == 0,
        "no_group_modulation": len(res["group_modulated"]) == 0,
        "per_module_shape_recorded": len(res["per_module_shapes"]) == len(tp.time_names),
        "all_shapes_match_c": all(a["a_out"] == a["c"] for a in res["per_module_shapes"]),
        "group_lora_still_trainable": all(
            any(q.requires_grad for n_,q in dict(base.named_modules())[g].named_parameters() if "lora" in n_)
            for g in sorted(tp.group_names))}
    res["pass"] = all(res["checks"].values())
    tp.remove_hooks()
    for k,v in res["checks"].items(): print(f"  [{'PASS' if v else 'FAIL'}] {k}", flush=True)
    print(f"  time {res['time_lora_modules']} / group {res['group_lora_modules']} / hook 호출 {res['hook_invocations']}", flush=True)
    json.dump(res, open(OUT/"MODULE_SCOPE.json","w"), indent=2, ensure_ascii=False)
    return res["pass"]

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
