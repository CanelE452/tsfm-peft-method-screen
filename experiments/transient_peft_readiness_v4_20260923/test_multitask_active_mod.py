"""§4·§11 active MOD 다중 task 정렬·격리 7개. B≠0, c≠0 인 활성 경로에서만 검사한다."""
import json, sys, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build, OUT, fwd
from fixture import make_tasks, raw_rows, task_states

def activate(tp, bscale=0.01, mscale=0.02, seed=1):
    torch.manual_seed(seed)
    import peft.tuners.lora as L
    base = tp.peft_model.base_model.model
    with torch.no_grad():
        for n_, mod in base.named_modules():
            if isinstance(mod, L.Linear): mod.lora_B["default"].weight.normal_(0, bscale)
        tp.modulator.lin.weight.normal_(0, mscale)

def main():
    tp, _ = build(with_modulator=True)
    base = tp.peft_model.base_model.model
    p = base.chronos_config.input_patch_size
    activate(tp)
    def scenario(cmd_shift=0.0, n_ctx_patch=2, n_tasks=2, shift_task=1):
        tasks, ctx_len, H = make_tasks(p, n_ctx_patch, 1, cmd_shift=cmd_shift,
                                       n_tasks=n_tasks, shift_task=shift_task)
        rows = raw_rows(tasks, ctx_len, H)
        m_task, tv, _ = task_states(tasks, p, n_ctx_patch, 1, ctx_len, H)
        return tasks, rows, m_task, tv
    T = {}
    tasks, rows, m_task, tv = scenario()
    ctx, fc, fcm, ft, ftm, gids, meta = rows
    c_rows = tp.build_c_rows(m_task, tv, gids)
    with torch.no_grad(): base_pred = fwd(tp, ctx, fc, fcm, ft, ftm, gids, 1, c_rows).quantile_preds.clone()

    # 4) 같은 task 의 target/command row 가 같은 state 를 받는가
    same = float((c_rows[0]-c_rows[1]).abs().max())
    # 5) 다른 task row 는 다른 state
    diff = float((c_rows[0]-c_rows[2]).abs().max())
    T["4_same_task_rows_share_state"] = {"max_abs_diff": same, "pass": bool(same == 0.0)}
    T["5_different_task_rows_differ"] = {"max_abs_diff": diff, "pass": bool(diff > 0)}

    # 1) row permutation + group_id/state mapping 동시 변경 -> 재정렬 후 동일
    perm = [2,0,3,1]
    gp = gids[perm]
    cp = tp.build_c_rows(m_task, tv, gp)
    with torch.no_grad():
        q = fwd(tp, ctx[perm], fc[perm], fcm[perm], ft[perm], ftm[perm], gp, 1, cp).quantile_preds
    inv = [perm.index(i) for i in range(len(perm))]
    d1 = float((q[inv]-base_pred).abs().max())
    T["1_row_permutation_invariance"] = {"max_abs_diff": d1, "atol":1e-5, "pass": bool(d1 < 1e-5)}

    # 2) task B command 만 바꿔도 task A 불변
    tasks2, rows2, m2, tv2 = scenario(cmd_shift=3.0)
    ctx2, fc2, fcm2, ft2, ftm2, g2, _ = rows2
    ctxB = ctx.clone(); ctxB[2:4] = ctx2[2:4]      # task B 행만 교체
    fcB = fc.clone(); fcB[2:4] = fc2[2:4]
    mB = m_task.clone(); mB[1] = m2[1]
    cB = tp.build_c_rows(mB, tv, gids)
    with torch.no_grad(): qB = fwd(tp, ctxB, fcB, fcm, ft, ftm, gids, 1, cB).quantile_preds
    d2 = float((qB[0]-base_pred[0]).abs().max())
    T["2_taskB_change_does_not_affect_taskA"] = {"max_abs_diff_taskA": d2, "atol":1e-5, "pass": bool(d2 < 1e-5)}

    # 3) task A command 를 바꾸면 task A 가 실제로 변함
    tasks3, rows3, m3, tv3 = scenario(cmd_shift=3.0, shift_task=0)   # task A 를 바꾼다
    ctx3, fc3 = rows3[0], rows3[1]
    ctxA = ctx.clone(); ctxA[0:2] = ctx3[0:2]
    mA = m_task.clone(); mA[0] = m3[0]
    cA = tp.build_c_rows(mA, tv, gids)
    with torch.no_grad(): qA = fwd(tp, ctxA, fc, fcm, ft, ftm, gids, 1, cA).quantile_preds
    d3 = float((qA[0]-base_pred[0]).abs().max())
    cdiff = float((cA[0]-c_rows[0]).abs().max())
    T["3_taskA_command_changes_taskA"] = {"pred_diff": d3, "c_diff": cdiff,
                                          "pass": bool(d3 > 0 and cdiff > 0)}

    # 6) patch boundary impulse 가 의도한 토큰에 반영
    tk = make_tasks(p, 2, 1)[0]
    u = torch.zeros(2*p+p); u[p] = 1.0             # 정확히 두 번째 ctx 패치 시작
    from transient_model import ResponseState, patch_aggregate
    from fixture import TAU_INIT
    rs = ResponseState(TAU_INIT, learned=False); mm = rs(u)
    st, vv = patch_aggregate(mm, torch.ones(3*p), p, 2, 1, 2*p, p)
    T["6_patch_boundary_impulse"] = {
        "patch0_state": float(st[:,0].abs().max()), "patch1_state": float(st[:,1].abs().max()),
        "REG_state": float(st[:,2].abs().max()),
        "pass": bool(float(st[:,0].abs().max())==0.0 and float(st[:,1].abs().max())>0 and float(st[:,2].abs().max())==0.0)}

    # 7) batch/episode 를 바꿔 호출해도 carry-over 없음
    with torch.no_grad():
        _ = fwd(tp, ctx2, fc2, fcm2, ft2, ftm2, g2, 1, tp.build_c_rows(m2, tv2, g2))
        q7 = fwd(tp, ctx, fc, fcm, ft, ftm, gids, 1, c_rows).quantile_preds
    d7 = float((q7-base_pred).abs().max())
    T["7_no_state_carryover"] = {"max_abs_diff": d7, "pass": bool(d7 == 0.0),
                                 "holder_cleared": bool(len(tp._holder)==0)}

    # D 반례: row 수 == token 수 인 fixture (과거 shape hook 이면 오작동)
    # tok = 2(ctx) + 1(REG) + 1(out) = 4, rows = 2 task x 2 row = 4 -> 두 축의 길이가 같다
    tasksD, ctxD_len, HD = make_tasks(p, n_ctx_patch=2, n_out_patch=1, n_tasks=2)
    rowsD = raw_rows(tasksD, ctxD_len, HD)
    mD, tvD, _ = task_states(tasksD, p, 2, 1, ctxD_len, HD)
    cD = tp.build_c_rows(mD, tvD, rowsD[5])
    n_tok_D = mD.shape[2]; n_rows_D = rowsD[0].shape[0]
    # 시도6 실패 원인: n_ctx_patch=1 이면 tok=3, rows=4 로 "행 수 == 토큰 수" 가 성립하지 않았다.
    assert n_rows_D == n_tok_D, f"D 반례 전제 불성립: rows {n_rows_D} != tokens {n_tok_D}"
    tp.reset_audit()
    with torch.no_grad(): fwd(tp, *rowsD[:5], rowsD[5], 1, cD)
    T["D_rows_equals_tokens_counterexample"] = {
        "n_rows": n_rows_D, "n_tokens": n_tok_D, "hook_calls": tp.hook_calls,
        "pass": bool(tp.hook_calls == len(tp.time_names)),
        "note": "축을 이름으로 알므로 행 수와 토큰 수가 같아도 오작동하지 않는다"}
    tp.remove_hooks()
    n_pass = sum(1 for v in T.values() if v["pass"])
    # v4 §4: 각 hook 이 실제로 쓴 module_name / row / group_id / token index / c 요약
    debug_audit = tp.audit()
    res = {"tests": T, "hook_debug_audit": debug_audit,
           "hook_debug_audit_fields": ["module", "n_rows", "n_tokens", "group_ids",
                                       "c_absmax_by_token", "c_absmax_by_row", "a_out", "c"],
           "summary": {"pass": n_pass, "total": len(T)}, "all_pass": n_pass == len(T)}
    for k,v in T.items(): print(f"  [{'PASS' if v['pass'] else 'FAIL'}] {k}", flush=True)
    print(f"  -> {n_pass}/{len(T)}", flush=True)
    json.dump(res, open(OUT/"ACTIVE_MOD_ALIGNMENT.json","w"), indent=2, ensure_ascii=False)
    return res["all_pass"]

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
