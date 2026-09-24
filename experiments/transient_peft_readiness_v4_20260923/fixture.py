"""v4 공통 fixture — 모든 arm 이 동일한 raw command 정보 권한을 받는다 (P5).
가상 텐서이며 실제 HVAC/설비 데이터가 아니다."""
import torch, numpy as np
from transient_model import ResponseState, patch_aggregate

TAU_INIT = [4.0, 16.0]

def make_tasks(p, n_ctx_patch=2, n_out_patch=1, seed=20260923, cmd_shift=0.0, n_tasks=2, shift_task=1):
    """task 마다 target row 1 + raw command row 1. 서로 다른 command history/plan."""
    ctx_len = n_ctx_patch * p
    H = n_out_patch * p
    tasks = []
    for t in range(n_tasks):
        g = np.random.default_rng(seed + t)
        tgt = torch.tensor((np.sin(np.arange(ctx_len+H)/5.0 + t)*3 + 20 + g.normal(0,.05,ctx_len+H)).astype(np.float32))
        u = torch.zeros(ctx_len+H)
        s = 5 + 7*t; e = s + 10 + 3*t
        # cmd_shift 는 shift_task 로 지정한 task 에만 적용한다.
        # (v4 시도1 실패 원인: t==1 고정이라 task A 를 바꾸는 시나리오가 성립하지 않았다)
        u[s:e] = 1.0 + (cmd_shift if t == shift_task else 0.0)
        u[ctx_len:] = 0.5*(t+1)                  # 원점에 확정된 미래 계획
        obs = torch.ones(ctx_len+H)
        tasks.append({"target": tgt, "command": u, "obs_mask": obs,
                      "ctx_len": ctx_len, "H": H})
    return tasks, ctx_len, H

def raw_rows(tasks, ctx_len, H):
    """모든 arm 공통: target row + raw command row. (P5)
    반환 ctx (rows, ctx_len), fc (rows, H), fcm, ft, ftm, gids, row_meta"""
    ctx, fc, fcm, ft, ftm, gids, meta = [], [], [], [], [], [], []
    for gi, t in enumerate(tasks):
        ctx.append(t["target"][:ctx_len]); fc.append(torch.zeros(H)); fcm.append(torch.zeros(H))
        ft.append(t["target"][ctx_len:]);  ftm.append(torch.ones(H))
        gids.append(gi); meta.append({"task": gi, "kind": "target"})
        ctx.append(t["command"][:ctx_len]); fc.append(t["command"][ctx_len:]); fcm.append(torch.ones(H))
        ft.append(torch.zeros(H));          ftm.append(torch.zeros(H))
        gids.append(gi); meta.append({"task": gi, "kind": "command"})
    return (torch.stack(ctx), torch.stack(fc), torch.stack(fcm),
            torch.stack(ft), torch.stack(ftm), torch.tensor(gids, dtype=torch.long), meta)

def state_rows(tasks, ctx_len, H, learned=False, gens=None):
    """INPUT arm 전용 추가 행 (FI/LI). raw rows 위에 상태 행을 덧붙인다."""
    ctx, fc, fcm, ft, ftm, gids, meta = raw_rows(tasks, ctx_len, H)
    extra_c, extra_fc, extra_fcm, extra_ft, extra_ftm, extra_g, extra_m = [], [], [], [], [], [], []
    _g = list(gens) if gens else []
    for gi, t in enumerate(tasks):
        if gens: rs = _g[gi]
        else:    rs = ResponseState(TAU_INIT, learned=learned); _g.append(rs)
        m = rs(t["command"])
        for j in range(m.shape[0]):
            extra_c.append(m[j, :ctx_len]); extra_fc.append(m[j, ctx_len:])
            extra_fcm.append(torch.ones(H)); extra_ft.append(torch.zeros(H)); extra_ftm.append(torch.zeros(H))
            extra_g.append(gi); extra_m.append({"task": gi, "kind": f"state_m{j}"})
    return (torch.cat([ctx, torch.stack(extra_c)]), torch.cat([fc, torch.stack(extra_fc)]),
            torch.cat([fcm, torch.stack(extra_fcm)]), torch.cat([ft, torch.stack(extra_ft)]),
            torch.cat([ftm, torch.stack(extra_ftm)]),
            torch.cat([gids, torch.tensor(extra_g, dtype=torch.long)]), meta + extra_m, gens)

def task_states(tasks, p, n_ctx_patch, n_out_patch, ctx_len, H, learned=False, gens=None):
    """MOD arm 용 task 별 상태. 반환 (m_task (n_tasks,J,n_tok), token_valid (n_tasks,n_tok), gens)"""
    ms, vs, out_gens = [], [], []
    for i, t in enumerate(tasks):
        rs = gens[i] if gens else ResponseState(TAU_INIT, learned=learned)
        out_gens.append(rs)
        m = rs(t["command"])
        st, valid = patch_aggregate(m, t["obs_mask"], p, n_ctx_patch, n_out_patch, ctx_len, H)
        ms.append(st); vs.append(valid)
    return torch.stack(ms), torch.stack(vs), out_gens
