"""§7 초기 동일성 — F0_BASE / F0_STATE_INIT.

v4 §7 요구:
  L0/FM/LM step0 prediction == F0_BASE
  FI/LI step0 prediction == F0_STATE_INIT
  FI 와 LI 는 같은 tau init/state 정의에서 시작
  서로 다른 input view 끼리 동일 prediction 을 요구하지 않는다
  F0_BASE 대 F0_STATE_INIT 차이는 정보/표현 효과로 별도 기록
optimizer 를 쓰지 않는 forward 전용 검사다 (update 예산 소모 0).
"""
import json, sys, torch, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build, OUT, fwd
from fixture import make_tasks, raw_rows, state_rows, task_states, TAU_INIT

N_CTX, N_OUT = 2, 1

def sha(t): return hashlib.sha256(t.detach().numpy().tobytes()).hexdigest()[:16]

def target_row_pred(q, meta):
    """행 구성이 다른 view 끼리 비교하려면 target row 만 본다."""
    idx = [i for i, m in enumerate(meta) if m["kind"] == "target"]
    return q[idx]

def trainable_names(tp, with_mod):
    """§8 'trainable name/count' 보완 — UPDATE_AUDIT 은 수만 남겼다."""
    lora = sorted(n for n, q in tp.peft_model.named_parameters() if q.requires_grad)
    sg   = sorted(n for n, q in tp.state_gen.named_parameters() if q.requires_grad)
    md   = sorted(n for n, q in tp.modulator.named_parameters() if q.requires_grad) if with_mod else []
    return {"lora_count": len(lora), "lora_names_all": lora,
            "state_gen_trainable_names": sg, "modulator_trainable_names": md}

def main():
    T = {}
    INV = {}
    # ---------- F0_BASE : raw rows, LoRA B=0, modulation 없음 ----------
    tp0, _ = build(seed=0)                      # modulator 없음 = L0 구성
    base = tp0.peft_model.base_model.model
    p = base.chronos_config.input_patch_size
    tasks, ctx_len, H = make_tasks(p, N_CTX, N_OUT)
    rows = raw_rows(tasks, ctx_len, H)
    meta_raw = rows[6]
    with torch.no_grad():
        q_base = fwd(tp0, *rows[:5], rows[5], N_OUT, None).quantile_preds
    F0_BASE = target_row_pred(q_base, meta_raw)
    tp0.remove_hooks()

    # ---------- L0 / FM / LM step0 ----------
    arms = {}
    for arm, learned, with_mod in [("L0", False, False), ("FM", False, True), ("LM", True, True)]:
        tp, _ = build(seed=0, learned_state=learned, with_modulator=with_mod)
        if with_mod:
            m_task, tv, _ = task_states(tasks, p, N_CTX, N_OUT, ctx_len, H,
                                        learned=learned, gens=[tp.state_gen, tp.state_gen])
            c_rows = tp.build_c_rows(m_task, tv, rows[5])     # modulator init=0 -> c=0
            c_absmax = float(c_rows.abs().max())
        else:
            c_rows, c_absmax = None, None
        with torch.no_grad():
            q = fwd(tp, *rows[:5], rows[5], N_OUT, c_rows).quantile_preds
        pr = target_row_pred(q, meta_raw)
        d = float((pr - F0_BASE).abs().max())
        arms[arm] = {"max_abs_diff_vs_F0_BASE": d, "sha": sha(pr), "c_absmax_at_init": c_absmax,
                     "exact": bool(d == 0.0)}
        INV[arm] = trainable_names(tp, with_mod)
        tp.remove_hooks()
        T[f"{arm}_step0_equals_F0_BASE"] = {"max_abs": d, "pass": bool(d == 0.0)}

    # ---------- F0_STATE_INIT : raw rows + 초기 고정 상태 행 ----------
    tpF, _ = build(seed=0, learned_state=False)
    srF = state_rows(tasks, ctx_len, H, learned=False, gens=[tpF.state_gen, tpF.state_gen])
    meta_st = srF[6]
    with torch.no_grad():
        qF = fwd(tpF, *srF[:5], srF[5], N_OUT, None).quantile_preds
    F0_STATE_INIT = target_row_pred(qF, meta_st)
    INV["FI"] = trainable_names(tpF, False)
    tpF.remove_hooks()

    tpL, _ = build(seed=0, learned_state=True)
    srL = state_rows(tasks, ctx_len, H, learned=True, gens=[tpL.state_gen, tpL.state_gen])
    with torch.no_grad():
        qL = fwd(tpL, *srL[:5], srL[5], N_OUT, None).quantile_preds
    prL = target_row_pred(qL, srL[6])
    INV["LI"] = trainable_names(tpL, False)
    tpL.remove_hooks()

    dFI = 0.0                                        # F0_STATE_INIT 자신이 FI step0
    dLI = float((prL - F0_STATE_INIT).abs().max())
    T["FI_step0_equals_F0_STATE_INIT"] = {"max_abs": dFI, "pass": True,
        "note": "F0_STATE_INIT 을 FI 의 step0 으로 정의했으므로 이 항목은 정의상 0 이다. "
                "실질 검사는 LI 쪽(다른 tau 파라미터화가 같은 값을 내는가)이다."}
    T["LI_step0_equals_F0_STATE_INIT"] = {"max_abs": dLI, "pass": bool(dLI == 0.0)}

    # FI/LI 가 같은 tau init 에서 출발하는지 — 상태 행 자체를 비교
    d_state = float((srF[0] - srL[0]).abs().max())
    T["FI_LI_same_tau_init_state_rows"] = {
        "tau_init": TAU_INIT, "max_abs": d_state,
        "fixed_log_tau": [float(x) for x in tpF.state_gen.log_tau],
        "learned_log_tau": [float(x) for x in tpL.state_gen.log_tau],
        "pass": bool(d_state == 0.0)}

    # 서로 다른 view 끼리는 같기를 요구하지 않는다 — 차이를 기록만 한다
    view_gap = float((F0_STATE_INIT - F0_BASE).abs().max())
    T["view_gap_recorded_not_asserted"] = {
        "F0_BASE_vs_F0_STATE_INIT_max_abs": view_gap,
        "note": "정보/표현 효과. 동일성을 요구하지 않고 기록만 한다 (v4 §7).",
        "pass": True}

    n_pass = sum(1 for v in T.values() if v["pass"])
    res = {"contract": "F0_BASE = raw rows(target+command), B=0, c=0 / "
                       "F0_STATE_INIT = F0_BASE rows + 초기 고정 상태 행",
           "n_rows_F0_BASE": int(rows[0].shape[0]), "n_rows_F0_STATE_INIT": int(srF[0].shape[0]),
           "arms": arms, "tests": T,
           "trainable_inventory_names": INV,
           "trainable_inventory_note": "v4 §8 의 'trainable name/count' 중 이름 부분. "
               "UPDATE_AUDIT.json 의 수치와 같은 builder 에서 나온 목록이며, optimizer step 은 수행하지 않았다.",
           "summary": {"pass": n_pass, "total": len(T)}, "all_pass": n_pass == len(T)}
    for k, v in T.items():
        print(f"  [{'PASS' if v['pass'] else 'FAIL'}] {k}  max_abs={v.get('max_abs', v.get('F0_BASE_vs_F0_STATE_INIT_max_abs'))}", flush=True)
    print(f"  -> {n_pass}/{len(T)}", flush=True)
    json.dump(res, open(OUT/"INITIAL_VIEWS.json", "w"), indent=2, ensure_ascii=False)
    return res["all_pass"]

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
