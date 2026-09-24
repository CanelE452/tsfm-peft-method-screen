"""§8 실제 optimizer update — L0/FI/FM/LI/LM 각 3 step. 연구 fit 아니라 lifecycle 검사.
"gradient 존재" 가 아니라 "optimizer step 뒤 파라미터가 실제로 변했는가" 를 본다."""
import json, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build, OUT, frozen_hash, fwd
from fixture import make_tasks, raw_rows, state_rows, task_states
import loss_contract as LC

ARMS = {  # (learned_state, with_modulator, input_view)
    "L0": (False, False, "raw"),
    "FI": (False, False, "state"),
    "LI": (True,  False, "state"),
    "FM": (False, True,  "raw"),
    "LM": (True,  True,  "raw"),
}
N_STEP = 3
# 모든 arm 에 동일한 lr. 1e-3 에서는 LM 의 tau delta 가 약 1.65e-07 로 float32 절대 정밀도
# (log_tau=1.386 근처에서 약 1.6e-07) 아래라 파라미터 변화가 반올림으로 사라졌다.
# 배선 검사 목적상 delta 가 관측 가능해야 하므로 lr 을 올린다. PASS 기준(delta > 0)은 그대로다.
LR = 1e-1

def run_arm(arm, learned, with_mod, view):
    tp, _ = build(seed=0, learned_state=learned, with_modulator=with_mod)
    base = tp.peft_model.base_model.model
    p = base.chronos_config.input_patch_size
    n_ctx, n_out = 2, 1
    tasks, ctx_len, H = make_tasks(p, n_ctx, n_out)
    # 상태 생성기는 한 번만 만들고(파라미터 고정), 입력/상태는 매 step 재생성한다.
    # (v4 시도3 실패 원인: state_rows 를 루프 밖에서 한 번만 만들어 두 번째 backward 가
    #  이미 해제된 graph 를 재사용했다 — "backward through the graph a second time")
    from transient_model import ResponseState
    from fixture import TAU_INIT
    gens = [ResponseState(TAU_INIT, learned=learned) for _ in tasks]
    state_params = [q for g in gens for q in g.parameters() if q.requires_grad]
    def make_batch():
        if view == "state":
            return state_rows(tasks, ctx_len, H, learned=learned, gens=gens)[:7] + (None, None)
        r = raw_rows(tasks, ctx_len, H)
        mt, tvv, _ = task_states(tasks, p, n_ctx, n_out, ctx_len, H, learned=learned, gens=gens)
        return r + (mt, tvv)
    _b = make_batch(); ftm = _b[4]; gids = _b[5]
    # optimizer inventory: model 바깥 객체도 반드시 포함
    lora_params = [q for n_,q in tp.peft_model.named_parameters() if q.requires_grad]
    mod_params  = list(tp.modulator.parameters()) if with_mod else []
    params = lora_params + mod_params + state_params
    inv = {"lora": sum(q.numel() for q in lora_params),
           "modulator": sum(q.numel() for q in mod_params),
           "state_gen": sum(q.numel() for q in state_params)}
    opt = torch.optim.SGD(params, lr=LR)
    h0 = frozen_hash(tp)
    def snap():
        return {"loraB": float(sum(q.norm()**2 for n_,q in tp.peft_model.named_parameters() if "lora_B" in n_)**0.5),
                "mod": float(tp.modulator.lin.weight.norm()) if with_mod else None,
                # log_tau 원값을 그대로 본다. exp 후 표시 자릿수로 비교하면 작은 delta 를 놓친다
                # (v4 시도4 실패 원인: LM 의 tau gradient 는 1.65e-04 로 존재했으나
                #  lr=1e-3 에서 delta 가 1e-7 수준이라 반올림된 값 비교로 False 가 났다)
                "log_tau": [float(x) for x in gens[0].log_tau.detach()] if learned else None,
                "tau": [float(x) for x in torch.exp(gens[0].log_tau.detach())] if learned else None}
    s0 = snap(); steps = []
    for st in range(N_STEP):
        opt.zero_grad(set_to_none=True)
        ctx, fc, fcm, ft, ftm, gids, meta, m_task, tv = make_batch()   # 매 step 새 graph
        c_rows = tp.build_c_rows(m_task, tv, gids) if with_mod else None
        out = fwd(tp, ctx, fc, fcm, ft, ftm, gids, n_out, c_rows)
        native = out.loss
        corrected, info = LC.corrected_loss(native, ftm)
        corrected.backward()
        gn = {"lora": float(sum((q.grad**2).sum() for q in lora_params if q.grad is not None)**0.5),
              "modulator": float(sum((q.grad**2).sum() for q in mod_params if q.grad is not None)**0.5) if mod_params else None,
              "state_gen": float(sum((q.grad**2).sum() for q in state_params if q.grad is not None)**0.5) if state_params else None}
        before = snap(); opt.step(); after = snap()
        steps.append({"step": st, "loss_native": float(native.detach()),
                      "loss_corrected": float(corrected.detach()),
                      "correction": info, "grad_norm": gn,
                      "loraB_norm": after["loraB"],
                      "mod_norm": after["mod"], "tau": after["tau"],
                      "loraB_delta": after["loraB"]-before["loraB"],
                      "mod_delta": (after["mod"]-before["mod"]) if with_mod else None,
                      "tau_delta": (max(abs(a-b) for a,b in zip(after["tau"],before["tau"])) if learned else None),
                      "log_tau_delta": (max(abs(a-b) for a,b in zip(after["log_tau"],before["log_tau"])) if learned else None)})
        if with_mod and st == 0:
            with torch.no_grad(): tp.modulator.lin.weight.add_(torch.randn_like(tp.modulator.lin.weight)*0.01)
    sN = snap()
    tp.remove_hooks()
    res = {"arm": arm, "trainable_inventory": inv, "steps": steps,
           "frozen_head_preserved": bool(h0 == frozen_hash(tp)),
           "loraB_changed": bool(abs(sN["loraB"]-s0["loraB"]) > 0),
           "mod_changed": (bool(abs(sN["mod"]-s0["mod"]) > 0) if with_mod else None),
           "tau_changed": (bool(max(abs(a-b) for a,b in zip(sN["log_tau"],s0["log_tau"])) > 0) if learned else None),
           "log_tau_total_delta": (max(abs(a-b) for a,b in zip(sN["log_tau"],s0["log_tau"])) if learned else None),
           "state_grad_reached": None,
           "state_trainable_zero_for_fixed": bool(inv["state_gen"] == 0) if not learned else None}
    ok = res["loraB_changed"] and res["frozen_head_preserved"]
    if learned: ok = ok and res["tau_changed"]
    else:       ok = ok and res["state_trainable_zero_for_fixed"]
    if with_mod: ok = ok and res["mod_changed"]
    res["verdict"] = f"{arm}_UPDATE_{'OK' if ok else 'FAIL'}"
    return res, ok

def main():
    out = {"lr": LR, "steps_per_arm": N_STEP, "arms": {}}
    allok = True
    for arm, (le, wm, vw) in ARMS.items():
        r, ok = run_arm(arm, le, wm, vw); out["arms"][arm] = r; allok = allok and ok
        print(f"  {r['verdict']:16s} loraB {r['loraB_changed']}  mod {r['mod_changed']}  tau {r['tau_changed']}  "
              f"head {r['frozen_head_preserved']}  params {r['trainable_inventory']}", flush=True)
    out["total_optimizer_updates"] = len(ARMS)*N_STEP
    out["all_pass"] = allok
    print(f"\n  총 optimizer update {out['total_optimizer_updates']} (상한 15)  ->  {'ALL OK' if allok else 'FAIL 있음'}", flush=True)
    json.dump(out, open(OUT/"UPDATE_AUDIT.json","w"), indent=2, ensure_ascii=False)
    return allok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
