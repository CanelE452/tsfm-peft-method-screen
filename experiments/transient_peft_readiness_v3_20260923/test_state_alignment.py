"""§4.4 상태 정렬 검사 6개 + §4.3 토큰 대응 표. 가상 fixture 전용, 성능 실험 아님."""
import json, torch, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_state_gradient import ResponseState
OUT = Path(__file__).resolve().parents[2] / "results/transient_peft_readiness_v3_20260923"
LED = {"forward":0,"backward":0}

def patch_aggregate(m_raw, p, n_ctx_patch, n_out_patch, ctx_len, H):
    """§4.4: 패치 상태는 '마지막 유효 시점' 으로 고정. 좌측 padding 과 REG 는 중립(0)."""
    J = m_raw.shape[0]
    pad = n_ctx_patch*p - ctx_len                      # 좌측 padding 길이
    out = torch.zeros(J, n_ctx_patch + 1 + n_out_patch)   # [ctx patches][REG][out patches]
    for k in range(n_ctx_patch):
        s, e = k*p - pad, (k+1)*p - pad
        s_, e_ = max(s,0), min(e, ctx_len)
        out[:, k] = m_raw[:, e_-1] if e_ > s_ else 0.0     # 전부 padding 이면 중립
    out[:, n_ctx_patch] = 0.0                              # REG 중립
    for k in range(n_out_patch):
        s = ctx_len + k*p; e = min(ctx_len + (k+1)*p, ctx_len+H)
        out[:, n_ctx_patch+1+k] = m_raw[:, e-1] if e > s else 0.0   # H 초과는 중립
    return out

def main():
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu", dtype=torch.float32)
    m = pipe.inner_model
    for q in m.parameters(): q.requires_grad_(False)
    p = m.chronos_config.input_patch_size
    ctx_len, H = 2*p+1, p+1                      # 패치 배수 아님 + padding 유발
    n_ctx = int(np.ceil(ctx_len/p)); n_out = int(np.ceil(H/p))
    g = np.random.default_rng(20260923)
    res = {"config":{"p":p,"ctx_len":ctx_len,"H":H,"n_ctx_patch":n_ctx,"n_out_patch":n_out}}

    # ---- §4.3 토큰 대응 표 ----
    pad = n_ctx*p - ctx_len
    tok = []
    for k in range(n_ctx):
        s,e = k*p-pad, (k+1)*p-pad
        tok.append({"idx":k,"kind":"context_patch",
                    "raw_start":max(s,0),"raw_end":min(e,ctx_len)-1,
                    "left_pad":max(0,-s),"right_pad":max(0,e-ctx_len),
                    "all_padding":bool(min(e,ctx_len)<=max(s,0))})
    tok.append({"idx":n_ctx,"kind":"REG","raw_start":None,"raw_end":None,
                "note":"model.py:600-604 use_reg_token. 중립 modulation"})
    for k in range(n_out):
        s = ctx_len+k*p; e = min(ctx_len+(k+1)*p, ctx_len+H)
        tok.append({"idx":n_ctx+1+k,"kind":"output_patch",
                    "raw_start":s,"raw_end":e-1,
                    "represents_command_window":[s,e-1],
                    "beyond_H_padding":int(ctx_len+(k+1)*p - (ctx_len+H)) if (k+1)*p > H else 0})
    res["token_map"] = tok
    res["row_map"] = [{"row":0,"task":0,"variable":"target"},
                      {"row":1,"task":0,"variable":"state_m0"},
                      {"row":2,"task":0,"variable":"state_m1"}]

    def build(u, tgt, seed_rows=None):
        rs = ResponseState([4.0,16.0], learned=False)
        mm = rs(u)
        ctx = torch.cat([tgt[:ctx_len][None,:], mm[:,:ctx_len]],0)
        fc  = torch.cat([torch.zeros(1,H), mm[:,ctx_len:]],0)
        fcm = torch.cat([torch.zeros(1,H), torch.ones(2,H)],0)
        ft  = torch.zeros(3,H); ft[0]=tgt[ctx_len:]
        ftm = torch.zeros(3,H); ftm[0]=1.0
        return mm, ctx, fc, fcm, ft, ftm

    def run(ctx, fc, fcm, ft, ftm, gids):
        with torch.no_grad():
            o = m(context=ctx, group_ids=gids, future_covariates=fc,
                  future_covariates_mask=fcm, future_target=ft, future_target_mask=ftm,
                  num_output_patches=n_out)
        LED["forward"] += 1
        return o.quantile_preds.clone()

    tgt = torch.tensor((np.sin(np.arange(ctx_len+H)/5.0)*3+20).astype(np.float32))
    u = torch.zeros(ctx_len+H); u[10:25] = 1.0
    mm, ctx, fc, fcm, ft, ftm = build(u, tgt)
    gid0 = torch.zeros(3, dtype=torch.long)
    base = run(ctx, fc, fcm, ft, ftm, gid0)
    T = {}

    # 1) row permutation + group id 재명명 -> 같은 task 결과 동일
    perm = [0,2,1]
    q = run(ctx[perm], fc[perm], fcm[perm], ft[perm], ftm[perm], gid0[perm]+7)
    inv = [perm.index(i) for i in range(3)]
    T["1_row_permutation_invariance"] = {
        "max_abs_diff_target_row": float((q[inv][0]-base[0]).abs().max()),
        "atol":1e-5, "pass": bool((q[inv][0]-base[0]).abs().max() < 1e-5)}

    # 2) 다른 task 의 값을 바꿔도 이 task 는 불변 (group id 분리)
    u2 = torch.zeros(ctx_len+H); u2[5:30] = 2.0
    mm2, ctx2, fc2, fcm2, ft2, ftm2 = build(u2, tgt+5)
    big_ctx = torch.cat([ctx, ctx2],0); big_fc = torch.cat([fc,fc2],0)
    big_fcm= torch.cat([fcm,fcm2],0);  big_ft = torch.cat([ft,ft2],0); big_ftm=torch.cat([ftm,ftm2],0)
    gids2 = torch.tensor([0,0,0,1,1,1])
    q2 = run(big_ctx, big_fc, big_fcm, big_ft, big_ftm, gids2)
    # task1 의 명령을 바꿔 다시
    u2b = torch.zeros(ctx_len+H); u2b[8:20] = -3.0
    _, ctx2b, fc2b, fcm2b, ft2b, ftm2b = build(u2b, tgt+5)
    q2b = run(torch.cat([ctx,ctx2b],0), torch.cat([fc,fc2b],0), torch.cat([fcm,fcm2b],0),
              torch.cat([ft,ft2b],0), torch.cat([ftm,ftm2b],0), gids2)
    T["2_cross_task_isolation"] = {
        "max_abs_diff_task0": float((q2[0]-q2b[0]).abs().max()),
        "atol":1e-5, "pass": bool((q2[0]-q2b[0]).abs().max() < 1e-5)}

    # 3) 패치 경계 명령 변화가 의도한 상태 위치에 반영되는지 (원시 recurrence 대조)
    agg = patch_aggregate(mm, p, n_ctx, n_out, ctx_len, H)
    manual = []
    for k in range(n_ctx):
        s,e = k*p-pad, (k+1)*p-pad
        e_ = min(e, ctx_len); s_ = max(s,0)
        manual.append(mm[:, e_-1].tolist() if e_>s_ else [0.0,0.0])
    T["3_patch_boundary_mapping"] = {
        "agg_ctx_patches": agg[:, :n_ctx].T.tolist(),
        "manual_last_valid": manual,
        "pass": bool(np.allclose(np.array(agg[:, :n_ctx].T.tolist()), np.array(manual), atol=0)),
        "REG_neutral": bool(float(agg[:, n_ctx].abs().max())==0.0),
        "left_pad": pad}

    # 4) batch 간 상태 누출 없음 — 원점마다 새로 초기화
    mmA = ResponseState([4.0,16.0], learned=False)(u)
    mmB = ResponseState([4.0,16.0], learned=False)(u)
    T["4_no_state_carryover"] = {"max_abs_diff": float((mmA-mmB).abs().max()),
        "pass": bool(torch.equal(mmA, mmB)), "note":"같은 입력이면 항상 같은 상태. 이전 batch 영향 없음"}

    # 5) 미허용 미래 target 을 바꿔도 입력·상태·예측 불변
    ft_alt = ft.clone(); ft_alt[0] = ft_alt[0] + 100.0
    q5 = run(ctx, fc, fcm, ft_alt, ftm, gid0)
    T["5_future_target_does_not_leak"] = {
        "max_abs_diff_pred": float((q5-base).abs().max()),
        "atol":1e-6, "pass": bool((q5-base).abs().max() < 1e-6),
        "note":"예측은 불변이어야 한다. loss 가 달라지는 것은 정상"}

    # 6) 허용된 명령 계획 변경은 상태에 반영
    u6 = u.clone(); u6[ctx_len:] = 5.0       # 미래 계획만 변경
    mm6 = ResponseState([4.0,16.0], learned=False)(u6)
    fut_changed = float((mm6[:, ctx_len:] - mm[:, ctx_len:]).abs().max())
    past_same   = float((mm6[:, :ctx_len] - mm[:, :ctx_len]).abs().max())
    T["6_command_plan_changes_state"] = {
        "future_state_diff": fut_changed, "past_state_diff": past_same,
        "pass": bool(fut_changed > 0 and past_same == 0),
        "note":"미래 계획 변경은 미래 상태만 바꾸고 과거 상태는 그대로여야 한다"}

    res["tests"] = T; res["ledger"] = dict(LED)
    n_pass = sum(1 for v in T.values() if v.get("pass"))
    res["summary"] = {"pass": n_pass, "total": len(T)}
    for k,v in T.items(): print(f"  [{'PASS' if v.get('pass') else 'FAIL'}] {k}", flush=True)
    print(f"  -> {n_pass}/{len(T)}", flush=True)
    json.dump(res, open(OUT/"STATE_ALIGNMENT.json","w"), indent=2, ensure_ascii=False)
    print(f"[saved] {OUT/'STATE_ALIGNMENT.json'}", flush=True)

if __name__ == "__main__":
    main()
