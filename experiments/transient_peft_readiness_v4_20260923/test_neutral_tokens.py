"""§5 중립 토큰 — 학습된 weight 에서도 REG/padding/all-masked/H밖 토큰의 c 가 정확히 0."""
import json, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build, OUT
from transient_model import ResponseState, Modulator, patch_aggregate
from fixture import TAU_INIT

def main():
    p = 16
    tp, _ = build(with_modulator=True)
    p = tp.peft_model.base_model.model.chronos_config.input_patch_size
    torch.manual_seed(3)
    with torch.no_grad(): tp.modulator.lin.weight.normal_(0, 0.5)   # 학습된 것처럼 크게
    T = {}

    # (a) 길이 padding: ctx = 2p+1 -> 첫 패치에 좌측 padding
    ctx_len, H = 2*p+1, p+1
    n_ctx, n_out = 3, 2
    u = torch.zeros(ctx_len+H); u[10:20] = 1.0
    obs = torch.ones(ctx_len+H)
    rs = ResponseState(TAU_INIT, learned=False); mm = rs(u)
    st, valid = patch_aggregate(mm, obs, p, n_ctx, n_out, ctx_len, H)
    c = tp.modulator(st.T) * valid[:, None]
    reg_idx = n_ctx
    T["REG_c_is_zero"] = {"max_abs": float(c[reg_idx].abs().max()), "pass": bool(float(c[reg_idx].abs().max())==0.0)}
    # H 밖 토큰: n_out 을 H 가 요구하는 수보다 크게 잡아 마지막 패치를 완전히 H 밖에 둔다.
    # (v4 시도2 실패 원인: H=p+1 이라 두 번째 출력 패치에도 1칸 유효 구간이 있었다.
    #  "두 번째 패치 = H 밖" 이라는 검사 전제 자체가 틀렸다.)
    n_out_pad = 3                      # 필요한 것은 ceil(H/p)=2. 세 번째는 전부 H 밖
    st_b, valid_b = patch_aggregate(mm, obs, p, n_ctx, n_out_pad, ctx_len, H)
    c_b = tp.modulator(st_b.T) * valid_b[:, None]
    beyond = n_ctx + 1 + 2             # 세 번째 출력 패치
    s_b = ctx_len + 2*p
    T["beyond_H_c_is_zero"] = {"token": beyond, "valid": float(valid_b[beyond]),
                               "raw_start": s_b, "H_end": ctx_len+H,
                               "fully_beyond_H": bool(s_b >= ctx_len+H),
                               "max_abs": float(c_b[beyond].abs().max()),
                               "pass": bool(float(c_b[beyond].abs().max())==0.0)}

    # (b) all-masked command patch
    obs2 = torch.ones(ctx_len+H); obs2[0:p] = 0.0      # 첫 패치 관측 전부 마스킹
    st2, valid2 = patch_aggregate(mm, obs2, p, n_ctx, n_out, ctx_len, H)
    c2 = tp.modulator(st2.T) * valid2[:, None]
    T["all_masked_patch_c_is_zero"] = {"valid": float(valid2[0]), "max_abs": float(c2[0].abs().max()),
                                       "pass": bool(float(c2[0].abs().max())==0.0)}

    # (c) partially observed patch 는 마지막 valid 시점만 사용
    obs3 = torch.ones(ctx_len+H); obs3[p+5:2*p] = 0.0   # 두 번째 패치 뒷부분 마스킹
    st3, valid3 = patch_aggregate(mm, obs3, p, n_ctx, n_out, ctx_len, H)
    pad = n_ctx*p - ctx_len
    s,e = max(1*p-pad,0), min(2*p-pad, ctx_len)
    last_valid = max(i for i in range(s,e) if obs3[i] > 0)
    T["partial_patch_uses_last_valid"] = {
        "last_valid_index": last_valid,
        "match": float((st3[:,1] - mm[:, last_valid]).abs().max()),
        "pass": bool(float((st3[:,1] - mm[:, last_valid]).abs().max())==0.0)}

    # (d) Modulator 에 bias 가 없어야 한다 — 있으면 c(0)=bias 로 중립이 깨진다
    has_bias = tp.modulator.lin.bias is not None
    m0 = torch.zeros(1, len(TAU_INIT))
    c0 = tp.modulator(m0)
    T["modulator_has_no_bias"] = {"bias_is_none": bool(not has_bias),
                                  "c_at_m_zero_max_abs": float(c0.abs().max()),
                                  "pass": bool((not has_bias) and float(c0.abs().max())==0.0)}
    # bias 를 넣으면 검사가 실패해야 한다 (규약 확인)
    probe = Modulator(len(TAU_INIT), 8)
    probe.lin = torch.nn.Linear(len(TAU_INIT), 8, bias=True)
    torch.nn.init.zeros_(probe.lin.weight); torch.nn.init.constant_(probe.lin.bias, 0.3)
    T["bias_would_break_neutrality"] = {
        "c_at_m_zero_with_bias": float(probe(m0).abs().max()),
        "pass": bool(float(probe(m0).abs().max()) > 0),
        "note": "bias 가 있으면 m=0 에서도 c!=0 이 되므로 bias=False 를 계약으로 둔다"}
    tp.remove_hooks()
    n = sum(1 for v in T.values() if v["pass"])
    res = {"tests": T, "summary": {"pass": n, "total": len(T)}, "all_pass": n == len(T)}
    for k,v in T.items(): print(f"  [{'PASS' if v['pass'] else 'FAIL'}] {k}  max_abs={v.get('max_abs', v.get('c_at_m_zero_max_abs',''))}", flush=True)
    json.dump(res, open(OUT/"NEUTRAL_TOKEN_AUDIT.json","w"), indent=2, ensure_ascii=False)
    return res["all_pass"]

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
