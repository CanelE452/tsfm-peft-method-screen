"""사전확인 (a): G1 판정이 QMEAN 꼬리 규칙과 표본 구성에 얼마나 의존하는가.

[소비처] MASTER CLI 지시문 v2 개정 결정
[문장]   "G1의 과소예측 판정이 모델 현상인가, QMEAN 추정기 편향과 층화 배분의 산물인가"

추론 1회(학습 분위수 전체)로 모든 꼬리 규칙·층 정의를 계산한다.
STEP 1 추론 -> 2 QMEAN 3종 -> 3 SBC 2정의 -> 4 PRB 3가중 -> 5 판정
"""
import sys, time, json
from pathlib import Path
import numpy as np
import torch

M5 = Path("/home/minjae/Documents/github/m5dataset")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(M5))
from common.data_split import get_split  # noqa

H, CONTEXT, BS = 28, 512, 256
SBC_ADI, SBC_CV2 = 1.32, 0.49
QLEV_SPEC = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]  # 지시문 §5.1
SPLIT_END = 1885  # 지시문 §4.3: 적격성·SBC 는 d_1885 이하만

def log(m): print(m, flush=True)

# ---------------- STEP 1 ----------------
log("[STEP 1] 데이터 + 추론")
ids, train, test = get_split()
log(f"  train {train.shape}  test {test.shape}")

from chronos import Chronos2Pipeline
from chronos.utils import interpolate_quantiles
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda",
                                        torch_dtype=torch.bfloat16)
qtrain = list(pipe.quantiles)
log(f"  [P3] 모델 학습 분위수 {len(qtrain)}개: {qtrain}")
outside = [q for q in QLEV_SPEC if q not in qtrain]
log(f"  [P3] 지시문 Qlev 중 학습목록 밖(보간 대상): {outside}")

ctx = train[:, -CONTEXT:]
inputs = [torch.tensor(ctx[i], dtype=torch.float32) for i in range(ctx.shape[0])]
t0 = time.time()
qp, mean_out = pipe.predict_quantiles(inputs, prediction_length=H,
                                      quantile_levels=qtrain, batch_size=BS)
torch.cuda.synchronize()
infer_s = time.time() - t0
peak = torch.cuda.max_memory_allocated()/1e9
Q = np.stack([q[0].float().cpu().numpy() for q in qp])          # (N,H,nq) 학습 분위수
MEANO = np.stack([m[0].float().cpu().numpy() for m in mean_out]) # (N,H) 패키지 mean
log(f"  추론 {infer_s:.1f}s  peak {peak:.2f}GB  Q{Q.shape}")
np.save(OUT/"Q_train_levels.npy", Q.astype(np.float32))

neg = float((Q < 0).mean())
log(f"  [관찰] 음수 분위수 비중 {neg:.4%}   최소값 {Q.min():.4f}")
# P2: mean vs 0.5 분위수
i50 = qtrain.index(0.5)
log(f"  [P2] max|mean - q0.5| = {np.abs(MEANO - Q[:,:,i50]).max():.3e}  (0이면 mean=중앙값)")

# ---------------- STEP 2 ----------------
log("[STEP 2] QMEAN 꼬리 규칙 3종")
def qmean_trapz(levels, vals, tail):
    """levels 오름차순, vals (...,nq). tail: const|linear"""
    lv = np.asarray(levels, float)
    if tail == "const":
        lv_e = np.concatenate([[0.0], lv, [1.0]])
        v_e = np.concatenate([vals[..., :1], vals, vals[..., -1:]], axis=-1)
    elif tail == "linear":
        lo_s = (vals[...,1]-vals[...,0])/(lv[1]-lv[0])
        hi_s = (vals[...,-1]-vals[...,-2])/(lv[-1]-lv[-2])
        v0 = vals[...,0] - lo_s*(lv[0]-0.0)
        v1 = vals[...,-1] + hi_s*(1.0-lv[-1])
        lv_e = np.concatenate([[0.0], lv, [1.0]])
        v_e = np.concatenate([v0[...,None], vals, v1[...,None]], axis=-1)
    else:
        raise ValueError(tail)
    return np.trapezoid(v_e, x=lv_e, axis=-1)

Qspec = interpolate_quantiles(QLEV_SPEC, qtrain, torch.from_numpy(Q)).numpy()  # 지시문 13개
variants = {
    "A_spec_const":  np.clip(qmean_trapz(QLEV_SPEC, Qspec, "const"),  0, None),
    "B_spec_linear": np.clip(qmean_trapz(QLEV_SPEC, Qspec, "linear"), 0, None),
    "C_train_const": np.clip(qmean_trapz(qtrain,    Q,     "const"),  0, None),
}
variants["MED"] = np.clip(Q[:,:,i50], 0, None)
for k,v in variants.items(): log(f"  {k:15s} mean={v.mean():.4f}")

snaive = np.load(M5/"models/naive/pred_snaive_week.npy")
variants["SNAIVE_week"] = snaive

# ---------------- STEP 3 ----------------
log("[STEP 3] SBC 2정의 (d_1885 기준)")
hist = train[:, :SPLIT_END]
def sbc_stats(mode):
    adi = np.full(len(hist), np.nan); cv2 = np.full(len(hist), np.nan)
    for i, r in enumerate(hist):
        nz = np.nonzero(r)[0]
        if len(nz) == 0: continue
        span = (len(r)-nz[0]) if mode=="after_first" else (nz[-1]-nz[0]+1)
        adi[i] = span/len(nz)
        pos = r[r>0]
        cv2[i] = (pos.std(ddof=0)/pos.mean())**2 if pos.mean()>0 else 0.0
    return adi, cv2
def classify(adi, cv2):
    lab = np.full(len(adi), "none", dtype=object)
    ok = ~np.isnan(adi)
    lab[ok & (adi< SBC_ADI)&(cv2< SBC_CV2)] = "smooth"
    lab[ok & (adi< SBC_ADI)&(cv2>=SBC_CV2)] = "erratic"
    lab[ok & (adi>=SBC_ADI)&(cv2< SBC_CV2)] = "intermittent"
    lab[ok & (adi>=SBC_ADI)&(cv2>=SBC_CV2)] = "lumpy"
    return lab
elig = (train[:, :SPLIT_END] > 0).sum(1) >= 20   # 지시문 §4.3 적격
log(f"  적격 {elig.sum()} / {len(elig)}  (제외 {(~elig).sum()})")
labels = {}
for mode in ["after_first","first_to_last"]:
    a,c = sbc_stats(mode); lab = classify(a,c); labels[mode]=lab
    cnt = {k:int(((lab==k)&elig).sum()) for k in ["smooth","erratic","intermittent","lumpy"]}
    log(f"  {mode:15s} {cnt}   (지시문 요구: 유형당 750)")

# ---------------- STEP 4 ----------------
log("[STEP 4] PRB 3가중")
def prb(p, mask, w=None):
    y = test[mask]; yh = p[mask]
    if w is None: return float((yh-y).sum()/y.sum())
    w = w[:,None]
    return float(((yh-y)*w).sum()/((y*w).sum()))

lab = labels["first_to_last"]  # 논문 정의를 주 정의로
res = {}
for name, p in variants.items():
    row = {"pooled_all": prb(p, elig)}
    for g in ["smooth","erratic","intermittent","lumpy"]:
        m = elig & (lab==g)
        row[f"regime_{g}"] = prb(p, m) if m.sum() else None
        row[f"n_{g}"] = int(m.sum())
    m_int = elig & ((lab=="intermittent")|(lab=="lumpy"))
    row["pooled_intermittent_lumpy"] = prb(p, m_int)
    # 지시문 층화배분 가중 (유형당 750, 부족하면 전부)
    w = np.zeros(len(test))
    for g in ["smooth","erratic","intermittent","lumpy"]:
        m = elig & (lab==g); n = m.sum()
        if n: w[m] = min(750, n)/n
    row["spec_stratified_750"] = prb(p, np.ones(len(test),bool), w)
    res[name] = row
    log(f"  {name:15s} all={row['pooled_all']:+.4f}  int+lumpy={row['pooled_intermittent_lumpy']:+.4f}  spec750={row['spec_stratified_750']:+.4f}")

# ---------------- STEP 5 ----------------
log("[STEP 5] 판정")
a,b = res["A_spec_const"]["pooled_all"], res["B_spec_linear"]["pooled_all"]
verdict = {
 "tail_rule_sensitivity_pp": round((b-a)*100, 3),
 "A_spec_const_PRB": a, "B_spec_linear_PRB": b,
 "C_train_const_PRB": res["C_train_const"]["pooled_all"],
 "int_lumpy_vs_all_gap_pp": round((res["A_spec_const"]["pooled_intermittent_lumpy"]-a)*100,3),
 "spec750_vs_all_gap_pp": round((res["A_spec_const"]["spec_stratified_750"]-a)*100,3),
 "all_QMEAN_PRB_negative": bool(a<0 and b<0 and res["C_train_const"]["pooled_all"]<0),
 "snaive_PRB": res["SNAIVE_week"]["pooled_all"],
 "negative_quantile_share": neg, "infer_s": infer_s, "peak_gb": peak,
 "train_quantile_levels": qtrain, "interpolated_spec_levels": outside,
 "context": CONTEXT, "n_eligible": int(elig.sum()),
}
for k,v in verdict.items():
    if not isinstance(v,list): log(f"  {k} = {v}")
json.dump({"verdict":verdict,"prb":res,
           "sbc_counts":{m:{k:int(((labels[m]==k)&elig).sum()) for k in ["smooth","erratic","intermittent","lumpy"]} for m in labels}},
          open(OUT/"precheck_a.json","w"), indent=2)
np.savez_compressed(OUT/"point_forecasts.npz", **{k:v.astype(np.float32) for k,v in variants.items()})
log(f"[DONE] {OUT/'precheck_a.json'}")
