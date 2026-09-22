#!/usr/bin/env bash
# V0a: 논문 환경에서 run_raf_instancenorm.py 재현. 환경구축 -> 실행 -> 대조 -> 판정까지 이어진다.
set -o pipefail
R=/home/minjae/Documents/github/tsfm-peft-method-screen
E=$R/external/icdm-2026-reproduction
OUT=$R/results/service_axis_v2_20260922
LOG=$OUT/v0a_log.txt
exec > >(tee -a "$LOG") 2>&1
echo "=== V0a 시작 $(date +%H:%M:%S) ==="

echo "--- STEP 1: uv sync --locked (main env) ---"
cd $E/src || exit 21
timeout 2700 uv sync --locked || { echo "FAIL: uv sync 실패"; exit 22; }
echo "STEP 1 완료 $(date +%H:%M:%S)"

echo "--- STEP 2: run_raf_instancenorm.py ---"
timeout 5400 uv run python scripts/10_experiments/run_raf_instancenorm.py || { echo "FAIL: 스크립트 실패"; exit 23; }
RES=$E/src/results/foundation_finetuning/2026-04-13_raf_instancenorm/results.csv
[ -f "$RES" ] || { echo "FAIL: 산출물 없음 $RES"; exit 24; }
echo "STEP 2 완료 $(date +%H:%M:%S)"

echo "--- STEP 3+4: 고정 참조와 대조 및 판정 ---"
$R/.venv/bin/python - "$RES" "$E/results_release/raf_instancenorm/raf_instancenorm_fill.csv" "$OUT/V0A_REPRO.json" <<'PY'
import sys, json, pandas as pd
got=pd.read_csv(sys.argv[1]); ref=pd.read_csv(sys.argv[2])
key=["config","dataset"]; cof=[c for c in ref.columns if c.startswith("cofr_")]
m=ref.merge(got,on=key,suffixes=("_ref","_got"))
if len(m)!=len(ref):
    print(f"FAIL: 행 매칭 {len(m)}/{len(ref)}"); json.dump({"pass":False,"reason":"row mismatch"},open(sys.argv[3],"w")); sys.exit(25)
rows=[]; worst=0.0
for _,r in m.iterrows():
    for c in cof:
        d=abs(r[f"{c}_ref"]-r[f"{c}_got"])*100
        worst=max(worst,d)
        rows.append({"config":r["config"],"metric":c,"ref":r[f"{c}_ref"],"got":r[f"{c}_got"],"diff_pp":d})
    dm=abs(r["mase_ref"]-r["mase_got"]); rows.append({"config":r["config"],"metric":"mase","ref":r["mase_ref"],"got":r["mase_got"],"diff_pp":dm})
ok = worst <= 0.5
print(f"\n{'config':14s}{'metric':12s}{'ref':>12s}{'got':>12s}{'diff(pp)':>10s}")
for x in rows: print(f"{x['config']:14s}{x['metric']:12s}{x['ref']:>12.6f}{x['got']:>12.6f}{x['diff_pp']:>10.4f}")
print(f"\n최대 충족 차이 {worst:.4f}pp  (기준 <= 0.5pp)")
print(">>> V0a", "PASS" if ok else "FAIL")
json.dump({"pass":bool(ok),"max_cofr_diff_pp":float(worst),"rows":rows},open(sys.argv[3],"w"),indent=2)
sys.exit(0 if ok else 26)
PY
rc=$?
echo "=== V0a 종료 $(date +%H:%M:%S) rc=$rc ==="
exit $rc
