"""빠른 과도응답이 30초 격자 아래에 있는지 확인한다.

testcase.py 의 저장 규칙: step >= 30 이면 ncp = step/30 (30초 해상도),
step < 30 이면 ncp = 1 이라 advance 당 1점, 즉 해상도 = step.
따라서 step=10 으로 두면 10초 해상도를 얻는다.
"""
import json, csv, sys
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8000"
OUT = Path("/home/minjae/Documents/github/tsfm-peft-method-screen/results/boptest_data_contract_20260924")
CASE = "bestest_hydronic_heat_pump"
STEP = 10.0
START = (23-1)*86400.0
WARMUP = 86400.0
MEAS = ["reaTSup_y","reaTRet_y","reaTZon_y","reaPHeaPum_y","reaQHeaPumCon_y","reaCOP_y"]
CTRL = ["oveHeaPumY_u","ovePum_u","oveFan_u"]

def req(method, path, payload=None, timeout=300):
    if payload is None and method in ("POST","PUT"): payload = {}
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(f"{URL}/{path}", data=data, method=method,
                               headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f: body = f.read().decode()
    try: return json.loads(body)
    except json.JSONDecodeError: return {"payload": body.strip(), "raw": True}

def main():
    tid = req("POST", f"testcases/{CASE}/select", timeout=120)["testid"]
    print("testid", tid, flush=True)
    try:
        req("PUT", f"step/{tid}", {"step": STEP})
        req("PUT", f"initialize/{tid}", {"start_time": START, "warmup_period": WARMUP})
        # OFF 5분 -> ON 20분 -> OFF 5분,  10초 간격
        sched = [0.0]*30 + [1.0]*120 + [0.0]*30
        for u in sched:
            req("POST", f"advance/{tid}", {
                "oveHeaPumY_activate":1, "oveHeaPumY_u":u,
                "ovePum_activate":1, "ovePum_u":1.0 if u>0 else 0.0,
                "oveFan_activate":1, "oveFan_u":1.0 if u>0 else 0.0})
        res = req("PUT", f"results/{tid}", {"point_names": MEAS+CTRL,
                  "start_time": START, "final_time": START+len(sched)*STEP+STEP})["payload"]
        n = len(res["time"]); dt = res["time"][1]-res["time"][0]
        print(f"{n} 점, 간격 {dt:.0f}s", flush=True)
        cols = ["time"]+[c for c in MEAS+CTRL if c in res]
        with open(OUT/"fast_probe.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(cols)
            for k in range(n): w.writerow([res[c][k] for c in cols])
        json.dump({"step_s":STEP,"sample_interval_s":dt,"n":n,
                   "schedule":"OFF 5분 -> ON 20분 -> OFF 5분","start_time_s":START,
                   "purpose":"빠른 과도응답이 30초 격자 아래인지 확인"},
                  open(OUT/"FAST_PROBE.json","w"), indent=2, ensure_ascii=False)
        # 전환 직후 출력
        import numpy as np
        t=np.array(res["time"]); t-=t[0]; u=np.array(res["oveHeaPumY_u"])
        i0=int(np.where(np.diff(u)>0.5)[0][0])+1
        print(f"\nON 전환 t={t[i0]:.0f}s 직후 ({dt:.0f}초 간격)", flush=True)
        print(f"{'경과':>7} {'reaTSup_y':>11} {'reaPHeaPum_y':>13} {'reaQHeaPumCon_y':>16}")
        for k in range(i0-2, min(i0+25, n)):
            print(f"{t[k]-t[i0]:7.0f} {res['reaTSup_y'][k]:11.3f} "
                  f"{res['reaPHeaPum_y'][k]:13.1f} {res['reaQHeaPumCon_y'][k]:16.1f}")
        return True
    finally:
        try: req("PUT", f"stop/{tid}", timeout=120); print("\nworker 반납", flush=True)
        except Exception as e: print("stop 실패", e, flush=True)

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
