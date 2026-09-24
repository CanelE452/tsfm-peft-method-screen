"""과도응답 수집 — bestest_hydronic_heat_pump 에 계단 명령을 넣고 응답을 받는다.

목적: NEXT_STAGE_CONTRACT §2 의 빈 칸 중 "과도응답의 물리적 정의" 와 tau0 추정 근거를 만든다.
성능 실험이 아니라 difficulty 전제 확인이다. testcase 1개, 모의 시간 하루 미만.
"""
import json, sys, time, csv
from pathlib import Path
import urllib.request

URL = "http://127.0.0.1:8000"
OUT = Path("/home/minjae/Documents/github/tsfm-peft-method-screen/results/boptest_data_contract_20260924")
CASE = "bestest_hydronic_heat_pump"
STEP = 300.0                      # 제어 주기 5분. 결과는 30초 간격으로 저장된다
PEAK_HEAT_DAY = 23                # days.json
START = (PEAK_HEAT_DAY - 1) * 86400.0
WARMUP = 86400.0

# measurements 에 있는 이름만 쓴다. ove*_y 는 measurements 에 없고, 제어 신호는
# u_store 에 _u 이름으로 저장된다 (testcase.py __get_results: key[:-2]+"_y" 로 값을 받아 _u 키에 넣음).
MEAS = ["reaTZon_y", "reaTSup_y", "reaTRet_y", "reaPHeaPum_y", "reaQHeaPumCon_y",
        "reaQHeaPumEva_y", "reaQFloHea_y", "reaPPumEmi_y", "reaPFan_y", "reaCOP_y",
        "reaTSetHea_y", "weaSta_reaWeaTDryBul_y"]
CTRL = ["oveHeaPumY_u", "ovePum_u", "oveFan_u"]

def req(method, path, payload=None, timeout=600):
    # 본문 없이 application/json 을 보내면 서버가 빈 문자열을 JSON.parse 하다 500 을 낸다.
    # POST/PUT 은 항상 최소 {} 를 보낸다.
    if payload is None and method in ("POST", "PUT"): payload = {}
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(f"{URL}/{path}", data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        body = f.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"payload": body.strip(), "raw": True}   # /stop 은 "OK" 평문을 준다

def main():
    # worker 는 1개고 실행 중 테스트를 조회하는 API 가 없다. 클라이언트가 죽으면 그 테스트가
    # worker 를 계속 잡아 다음 select 가 큐에서 대기하다 타임아웃된다 (2회 겪음).
    # 대응: select 타임아웃을 짧게 둬 빨리 실패하고, 아래 finally 에서 반드시 반납한다.
    print("[1] testcase 선택", flush=True)
    sel = req("POST", f"testcases/{CASE}/select", timeout=120)
    testid = sel["testid"] if "testid" in sel else sel["payload"]["testid"]
    print("   testid", testid, flush=True)

    try:
        return run(testid)
    finally:
        try:
            req("PUT", f"stop/{testid}", timeout=120)   # 어떤 경로로 끝나든 worker 반납
            print("   worker 반납 완료", flush=True)
        except Exception as e:
            print(f"   [경고] stop 실패: {e}", flush=True)

def run(testid):
    print("[2] 신호 메타데이터 (API 기준)", flush=True)
    inputs = req("GET", f"inputs/{testid}")["payload"]
    meas   = req("GET", f"measurements/{testid}")["payload"]
    json.dump({"inputs": inputs, "measurements": meas},
              open(OUT/"API_SIGNAL_METADATA.json","w"), indent=2, ensure_ascii=False)
    print(f"   inputs {len(inputs)} / measurements {len(meas)}", flush=True)

    print("[3] step / initialize", flush=True)
    req("PUT", f"step/{testid}", {"step": STEP})
    req("PUT", f"initialize/{testid}", {"start_time": START, "warmup_period": WARMUP})

    # 명령 일정: 안정 -> ON -> 안정 -> ON -> 안정  (전환 4회)
    hold_off, hold_on = 24, 48        # 5분 x 24 = 2시간 / x48 = 4시간
    schedule = ([0.0]*hold_off + [1.0]*hold_on) * 2 + [0.0]*hold_off
    print(f"[4] advance {len(schedule)} step = 모의 {len(schedule)*STEP/3600:.1f}시간", flush=True)

    t0 = time.time()
    for i, u in enumerate(schedule):
        req("POST", f"advance/{testid}", {
            "oveHeaPumY_activate": 1, "oveHeaPumY_u": u,
            "ovePum_activate": 1,     "ovePum_u": 1.0 if u > 0 else 0.0,
            "oveFan_activate": 1,     "oveFan_u": 1.0 if u > 0 else 0.0})
        if (i+1) % 24 == 0:
            print(f"   {i+1}/{len(schedule)}  {time.time()-t0:.0f}s", flush=True)

    print("[5] 결과 수집 (30초 해상도)", flush=True)
    final = START + WARMUP*0 + len(schedule)*STEP
    res = req("PUT", f"results/{testid}", {
        "point_names": MEAS + CTRL,
        "start_time": START, "final_time": START + len(schedule)*STEP + STEP})["payload"]
    n = len(res["time"])
    print(f"   {n} 점, 간격 {res['time'][1]-res['time'][0]:.0f}s", flush=True)

    cols = ["time"] + [m for m in (MEAS + CTRL) if m in res]
    missing = [m for m in (MEAS + CTRL) if m not in res]
    if missing: print(f"   [주의] 응답에 없는 point: {missing}", flush=True)
    print(f"   시각 범위 {res['time'][0]:.0f} ~ {res['time'][-1]:.0f}s "
          f"(= {(res['time'][-1]-res['time'][0])/3600:.2f}시간)", flush=True)
    with open(OUT/"transient_response.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for k in range(n): w.writerow([res[c][k] for c in cols])

    json.dump({"testcase": CASE, "testid": testid, "step_s": STEP,
               "start_time_s": START, "warmup_s": WARMUP, "peak_heat_day": PEAK_HEAT_DAY,
               "schedule": {"hold_off_steps": hold_off, "hold_on_steps": hold_on,
                            "pattern": "off-on-off-on-off", "n_steps": len(schedule),
                            "n_transitions": 4},
               "commanded": ["oveHeaPumY", "ovePum", "oveFan"],
               "points": cols, "n_samples": n,
               "sample_interval_s": res["time"][1]-res["time"][0],
               "elapsed_wall_s": round(time.time()-t0,1)},
              open(OUT/"TRANSIENT_RUN.json","w"), indent=2, ensure_ascii=False)
    print("   ==> transient_response.csv 저장", flush=True)
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
