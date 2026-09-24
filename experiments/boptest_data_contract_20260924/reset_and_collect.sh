#!/usr/bin/env bash
# 큐에 남은 유령 select 작업까지 비우고(=down/up) 과도응답 수집까지 한 번에 간다.
# worker 재시작만으로는 큐가 안 비워진다 — 재시작 직후 유령 작업을 다시 집어 점유한다 (실측).
set -u
B=/home/minjae/Documents/github/project1-boptest
R=/home/minjae/Documents/github/tsfm-peft-method-screen
L=$R/results/boptest_data_contract_20260924/raw_logs; mkdir -p $L

echo "[A] 전체 종료 (redis 영속 볼륨 없음 -> 큐 소멸)  $(date +%H:%M:%S)"
cd $B && docker compose down > $L/reset.txt 2>&1
echo "[B] 재기동"
docker compose up -d web worker provision >> $L/reset.txt 2>&1 || { echo "[실패] up"; tail -15 $L/reset.txt; exit 1; }

echo "[C] 준비 대기 (최대 5분)"
ok=0
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:8000/testcases 2>/dev/null)
  [ "$code" = "200" ] && { ok=1; echo "   준비됨 ($((i*5))초)"; break; }
  sleep 5
done
[ $ok -ne 1 ] && { echo "[실패] 준비 안 됨"; docker compose logs --tail 20 web >> $L/reset.txt 2>&1; exit 1; }

echo "[D] worker 유휴 확인 — select 가 빨리 오는지"
t0=$(date +%s)
tid=$(curl -s -m 60 -X POST -H "Content-Type: application/json" -d '{}' \
      http://127.0.0.1:8000/testcases/bestest_hydronic_heat_pump/select | python3 -c "import sys,json;print(json.load(sys.stdin).get('testid',''))" 2>/dev/null)
t1=$(date +%s)
if [ -z "$tid" ]; then echo "[실패] select 무응답 ($((t1-t0))초) — 큐가 여전히 막혀 있다"; exit 1; fi
echo "   select $((t1-t0))초, testid $tid — 즉시 반납"
curl -s -m 60 -X PUT -H "Content-Type: application/json" -d '{}' http://127.0.0.1:8000/stop/$tid > /dev/null

echo "[E] 과도응답 수집  $(date +%H:%M:%S)"
$R/.venv/bin/python $R/experiments/boptest_data_contract_20260924/collect_transient.py
rc=$?
echo "[F] 판정  $(date +%H:%M:%S)  collect exit=$rc"
CSV=$R/results/boptest_data_contract_20260924/transient_response.csv
if [ -f "$CSV" ]; then echo "   ==> COLLECTED  $(wc -l < $CSV) 행"; else echo "   ==> 산출물 없음"; exit 1; fi
