#!/usr/bin/env bash
# BOPTEST 로컬 기동 → 준비 대기 → testcase 목록 수집 → 판정까지 한 스크립트에서 끝낸다.
# 완료 판정은 종료코드가 아니라 산출물(TESTCASES.json)로 한다.
set -u
B=/home/minjae/Documents/github/project1-boptest
O=/home/minjae/Documents/github/tsfm-peft-method-screen/results/boptest_data_contract_20260924
L=$O/raw_logs; mkdir -p $L
URL=http://127.0.0.1:8000

echo "[STEP 1] 빌드·기동  $(date +%H:%M:%S)"
cd $B || exit 1
docker compose up -d --build web worker provision > $L/compose_up.txt 2>&1
rc=$?
echo "  compose exit=$rc"
if [ $rc -ne 0 ]; then
  echo "[실패] 빌드/기동 단계. 마지막 20줄:"; tail -20 $L/compose_up.txt; exit 1
fi

echo "[STEP 2] 서비스 준비 대기 (최대 20분)  $(date +%H:%M:%S)"
ok=0
for i in $(seq 1 240); do
  code=$(curl -s -o $L/testcases_body.json -w "%{http_code}" --max-time 10 $URL/testcases 2>/dev/null)
  if [ "$code" = "200" ]; then ok=1; echo "  준비됨 ($((i*5))초)"; break; fi
  sleep 5
done
if [ $ok -ne 1 ]; then
  echo "[실패] 20분 내 200 응답 없음. 마지막 code=$code"
  docker compose ps > $L/ps.txt 2>&1; docker compose logs --tail 40 web worker >> $L/ps.txt 2>&1
  tail -30 $L/ps.txt; exit 1
fi

echo "[STEP 3] testcase 목록 저장"
cp $L/testcases_body.json $O/TESTCASES.json
python3 - <<'PY'
import json
d = json.load(open("/home/minjae/Documents/github/tsfm-peft-method-screen/results/boptest_data_contract_20260924/TESTCASES.json"))
items = d if isinstance(d, list) else d.get("payload", d)
names = [t.get("testcaseid", t) if isinstance(t, dict) else t for t in items] if isinstance(items, list) else list(items)
print("  testcase 수:", len(names))
for n in names: print("   -", n)
print("  bestest_hydronic_heat_pump 존재:", any("bestest_hydronic_heat_pump" == str(n) for n in names))
PY

echo "[STEP 4] 판정  $(date +%H:%M:%S)"
docker compose ps > $L/ps.txt 2>&1
echo "  ==> SERVICE_UP. TESTCASES.json 저장 완료"
