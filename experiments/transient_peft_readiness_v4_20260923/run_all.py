"""v4 전체 검사 실행기. 사용: python run_all.py [--help]"""
import argparse, subprocess, sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
TESTS = ["test_module_scope.py","test_multitask_active_mod.py","test_neutral_tokens.py",
         "test_loss_contract.py","test_optimizer_updates.py","test_full_roundtrip.py"]
def main():
    ap = argparse.ArgumentParser(description="v4 readiness 검사 전체 실행 (배선 검사, 연구 fit 아님)")
    ap.add_argument("--only", nargs="*", help="일부만 실행")
    a = ap.parse_args()
    ts = a.only or TESTS
    rc = {}
    for t in ts:
        r = subprocess.run([sys.executable, str(HERE/t)], cwd=str(HERE))
        rc[t] = r.returncode
        print(f"[{'PASS' if r.returncode==0 else 'FAIL'}] {t}")
    return 0 if all(v==0 for v in rc.values()) else 1
if __name__ == "__main__":
    sys.exit(main())
