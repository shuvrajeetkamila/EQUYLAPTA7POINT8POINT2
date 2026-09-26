"""self_check.py — Master orchestrator running all EQUYLAPTA7.2 validation suites.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(WORKSPACE, "results_e7_2.json")):
    WORKSPACE = "/home/user"


def run_all_checks() -> bool:
    print("=" * 76)
    print("EQUYLAPTA7.2 MASTER SELF-CHECK SUITE")
    print("=" * 76)

    scripts = [
        "semantic_report_validator.py",
        "report_integrity_check.py",
        "optimization_integrity_check.py",
        "reconstruction_independence_check.py",
        "functional_signature_usage_check.py"
    ]

    val_dir = os.path.join(WORKSPACE, "validation")
    all_passed = True

    for s in scripts:
        sp = os.path.join(val_dir, s)
        if not os.path.exists(sp):
            sp = os.path.join(WORKSPACE, s)
        print(f"\n--> Running {s}...")
        res = subprocess.run([sys.executable, sp], cwd=WORKSPACE, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr, file=sys.stderr)
            print(f"FAILED: {s} returned code {res.returncode}", file=sys.stderr)
            all_passed = False
            break

    if not all_passed:
        print("MASTER SELF-CHECK FAILED!", file=sys.stderr)
        return False

    # Check workspace size
    with open(os.path.join(WORKSPACE, "size_report.json")) as f:
        sz = json.load(f)

    print(f"\nWorkspace Size Verification: {sz['total_mb']} MB (Hard limit: {sz['hard_limit_mb']} MB)")
    assert sz["within_limit"] is True, f"Workspace size {sz['total_mb']} MB exceeds hard limit!"
    assert sz["total_mb"] < 120.0, "Hard limit exceeded!"

    print("\n" + "=" * 76)
    print("ALL EQUYLAPTA7.2 CHECKS PASSED PERFECTLY!")
    print("=" * 76)
    return True


if __name__ == "__main__":
    if not run_all_checks():
        sys.exit(1)
