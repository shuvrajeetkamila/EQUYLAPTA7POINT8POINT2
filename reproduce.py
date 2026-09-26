"""reproduce.py — One-command reproduction script for EQUYLAPTA E7.6.

Supports:
  --mode smoke : Fast deterministic verification run (< 2s)
  --mode full  : Full multi-seed multi-scale cross-family fusion pipeline
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

WORKSPACE = os.path.dirname(os.path.abspath(__file__))


def run():
    parser = argparse.ArgumentParser(description="EQUYLAPTA E7.6 Reproduction Harness")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke",
                        help="Execution mode: 'smoke' for fast test or 'full' for full suite")
    args = parser.parse_args()

    print("=" * 78)
    print(f"EQUYLAPTA E7.6 REPRODUCTION HARNESS [MODE: {args.mode.upper()}]")
    print("=" * 78)
    t0 = time.time()

    if args.mode == "smoke":
        print("\n[1/3] Running Consistency Check & Validation...")
        validator_script = os.path.join(WORKSPACE, "validation", "report_consistency_validator.py")
        res = subprocess.run([sys.executable, validator_script], cwd=WORKSPACE, text=True)
        if res.returncode != 0:
            print("ERROR: Report consistency audit failed!", file=sys.stderr)
            sys.exit(res.returncode)

        print("\n[2/3] Verifying Canonical E7.6 Artifacts...")
        req_artifacts = [
            "e7.5_audit.json",
            "EQUYLAPTA_E7.5_AUDIT.md",
            "report_consistency_check.json",
            "architecture_family_report.json",
            "component_genome.json",
            "component_discovery_results.json",
            "component_causal_results.json",
            "functional_translation_results.json",
            "functional_composition_results.json",
            "interaction_matrix.json",
            "ablation_results.json",
            "restoration_results.json",
            "random_control_results.json",
            "host_learning_control.json",
            "depth_sweep_results.json",
            "seed_results.json",
            "claim_provenance.json",
            "e7_6_results.json",
        ]
        missing = [f for f in req_artifacts if not os.path.exists(os.path.join(WORKSPACE, f))]
        if missing:
            print(f"ERROR: Missing canonical artifacts: {missing}", file=sys.stderr)
            sys.exit(1)
        print(f"All {len(req_artifacts)} canonical artifacts verified present.")

        print("\n[3/3] Regenerating EQUYLAPTA E7.6 Reports...")
        rep_script = os.path.join(WORKSPACE, "scripts", "generate_e7_6_reports.py")
        res = subprocess.run([sys.executable, rep_script], cwd=WORKSPACE, text=True)
        if res.returncode != 0:
            print("ERROR: Report generation failed!", file=sys.stderr)
            sys.exit(res.returncode)

    else:
        # Full mode
        print("\n[1/3] Executing EQUYLAPTA 7.6 Master Experimental Pipeline...")
        exp_script = os.path.join(WORKSPACE, "demo", "run_equylapta_e7_6.py")
        res = subprocess.run([sys.executable, exp_script], cwd=WORKSPACE, text=True)
        if res.returncode != 0:
            print("ERROR: Master experiment failed!", file=sys.stderr)
            sys.exit(res.returncode)

        print("\n[2/3] Executing Consistency Audit Validator...")
        validator_script = os.path.join(WORKSPACE, "validation", "report_consistency_validator.py")
        res = subprocess.run([sys.executable, validator_script], cwd=WORKSPACE, text=True)
        if res.returncode != 0:
            print("ERROR: Consistency validator failed!", file=sys.stderr)
            sys.exit(res.returncode)

        print("\n[3/3] Regenerating EQUYLAPTA E7.6 Reports...")
        rep_script = os.path.join(WORKSPACE, "scripts", "generate_e7_6_reports.py")
        res = subprocess.run([sys.executable, rep_script], cwd=WORKSPACE, text=True)
        if res.returncode != 0:
            print("ERROR: Report generation failed!", file=sys.stderr)
            sys.exit(res.returncode)

    elapsed = time.time() - t0
    print("\n" + "=" * 78)
    print(f"REPRODUCTION COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    print(f"Reports: {os.path.join(WORKSPACE, 'EQUYLAPTA_E7.6_REPORT.md')}")
    print(f"         {os.path.join(WORKSPACE, 'EQUYLAPTA_E7.6_REPORT.txt')}")
    print("=" * 78)


if __name__ == "__main__":
    run()
