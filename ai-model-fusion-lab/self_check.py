"""self_check.py — Comprehensive Self-Audit for EQUYLAPTA7.

Performs 6-point validation:
  1. Data Integrity (JSON schemas, numeric validity, no NaN/inf, sample counts)
  2. Experiment Integrity (all required experimental sweeps, arms, and controls present)
  3. Report Numerical Integrity (every headline number matches machine-readable keys)
  4. Semantic & Logical Integrity (verifies comparisons, deltas, convergence, and directions)
  5. Artifact Integrity (workspace total < 120 MB, target 20-50 MB)
  6. Scientific Language Integrity (strictly prohibits unsupported inflated claims)
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Optional

CURR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURR) == "ai-model-fusion-lab":
    ROOT = CURR
    WORKSPACE = os.path.abspath(os.path.join(ROOT, os.pardir))
else:
    WORKSPACE = CURR
    ROOT = os.path.join(WORKSPACE, "ai-model-fusion-lab")

sys.path.insert(0, ROOT)
sys.path.insert(0, WORKSPACE)


def check_no_nan_or_inf(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            check_no_nan_or_inf(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_no_nan_or_inf(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise AssertionError(f"NaN or Inf found at path: {path}")


def run_full_self_check(results: Optional[dict] = None) -> bool:
    print("Executing Comprehensive EQUYLAPTA7 Self-Audit...")

    # 1. Load results if not passed
    if results is None:
        rpath = os.path.join(WORKSPACE, "results.json")
        if not os.path.exists(rpath):
            rpath = os.path.join(ROOT, "results.json")
        if not os.path.exists(rpath):
            raise FileNotFoundError(f"Missing results.json at {rpath}")
        with open(rpath) as f:
            results = json.load(f)

    # --- POINT 1: DATA INTEGRITY ---
    print("  [1/6] Checking Data Integrity...")
    check_no_nan_or_inf(results)
    mandatory_root_keys = [
        "milestone", "timestamp", "arch_a", "arch_b", "arch_c",
        "hierarchical_discovery", "causal_discovery", "minimality",
        "redundancy_and_synergy", "compositionality", "functional_signature",
        "three_reconstruction_methods", "target_controls",
        "independent_reconstructions", "depth_sweep", "capability_specificity",
        "information_budget", "leakage_audit", "evidence_level", "final_status",
        "twenty_questions"
    ]
    for k in mandatory_root_keys:
        assert k in results, f"Missing mandatory root key: {k}"

    assert results["causal_discovery"]["baseline"]["n_seeds"] >= 3
    assert results["independent_reconstructions"]["n_reconstructions"] >= 3
    print("      Data integrity verified: clean numerics, valid counts, 0 NaNs.")

    # --- POINT 2: EXPERIMENT INTEGRITY ---
    print("  [2/6] Checking Experiment Integrity...")
    # Depth sweep check
    assert all(d in results["depth_sweep"]["sweep"] for d in ["2L", "4L", "6L", "8L"]), "Depth sweep missing 2L, 4L, 6L, or 8L"
    # Controls check (7 controls)
    assert len(results["target_controls"]) >= 7, "Target controls battery has fewer than 7 controls"
    # Three reconstruction methods check
    assert "method_a_structural" in results["three_reconstruction_methods"]
    assert "method_b_subspace" in results["three_reconstruction_methods"]
    assert "method_c_distillation" in results["three_reconstruction_methods"]
    # Minimality check
    assert len(results["minimality"]["curve"]) >= 5
    print("      Experiment integrity verified: depth sweep, 7 controls, 3 methods, and minimality confirmed.")

    # --- POINT 3: REPORT NUMERICAL INTEGRITY ---
    print("  [3/6] Checking Report Numerical Integrity...")
    from report_integrity_check import verify_report_integrity
    verify_report_integrity(results)

    # --- POINT 4: SEMANTIC & LOGICAL INTEGRITY ---
    print("  [4/6] Checking Semantic & Logical Integrity...")
    from semantic_report_validator import run_semantic_validation
    run_semantic_validation(results)

    # --- POINT 5: ARTIFACT INTEGRITY (<120 MB) ---
    print("  [5/6] Checking Artifact Integrity (<120 MB)...")
    def get_dir_size(path):
        tot = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    tot += os.path.getsize(fp)
        return tot

    raw_mb = (get_dir_size(ROOT) + get_dir_size(os.path.join(WORKSPACE, "fusionlab_data"))) / (1024 * 1024)
    ws_mb = get_dir_size(WORKSPACE) / (1024 * 1024)
    print(f"      Workspace size: {ws_mb:.2f} MB (Hard limit: 120.00 MB, Target: 20-50 MB)")
    assert ws_mb < 120.0, f"Workspace exceeds 120 MB! ({ws_mb:.2f} MB)"

    # --- POINT 6: SCIENTIFIC INTEGRITY (LANGUAGE RULES) ---
    print("  [6/6] Checking Scientific Language Integrity...")
    report_path = os.path.join(WORKSPACE, "EQUYLAPTA7_REPORT.md")
    with open(report_path) as f:
        rep_text = f.read().lower()

    prohibited_phrases = [
        "proved transferable",
        "universal component",
        "general intelligence module",
        "model-independent intelligence",
        "portable reasoning"
    ]
    for ph in prohibited_phrases:
        assert ph not in rep_text, f"Prohibited unscientific phrase detected: '{ph}'"

    print("      Scientific language verified: zero prohibited inflated claims.")
    print("\nALL 6 SELF-CHECK AUDIT DIMENSIONS PASSED SUCCESSFULLY!")
    return True


if __name__ == "__main__":
    run_full_self_check()
