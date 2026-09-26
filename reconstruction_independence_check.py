"""reconstruction_independence_check.py — Verifies true independence of target reconstructions.
"""
from __future__ import annotations

import json
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(WORKSPACE, "aa_results.json")):
    WORKSPACE = "/home/user"


def run_check() -> bool:
    print("=" * 76)
    print("RECONSTRUCTION INDEPENDENCE CHECK")
    print("=" * 76)

    with open(os.path.join(WORKSPACE, "aa_results.json")) as f:
        aa = json.load(f)
    with open(os.path.join(WORKSPACE, "ab_results.json")) as f:
        ab = json.load(f)

    # Check A->A runs
    aa_runs = aa["runs"]
    assert len(aa_runs) >= 5, f"Fewer than 5 A->A runs: {len(aa_runs)}"
    aa_inits = [r["target_initialization_hash"] for r in aa_runs]
    assert len(set(aa_inits)) == len(aa_inits), f"A->A initialization hashes not distinct: {aa_inits}"
    print(f"Verified {len(aa_inits)} distinct A->A initialization hashes.")

    # Check A->B runs
    ab_runs = ab["method_2_functional_signature"]["runs"]
    assert len(ab_runs) >= 5, f"Fewer than 5 A->B runs: {len(ab_runs)}"
    ab_inits = [r["target_initialization_hash"] for r in ab_runs]
    assert len(set(ab_inits)) == len(ab_inits), f"A->B initialization hashes not distinct: {ab_inits}"
    print(f"Verified {len(ab_inits)} distinct A->B initialization hashes.")

    # Check parameter distance
    assert aa["parameter_relative_distance"] > 0.0, "Parameter distance is zero (identical weights)!"
    print(f"A->A Parameter relative distance: {aa['parameter_relative_distance']:.4f}")
    print(f"A->A Linear CKA: {aa['linear_cka']:.4f}")
    print(f"A->A Behavioral Agreement: {aa['behavioral_agreement_pct']:.2f}%")

    print("Reconstruction independence check PASSED!")
    print("=" * 76)
    return True


if __name__ == "__main__":
    if not run_check():
        sys.exit(1)
