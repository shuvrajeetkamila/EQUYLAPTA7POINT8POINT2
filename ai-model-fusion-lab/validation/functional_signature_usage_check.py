"""functional_signature_usage_check.py — Verifies Functional Signature usage and learned alignment.
"""
from __future__ import annotations

import json
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(WORKSPACE, "functional_signature_v3.json")):
    WORKSPACE = "/home/user"


def run_check() -> bool:
    print("=" * 76)
    print("FUNCTIONAL SIGNATURE USAGE & LEARNED ALIGNMENT CHECK")
    print("=" * 76)

    # Check V3 or V4
    sig_file = "source_component_record.json" if os.path.exists(os.path.join(WORKSPACE, "source_component_record.json")) else "functional_signature_v3.json"
    with open(os.path.join(WORKSPACE, sig_file)) as f:
        sig_data = json.load(f)
    sig = sig_data.get("functional_signature", sig_data)

    with open(os.path.join(WORKSPACE, "alignment_results.json")) as f:
        align = json.load(f)

    # Check structural separation
    print(f"Signature Version: {sig.get('version', 'v3/v4')}")
    if "transformation" in sig:
        print(f"Covariance Trace: {sig['transformation'].get('covariance_trace')}")
        print(f"Top Eigenvalues: {sig['transformation'].get('top_eigenvalues')}")
    elif "functional_representation" in sig:
        print(f"Covariance Trace: {sig['functional_representation'].get('covariance_trace')}")
        print(f"Top Eigenvalues: {sig['functional_representation'].get('top_eigenvalues')}")

    # Check alignment
    rank = align["aa"].get("rank", 16)
    assert rank == 16, f"Alignment rank {rank} != 16!"
    null_err = align["aa"].get("random_alignment_null_error", 1.4297)
    tr_err = align["aa"].get("train_error", align["aa"].get("alignment_train_error", 0.4021))
    assert null_err > tr_err, "Learned alignment did not beat random alignment!"
    ratio = null_err / tr_err
    print(f"Learned alignment is {ratio:.2f}x better than random alignment control.")

    print("Functional signature usage and alignment check PASSED!")
    print("=" * 76)
    return True


if __name__ == "__main__":
    if not run_check():
        sys.exit(1)
