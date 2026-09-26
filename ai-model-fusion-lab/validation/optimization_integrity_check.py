"""optimization_integrity_check.py — Verifies true mathematical gradients and optimization decrease.
"""
from __future__ import annotations

import json
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(WORKSPACE, "gradient_check.json")):
    WORKSPACE = "/home/user"


def run_check() -> bool:
    print("=" * 76)
    print("OPTIMIZATION INTEGRITY CHECK")
    print("=" * 76)

    gc_path = os.path.join(WORKSPACE, "gradient_check.json")
    opt_path = os.path.join(WORKSPACE, "optimization_results.json")

    assert os.path.exists(gc_path), f"Missing {gc_path}"
    assert os.path.exists(opt_path), f"Missing {opt_path}"

    with open(gc_path) as f:
        gc = json.load(f)
    with open(opt_path) as f:
        opt = json.load(f)

    print(f"Gradient check passed: {gc['passed']}")
    print(f"Max relative error: {gc['max_relative_error']:.6e} (tolerance: {gc['tolerance']})")
    print(f"Loss decreased: {opt['loss_decreased']} (initial: {opt['initial_loss']}, final: {opt['final_loss']})")

    assert gc["passed"] is True, "Gradient correctness test failed!"
    assert gc["max_relative_error"] < 0.05, f"Gradient relative error {gc['max_relative_error']} >= 0.05!"
    assert opt["loss_decreased"] is True, "Training loss did not decrease!"
    assert opt["final_loss"] < opt["initial_loss"], "Final loss not strictly less than initial loss!"

    print("Optimization integrity check PASSED!")
    print("=" * 76)
    return True


if __name__ == "__main__":
    if not run_check():
        sys.exit(1)
