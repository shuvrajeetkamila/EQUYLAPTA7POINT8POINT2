"""src/gradient_validation.py

Verifies that the analytical two-branch backpropagation gradients used in the actual training pipeline
strictly match numerical central finite differences across multiple parameter groups and locations:
- Input translator (W_in)
- Output translator (W_out)
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Any, List
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))
sys.path.append(os.path.join(WORKSPACE, "EQUYLAPTA7POINT8_1"))
sys.path.append(os.path.dirname(__file__))

from models.synthetic import load_model, MicroTransformer
try:
    from src.corrected_functional_objective import (
        compute_donor_causal_signature,
        compute_functional_loss,
        compute_two_branch_gradients
    )
except ImportError:
    from corrected_functional_objective import (
        compute_donor_causal_signature,
        compute_functional_loss,
        compute_two_branch_gradients
    )


def run_training_gradient_validation(epsilon: float = 1e-2,
                                     tolerance: float = 0.05,
                                     seed: int = 42) -> Dict[str, Any]:
    """Validates the exact two-branch gradient against central finite differences."""
    m_src = load_model("e4-math-4L")
    m_tgt = load_model("e6-base-4L")
    ids = np.array([[10, 25, 42, 88]], dtype=np.int64)

    # 1. Target causal effect from donor
    delta_target = compute_donor_causal_signature(m_src, ids, source_head=(0, 2))

    # 2. Bridge setup
    h_dim_src = m_src.spec.hidden // m_src.spec.heads  # 16
    d_src = m_src.spec.hidden                          # 64
    h_dim_tgt = m_tgt.spec.hidden // m_tgt.spec.heads  # 16
    d_tgt = m_tgt.spec.hidden                          # 96

    rng = np.random.default_rng(seed)
    W_comp = m_src.params["L0.attn.o.W"][2 * h_dim_src : 3 * h_dim_src, :].copy()  # [16, 64] FROZEN
    W_in = np.eye(h_dim_tgt, h_dim_src, dtype=np.float32) + rng.standard_normal((h_dim_tgt, h_dim_src)).astype(np.float32) * 0.01
    W_out = np.zeros((d_src, d_tgt), dtype=np.float32)
    W_out[:d_src, :d_src] = np.eye(d_src, dtype=np.float32)
    W_out += rng.standard_normal((d_src, d_tgt)).astype(np.float32) * 0.01

    def ablate_slot():
        m_tgt.params["L0.attn.o.W"][:h_dim_tgt, :] = 0.0

    def restore_slot(w_i=None, w_o=None):
        wi = W_in if w_i is None else w_i
        wo = W_out if w_o is None else w_o
        eff = (wi @ W_comp @ wo).astype(np.float32)
        m_tgt.params["L0.attn.o.W"][:h_dim_tgt, :] = eff

    def loss_fn(w_i, w_o):
        eff = (w_i @ W_comp @ w_o).astype(np.float32)
        m_tgt.params["L0.attn.o.W"][:h_dim_tgt, :] = eff
        m_tgt.head_mask = None
        l_int, _ = m_tgt.forward(ids)

        m_tgt.head_mask = np.ones((m_tgt.spec.layers, m_tgt.spec.heads), dtype=np.float32)
        m_tgt.head_mask[0, 0] = 0.0
        l_abl, _ = m_tgt.forward(ids)
        m_tgt.head_mask = None

        delta_recip = (l_int[0, -1, :] - l_abl[0, -1, :]).astype(np.float32)
        diff = delta_recip - delta_target
        return 0.5 * float(np.mean(diff ** 2))

    # Analytical Gradients
    restore_slot()
    loss, diff, l_int, l_abl = compute_functional_loss(m_tgt, ids, delta_target, ablate_slot, restore_slot)
    g_eff = compute_two_branch_gradients(m_tgt, ids, diff, l_int, l_abl, ablate_slot, restore_slot, head_slot=0)

    g_W_in_ana = g_eff @ (W_comp @ W_out).T
    g_W_out_ana = (W_in @ W_comp).T @ g_eff

    # Coordinate tests
    test_coords_in = [(0, 0), (2, 3), (7, 5), (10, 12), (15, 15)]
    test_coords_out = [(0, 0), (5, 8), (15, 20), (35, 50), (60, 90)]

    abs_errors = []
    rel_errors = []
    param_checks = []

    # Check W_in
    for i, j in test_coords_in:
        wp = W_in.copy(); wp[i, j] += epsilon
        wm = W_in.copy(); wm[i, j] -= epsilon
        g_fd = (loss_fn(wp, W_out) - loss_fn(wm, W_out)) / (2 * epsilon)
        g_ana = float(g_W_in_ana[i, j])
        abs_err = abs(g_fd - g_ana)
        rel_err = abs_err / (abs(g_fd) + abs(g_ana) + 1e-12)
        abs_errors.append(abs_err)
        rel_errors.append(rel_err)
        param_checks.append({
            "param": "W_in",
            "coord": [i, j],
            "g_fd": round(g_fd, 8),
            "g_ana": round(g_ana, 8),
            "abs_err": round(abs_err, 8),
            "rel_err": round(rel_err, 8)
        })

    # Check W_out
    for i, j in test_coords_out:
        wp = W_out.copy(); wp[i, j] += epsilon
        wm = W_out.copy(); wm[i, j] -= epsilon
        g_fd = (loss_fn(W_in, wp) - loss_fn(W_in, wm)) / (2 * epsilon)
        g_ana = float(g_W_out_ana[i, j])
        abs_err = abs(g_fd - g_ana)
        rel_err = abs_err / (abs(g_fd) + abs(g_ana) + 1e-12)
        abs_errors.append(abs_err)
        rel_errors.append(rel_err)
        param_checks.append({
            "param": "W_out",
            "coord": [i, j],
            "g_fd": round(g_fd, 8),
            "g_ana": round(g_ana, 8),
            "abs_err": round(abs_err, 8),
            "rel_err": round(rel_err, 8)
        })

    max_rel_error = float(np.max(rel_errors))
    mean_rel_error = float(np.mean(rel_errors))
    max_abs_error = float(np.max(abs_errors))
    mean_abs_error = float(np.mean(abs_errors))

    passed = max_rel_error < tolerance

    return {
        "status": "PASS" if passed else "FAIL",
        "epsilon": epsilon,
        "tolerance": tolerance,
        "total_parameters_tested": len(param_checks),
        "max_relative_error": round(max_rel_error, 8),
        "mean_relative_error": round(mean_rel_error, 8),
        "max_absolute_error": round(max_abs_error, 8),
        "mean_absolute_error": round(mean_abs_error, 8),
        "gradient_check_passed": bool(passed),
        "parameter_checks": param_checks
    }


if __name__ == "__main__":
    res = run_training_gradient_validation()
    print("Training Gradient Validation Result:")
    print(f"  Status: {res['status']}")
    print(f"  Max Relative Error: {res['max_relative_error']} (Tolerance: {res['tolerance']})")
    print(f"  Max Absolute Error: {res['max_absolute_error']}")
