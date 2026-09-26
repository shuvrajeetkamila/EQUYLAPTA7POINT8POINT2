"""validation/gradient_check.py

Mandatory Finite-Difference vs. Analytical Gradient Verification for EQUYLAPTA 7.8.
Verifies that the analytical two-branch functional effect gradients:
    L_func = 0.5 * mean( ((Y_intact(x, theta) - Y_ablated(x, theta)) - F_src(x))^2 )
strictly match numerical central finite differences across all adapter parameter groups with rel err < 0.05.
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(WORKSPACE, "ai-model-fusion-lab"))

from models.synthetic import load_model


def verify_functional_effect_gradient(epsilon: float = 1e-4, tolerance: float = 0.05) -> dict:
    model = load_model("e4-base-4L")
    spec = model.spec
    head_dim = spec.hidden // spec.heads
    d_comp = 64
    d_tgt = spec.hidden

    np.random.seed(42)
    W_in = np.random.randn(head_dim, head_dim).astype(np.float32) * 0.1
    W_comp = np.random.randn(head_dim, d_comp).astype(np.float32) * 0.1  # STRICTLY FROZEN
    W_out = np.random.randn(d_comp, d_tgt).astype(np.float32) * 0.1
    F_src = np.random.randn(spec.vocab).astype(np.float32)
    arr = np.array([[1, 2, 3, 4]], dtype=np.int64)

    def loss_fn(w_in: np.ndarray, w_out: np.ndarray) -> float:
        w_eff = (w_in @ W_comp @ w_out).astype(np.float32)
        model.params["L0.attn.o.W"][:head_dim, :] = w_eff
        model.head_mask = np.ones((spec.layers, spec.heads), dtype=np.float32)
        l_int, _ = model.forward(arr)
        model.head_mask[0, 0] = 0.0
        l_abl, _ = model.forward(arr)
        diff = (l_int[0, -1, :] - l_abl[0, -1, :]) - F_src
        return 0.5 * float(np.mean(diff ** 2))

    def analytical_grads(w_in: np.ndarray, w_out: np.ndarray):
        w_eff = (w_in @ W_comp @ w_out).astype(np.float32)
        model.params["L0.attn.o.W"][:head_dim, :] = w_eff
        model.head_mask = np.ones((spec.layers, spec.heads), dtype=np.float32)
        l_int, _ = model.forward(arr)
        model.head_mask[0, 0] = 0.0
        l_abl, _ = model.forward(arr)

        diff = (l_int[0, -1, :] - l_abl[0, -1, :]) - F_src
        dlogits = (diff / spec.vocab).astype(np.float32)

        # Intact backward
        model.head_mask = np.ones((spec.layers, spec.heads), dtype=np.float32)
        dint = np.zeros_like(l_int); dint[0, -1, :] = dlogits
        g_int = model.backward(arr, dint)

        # Ablated backward
        model.head_mask[0, 0] = 0.0
        dabl = np.zeros_like(l_abl); dabl[0, -1, :] = -dlogits
        g_abl = model.backward(arr, dabl)

        g_eff = g_int["L0.attn.o.W"][:head_dim, :] + g_abl["L0.attn.o.W"][:head_dim, :]
        dW_in = g_eff @ (W_comp @ w_out).T
        dW_out = (w_in @ W_comp).T @ g_eff
        return dW_in, dW_out

    dW_in_ana, dW_out_ana = analytical_grads(W_in, W_out)
    rel_errors = []

    # Check W_in
    coords_in = [(0, 0), (1, 2), (2, 3), (3, 1), (0, 3)]
    for i, j in coords_in:
        wp = W_in.copy(); wp[i, j] += epsilon
        wm = W_in.copy(); wm[i, j] -= epsilon
        num = (loss_fn(wp, W_out) - loss_fn(wm, W_out)) / (2 * epsilon)
        ana = float(dW_in_ana[i, j])
        err = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
        rel_errors.append(err)

    # Check W_out
    coords_out = [(0, 0), (5, 8), (12, 15), (20, 2), (30, 10)]
    for i, j in coords_out:
        wp = W_out.copy(); wp[i, j] += epsilon
        wm = W_out.copy(); wm[i, j] -= epsilon
        num = (loss_fn(W_in, wp) - loss_fn(W_in, wm)) / (2 * epsilon)
        ana = float(dW_out_ana[i, j])
        err = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
        rel_errors.append(err)

    max_rel_error = float(max(rel_errors))
    passed = bool(max_rel_error < tolerance)

    return {
        "status": "PASS" if passed else "FAIL",
        "max_relative_error": round(max_rel_error, 6),
        "tolerance": tolerance,
        "gradient_check_passed": passed
    }


if __name__ == "__main__":
    res = verify_functional_effect_gradient()
    print("Gradient Check Result:", res)
